from datetime import datetime, timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from .forms import LABORATORY_PARAMETERS, ManualBloodTestForm
from .models import BloodTest, BloodTestResult
from .services import ReferenceAnalysisService


@login_required
def manual_blood_test_create(request):
    if not request.user.profile_is_complete:
        messages.warning(
            request,
            "Prije unosa nalaza dovršite profil kako bi se dob i spol ispravno spremili uz nalaz.",
        )
        return redirect("profile")

    if request.method == "POST":
        form = ManualBloodTestForm(request.POST)
        if form.is_valid():
            with transaction.atomic():
                blood_test = BloodTest.objects.create(
                    user=request.user,
                    sampling_date=form.cleaned_data["sampling_date"],
                    input_method=BloodTest.InputMethod.MANUAL,
                    processing_status=BloodTest.ProcessingStatus.COMPLETED,
                )
                results = [
                    BloodTestResult(blood_test=blood_test, **parameter_data)
                    for parameter_data in form.iter_parameter_data()
                ]
                for result in results:
                    ReferenceAnalysisService.analyze_result(result, save=False)
                BloodTestResult.objects.bulk_create(results)

            messages.success(request, "Krvni nalaz uspješno je spremljen.")
            return redirect("laboratory:detail", pk=blood_test.pk)
    else:
        form = ManualBloodTestForm()

    groups = []
    for group in LABORATORY_PARAMETERS:
        parameters = []
        for code, name, default_unit in group["parameters"]:
            parameters.append(
                {
                    "code": code,
                    "name": name,
                    "default_unit": default_unit,
                    "value_field": form[f"value_{code}"],
                    "unit_field": form[f"unit_{code}"],
                    "reference_min_field": form[f"reference_min_{code}"],
                    "reference_max_field": form[f"reference_max_{code}"],
                }
            )
        groups.append({"name": group["group"], "parameters": parameters})

    return render(
        request,
        "laboratory/manual_blood_test_form.html",
        {"form": form, "parameter_groups": groups},
    )


@login_required
def blood_test_history(request):
    """Display only the signed-in user's tests with search, filtering and pagination."""
    query = request.GET.get("q", "").strip()
    status = request.GET.get("status", "").strip()
    sort = request.GET.get("sort", "newest").strip()

    blood_tests = request.user.blood_tests.prefetch_related("results")

    if query:
        search_filter = Q(results__parameter_name__icontains=query) | Q(
            results__parameter_code__icontains=query
        )
        # Also allow dates entered in Croatian display format or ISO format.
        parsed_date = None
        for date_format in ("%d.%m.%Y", "%d.%m.%Y.", "%Y-%m-%d"):
            try:
                parsed_date = datetime.strptime(query, date_format).date()
                break
            except ValueError:
                continue
        if parsed_date:
            search_filter |= Q(sampling_date=parsed_date)
        blood_tests = blood_tests.filter(search_filter).distinct()

    valid_statuses = {
        BloodTestResult.Status.NORMAL,
        BloodTestResult.Status.LOW,
        BloodTestResult.Status.HIGH,
        BloodTestResult.Status.UNKNOWN,
        "abnormal",
    }
    if status in valid_statuses:
        if status == "abnormal":
            blood_tests = blood_tests.filter(
                results__status__in=(
                    BloodTestResult.Status.LOW,
                    BloodTestResult.Status.HIGH,
                )
            ).distinct()
        else:
            blood_tests = blood_tests.filter(results__status=status).distinct()

    ordering = {
        "newest": ("-sampling_date", "-created_at"),
        "oldest": ("sampling_date", "created_at"),
        "recently_added": ("-created_at",),
    }
    if sort not in ordering:
        sort = "newest"
    blood_tests = blood_tests.order_by(*ordering[sort])

    paginator = Paginator(blood_tests, 8)
    page_obj = paginator.get_page(request.GET.get("page"))

    # Results are prefetched, so these summaries do not cause extra queries.
    for blood_test in page_obj.object_list:
        counts = {"normal": 0, "low": 0, "high": 0, "unknown": 0}
        for result in blood_test.results.all():
            counts[result.status] = counts.get(result.status, 0) + 1
        blood_test.history_summary = counts
        blood_test.history_total = sum(counts.values())
        blood_test.history_abnormal = counts["low"] + counts["high"]

    query_params = request.GET.copy()
    query_params.pop("page", None)

    return render(
        request,
        "laboratory/blood_test_history.html",
        {
            "page_obj": page_obj,
            "search_query": query,
            "selected_status": status,
            "selected_sort": sort,
            "query_string": query_params.urlencode(),
            "total_matching": paginator.count,
        },
    )


@login_required
def laboratory_trends(request):
    """Show a chronological trend for one numeric laboratory parameter."""
    numeric_results = BloodTestResult.objects.filter(
        blood_test__user=request.user,
        numeric_value__isnull=False,
    )

    parameter_rows = (
        numeric_results.values("parameter_code", "parameter_name", "unit")
        .order_by("parameter_name", "parameter_code", "unit")
        .distinct()
    )
    parameters = list(parameter_rows)

    selected_code = request.GET.get("parameter", "").strip()
    selected_unit = request.GET.get("unit", "").strip()
    selected_range = request.GET.get("range", "all").strip()
    if selected_range not in {"all", "6m", "1y", "2y"}:
        selected_range = "all"

    available_codes = {row["parameter_code"] for row in parameters}
    if not selected_code and parameters:
        selected_code = parameters[0]["parameter_code"]
    if selected_code not in available_codes:
        selected_code = ""

    results = numeric_results.none()
    selected_parameter = None
    if selected_code:
        results = numeric_results.filter(parameter_code=selected_code)
        units_for_code = list(
            results.values_list("unit", flat=True).order_by("unit").distinct()
        )
        if selected_unit not in units_for_code:
            selected_unit = units_for_code[0] if units_for_code else ""
        results = results.filter(unit=selected_unit)

        today = timezone.localdate()
        range_days = {"6m": 183, "1y": 365, "2y": 730}
        if selected_range in range_days:
            results = results.filter(
                blood_test__sampling_date__gte=today - timedelta(days=range_days[selected_range])
            )

        results = results.select_related("blood_test").order_by(
            "blood_test__sampling_date", "blood_test__created_at"
        )
        selected_parameter = next(
            (row for row in parameters if row["parameter_code"] == selected_code and row["unit"] == selected_unit),
            next((row for row in parameters if row["parameter_code"] == selected_code), None),
        )

    points = []
    for result in results:
        points.append(
            {
                "date": result.blood_test.sampling_date.isoformat(),
                "display_date": result.blood_test.sampling_date.strftime("%d.%m.%Y."),
                "value": float(result.numeric_value),
                "reference_min": float(result.reference_min) if result.reference_min is not None else None,
                "reference_max": float(result.reference_max) if result.reference_max is not None else None,
                "status": result.status,
                "detail_url": reverse("laboratory:detail", args=[result.blood_test_id]),
            }
        )

    values = [point["value"] for point in points]
    statistics = {
        "count": len(values),
        "latest": values[-1] if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "change": (values[-1] - values[-2]) if len(values) >= 2 else None,
    }

    return render(
        request,
        "laboratory/laboratory_trends.html",
        {
            "parameters": parameters,
            "selected_code": selected_code,
            "selected_unit": selected_unit,
            "selected_range": selected_range,
            "selected_parameter": selected_parameter,
            "trend_points": points,
            "statistics": statistics,
        },
    )


@login_required
def blood_test_detail(request, pk):
    blood_test = get_object_or_404(
        BloodTest.objects.prefetch_related("results"),
        pk=pk,
        user=request.user,
    )
    analysis_summary = ReferenceAnalysisService.analyze_blood_test(blood_test)
    return render(
        request,
        "laboratory/blood_test_detail.html",
        {
            "blood_test": blood_test,
            "analysis_summary": analysis_summary,
        },
    )


@login_required
def blood_test_delete(request, pk):
    blood_test = get_object_or_404(BloodTest, pk=pk, user=request.user)

    if request.method == "POST":
        sampling_date = blood_test.sampling_date
        blood_test.delete()
        messages.success(
            request,
            f"Krvni nalaz od {sampling_date:%d.%m.%Y.} uspješno je obrisan.",
        )
        return redirect("laboratory:history")

    cancel_url = request.GET.get("next")
    if not cancel_url or not cancel_url.startswith("/"):
        cancel_url = reverse("laboratory:detail", kwargs={"pk": blood_test.pk})

    return render(
        request,
        "laboratory/blood_test_confirm_delete.html",
        {"blood_test": blood_test, "cancel_url": cancel_url},
    )
