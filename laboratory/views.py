from datetime import datetime

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Q
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
