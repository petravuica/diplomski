from datetime import datetime, timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from .forms import (
    LABORATORY_PARAMETERS,
    ManualBloodTestForm,
    PdfBloodTestUploadForm,
    PdfReviewMetadataForm,
    PdfResultReviewFormSet,
)
from .models import BloodTest, BloodTestResult
from .pdf_parser import LaboratoryPdfParser, PdfParsingError
from .services import ReferenceAnalysisService
from .ml_services.anemia_prediction import predict_anemia
from .ml_services.liver_prediction import predict_liver


@login_required
def pdf_blood_test_upload(request):
    if not request.user.profile_is_complete:
        messages.warning(
            request,
            "Prije učitavanja nalaza dovršite profil kako bi se dob i spol ispravno spremili uz nalaz.",
        )
        return redirect("profile")

    if request.method == "POST":
        form = PdfBloodTestUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data["source_file"]
            try:
                parsed_report = LaboratoryPdfParser.parse(uploaded_file)
            except PdfParsingError as exc:
                form.add_error("source_file", str(exc))
            else:
                if not parsed_report.results:
                    form.add_error(
                        "source_file",
                        "Nije prepoznat nijedan krvni parametar. Pokušajte s drugim PDF-om ili upotrijebite ručni unos.",
                    )
                    return render(request, "laboratory/pdf_blood_test_upload.html", {"form": form})
                sampling_date = parsed_report.sampling_date or timezone.localdate()
                with transaction.atomic():
                    blood_test = BloodTest.objects.create(
                        user=request.user,
                        sampling_date=sampling_date,
                        input_method=BloodTest.InputMethod.PDF,
                        source_file=uploaded_file,
                        processing_status=BloodTest.ProcessingStatus.PENDING_REVIEW,
                    )
                    pending_results = []
                    for item in parsed_report.results:
                        result = BloodTestResult(
                            blood_test=blood_test,
                            parameter_code=item.parameter_code,
                            parameter_name=item.parameter_name,
                            numeric_value=item.numeric_value,
                            text_value=item.text_value,
                            unit=item.unit,
                            reference_min=item.reference_min,
                            reference_max=item.reference_max,
                            reference_text=item.reference_text,
                            parser_confidence=item.confidence,
                            parser_source_line=item.source_line,
                            normalization_note=item.normalization_note,
                        )
                        ReferenceAnalysisService.analyze_result(result, save=False)
                        pending_results.append(result)
                    if pending_results:
                        BloodTestResult.objects.bulk_create(pending_results)

                # Demographic discrepancies are warnings only; profile data is never overwritten.
                if parsed_report.birth_date and request.user.date_of_birth:
                    if parsed_report.birth_date != request.user.date_of_birth:
                        messages.warning(
                            request,
                            "Datum rođenja u PDF-u razlikuje se od datuma u profilu. Provjerite je li dokument vaš.",
                        )
                if parsed_report.gender and request.user.gender:
                    if parsed_report.gender != request.user.gender:
                        messages.warning(
                            request,
                            "Spol naveden u PDF-u razlikuje se od podatka u profilu.",
                        )
                for warning in parsed_report.warnings:
                    messages.warning(request, warning)
                messages.info(
                    request,
                    f"Prepoznato je {len(parsed_report.results)} parametara. Provjerite podatke prije spremanja.",
                )
                return redirect("laboratory:pdf-review", pk=blood_test.pk)
    else:
        form = PdfBloodTestUploadForm()

    return render(request, "laboratory/pdf_blood_test_upload.html", {"form": form})


@login_required
def pdf_blood_test_review(request, pk):
    blood_test = get_object_or_404(
        BloodTest.objects.prefetch_related("results"),
        pk=pk,
        user=request.user,
        input_method=BloodTest.InputMethod.PDF,
    )
    if blood_test.processing_status == BloodTest.ProcessingStatus.COMPLETED:
        return redirect("laboratory:detail", pk=blood_test.pk)

    existing_results = list(blood_test.results.all())
    initial_rows = [
        {
            "include": True,
            "parameter_code": result.parameter_code,
            "parameter_name": result.parameter_name,
            "numeric_value": result.numeric_value,
            "text_value": result.text_value,
            "unit": result.unit,
            "reference_min": result.reference_min,
            "reference_max": result.reference_max,
            "reference_text": result.reference_text,
            "parser_confidence": result.parser_confidence,
            "parser_source_line": result.parser_source_line,
            "normalization_note": result.normalization_note,
        }
        for result in existing_results
    ]

    if request.method == "POST":
        metadata_form = PdfReviewMetadataForm(request.POST)
        formset = PdfResultReviewFormSet(request.POST, prefix="results")
        if metadata_form.is_valid() and formset.is_valid():
            included_rows = [
                row.cleaned_data for row in formset
                if row.cleaned_data and row.cleaned_data.get("include")
            ]
            codes = [row["parameter_code"].strip().upper() for row in included_rows]
            if not included_rows:
                formset._non_form_errors = formset.error_class(["Odaberite barem jedan rezultat za spremanje."])
            elif len(codes) != len(set(codes)):
                formset._non_form_errors = formset.error_class(["Svaki parametar smije se pojaviti samo jednom."])
            else:
                with transaction.atomic():
                    blood_test.sampling_date = metadata_form.cleaned_data["sampling_date"]
                    blood_test.age_at_test = request.user.age_on(blood_test.sampling_date)
                    blood_test.gender_at_test = request.user.gender
                    blood_test.processing_status = BloodTest.ProcessingStatus.COMPLETED
                    blood_test.save()
                    blood_test.results.all().delete()

                    final_results = []
                    for row in included_rows:
                        result = BloodTestResult(
                            blood_test=blood_test,
                            parameter_code=row["parameter_code"].strip().upper(),
                            parameter_name=row["parameter_name"].strip(),
                            numeric_value=row.get("numeric_value"),
                            text_value=row.get("text_value", "").strip(),
                            unit=row.get("unit", "").strip(),
                            reference_min=row.get("reference_min"),
                            reference_max=row.get("reference_max"),
                            reference_text=row.get("reference_text", "").strip(),
                            parser_confidence=row.get("parser_confidence", "").strip(),
                            parser_source_line=row.get("parser_source_line", "").strip()[:500],
                            normalization_note=row.get("normalization_note", "").strip(),
                        )
                        ReferenceAnalysisService.analyze_result(result, save=False)
                        final_results.append(result)
                    BloodTestResult.objects.bulk_create(final_results)

                messages.success(request, "PDF nalaz provjeren je i uspješno spremljen.")
                return redirect("laboratory:detail", pk=blood_test.pk)
    else:
        metadata_form = PdfReviewMetadataForm(initial={"sampling_date": blood_test.sampling_date})
        formset = PdfResultReviewFormSet(initial=initial_rows, prefix="results")

    return render(
        request,
        "laboratory/pdf_blood_test_review.html",
        {
            "blood_test": blood_test,
            "metadata_form": metadata_form,
            "formset": formset,
            "recognized_count": len(existing_results),
            "high_confidence_count": sum(r.parser_confidence == "high" for r in existing_results),
            "review_count": sum(r.parser_confidence in {"medium", "low"} for r in existing_results),
        },
    )


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

    analysis_summary = ReferenceAnalysisService.analyze_blood_test(
        blood_test
    )

    # ---------------------------------------------------------
    # PRIPREMA REZULTATA
    # ---------------------------------------------------------

    result_values = {
        result.parameter_code.strip().upper(): result.numeric_value
        for result in blood_test.results.all()
        if result.numeric_value is not None
    }

    # ---------------------------------------------------------
    # ANEMIA ML PREDIKCIJA
    # ---------------------------------------------------------

    anemia_prediction = None
    anemia_prediction_message = None

    anemia_required_parameters = {
        "RBC",
        "HGB",
        "HCT",
        "MCV",
        "MCH",
        "MCHC",
    }

    anemia_missing_parameters = sorted(
        anemia_required_parameters - result_values.keys()
    )

    if anemia_missing_parameters:
        anemia_prediction_message = (
            "Predikcija anemije nije dostupna jer nedostaju parametri: "
            + ", ".join(anemia_missing_parameters)
            + "."
        )

    elif blood_test.age_at_test is None:
        anemia_prediction_message = (
            "Predikcija anemije nije dostupna jer nije spremljena "
            "dob korisnika u trenutku nalaza."
        )

    elif blood_test.gender_at_test not in {"female", "male"}:
        anemia_prediction_message = (
            "Predikcija anemije trenutačno je dostupna samo za "
            "ženski ili muški spol."
        )

    else:
        try:
            anemia_prediction = predict_anemia(
                gender=blood_test.gender_at_test,
                age=blood_test.age_at_test,
                hgb=result_values["HGB"],
                rbc=result_values["RBC"],
                hct=result_values["HCT"],
                mcv=result_values["MCV"],
                mch=result_values["MCH"],
                mchc=result_values["MCHC"],
            )

        except (ValueError, FileNotFoundError) as exc:
            anemia_prediction_message = str(exc)

        except Exception:
            anemia_prediction_message = (
                "Predikciju anemije trenutačno nije moguće izračunati."
            )

    # ---------------------------------------------------------
    # JETRENI ML MODEL
    # ---------------------------------------------------------

    liver_prediction = None
    liver_prediction_message = None

    # Konačni practical model koristi:
    # Gender
    # Age
    # Total Bilirubin
    # ALP
    # ALT
    # AST
    # Total Proteins
    # Albumin

    liver_parameter_codes = {
        "Total_Bilirubin": ("BILI", "TBIL", "BIL"),
        "Alkaline_Phosphatase": ("ALP", "AP"),
        "Alanine_Aminotransferase": ("ALT",),
        "Aspartate_Aminotransferase": ("AST",),
        "Total_Proteins": ("TP", "PROT"),
        "Albumin": ("ALB", "ALBUMIN"),
    }

    def get_result_value(possible_codes):
        for code in possible_codes:
            value = result_values.get(code)

            if value is not None:
                return value

        return None

    liver_values = {
        parameter_name: get_result_value(possible_codes)
        for parameter_name, possible_codes
        in liver_parameter_codes.items()
    }

    liver_missing_parameters = [
        parameter_name
        for parameter_name, value in liver_values.items()
        if value is None
    ]

    if liver_missing_parameters:
        readable_names = {
            "Total_Bilirubin": "ukupni bilirubin",
            "Alkaline_Phosphatase": "ALP",
            "Alanine_Aminotransferase": "ALT",
            "Aspartate_Aminotransferase": "AST",
            "Total_Proteins": "ukupni proteini",
            "Albumin": "albumin",
        }

        missing_names = [
            readable_names[name]
            for name in liver_missing_parameters
        ]

        liver_prediction_message = (
            "ML analiza jetrenih parametara nije dostupna jer "
            "nedostaju parametri: "
            + ", ".join(missing_names)
            + "."
        )

    elif blood_test.age_at_test is None:
        liver_prediction_message = (
            "ML analiza jetrenih parametara nije dostupna jer "
            "nije spremljena dob korisnika u trenutku nalaza."
        )

    elif blood_test.gender_at_test not in {"female", "male"}:
        liver_prediction_message = (
            "ML analiza jetrenih parametara trenutačno je dostupna "
            "samo za ženski ili muški spol."
        )

    else:
        try:
            liver_prediction = predict_liver(
                gender=blood_test.gender_at_test,
                age=blood_test.age_at_test,
                total_bilirubin=liver_values["Total_Bilirubin"],
                alkaline_phosphatase=liver_values[
                    "Alkaline_Phosphatase"
                ],
                alt=liver_values[
                    "Alanine_Aminotransferase"
                ],
                ast=liver_values[
                    "Aspartate_Aminotransferase"
                ],
                total_proteins=liver_values[
                    "Total_Proteins"
                ],
                albumin=liver_values["Albumin"],
            )

        except (ValueError, FileNotFoundError) as exc:
            liver_prediction_message = str(exc)

        except Exception:
            liver_prediction_message = (
                "ML analizu jetrenih parametara trenutačno "
                "nije moguće izračunati."
            )

    # ---------------------------------------------------------
    # TEMPLATE
    # ---------------------------------------------------------

    return render(
        request,
        "laboratory/blood_test_detail.html",
        {
            "blood_test": blood_test,
            "analysis_summary": analysis_summary,

            "anemia_prediction": anemia_prediction,
            "anemia_prediction_message": anemia_prediction_message,

            "liver_prediction": liver_prediction,
            "liver_prediction_message": liver_prediction_message,
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
