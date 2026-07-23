from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect, render

from .forms import LABORATORY_PARAMETERS, ManualBloodTestForm
from .models import BloodTest, BloodTestResult


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
                BloodTestResult.objects.bulk_create(
                    [
                        BloodTestResult(blood_test=blood_test, **parameter_data)
                        for parameter_data in form.iter_parameter_data()
                    ]
                )

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
def blood_test_detail(request, pk):
    blood_test = get_object_or_404(
        BloodTest.objects.prefetch_related("results"),
        pk=pk,
        user=request.user,
    )
    return render(
        request,
        "laboratory/blood_test_detail.html",
        {"blood_test": blood_test},
    )
