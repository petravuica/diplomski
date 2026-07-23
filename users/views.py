from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .forms import UserProfileForm, UserRegisterForm


@login_required
def home(request):
    profile_complete = request.user.profile_is_complete
    blood_tests = request.user.blood_tests.prefetch_related("results")
    recent_blood_tests = blood_tests[:5]
    total_blood_tests = blood_tests.count()
    latest_blood_test = blood_tests.first()

    if latest_blood_test:
        onboarding_progress = 75
    elif profile_complete:
        onboarding_progress = 50
    else:
        onboarding_progress = 25

    return render(
        request,
        "registration/home.html",
        {
            "profile_complete": profile_complete,
            "onboarding_progress": onboarding_progress,
            "recent_blood_tests": recent_blood_tests,
            "total_blood_tests": total_blood_tests,
            "latest_blood_test": latest_blood_test,
        },
    )


def register(request):
    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Registracija je uspješna. Prijavite se i dovršite svoj profil.")
            return redirect("login")
    else:
        form = UserRegisterForm()

    return render(request, "registration/register.html", {"form": form})


@login_required
def profile(request):
    if request.method == "POST":
        form = UserProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Korisnički profil uspješno je spremljen.")
            return redirect("profile")
    else:
        form = UserProfileForm(instance=request.user)

    return render(
        request,
        "registration/profile.html",
        {
            "form": form,
            "profile_complete": request.user.profile_is_complete,
            "calculated_age": request.user.age_on(),
        },
    )
