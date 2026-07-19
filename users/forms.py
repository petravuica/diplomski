from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

from .models import User


class UserRegisterForm(UserCreationForm):
    class Meta:
        model = User
        fields = [
            "username",
            "email",
            "role",
            "age",
            "password1",
            "password2",
        ]
        widgets = {
            "username": forms.TextInput(attrs={"placeholder": "Unesite korisničko ime"}),
            "email": forms.EmailInput(attrs={"placeholder": "ime@primjer.hr"}),
            "age": forms.NumberInput(attrs={"placeholder": "Unesite dob", "min": 1, "max": 120}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-control"
            field.help_text = None

        self.fields["email"].required = True
        self.fields["role"].widget.attrs["class"] = "form-select"
        self.fields["role"].choices = [
            ("", "Odaberite ulogu"),
            *self.fields["role"].choices,
        ]
        self.fields["password1"].widget.attrs["placeholder"] = "Najmanje 8 znakova"
        self.fields["password2"].widget.attrs["placeholder"] = "Ponovite lozinku"


class StyledAuthenticationForm(AuthenticationForm):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "Korisničko ime",
                "autocomplete": "username",
            }
        )
    )
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "class": "form-control",
                "placeholder": "Lozinka",
                "autocomplete": "current-password",
            }
        )
    )
