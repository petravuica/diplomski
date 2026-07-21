from datetime import date

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
            "password1",
            "password2",
        ]
        widgets = {
            "username": forms.TextInput(attrs={"placeholder": "Unesite korisničko ime"}),
            "email": forms.EmailInput(attrs={"placeholder": "ime@primjer.hr"}),
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


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = [
            "first_name",
            "last_name",
            "email",
            "date_of_birth",
            "gender",
            "height_cm",
            "weight_kg",
        ]
        labels = {
            "first_name": "Ime",
            "last_name": "Prezime",
            "email": "Adresa e-pošte",
            "date_of_birth": "Datum rođenja",
            "gender": "Spol",
            "height_cm": "Visina (cm)",
            "weight_kg": "Težina (kg)",
        }
        widgets = {
            "first_name": forms.TextInput(attrs={"placeholder": "Unesite ime"}),
            "last_name": forms.TextInput(attrs={"placeholder": "Unesite prezime"}),
            "email": forms.EmailInput(attrs={"placeholder": "ime@primjer.hr"}),
            "date_of_birth": forms.DateInput(attrs={"type": "date"}),
            "gender": forms.Select(),
            "height_cm": forms.NumberInput(attrs={"step": "0.1", "min": "50", "max": "250", "placeholder": "npr. 170"}),
            "weight_kg": forms.NumberInput(attrs={"step": "0.1", "min": "20", "max": "400", "placeholder": "npr. 65"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.help_text = None
            field.widget.attrs["class"] = "form-select" if name == "gender" else "form-control"

        self.fields["first_name"].required = True
        self.fields["last_name"].required = True
        self.fields["email"].required = True
        self.fields["date_of_birth"].required = True
        self.fields["gender"].required = True
        self.fields["gender"].choices = [("", "Odaberite spol"), *User.GENDER_CHOICES]

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.exclude(pk=self.instance.pk).filter(email__iexact=email).exists():
            raise forms.ValidationError("Korisnik s ovom adresom e-pošte već postoji.")
        return email

    def clean_date_of_birth(self):
        value = self.cleaned_data.get("date_of_birth")
        if value and value > date.today():
            raise forms.ValidationError("Datum rođenja ne može biti u budućnosti.")
        if value and value.year < 1900:
            raise forms.ValidationError("Unesite ispravan datum rođenja.")
        return value


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
