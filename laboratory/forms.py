from datetime import date
from decimal import Decimal

from django import forms


LABORATORY_PARAMETERS = (
    {
        "group": "Kompletna krvna slika",
        "parameters": (
            ("WBC", "Leukociti (WBC)", "10^9/L"),
            ("RBC", "Eritrociti (RBC)", "10^12/L"),
            ("HGB", "Hemoglobin (HGB)", "g/L"),
            ("HCT", "Hematokrit (HCT)", "%"),
            ("MCV", "Prosječni volumen eritrocita (MCV)", "fL"),
            ("MCH", "Prosječna količina hemoglobina (MCH)", "pg"),
            ("MCHC", "Prosječna koncentracija hemoglobina (MCHC)", "g/L"),
            ("RDW", "Raspodjela eritrocita (RDW)", "%"),
            ("PLT", "Trombociti (PLT)", "10^9/L"),
            ("MPV", "Prosječni volumen trombocita (MPV)", "fL"),
        ),
    },
    {
        "group": "Biokemija",
        "parameters": (
            ("GLU", "Glukoza", "mmol/L"),
            ("CREA", "Kreatinin", "µmol/L"),
            ("UREA", "Urea", "mmol/L"),
            ("CRP", "C-reaktivni protein (CRP)", "mg/L"),
            ("FE", "Željezo", "µmol/L"),
        ),
    },
)


class ManualBloodTestForm(forms.Form):
    sampling_date = forms.DateField(
        label="Datum uzorkovanja",
        widget=forms.DateInput(attrs={"type": "date", "class": "form-control"}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for group in LABORATORY_PARAMETERS:
            for code, name, unit in group["parameters"]:
                self.fields[f"value_{code}"] = forms.DecimalField(
                    label=name,
                    required=False,
                    max_digits=14,
                    decimal_places=6,
                    widget=forms.NumberInput(
                        attrs={
                            "class": "form-control",
                            "step": "any",
                            "inputmode": "decimal",
                            "placeholder": "Vrijednost",
                        }
                    ),
                )
                self.fields[f"unit_{code}"] = forms.CharField(
                    label=f"Mjerna jedinica za {name}",
                    required=False,
                    initial=unit,
                    max_length=50,
                    widget=forms.TextInput(attrs={"class": "form-control"}),
                )
                self.fields[f"reference_min_{code}"] = forms.DecimalField(
                    label=f"Donja referentna granica za {name}",
                    required=False,
                    max_digits=14,
                    decimal_places=6,
                    widget=forms.NumberInput(
                        attrs={"class": "form-control", "step": "any", "placeholder": "Od"}
                    ),
                )
                self.fields[f"reference_max_{code}"] = forms.DecimalField(
                    label=f"Gornja referentna granica za {name}",
                    required=False,
                    max_digits=14,
                    decimal_places=6,
                    widget=forms.NumberInput(
                        attrs={"class": "form-control", "step": "any", "placeholder": "Do"}
                    ),
                )

    def clean_sampling_date(self):
        sampling_date = self.cleaned_data["sampling_date"]
        if sampling_date > date.today():
            raise forms.ValidationError("Datum uzorkovanja ne može biti u budućnosti.")
        return sampling_date

    def clean(self):
        cleaned_data = super().clean()
        has_result = False

        for group in LABORATORY_PARAMETERS:
            for code, _name, _unit in group["parameters"]:
                value = cleaned_data.get(f"value_{code}")
                reference_min = cleaned_data.get(f"reference_min_{code}")
                reference_max = cleaned_data.get(f"reference_max_{code}")

                if value is not None:
                    has_result = True
                elif reference_min is not None or reference_max is not None:
                    self.add_error(
                        f"value_{code}",
                        "Unesite vrijednost parametra ili uklonite referentne granice.",
                    )

                if (
                    reference_min is not None
                    and reference_max is not None
                    and reference_min > reference_max
                ):
                    self.add_error(
                        f"reference_max_{code}",
                        "Gornja granica mora biti veća ili jednaka donjoj.",
                    )

        if not has_result:
            raise forms.ValidationError("Unesite barem jedan laboratorijski parametar.")

        return cleaned_data

    def iter_parameter_data(self):
        for group in LABORATORY_PARAMETERS:
            for code, name, default_unit in group["parameters"]:
                value = self.cleaned_data.get(f"value_{code}")
                if value is None:
                    continue
                yield {
                    "parameter_code": code,
                    "parameter_name": name,
                    "numeric_value": Decimal(value),
                    "unit": self.cleaned_data.get(f"unit_{code}") or default_unit,
                    "reference_min": self.cleaned_data.get(f"reference_min_{code}"),
                    "reference_max": self.cleaned_data.get(f"reference_max_{code}"),
                }
