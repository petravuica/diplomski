from datetime import date

from django.contrib.auth.models import AbstractUser
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class User(AbstractUser):
    ROLE_CHOICES = (
        ("patient", "Pacijent"),
        ("doctor", "Liječnik"),
    )
    GENDER_CHOICES = (
        ("female", "Ženski"),
        ("male", "Muški"),
        ("other", "Drugo / ne želim navesti"),
    )

    role = models.CharField(max_length=10, choices=ROLE_CHOICES)

    # Polje age ostavljeno je radi kompatibilnosti s postojećom bazom. Nova logika
    # koristi datum rođenja kako bi se dob izračunala na datum uzorkovanja nalaza.
    age = models.IntegerField(null=True, blank=True)
    date_of_birth = models.DateField("Datum rođenja", null=True, blank=True)
    gender = models.CharField(
        "Spol",
        max_length=10,
        choices=GENDER_CHOICES,
        blank=True,
    )
    height_cm = models.DecimalField(
        "Visina (cm)",
        max_digits=5,
        decimal_places=1,
        null=True,
        blank=True,
        validators=[MinValueValidator(50), MaxValueValidator(250)],
    )
    weight_kg = models.DecimalField(
        "Težina (kg)",
        max_digits=5,
        decimal_places=1,
        null=True,
        blank=True,
        validators=[MinValueValidator(20), MaxValueValidator(400)],
    )

    @property
    def profile_is_complete(self):
        """Minimalni podaci potrebni za buduću laboratorijsku analizu."""
        return bool(self.first_name and self.last_name and self.date_of_birth and self.gender)

    def age_on(self, on_date=None):
        """Vraća dob korisnika na zadani datum (npr. datum uzorkovanja)."""
        if not self.date_of_birth:
            return None

        on_date = on_date or date.today()
        return (
            on_date.year
            - self.date_of_birth.year
            - (
                (on_date.month, on_date.day)
                < (self.date_of_birth.month, self.date_of_birth.day)
            )
        )
