from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models


class BloodTest(models.Model):
    class InputMethod(models.TextChoices):
        MANUAL = "manual", "Ručni unos"
        PDF = "pdf", "PDF dokument"

    class ProcessingStatus(models.TextChoices):
        DRAFT = "draft", "Skica"
        PENDING_REVIEW = "pending_review", "Čeka provjeru"
        COMPLETED = "completed", "Dovršeno"
        FAILED = "failed", "Neuspjela obrada"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="blood_tests",
        verbose_name="Korisnik",
    )
    sampling_date = models.DateField("Datum uzorkovanja")
    age_at_test = models.PositiveSmallIntegerField(
        "Dob u trenutku nalaza",
        null=True,
        blank=True,
        editable=False,
    )
    gender_at_test = models.CharField(
        "Spol u trenutku nalaza",
        max_length=10,
        choices=(
            ("female", "Ženski"),
            ("male", "Muški"),
            ("other", "Drugo / ne želim navesti"),
        ),
        blank=True,
        editable=False,
    )
    input_method = models.CharField(
        "Način unosa",
        max_length=10,
        choices=InputMethod.choices,
        default=InputMethod.MANUAL,
    )
    source_file = models.FileField(
        "Izvorni dokument",
        upload_to="blood_tests/%Y/%m/",
        null=True,
        blank=True,
    )
    processing_status = models.CharField(
        "Status obrade",
        max_length=20,
        choices=ProcessingStatus.choices,
        default=ProcessingStatus.DRAFT,
    )
    created_at = models.DateTimeField("Datum unosa", auto_now_add=True)
    updated_at = models.DateTimeField("Zadnja izmjena", auto_now=True)

    class Meta:
        verbose_name = "Krvni nalaz"
        verbose_name_plural = "Krvni nalazi"
        ordering = ("-sampling_date", "-created_at")
        indexes = [
            models.Index(fields=("user", "-sampling_date"), name="bloodtest_user_date_idx"),
        ]

    def clean(self):
        super().clean()
        if self.input_method == self.InputMethod.PDF and not self.source_file:
            raise ValidationError(
                {"source_file": "Za PDF način unosa potrebno je priložiti dokument."}
            )

    def save(self, *args, **kwargs):
        # Dob i spol čuvaju se kao povijesna snimka jer se profil korisnika
        # kasnije može promijeniti.
        if self.user_id:
            if self.age_at_test is None and self.sampling_date:
                self.age_at_test = self.user.age_on(self.sampling_date)
            if not self.gender_at_test:
                self.gender_at_test = self.user.gender
        super().save(*args, **kwargs)

    @property
    def abnormal_results_count(self):
        return self.results.filter(
            status__in=(BloodTestResult.Status.LOW, BloodTestResult.Status.HIGH)
        ).count()

    def __str__(self):
        return f"{self.user} – {self.sampling_date:%d.%m.%Y.}"


class BloodTestResult(models.Model):
    class Status(models.TextChoices):
        LOW = "low", "Sniženo"
        NORMAL = "normal", "Uredno"
        HIGH = "high", "Povišeno"
        UNKNOWN = "unknown", "Nepoznato"

    blood_test = models.ForeignKey(
        BloodTest,
        on_delete=models.CASCADE,
        related_name="results",
        verbose_name="Krvni nalaz",
    )
    parameter_code = models.CharField("Šifra parametra", max_length=50)
    parameter_name = models.CharField("Naziv parametra", max_length=150)
    numeric_value = models.DecimalField(
        "Brojčana vrijednost",
        max_digits=14,
        decimal_places=6,
        null=True,
        blank=True,
    )
    text_value = models.CharField(
        "Tekstualna vrijednost",
        max_length=255,
        blank=True,
    )
    unit = models.CharField("Mjerna jedinica", max_length=50, blank=True)
    reference_min = models.DecimalField(
        "Donja referentna granica",
        max_digits=14,
        decimal_places=6,
        null=True,
        blank=True,
    )
    reference_max = models.DecimalField(
        "Gornja referentna granica",
        max_digits=14,
        decimal_places=6,
        null=True,
        blank=True,
    )
    reference_text = models.CharField(
        "Referentni interval",
        max_length=255,
        blank=True,
    )
    status = models.CharField(
        "Status",
        max_length=10,
        choices=Status.choices,
        default=Status.UNKNOWN,
    )
    created_at = models.DateTimeField("Datum stvaranja", auto_now_add=True)
    updated_at = models.DateTimeField("Zadnja izmjena", auto_now=True)

    class Meta:
        verbose_name = "Rezultat krvnog nalaza"
        verbose_name_plural = "Rezultati krvnih nalaza"
        ordering = ("parameter_name",)
        constraints = [
            models.UniqueConstraint(
                fields=("blood_test", "parameter_code"),
                name="unique_parameter_per_blood_test",
            ),
            models.CheckConstraint(
                check=(
                    models.Q(numeric_value__isnull=False)
                    | ~models.Q(text_value="")
                ),
                name="blood_result_has_value",
            ),
        ]
        indexes = [
            models.Index(fields=("parameter_code",), name="bloodresult_param_idx"),
            models.Index(fields=("status",), name="bloodresult_status_idx"),
        ]

    def clean(self):
        super().clean()
        if self.numeric_value is None and not self.text_value.strip():
            raise ValidationError(
                "Potrebno je unijeti brojčanu ili tekstualnu vrijednost rezultata."
            )
        if self.reference_min is not None and self.reference_max is not None:
            if self.reference_min > self.reference_max:
                raise ValidationError(
                    {"reference_max": "Gornja granica mora biti veća ili jednaka donjoj."}
                )

    @property
    def display_value(self):
        if self.numeric_value is not None:
            return f"{self.numeric_value:g}"
        return self.text_value

    def __str__(self):
        return f"{self.parameter_name}: {self.display_value} {self.unit}".strip()
