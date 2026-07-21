from django.db import migrations, models
import django.core.validators


class Migration(migrations.Migration):
    dependencies = [("users", "0001_initial")]

    operations = [
        migrations.AddField(
            model_name="user",
            name="date_of_birth",
            field=models.DateField(blank=True, null=True, verbose_name="Datum rođenja"),
        ),
        migrations.AddField(
            model_name="user",
            name="gender",
            field=models.CharField(
                blank=True,
                choices=[
                    ("female", "Ženski"),
                    ("male", "Muški"),
                    ("other", "Drugo / ne želim navesti"),
                ],
                max_length=10,
                verbose_name="Spol",
            ),
        ),
        migrations.AddField(
            model_name="user",
            name="height_cm",
            field=models.DecimalField(
                blank=True,
                decimal_places=1,
                max_digits=5,
                null=True,
                validators=[
                    django.core.validators.MinValueValidator(50),
                    django.core.validators.MaxValueValidator(250),
                ],
                verbose_name="Visina (cm)",
            ),
        ),
        migrations.AddField(
            model_name="user",
            name="weight_kg",
            field=models.DecimalField(
                blank=True,
                decimal_places=1,
                max_digits=5,
                null=True,
                validators=[
                    django.core.validators.MinValueValidator(20),
                    django.core.validators.MaxValueValidator(400),
                ],
                verbose_name="Težina (kg)",
            ),
        ),
        migrations.AlterField(
            model_name="user",
            name="role",
            field=models.CharField(
                choices=[("patient", "Pacijent"), ("doctor", "Liječnik")],
                max_length=10,
            ),
        ),
    ]
