from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("laboratory", "0001_initial")]

    operations = [
        migrations.AddField(
            model_name="bloodtestresult",
            name="parser_confidence",
            field=models.CharField(blank=True, max_length=10, verbose_name="Pouzdanost automatskog prepoznavanja"),
        ),
        migrations.AddField(
            model_name="bloodtestresult",
            name="parser_source_line",
            field=models.CharField(blank=True, max_length=500, verbose_name="Izvorni redak iz PDF-a"),
        ),
        migrations.AddField(
            model_name="bloodtestresult",
            name="normalization_note",
            field=models.CharField(blank=True, max_length=255, verbose_name="Napomena o normalizaciji"),
        ),
    ]
