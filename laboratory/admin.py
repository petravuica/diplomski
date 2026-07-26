from django.contrib import admin

from .models import BloodTest, BloodTestResult


class BloodTestResultInline(admin.TabularInline):
    model = BloodTestResult
    extra = 0
    fields = (
        "parameter_code",
        "parameter_name",
        "numeric_value",
        "text_value",
        "unit",
        "reference_min",
        "reference_max",
        "status",
        "parser_confidence",
    )


@admin.register(BloodTest)
class BloodTestAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "sampling_date",
        "input_method",
        "processing_status",
        "age_at_test",
        "created_at",
    )
    list_filter = ("input_method", "processing_status", "sampling_date")
    search_fields = ("user__username", "user__first_name", "user__last_name")
    readonly_fields = ("age_at_test", "gender_at_test", "created_at", "updated_at")
    inlines = (BloodTestResultInline,)


@admin.register(BloodTestResult)
class BloodTestResultAdmin(admin.ModelAdmin):
    list_display = (
        "parameter_code",
        "parameter_name",
        "blood_test",
        "numeric_value",
        "text_value",
        "unit",
        "status",
        "parser_confidence",
    )
    list_filter = ("status",
        "parser_confidence", "parameter_code")
    search_fields = (
        "parameter_code",
        "parameter_name",
        "blood_test__user__username",
    )
