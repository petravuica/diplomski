from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import User


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (
        (
            "BloodLab profil",
            {
                "fields": (
                    "role",
                    "date_of_birth",
                    "gender",
                    "height_cm",
                    "weight_kg",
                )
            },
        ),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ("BloodLab profil", {"fields": ("email", "role")}),
    )
    list_display = ("username", "email", "first_name", "last_name", "role", "is_staff")
