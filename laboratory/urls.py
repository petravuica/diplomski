from django.urls import path

from . import views

app_name = "laboratory"

urlpatterns = [
    path("new/", views.manual_blood_test_create, name="manual-create"),
    path("<int:pk>/", views.blood_test_detail, name="detail"),
]
