from django.urls import path

from . import views

app_name = "laboratory"

urlpatterns = [
    path("", views.blood_test_history, name="history"),
    path("new/", views.manual_blood_test_create, name="manual-create"),
    path("trends/", views.laboratory_trends, name="trends"),
    path("<int:pk>/", views.blood_test_detail, name="detail"),
    path("<int:pk>/delete/", views.blood_test_delete, name="delete"),
]
