from django.db import models
from django.contrib.auth.models import AbstractUser
# Create your models here.

class User(AbstractUser):
    ROLE_CHOICES = (
        ('patient', 'Patient'),
        ('doctor', 'Doctor'),
    )

    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    age = models.IntegerField(null=True, blank=True)