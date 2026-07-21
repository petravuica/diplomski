from datetime import date
from decimal import Decimal

from django.core.exceptions import ValidationError
from django.test import TestCase

from users.models import User

from .models import BloodTest, BloodTestResult


class BloodTestModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="ana",
            password="test-password",
            role="patient",
            date_of_birth=date(2000, 8, 20),
            gender="female",
        )

    def test_profile_data_is_snapshotted_when_test_is_created(self):
        blood_test = BloodTest.objects.create(
            user=self.user,
            sampling_date=date(2026, 7, 21),
        )

        self.assertEqual(blood_test.age_at_test, 25)
        self.assertEqual(blood_test.gender_at_test, "female")

    def test_pdf_input_requires_source_file(self):
        blood_test = BloodTest(
            user=self.user,
            sampling_date=date(2026, 7, 21),
            input_method=BloodTest.InputMethod.PDF,
        )

        with self.assertRaises(ValidationError):
            blood_test.full_clean()

    def test_result_requires_numeric_or_text_value(self):
        blood_test = BloodTest.objects.create(
            user=self.user,
            sampling_date=date(2026, 7, 21),
        )
        result = BloodTestResult(
            blood_test=blood_test,
            parameter_code="HGB",
            parameter_name="Hemoglobin",
        )

        with self.assertRaises(ValidationError):
            result.full_clean()

    def test_abnormal_results_count(self):
        blood_test = BloodTest.objects.create(
            user=self.user,
            sampling_date=date(2026, 7, 21),
        )
        BloodTestResult.objects.create(
            blood_test=blood_test,
            parameter_code="HGB",
            parameter_name="Hemoglobin",
            numeric_value=Decimal("118"),
            unit="g/L",
            status=BloodTestResult.Status.LOW,
        )
        BloodTestResult.objects.create(
            blood_test=blood_test,
            parameter_code="WBC",
            parameter_name="Leukociti",
            numeric_value=Decimal("6.2"),
            unit="10^9/L",
            status=BloodTestResult.Status.NORMAL,
        )

        self.assertEqual(blood_test.abnormal_results_count, 1)
