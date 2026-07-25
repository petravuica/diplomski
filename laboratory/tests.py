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

from django.urls import reverse


class ManualBloodTestViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="petra",
            password="test-password",
            role="patient",
            first_name="Petra",
            last_name="Vuica",
            date_of_birth=date(2000, 8, 20),
            gender="female",
        )
        self.client.login(username="petra", password="test-password")

    def test_manual_entry_creates_test_and_only_filled_results(self):
        response = self.client.post(
            reverse("laboratory:manual-create"),
            {
                "sampling_date": "2026-07-21",
                "value_HGB": "132",
                "unit_HGB": "g/L",
                "reference_min_HGB": "119",
                "reference_max_HGB": "157",
                "value_CRP": "4.2",
                "unit_CRP": "mg/L",
            },
        )

        blood_test = BloodTest.objects.get(user=self.user)
        self.assertRedirects(response, reverse("laboratory:detail", args=[blood_test.pk]))
        self.assertEqual(blood_test.results.count(), 2)
        self.assertEqual(blood_test.processing_status, BloodTest.ProcessingStatus.COMPLETED)
        self.assertEqual(
            blood_test.results.get(parameter_code="HGB").status,
            BloodTestResult.Status.NORMAL,
        )
        self.assertEqual(
            blood_test.results.get(parameter_code="CRP").status,
            BloodTestResult.Status.UNKNOWN,
        )

    def test_manual_entry_requires_at_least_one_parameter(self):
        response = self.client.post(
            reverse("laboratory:manual-create"),
            {"sampling_date": "2026-07-21"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Unesite barem jedan laboratorijski parametar.")
        self.assertFalse(BloodTest.objects.filter(user=self.user).exists())

    def test_user_cannot_open_another_users_test(self):
        other_user = User.objects.create_user(
            username="other",
            password="test-password",
            role="patient",
            date_of_birth=date(1990, 1, 1),
            gender="male",
        )
        blood_test = BloodTest.objects.create(
            user=other_user,
            sampling_date=date(2026, 7, 21),
        )

        response = self.client.get(reverse("laboratory:detail", args=[blood_test.pk]))
        self.assertEqual(response.status_code, 404)

from .services import ReferenceAnalysisService


class ReferenceAnalysisServiceTests(TestCase):
    def test_status_is_low_below_minimum(self):
        status = ReferenceAnalysisService.determine_status(
            Decimal("3.8"), Decimal("4.2"), Decimal("5.4")
        )
        self.assertEqual(status, BloodTestResult.Status.LOW)

    def test_status_is_high_above_maximum(self):
        status = ReferenceAnalysisService.determine_status(
            Decimal("12"), None, Decimal("5")
        )
        self.assertEqual(status, BloodTestResult.Status.HIGH)

    def test_status_is_normal_inside_range(self):
        status = ReferenceAnalysisService.determine_status(
            Decimal("132"), Decimal("119"), Decimal("157")
        )
        self.assertEqual(status, BloodTestResult.Status.NORMAL)

    def test_status_is_unknown_without_reference_range(self):
        status = ReferenceAnalysisService.determine_status(
            Decimal("4.2"), None, None
        )
        self.assertEqual(status, BloodTestResult.Status.UNKNOWN)

    def test_inclusive_reference_boundaries_are_normal(self):
        self.assertEqual(
            ReferenceAnalysisService.determine_status(
                Decimal("5"), None, Decimal("5")
            ),
            BloodTestResult.Status.NORMAL,
        )
        self.assertEqual(
            ReferenceAnalysisService.determine_status(
                Decimal("4.2"), Decimal("4.2"), None
            ),
            BloodTestResult.Status.NORMAL,
        )


class BloodTestHistoryViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="history-user",
            password="test-password",
            role="patient",
            date_of_birth=date(1998, 3, 10),
            gender="female",
        )
        self.other_user = User.objects.create_user(
            username="other-history-user",
            password="test-password",
            role="patient",
            date_of_birth=date(1990, 1, 1),
            gender="male",
        )
        self.client.login(username="history-user", password="test-password")

        self.blood_test = BloodTest.objects.create(
            user=self.user,
            sampling_date=date(2026, 7, 20),
            processing_status=BloodTest.ProcessingStatus.COMPLETED,
        )
        BloodTestResult.objects.create(
            blood_test=self.blood_test,
            parameter_code="CRP",
            parameter_name="C-reaktivni protein",
            numeric_value=Decimal("12"),
            reference_max=Decimal("5"),
            status=BloodTestResult.Status.HIGH,
        )
        self.other_test = BloodTest.objects.create(
            user=self.other_user,
            sampling_date=date(2026, 7, 19),
        )

    def test_history_displays_only_signed_in_users_tests(self):
        response = self.client.get(reverse("laboratory:history"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "20.07.2026.")
        self.assertNotContains(response, "19.07.2026.")

    def test_history_can_filter_abnormal_tests(self):
        response = self.client.get(
            reverse("laboratory:history"), {"status": "abnormal"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "1 odstupanja")

    def test_history_searches_parameter_name(self):
        response = self.client.get(
            reverse("laboratory:history"), {"q": "reaktivni"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "20.07.2026.")

    def test_delete_requires_post_and_deletes_own_test(self):
        delete_url = reverse("laboratory:delete", args=[self.blood_test.pk])
        get_response = self.client.get(delete_url)
        self.assertEqual(get_response.status_code, 200)
        self.assertTrue(BloodTest.objects.filter(pk=self.blood_test.pk).exists())

        post_response = self.client.post(delete_url)
        self.assertRedirects(post_response, reverse("laboratory:history"))
        self.assertFalse(BloodTest.objects.filter(pk=self.blood_test.pk).exists())

    def test_user_cannot_delete_another_users_test(self):
        response = self.client.post(
            reverse("laboratory:delete", args=[self.other_test.pk])
        )

        self.assertEqual(response.status_code, 404)
        self.assertTrue(BloodTest.objects.filter(pk=self.other_test.pk).exists())
