from dataclasses import dataclass
from decimal import Decimal

from .models import BloodTest, BloodTestResult


@dataclass(frozen=True)
class ReferenceAnalysisSummary:
    normal: int = 0
    low: int = 0
    high: int = 0
    unknown: int = 0

    @property
    def total(self):
        return self.normal + self.low + self.high + self.unknown


class ReferenceAnalysisService:
    """Determines result status using the reference range supplied by the laboratory."""

    @staticmethod
    def determine_status(
        numeric_value: Decimal | None,
        reference_min: Decimal | None,
        reference_max: Decimal | None,
    ) -> str:
        if numeric_value is None:
            return BloodTestResult.Status.UNKNOWN

        if reference_min is None and reference_max is None:
            return BloodTestResult.Status.UNKNOWN

        if reference_min is not None and numeric_value < reference_min:
            return BloodTestResult.Status.LOW

        if reference_max is not None and numeric_value > reference_max:
            return BloodTestResult.Status.HIGH

        return BloodTestResult.Status.NORMAL

    @classmethod
    def analyze_result(cls, result: BloodTestResult, *, save: bool = True) -> str:
        status = cls.determine_status(
            result.numeric_value,
            result.reference_min,
            result.reference_max,
        )
        result.status = status
        if save and result.pk:
            result.save(update_fields=["status", "updated_at"])
        return status

    @classmethod
    def analyze_blood_test(cls, blood_test: BloodTest) -> ReferenceAnalysisSummary:
        results = list(blood_test.results.all())
        changed_results = []

        for result in results:
            new_status = cls.determine_status(
                result.numeric_value,
                result.reference_min,
                result.reference_max,
            )
            if result.status != new_status:
                result.status = new_status
                changed_results.append(result)

        if changed_results:
            BloodTestResult.objects.bulk_update(changed_results, ["status", "updated_at"])

        counts = {
            BloodTestResult.Status.NORMAL: 0,
            BloodTestResult.Status.LOW: 0,
            BloodTestResult.Status.HIGH: 0,
            BloodTestResult.Status.UNKNOWN: 0,
        }
        for result in results:
            counts[result.status] += 1

        return ReferenceAnalysisSummary(
            normal=counts[BloodTestResult.Status.NORMAL],
            low=counts[BloodTestResult.Status.LOW],
            high=counts[BloodTestResult.Status.HIGH],
            unknown=counts[BloodTestResult.Status.UNKNOWN],
        )
