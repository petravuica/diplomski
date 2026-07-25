"""PDF extraction and normalization for laboratory reports.

The parser intentionally separates extraction from persistence. It recognizes common
Croatian laboratory naming conventions, keeps the laboratory's original reference
interval and assigns stable parameter codes for history and trend charts.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


class PdfParsingError(Exception):
    """Raised when a PDF cannot be read safely or contains no extractable text."""


@dataclass(frozen=True)
class ParameterDefinition:
    code: str
    name: str
    aliases: tuple[str, ...]
    default_unit: str = ""


@dataclass
class ParsedLaboratoryResult:
    parameter_code: str
    parameter_name: str
    numeric_value: Decimal | None = None
    text_value: str = ""
    unit: str = ""
    reference_min: Decimal | None = None
    reference_max: Decimal | None = None
    reference_text: str = ""
    source_line: str = ""
    confidence: str = "high"


@dataclass
class ParsedPdfReport:
    sampling_date: date | None
    birth_date: date | None
    gender: str
    results: list[ParsedLaboratoryResult]
    raw_text: str
    warnings: list[str]


PARAMETERS: tuple[ParameterDefinition, ...] = (
    ParameterDefinition("WBC", "Leukociti (WBC)", ("leukociti", "wbc"), "10^9/L"),
    ParameterDefinition("RBC", "Eritrociti (RBC)", ("eritrociti", "rbc"), "10^12/L"),
    ParameterDefinition("HGB", "Hemoglobin (HGB)", ("hemoglobin", "hgb", "hb"), "g/L"),
    ParameterDefinition("HCT", "Hematokrit (HCT)", ("hematokrit", "hct"), "L/L"),
    ParameterDefinition("MCV", "Prosječni volumen eritrocita (MCV)", ("mcv",), "fL"),
    ParameterDefinition("MCH", "Prosječna količina hemoglobina (MCH)", ("mch",), "pg"),
    ParameterDefinition("MCHC", "Prosječna koncentracija hemoglobina (MCHC)", ("mchc",), "g/L"),
    ParameterDefinition("RDW", "Raspodjela eritrocita (RDW)", ("rdw-kv", "rdw-cv", "rdw"), "%"),
    ParameterDefinition("PLT", "Trombociti (PLT)", ("trombociti", "plt"), "10^9/L"),
    ParameterDefinition("MPV", "Prosječni volumen trombocita (MPV)", ("mpv",), "fL"),
    ParameterDefinition("GLU", "Glukoza", ("glukoza", "glucose"), "mmol/L"),
    ParameterDefinition("UREA", "Urea", ("urea",), "mmol/L"),
    ParameterDefinition("CREA", "Kreatinin", ("kreatinin", "creatinine"), "µmol/L"),
    ParameterDefinition("FE", "Željezo", ("željezo", "zeljezo", "iron"), "µmol/L"),
    ParameterDefinition("UIBC", "Nezasićeni kapacitet vezanja željeza (UIBC)", ("nez.kap.vez.željeza", "uibc"), "µmol/L"),
    ParameterDefinition("TIBC", "Ukupni kapacitet vezanja željeza (TIBC)", ("uk.kap.vez.željeza", "tibc"), "µmol/L"),
    ParameterDefinition("CRP", "C-reaktivni protein (CRP)", ("c-reaktivni protein", "crp"), "mg/L"),
    ParameterDefinition("CHOL", "Kolesterol", ("kolesterol", "cholesterol"), "mmol/L"),
    ParameterDefinition("TG", "Trigliceridi", ("trigliceridi", "triglycerides"), "mmol/L"),
    ParameterDefinition("TSH", "TSH", ("tsh",), "mIU/L"),
    ParameterDefinition("ALT", "Alanin-aminotransferaza (ALT)", ("alt", "alanin-aminotransferaza"), "U/L"),
    ParameterDefinition("AST", "Aspartat-aminotransferaza (AST)", ("ast", "aspartat-aminotransferaza"), "U/L"),
    ParameterDefinition("GGT", "Gama-glutamiltransferaza (GGT)", ("ggt", "gama-glutamiltransferaza"), "U/L"),
    ParameterDefinition("BILI", "Bilirubin ukupni", ("bilirubin ukupni", "ukupni bilirubin"), "µmol/L"),
    ParameterDefinition("ALP", "Alkalna fosfataza (ALP)", ("alkalna fosfataza", "alp"), "U/L"),
    ParameterDefinition("NA", "Natrij", ("natrij", "sodium"), "mmol/L"),
    ParameterDefinition("K", "Kalij", ("kalij", "potassium"), "mmol/L"),
    ParameterDefinition("CA", "Kalcij", ("kalcij", "calcium"), "mmol/L"),
    ParameterDefinition("FERR", "Feritin", ("feritin", "ferritin"), "µg/L"),
    ParameterDefinition("B12", "Vitamin B12", ("vitamin b12", "b12"), "pmol/L"),
    ParameterDefinition("FOL", "Folat", ("folat", "folna kiselina"), "nmol/L"),
    ParameterDefinition("VITD", "Vitamin D", ("25-oh vitamin d", "vitamin d"), "nmol/L"),
    ParameterDefinition("HBA1C", "HbA1c", ("hba1c", "hemoglobin a1c"), "%"),
)

# Longest aliases first prevents "MCH" from matching the beginning of "MCHC".
_ALIAS_ENTRIES = sorted(
    ((alias, definition) for definition in PARAMETERS for alias in definition.aliases),
    key=lambda item: len(item[0]),
    reverse=True,
)

_NUM = r"[-+]?\d+(?:[.,]\d+)?"
_UNIT_PATTERN = re.compile(
    r"^(?P<unit>(?:10\s*\^?\s*\d+\s*/\s*[lL]|[gmkµu]?mol\s*/\s*[lL]|mg\s*/\s*[lL]|g\s*/\s*[lL]|mIU\s*/\s*[lL]|IU\s*/\s*[lL]|U\s*/\s*[lL]|L\s*/\s*L|kg\s*/\s*L|fL|pL|pg|ng|µg|ug|%))(?=\s|$)",
    re.IGNORECASE,
)
_RANGE_PATTERN = re.compile(rf"(?P<min>{_NUM})\s*[-–—]\s*(?P<max>{_NUM})")
_LESS_PATTERN = re.compile(rf"(?:preporuka\s*)?<\s*(?P<max>{_NUM})", re.IGNORECASE)
_GREATER_PATTERN = re.compile(rf">\s*(?P<min>{_NUM})", re.IGNORECASE)


def _fold(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", value).strip().casefold()


def _decimal(value: str | None) -> Decimal | None:
    if not value:
        return None
    try:
        return Decimal(value.replace(" ", "").replace(",", "."))
    except InvalidOperation:
        return None


def _parse_date(value: str) -> date | None:
    value = value.strip().rstrip(".")
    for format_string in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, format_string).date()
        except ValueError:
            continue
    return None


class LaboratoryPdfParser:
    """Extracts common laboratory parameters from text-based PDF reports."""

    MAX_PAGES = 20

    @classmethod
    def extract_text(cls, file_object: BinaryIO) -> str:
        try:
            file_object.seek(0)
            reader = PdfReader(file_object)
        except Exception as exc:  # pypdf raises several PDF-specific exceptions
            raise PdfParsingError("PDF dokument nije moguće otvoriti ili je oštećen.") from exc

        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception as exc:
                raise PdfParsingError("PDF je zaštićen lozinkom i nije ga moguće obraditi.") from exc
            if not unlocked:
                raise PdfParsingError("PDF je zaštićen lozinkom i nije ga moguće obraditi.")

        if len(reader.pages) > cls.MAX_PAGES:
            raise PdfParsingError(f"PDF može imati najviše {cls.MAX_PAGES} stranica.")

        pages: list[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
            except TypeError:  # compatibility with older pypdf releases
                text = page.extract_text() or ""
            pages.append(text)

        extracted = "\n".join(pages).replace("\x00", "")
        if len(re.sub(r"\s", "", extracted)) < 30:
            raise PdfParsingError(
                "U dokumentu nije pronađen čitljiv tekst. Vjerojatno je riječ o skeniranom nalazu; "
                "za njega će biti potreban OCR ili ručni unos."
            )
        return extracted

    @classmethod
    def parse(cls, file_object: BinaryIO) -> ParsedPdfReport:
        text = cls.extract_text(file_object)
        sampling_date = cls._extract_sampling_date(text)
        birth_date = cls._extract_birth_date(text)
        gender = cls._extract_gender(text)
        results = cls._extract_results(text)
        warnings: list[str] = []

        if not sampling_date:
            warnings.append("Datum uzorkovanja nije pouzdano prepoznat pa ga provjerite.")
        if not results:
            warnings.append("Nije automatski prepoznat nijedan laboratorijski parametar.")

        return ParsedPdfReport(
            sampling_date=sampling_date,
            birth_date=birth_date,
            gender=gender,
            results=results,
            raw_text=text,
            warnings=warnings,
        )

    @staticmethod
    def _extract_sampling_date(text: str) -> date | None:
        patterns = (
            r"Vrijeme\s+uzorkovanja\s*:\s*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"Datum\s+uzorkovanja\s*:\s*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            r"Datum\s+nalaza\s*:\s*(\d{1,2}\.\d{1,2}\.\d{2,4})",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and (parsed := _parse_date(match.group(1))):
                return parsed
        return None

    @staticmethod
    def _extract_birth_date(text: str) -> date | None:
        match = re.search(
            r"Datum\s+rođenja\s*:\s*(\d{1,2}\.\d{1,2}\.\d{2,4})",
            text,
            re.IGNORECASE,
        )
        return _parse_date(match.group(1)) if match else None

    @staticmethod
    def _extract_gender(text: str) -> str:
        match = re.search(r"Spol\s*:\s*([A-Za-zčćžšđČĆŽŠĐ]+)", text, re.IGNORECASE)
        if not match:
            return ""
        value = _fold(match.group(1))
        if value.startswith(("zensk", "female", "f")):
            return "female"
        if value.startswith(("musk", "male", "m")):
            return "male"
        return "other"

    @classmethod
    def _extract_results(cls, text: str) -> list[ParsedLaboratoryResult]:
        # Preserve layout lines because table columns are often represented by spacing.
        raw_lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        results_by_code: dict[str, ParsedLaboratoryResult] = {}

        for line in raw_lines:
            if not line or cls._should_skip_line(line):
                continue
            parsed = cls._parse_result_line(line)
            if parsed and parsed.parameter_code not in results_by_code:
                results_by_code[parsed.parameter_code] = parsed

        # Stable catalog order makes the review form easier to inspect.
        order = {definition.code: index for index, definition in enumerate(PARAMETERS)}
        return sorted(results_by_code.values(), key=lambda item: order.get(item.parameter_code, 999))

    @staticmethod
    def _should_skip_line(line: str) -> bool:
        folded = _fold(line)
        if line.lstrip().lower().startswith("(u)"):
            return True  # urinalysis is deliberately excluded from the blood-test module
        if folded.startswith("rezultat jedinica ref.interval"):
            return True
        blocked = (
            "vrijeme validacije",
            "nalaz pregledao",
            "laboratorijski broj",
            "datum rodenja",
            "vrijeme uzorkovanja",
            "prezime i ime",
        )
        return any(phrase in folded for phrase in blocked)

    @classmethod
    def _parse_result_line(cls, line: str) -> ParsedLaboratoryResult | None:
        cleaned = re.sub(r"^(?:\((?:vk|s|p|k|b)\)\s*)+", "", line, flags=re.IGNORECASE).strip()
        folded = _fold(cleaned)

        matched_definition = None
        matched_alias = ""
        alias_end = 0
        for alias, definition in _ALIAS_ENTRIES:
            alias_folded = _fold(alias)
            # Alias must be at the beginning, apart from common punctuation/spacing.
            if folded == alias_folded or folded.startswith(alias_folded + " "):
                matched_definition = definition
                matched_alias = alias
                # Find the end in the original string using an accent-insensitive token count.
                alias_word_count = len(alias_folded.split())
                parts = cleaned.split()
                alias_end = len(" ".join(parts[:alias_word_count]))
                break

        if not matched_definition:
            return None

        tail = cleaned[alias_end:].strip(" :.-")
        value_match = re.match(rf"^(?P<value>{_NUM}|norm(?:alno)?|negativno|pozitivno|reaktivno|nereaktivno)\b", tail, re.IGNORECASE)
        if not value_match:
            return None

        raw_value = value_match.group("value")
        numeric_value = _decimal(raw_value)
        text_value = "" if numeric_value is not None else raw_value
        remaining = tail[value_match.end():].strip()

        # Laboratory flag (H/L) is retained only as a confidence hint; reference limits decide status.
        flag_match = re.match(r"^[HL]\b", remaining, re.IGNORECASE)
        if flag_match:
            remaining = remaining[flag_match.end():].strip()

        unit = ""
        unit_match = _UNIT_PATTERN.match(remaining)
        if unit_match:
            unit = re.sub(r"\s+", "", unit_match.group("unit"))
            remaining = remaining[unit_match.end():].strip()

        raw_reference = remaining.strip()
        reference_text = raw_reference
        reference_min = reference_max = None
        range_match = _RANGE_PATTERN.search(raw_reference)
        if range_match:
            reference_min = _decimal(range_match.group("min"))
            reference_max = _decimal(range_match.group("max"))
            reference_text = range_match.group(0)
        elif less_match := _LESS_PATTERN.search(raw_reference):
            reference_max = _decimal(less_match.group("max"))
            reference_text = less_match.group(0)
        elif greater_match := _GREATER_PATTERN.search(raw_reference):
            reference_min = _decimal(greater_match.group("min"))
            reference_text = greater_match.group(0)

        confidence = "high" if reference_text and (reference_min is not None or reference_max is not None) else "medium"
        return ParsedLaboratoryResult(
            parameter_code=matched_definition.code,
            parameter_name=matched_definition.name,
            numeric_value=numeric_value,
            text_value=text_value,
            unit=unit or matched_definition.default_unit,
            reference_min=reference_min,
            reference_max=reference_max,
            reference_text=reference_text,
            source_line=line,
            confidence=confidence,
        )
