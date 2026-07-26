"""Reliable PDF extraction and normalization for laboratory reports.

The parser keeps the original laboratory interval for auditability, normalizes common
Croatian/English parameter aliases and converts only well-defined unit combinations.
Every result receives a confidence level so the review screen can focus the user's
attention on rows that genuinely need checking.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import BinaryIO, Callable

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
    normalization_note: str = ""


@dataclass
class ParsedPdfReport:
    sampling_date: date | None
    birth_date: date | None
    gender: str
    results: list[ParsedLaboratoryResult]
    raw_text: str
    warnings: list[str]


PARAMETERS: tuple[ParameterDefinition, ...] = (
    ParameterDefinition("WBC", "Leukociti (WBC)", ("leukociti", "leukocytes", "wbc", "lkc"), "10^9/L"),
    ParameterDefinition("RBC", "Eritrociti (RBC)", ("eritrociti", "erythrocytes", "rbc", "erc"), "10^12/L"),
    ParameterDefinition("HGB", "Hemoglobin (HGB)", ("hemoglobin", "haemoglobin", "hgb", "hb"), "g/L"),
    ParameterDefinition("HCT", "Hematokrit (HCT)", ("hematokrit", "haematocrit", "hct"), "L/L"),
    ParameterDefinition("MCV", "Prosječni volumen eritrocita (MCV)", ("mcv",), "fL"),
    ParameterDefinition("MCH", "Prosječna količina hemoglobina (MCH)", ("mch",), "pg"),
    ParameterDefinition("MCHC", "Prosječna koncentracija hemoglobina (MCHC)", ("mchc",), "g/L"),
    ParameterDefinition("RDW", "Raspodjela eritrocita (RDW)", ("rdw-cv", "rdw-kv", "rdw cv", "rdw"), "%"),
    ParameterDefinition("PLT", "Trombociti (PLT)", ("trombociti", "platelets", "plt", "trc"), "10^9/L"),
    ParameterDefinition("MPV", "Prosječni volumen trombocita (MPV)", ("mpv",), "fL"),
    ParameterDefinition("NEUT", "Neutrofili", ("neutrofili aps", "neutrophils abs", "neutrofili", "neut"), "10^9/L"),
    ParameterDefinition("LYMPH", "Limfociti", ("limfociti aps", "lymphocytes abs", "limfociti", "lymph"), "10^9/L"),
    ParameterDefinition("MONO", "Monociti", ("monociti aps", "monocytes abs", "monociti", "mono"), "10^9/L"),
    ParameterDefinition("EOS", "Eozinofili", ("eozinofili aps", "eosinophils abs", "eozinofili", "eos"), "10^9/L"),
    ParameterDefinition("BASO", "Bazofili", ("bazofili aps", "basophils abs", "bazofili", "baso"), "10^9/L"),
    ParameterDefinition("GLU", "Glukoza", ("glukoza u plazmi", "glukoza", "glucose", "glu"), "mmol/L"),
    ParameterDefinition("UREA", "Urea", ("urea", "blood urea"), "mmol/L"),
    ParameterDefinition("CREA", "Kreatinin", ("kreatinin", "creatinine", "crea"), "µmol/L"),
    ParameterDefinition("EGFR", "Procijenjena glomerularna filtracija (eGFR)", ("egfr", "procijenjena glomerularna filtracija"), "mL/min/1.73m2"),
    ParameterDefinition("FE", "Željezo", ("serumsko zeljezo", "serumsko željezo", "željezo", "zeljezo", "iron", "fe"), "µmol/L"),
    ParameterDefinition("UIBC", "Nezasićeni kapacitet vezanja željeza (UIBC)", ("nez.kap.vez.željeza", "nezasiceni kapacitet vezanja zeljeza", "uibc"), "µmol/L"),
    ParameterDefinition("TIBC", "Ukupni kapacitet vezanja željeza (TIBC)", ("uk.kap.vez.željeza", "ukupni kapacitet vezanja zeljeza", "tibc"), "µmol/L"),
    ParameterDefinition("FERR", "Feritin", ("feritin", "ferritin", "ferr"), "µg/L"),
    ParameterDefinition("CRP", "C-reaktivni protein (CRP)", ("c-reaktivni protein", "c reactive protein", "crp"), "mg/L"),
    ParameterDefinition("CHOL", "Kolesterol", ("ukupni kolesterol", "kolesterol", "total cholesterol", "cholesterol"), "mmol/L"),
    ParameterDefinition("HDL", "HDL kolesterol", ("hdl kolesterol", "hdl cholesterol", "hdl"), "mmol/L"),
    ParameterDefinition("LDL", "LDL kolesterol", ("ldl kolesterol", "ldl cholesterol", "ldl"), "mmol/L"),
    ParameterDefinition("TG", "Trigliceridi", ("trigliceridi", "triglycerides", "tg"), "mmol/L"),
    ParameterDefinition("TSH", "TSH", ("tireotropin", "thyroid stimulating hormone", "tsh"), "mIU/L"),
    ParameterDefinition("FT4", "Slobodni tiroksin (fT4)", ("slobodni t4", "free t4", "ft4"), "pmol/L"),
    ParameterDefinition("FT3", "Slobodni trijodtironin (fT3)", ("slobodni t3", "free t3", "ft3"), "pmol/L"),
    ParameterDefinition("ALT", "Alanin-aminotransferaza (ALT)", ("alanin aminotransferaza", "alanin-aminotransferaza", "alat", "alt"), "U/L"),
    ParameterDefinition("AST", "Aspartat-aminotransferaza (AST)", ("aspartat aminotransferaza", "aspartat-aminotransferaza", "asat", "ast"), "U/L"),
    ParameterDefinition("GGT", "Gama-glutamiltransferaza (GGT)", ("gama glutamiltransferaza", "gama-glutamiltransferaza", "gamma gt", "ggt"), "U/L"),
    ParameterDefinition("BILI", "Bilirubin ukupni", ("bilirubin ukupni", "ukupni bilirubin", "total bilirubin"), "µmol/L"),
    ParameterDefinition("ALP", "Alkalna fosfataza (ALP)", ("alkalna fosfataza", "alkaline phosphatase", "alp"), "U/L"),
    ParameterDefinition("NA", "Natrij", ("natrij", "sodium", "na"), "mmol/L"),
    ParameterDefinition("K", "Kalij", ("kalij", "potassium"), "mmol/L"),
    ParameterDefinition("CA", "Kalcij", ("ukupni kalcij", "kalcij", "calcium"), "mmol/L"),
    ParameterDefinition("MG", "Magnezij", ("magnezij", "magnesium"), "mmol/L"),
    ParameterDefinition("B12", "Vitamin B12", ("vitamin b12", "cobalamin", "b12"), "pmol/L"),
    ParameterDefinition("FOL", "Folat", ("folna kiselina", "folat", "folate"), "nmol/L"),
    ParameterDefinition("VITD", "Vitamin D", ("25-oh vitamin d", "25 oh vitamin d", "25-hidroksi vitamin d", "vitamin d"), "nmol/L"),
    ParameterDefinition("HBA1C", "HbA1c", ("hemoglobin a1c", "glikirani hemoglobin", "hba1c"), "%"),
    ParameterDefinition("PROT", "Ukupni proteini", ("ukupni proteini", "total protein"), "g/L"),
    ParameterDefinition("ALB", "Albumin", ("albumin",), "g/L"),
)

_ALIAS_ENTRIES = sorted(
    ((_fold_alias := alias, definition) for definition in PARAMETERS for alias in definition.aliases),
    key=lambda item: len(item[0]), reverse=True,
)

_NUM = r"[-+]?\d+(?:[.,]\d+)?"
_RANGE_PATTERN = re.compile(rf"(?P<min>{_NUM})\s*(?:-|–|—|do)\s*(?P<max>{_NUM})", re.IGNORECASE)
_LESS_PATTERN = re.compile(rf"(?:preporuka\s*)?(?:<|≤|do)\s*(?P<max>{_NUM})", re.IGNORECASE)
_GREATER_PATTERN = re.compile(rf"(?:>|≥|iznad)\s*(?P<min>{_NUM})", re.IGNORECASE)
_TEXT_VALUES = r"norm(?:alno)?|negativno|pozitivno|reaktivno|nereaktivno|prisutan|odsutan"


def _fold(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = value.replace("×", "x").replace("−", "-")
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
    for format_string in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, format_string).date()
        except ValueError:
            continue
    return None


def _normalize_unit(raw: str) -> str:
    unit = raw.strip().replace("μ", "µ").replace("×", "x")
    unit = re.sub(r"\s+", "", unit)
    folded = unit.casefold().replace("µ", "u")
    replacements = {
        "g/l": "g/L", "g/dl": "g/dL", "mg/l": "mg/L", "mg/dl": "mg/dL",
        "mmol/l": "mmol/L", "umol/l": "µmol/L", "µmol/l": "µmol/L",
        "ug/l": "µg/L", "µg/l": "µg/L", "ug/dl": "µg/dL", "ng/ml": "ng/mL",
        "miu/l": "mIU/L", "uiu/ml": "µIU/mL", "u/l": "U/L", "iu/l": "IU/L",
        "l/l": "L/L", "%": "%", "fl": "fL", "pl": "pL", "pg": "pg",
        "10^9/l": "10^9/L", "10*9/l": "10^9/L", "10x9/l": "10^9/L",
        "10^12/l": "10^12/L", "10*12/l": "10^12/L", "10x12/l": "10^12/L",
        "ml/min/1.73m2": "mL/min/1.73m2",
    }
    return replacements.get(folded, unit)


def _conversion(code: str, unit: str) -> tuple[str, Decimal, str] | None:
    """Return canonical unit and multiplier for clinically unambiguous conversions."""
    table: dict[tuple[str, str], tuple[str, str, str]] = {
        ("HGB", "g/dL"): ("g/L", "10", "Pretvoreno iz g/dL u g/L."),
        ("MCHC", "g/dL"): ("g/L", "10", "Pretvoreno iz g/dL u g/L."),
        ("CREA", "mg/dL"): ("µmol/L", "88.4", "Pretvoreno iz mg/dL u µmol/L."),
        ("GLU", "mg/dL"): ("mmol/L", "0.0555", "Pretvoreno iz mg/dL u mmol/L."),
        ("CHOL", "mg/dL"): ("mmol/L", "0.02586", "Pretvoreno iz mg/dL u mmol/L."),
        ("HDL", "mg/dL"): ("mmol/L", "0.02586", "Pretvoreno iz mg/dL u mmol/L."),
        ("LDL", "mg/dL"): ("mmol/L", "0.02586", "Pretvoreno iz mg/dL u mmol/L."),
        ("TG", "mg/dL"): ("mmol/L", "0.01129", "Pretvoreno iz mg/dL u mmol/L."),
        ("FE", "µg/dL"): ("µmol/L", "0.1791", "Pretvoreno iz µg/dL u µmol/L."),
        ("CA", "mg/dL"): ("mmol/L", "0.2495", "Pretvoreno iz mg/dL u mmol/L."),
    }
    target = table.get((code, unit))
    if not target:
        return None
    return target[0], Decimal(target[1]), target[2]


class LaboratoryPdfParser:
    MAX_PAGES = 20

    @classmethod
    def extract_text(cls, file_object: BinaryIO) -> str:
        try:
            file_object.seek(0)
            reader = PdfReader(file_object)
        except Exception as exc:
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
        pages = []
        for page in reader.pages:
            try:
                page_text = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
            except TypeError:
                page_text = page.extract_text() or ""
            pages.append(page_text)
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
        results = cls._extract_results(text)
        warnings = []
        sampling_date = cls._extract_sampling_date(text)
        if not sampling_date:
            warnings.append("Datum uzorkovanja nije pouzdano prepoznat pa ga provjerite.")
        low_count = sum(item.confidence == "low" for item in results)
        medium_count = sum(item.confidence == "medium" for item in results)
        if low_count:
            warnings.append(f"{low_count} prepoznatih redaka ima nisku pouzdanost i označeno je za obaveznu provjeru.")
        elif medium_count:
            warnings.append(f"{medium_count} prepoznatih redaka preporučuje se kratko provjeriti.")
        if not results:
            warnings.append("Nije automatski prepoznat nijedan laboratorijski parametar.")
        return ParsedPdfReport(
            sampling_date=sampling_date,
            birth_date=cls._extract_birth_date(text),
            gender=cls._extract_gender(text),
            results=results,
            raw_text=text,
            warnings=warnings,
        )

    @staticmethod
    def _extract_sampling_date(text: str) -> date | None:
        patterns = (
            r"(?:Vrijeme|Datum)\s+uzorkovanja\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{2,4})",
            r"Datum\s+(?:nalaza|prijema|izdavanja)\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{2,4})",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and (parsed := _parse_date(match.group(1))):
                return parsed
        return None

    @staticmethod
    def _extract_birth_date(text: str) -> date | None:
        match = re.search(r"Datum\s+ro(?:đ|d)enja\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{2,4})", text, re.IGNORECASE)
        return _parse_date(match.group(1)) if match else None

    @staticmethod
    def _extract_gender(text: str) -> str:
        match = re.search(r"Spol\s*:?\s*([A-Za-zčćžšđČĆŽŠĐ]+)", text, re.IGNORECASE)
        if not match:
            return ""
        value = _fold(match.group(1))
        if value.startswith(("zensk", "female", "f")):
            return "female"
        if value.startswith(("musk", "male", "m")):
            return "male"
        return "other"

    @classmethod
    def _candidate_lines(cls, text: str) -> list[str]:
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        candidates: list[str] = []
        for index, line in enumerate(lines):
            if not line:
                continue
            candidates.append(line)
            # Some PDFs split a table row after the value or unit. Join only short continuations.
            if index + 1 < len(lines):
                following = lines[index + 1]
                if following and len(following) < 80 and not cls._line_starts_with_parameter(following):
                    candidates.append(f"{line} {following}")
        return candidates

    @classmethod
    def _line_starts_with_parameter(cls, line: str) -> bool:
        cleaned = re.sub(r"^(?:\((?:vk|s|p|k|b)\)\s*)+", "", line, flags=re.IGNORECASE).strip()
        folded = _fold(cleaned)
        return any(folded == _fold(alias) or folded.startswith(_fold(alias) + " ") for alias, _ in _ALIAS_ENTRIES)

    @classmethod
    def _extract_results(cls, text: str) -> list[ParsedLaboratoryResult]:
        results_by_code: dict[str, ParsedLaboratoryResult] = {}
        for line in cls._candidate_lines(text):
            if cls._should_skip_line(line):
                continue
            parsed = cls._parse_result_line(line)
            if not parsed:
                continue
            previous = results_by_code.get(parsed.parameter_code)
            if previous is None or cls._quality_score(parsed) > cls._quality_score(previous):
                results_by_code[parsed.parameter_code] = parsed
        order = {definition.code: index for index, definition in enumerate(PARAMETERS)}
        return sorted(results_by_code.values(), key=lambda item: order.get(item.parameter_code, 999))

    @staticmethod
    def _quality_score(item: ParsedLaboratoryResult) -> int:
        score = {"low": 1, "medium": 3, "high": 5}.get(item.confidence, 0)
        score += 2 if item.unit else 0
        score += 2 if item.reference_min is not None or item.reference_max is not None else 0
        score += 1 if item.numeric_value is not None else 0
        return score

    @staticmethod
    def _should_skip_line(line: str) -> bool:
        folded = _fold(line)
        if line.lstrip().lower().startswith("(u)"):
            return True
        blocked = (
            "vrijeme validacije", "nalaz pregledao",
            "laboratorijski broj", "datum rodenja", "vrijeme uzorkovanja", "prezime i ime",
            "napomena", "materijal",
        )
        return not line or any(phrase in folded for phrase in blocked)

    @classmethod
    def _parse_result_line(cls, line: str) -> ParsedLaboratoryResult | None:
        cleaned = re.sub(r"^(?:\((?:vk|s|p|k|b)\)\s*)+", "", line, flags=re.IGNORECASE).strip()
        folded = _fold(cleaned)
        matched_definition = None
        alias_end = 0
        for alias, definition in _ALIAS_ENTRIES:
            alias_folded = _fold(alias)
            if folded == alias_folded or folded.startswith(alias_folded + " ") or folded.startswith(alias_folded + ":"):
                matched_definition = definition
                alias_word_count = len(alias_folded.split())
                parts = cleaned.split()
                alias_end = len(" ".join(parts[:alias_word_count]))
                break
        if not matched_definition:
            return None

        tail = cleaned[alias_end:].strip(" :.-")
        value_match = re.match(rf"^(?P<value>{_NUM}|{_TEXT_VALUES})(?:\s|$)", tail, re.IGNORECASE)
        if not value_match:
            return None
        raw_value = value_match.group("value")
        numeric_value = _decimal(raw_value)
        text_value = "" if numeric_value is not None else raw_value.strip()
        remaining = tail[value_match.end():].strip()

        flag = ""
        flag_match = re.match(r"^(?:\*\s*)?(?P<flag>HI|LO|H|L|↑|↓)(?:\s|$)", remaining, re.IGNORECASE)
        if flag_match:
            flag = flag_match.group("flag")
            remaining = remaining[flag_match.end():].strip()

        unit = ""
        # Unit ends before a reference relation/range. Supports compact and spaced variants.
        unit_match = re.match(
            r"^(?P<unit>(?:10\s*(?:\^|\*)?\s*(?:9|12)\s*/\s*[lL]|mL/min/1[.,]73m2|[mµu]?g\s*/\s*d?[lL]|[npmkµu]?mol\s*/\s*[lL]|m?IU\s*/\s*[lL]|U\s*/\s*[lL]|L\s*/\s*L|fL|pL|pg|ng/mL|%))(?=\s|$)",
            remaining, re.IGNORECASE,
        )
        if unit_match:
            unit = _normalize_unit(unit_match.group("unit"))
            remaining = remaining[unit_match.end():].strip()

        raw_reference = remaining.strip(" ;")
        reference_text = raw_reference
        reference_min = reference_max = None
        if range_match := _RANGE_PATTERN.search(raw_reference):
            reference_min = _decimal(range_match.group("min"))
            reference_max = _decimal(range_match.group("max"))
            reference_text = range_match.group(0)
        elif less_match := _LESS_PATTERN.search(raw_reference):
            reference_max = _decimal(less_match.group("max"))
            reference_text = less_match.group(0)
        elif greater_match := _GREATER_PATTERN.search(raw_reference):
            reference_min = _decimal(greater_match.group("min"))
            reference_text = greater_match.group(0)

        normalization_note = ""
        final_unit = unit or matched_definition.default_unit
        conversion = _conversion(matched_definition.code, final_unit)
        if conversion and numeric_value is not None:
            target_unit, factor, normalization_note = conversion
            numeric_value *= factor
            if reference_min is not None:
                reference_min *= factor
            if reference_max is not None:
                reference_max *= factor
            final_unit = target_unit

        confidence = "high"
        if numeric_value is None and not text_value:
            confidence = "low"
        elif not unit:
            confidence = "medium"
        elif reference_min is None and reference_max is None:
            confidence = "medium"
        if raw_reference and reference_min is None and reference_max is None:
            confidence = "low"
        if flag and reference_min is None and reference_max is None:
            confidence = "low"

        return ParsedLaboratoryResult(
            parameter_code=matched_definition.code,
            parameter_name=matched_definition.name,
            numeric_value=numeric_value,
            text_value=text_value,
            unit=final_unit,
            reference_min=reference_min,
            reference_max=reference_max,
            reference_text=reference_text,
            source_line=line[:500],
            confidence=confidence,
            normalization_note=normalization_note,
        )
