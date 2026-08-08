from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PUTANJE
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parents[1]

DATASET_PATH = ML_DIR / "data" / "raw" / "anemia_raw.xlsx"
REPORT_DIR = ML_DIR / "reports" / "data_quality"

TEXT_REPORT_PATH = REPORT_DIR / "data_quality_report.txt"
SUSPICIOUS_ROWS_PATH = REPORT_DIR / "suspicious_rows.xlsx"
DUPLICATE_ROWS_PATH = REPORT_DIR / "duplicate_rows.xlsx"
MISSING_ROWS_PATH = REPORT_DIR / "rows_with_missing_values.xlsx"
IQR_SUMMARY_PATH = REPORT_DIR / "iqr_outlier_summary.csv"


# ---------------------------------------------------------
# KONFIGURACIJA
# ---------------------------------------------------------

TARGET_COLUMN = "Decision_Class"
CATEGORICAL_COLUMNS = ["Gender"]
NUMERIC_COLUMNS = [
    "Age",
    "HGB",
    "RBC",
    "HCT",
    "MCV",
    "MCH",
    "MCHC",
]

EXPECTED_COLUMNS = [
    "Gender",
    "Age",
    "HGB",
    "RBC",
    "HCT",
    "MCV",
    "MCH",
    "MCHC",
    "Decision_Class",
]


# Pragovi služe za označavanje vrlo sumnjivih vrijednosti.
# Ne znače automatski da će redak biti obrisan.
REVIEW_THRESHOLDS = {
    "Age": {"min": 0, "max": 120},
    "HGB": {"min": 2, "max": 25},
    "RBC": {"min": 0.5, "max": 10},
    "HCT": {"min": 5, "max": 75},
    "MCV": {"min": 40, "max": 160},
    "MCH": {"min": 10, "max": 60},
    "MCHC": {"min": 20, "max": 60},
}


# ---------------------------------------------------------
# UČITAVANJE I STANDARDIZACIJA
# ---------------------------------------------------------

def create_output_directory() -> None:
    """
    Stvara mapu za izvještaje ako ona ne postoji.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizira nazive stupaca kako bi sve skripte koristile
    jednake nazive.
    """
    df = df.copy()

    df.columns = df.columns.str.strip()

    df = df.rename(
        columns={
            "HGB(Hemoglobin)": "HGB",
            "PCV/HCT": "HCT",
        }
    )

    return df


def load_dataset() -> pd.DataFrame:
    """
    Učitava originalni Excel dataset i standardizira nazive stupaca.

    Returns:
        pd.DataFrame: Učitani dataset.

    Raises:
        FileNotFoundError: Ako dataset ne postoji.
        ValueError: Ako nedostaje očekivani stupac.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset nije pronađen na putanji:\n{DATASET_PATH}"
        )

    df = pd.read_excel(DATASET_PATH)
    df = standardize_column_names(df)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "U datasetu nedostaju očekivani stupci: "
            + ", ".join(missing_columns)
        )

    return df


# ---------------------------------------------------------
# OSNOVNE PROVJERE
# ---------------------------------------------------------

def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vraća broj i postotak nedostajućih vrijednosti po stupcu.
    """
    summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percentage": (
                df.isna().mean() * 100
            ).round(2),
        }
    )

    return summary


def get_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vraća sve retke koji sudjeluju u duplikatima.

    keep=False znači da se prikazuje i originalni redak
    i njegova ponovljena kopija.
    """
    duplicate_rows = df[df.duplicated(keep=False)].copy()

    return duplicate_rows


def get_rows_with_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vraća sve retke koji sadrže barem jednu nedostajuću vrijednost.
    """
    return df[df.isna().any(axis=1)].copy()


def get_gender_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Vraća raspodjelu originalnih vrijednosti spola.

    Ovdje namjerno ne koristimo str.strip(), jer želimo otkriti
    probleme poput vrijednosti ' F'.
    """
    return df["Gender"].value_counts(dropna=False)


def get_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vraća broj i postotak zapisa za svaku ciljnu klasu.
    """
    distribution = (
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
    )

    distribution["percentage"] = (
        distribution["count"] / len(df) * 100
    ).round(2)

    return distribution


# ---------------------------------------------------------
# PROVJERA NUMERIČKIH VRIJEDNOSTI
# ---------------------------------------------------------

def get_numeric_range_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vraća osnovne raspone numeričkih varijabli.
    """
    summary = pd.DataFrame(
        {
            "minimum": df[NUMERIC_COLUMNS].min(),
            "maximum": df[NUMERIC_COLUMNS].max(),
            "mean": df[NUMERIC_COLUMNS].mean(),
            "median": df[NUMERIC_COLUMNS].median(),
            "standard_deviation": df[NUMERIC_COLUMNS].std(),
        }
    )

    return summary.round(4)


def get_non_positive_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provjerava negativne i nulte vrijednosti numeričkih parametara.
    """
    rows = []

    for column in NUMERIC_COLUMNS:
        negative_count = int((df[column] < 0).sum())
        zero_count = int((df[column] == 0).sum())

        rows.append(
            {
                "parameter": column,
                "negative_count": negative_count,
                "zero_count": zero_count,
            }
        )

    return pd.DataFrame(rows)


def detect_threshold_violations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pronalazi vrijednosti izvan unaprijed definiranih pragova
    za ručni pregled.

    Ovi pragovi nisu dijagnostički referentni intervali.
    Služe samo za detekciju potencijalno pogrešnih ili vrlo
    ekstremnih vrijednosti.
    """
    suspicious_rows = []

    for column, limits in REVIEW_THRESHOLDS.items():
        lower_limit = limits["min"]
        upper_limit = limits["max"]

        mask = (
            (df[column] < lower_limit)
            | (df[column] > upper_limit)
        )

        matching_rows = df.loc[mask].copy()

        for index, row in matching_rows.iterrows():
            value = row[column]

            reason = (
                f"{column} izvan raspona za ručni pregled "
                f"[{lower_limit}, {upper_limit}]"
            )

            record = row.to_dict()
            record["original_index"] = index
            record["flagged_parameter"] = column
            record["flagged_value"] = value
            record["flag_reason"] = reason

            suspicious_rows.append(record)

    if not suspicious_rows:
        return pd.DataFrame()

    suspicious_df = pd.DataFrame(suspicious_rows)

    preferred_columns = [
        "original_index",
        "flagged_parameter",
        "flagged_value",
        "flag_reason",
        *EXPECTED_COLUMNS,
    ]

    return suspicious_df[preferred_columns]


# ---------------------------------------------------------
# IQR ANALIZA OUTLIERA
# ---------------------------------------------------------

def calculate_iqr_outliers(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detektira statističke outliere IQR metodom.

    Vrijednost je označena ako je manja od:
        Q1 - 1.5 * IQR

    ili veća od:
        Q3 + 1.5 * IQR

    IQR outlier nije automatski pogrešan podatak.
    """
    summary_rows = []
    outlier_records = []

    for column in NUMERIC_COLUMNS:
        series = df[column].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        )

        outlier_count = int(mask.sum())

        summary_rows.append(
            {
                "parameter": column,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "outlier_count": outlier_count,
                "outlier_percentage": round(
                    outlier_count / len(df) * 100,
                    2,
                ),
            }
        )

        matching_rows = df.loc[mask].copy()

        for index, row in matching_rows.iterrows():
            record = row.to_dict()
            record["original_index"] = index
            record["iqr_parameter"] = column
            record["iqr_value"] = row[column]
            record["iqr_lower_bound"] = lower_bound
            record["iqr_upper_bound"] = upper_bound

            outlier_records.append(record)

    summary_df = pd.DataFrame(summary_rows)

    if outlier_records:
        outliers_df = pd.DataFrame(outlier_records)
    else:
        outliers_df = pd.DataFrame()

    return summary_df, outliers_df


# ---------------------------------------------------------
# PROVJERA POVEZANIH HEMATOLOŠKIH PARAMETARA
# ---------------------------------------------------------

def calculate_derived_consistency_checks(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Provjerava približnu matematičku konzistentnost povezanih
    hematoloških parametara.

    Standardne približne formule:

    MCV ≈ HCT / RBC * 10
    MCH ≈ HGB / RBC * 10
    MCHC ≈ HGB / HCT * 100

    Veća odstupanja mogu upućivati na pogrešku unosa ili
    nekonzistentan zapis, ali nisu automatski razlog za brisanje.
    """
    result = df.copy()

    result["Calculated_MCV"] = (
        result["HCT"] / result["RBC"] * 10
    )

    result["Calculated_MCH"] = (
        result["HGB"] / result["RBC"] * 10
    )

    result["Calculated_MCHC"] = (
        result["HGB"] / result["HCT"] * 100
    )

    result["MCV_absolute_difference"] = (
        result["MCV"] - result["Calculated_MCV"]
    ).abs()

    result["MCH_absolute_difference"] = (
        result["MCH"] - result["Calculated_MCH"]
    ).abs()

    result["MCHC_absolute_difference"] = (
        result["MCHC"] - result["Calculated_MCHC"]
    ).abs()

    # Pragovi su namijenjeni ručnom pregledu, ne automatskom brisanju.
    suspicious_mask = (
        (result["MCV_absolute_difference"] > 10)
        | (result["MCH_absolute_difference"] > 5)
        | (result["MCHC_absolute_difference"] > 5)
    )

    suspicious = result.loc[suspicious_mask].copy()

    selected_columns = [
        *EXPECTED_COLUMNS,
        "Calculated_MCV",
        "MCV_absolute_difference",
        "Calculated_MCH",
        "MCH_absolute_difference",
        "Calculated_MCHC",
        "MCHC_absolute_difference",
    ]

    return suspicious[selected_columns]


# ---------------------------------------------------------
# SPREMANJE REZULTATA
# ---------------------------------------------------------

def save_excel_reports(
    duplicate_rows: pd.DataFrame,
    missing_rows: pd.DataFrame,
    threshold_violations: pd.DataFrame,
    iqr_outliers: pd.DataFrame,
    consistency_issues: pd.DataFrame,
) -> None:
    """
    Sprema detaljne tablice u Excel datoteke.
    """
    duplicate_rows.to_excel(
        DUPLICATE_ROWS_PATH,
        index=True,
    )

    missing_rows.to_excel(
        MISSING_ROWS_PATH,
        index=True,
    )

    with pd.ExcelWriter(
        SUSPICIOUS_ROWS_PATH,
        engine="openpyxl",
    ) as writer:
        threshold_violations.to_excel(
            writer,
            sheet_name="Threshold violations",
            index=False,
        )

        iqr_outliers.to_excel(
            writer,
            sheet_name="IQR outliers",
            index=False,
        )

        consistency_issues.to_excel(
            writer,
            sheet_name="Consistency issues",
            index=True,
        )


def save_text_report(
    df: pd.DataFrame,
    missing_summary: pd.DataFrame,
    duplicate_rows: pd.DataFrame,
    gender_distribution: pd.Series,
    target_distribution: pd.DataFrame,
    numeric_range_summary: pd.DataFrame,
    non_positive_summary: pd.DataFrame,
    threshold_violations: pd.DataFrame,
    iqr_summary: pd.DataFrame,
    consistency_issues: pd.DataFrame,
) -> None:
    """
    Sprema tekstualni izvještaj o kvaliteti podataka.
    """
    total_duplicate_rows = int(df.duplicated().sum())

    with TEXT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write("DATA QUALITY REPORT – ANEMIA DATASET\n")
        report.write("=" * 70 + "\n\n")

        report.write("1. OSNOVNE INFORMACIJE\n")
        report.write("-" * 70 + "\n")
        report.write(f"Broj zapisa: {len(df)}\n")
        report.write(f"Broj stupaca: {df.shape[1]}\n")
        report.write(
            f"Broj ponovljenih zapisa prema duplicated(): "
            f"{total_duplicate_rows}\n"
        )
        report.write(
            f"Broj svih redaka uključenih u skupine duplikata: "
            f"{len(duplicate_rows)}\n"
        )
        report.write(
            f"Ukupan broj nedostajućih vrijednosti: "
            f"{int(df.isna().sum().sum())}\n\n"
        )

        report.write("2. NEDOSTAJUĆE VRIJEDNOSTI\n")
        report.write("-" * 70 + "\n")
        report.write(missing_summary.to_string())
        report.write("\n\n")

        report.write("3. ORIGINALNE VRIJEDNOSTI SPOLA\n")
        report.write("-" * 70 + "\n")
        report.write(gender_distribution.to_string())
        report.write("\n\n")

        report.write("4. RASPODJELA CILJNE VARIJABLE\n")
        report.write("-" * 70 + "\n")
        report.write(target_distribution.to_string(index=False))
        report.write("\n\n")

        report.write("5. RASPONI NUMERIČKIH VARIJABLI\n")
        report.write("-" * 70 + "\n")
        report.write(numeric_range_summary.to_string())
        report.write("\n\n")

        report.write("6. NEGATIVNE I NULTE VRIJEDNOSTI\n")
        report.write("-" * 70 + "\n")
        report.write(non_positive_summary.to_string(index=False))
        report.write("\n\n")

        report.write("7. VRIJEDNOSTI IZVAN PRAGOVA ZA RUČNI PREGLED\n")
        report.write("-" * 70 + "\n")
        report.write(
            f"Broj označenih vrijednosti: "
            f"{len(threshold_violations)}\n"
        )

        if not threshold_violations.empty:
            columns = [
                "original_index",
                "flagged_parameter",
                "flagged_value",
                "flag_reason",
            ]

            report.write(
                threshold_violations[columns].to_string(index=False)
            )
        else:
            report.write("Nisu pronađene označene vrijednosti.")

        report.write("\n\n")

        report.write("8. IQR OUTLIERI\n")
        report.write("-" * 70 + "\n")
        report.write(iqr_summary.to_string(index=False))
        report.write("\n\n")

        report.write("9. NEKONZISTENTNI HEMATOLOŠKI IZRAČUNI\n")
        report.write("-" * 70 + "\n")
        report.write(
            f"Broj zapisa označenih za ručni pregled: "
            f"{len(consistency_issues)}\n"
        )

        if not consistency_issues.empty:
            preview_columns = [
                "HGB",
                "RBC",
                "HCT",
                "MCV",
                "Calculated_MCV",
                "MCV_absolute_difference",
                "MCH",
                "Calculated_MCH",
                "MCH_absolute_difference",
                "MCHC",
                "Calculated_MCHC",
                "MCHC_absolute_difference",
            ]

            report.write(
                consistency_issues[preview_columns]
                .head(30)
                .to_string()
            )
        else:
            report.write(
                "Nisu pronađeni zapisi s većim odstupanjima."
            )

        report.write("\n\n")

        report.write("10. NAPOMENA O TUMAČENJU\n")
        report.write("-" * 70 + "\n")
        report.write(
            "Statistički outlier ili vrijednost izvan zadanog praga "
            "nije automatski pogrešan podatak. Rezultati ove skripte "
            "služe za ručni pregled i dokumentiranje odluka prije "
            "čišćenja skupa podataka.\n"
        )


# ---------------------------------------------------------
# ISPIS U TERMINAL
# ---------------------------------------------------------

def print_console_summary(
    df: pd.DataFrame,
    threshold_violations: pd.DataFrame,
    iqr_summary: pd.DataFrame,
    consistency_issues: pd.DataFrame,
) -> None:
    """
    Ispisuje najvažnije rezultate u terminal.
    """
    print("=" * 70)
    print("DATA QUALITY CHECK – ANEMIA DATASET")
    print("=" * 70)

    print(f"Broj zapisa: {len(df)}")
    print(f"Broj stupaca: {df.shape[1]}")
    print(f"Broj duplikata: {df.duplicated().sum()}")
    print(
        "Ukupan broj nedostajućih vrijednosti: "
        f"{df.isna().sum().sum()}"
    )

    print("\nOriginalne vrijednosti spola:")
    print(df["Gender"].value_counts(dropna=False))

    print("\nMinimalne vrijednosti:")
    print(df[NUMERIC_COLUMNS].min())

    print("\nMaksimalne vrijednosti:")
    print(df[NUMERIC_COLUMNS].max())

    print("\nVrijednosti izvan pragova za ručni pregled:")
    if threshold_violations.empty:
        print("Nema označenih vrijednosti.")
    else:
        print(
            threshold_violations[
                [
                    "original_index",
                    "flagged_parameter",
                    "flagged_value",
                ]
            ].to_string(index=False)
        )

    print("\nIQR sažetak:")
    print(
        iqr_summary[
            [
                "parameter",
                "lower_bound",
                "upper_bound",
                "outlier_count",
            ]
        ].to_string(index=False)
    )

    print(
        "\nBroj zapisa s većim odstupanjem između izvornih "
        "i izračunatih hematoloških parametara:"
    )
    print(len(consistency_issues))

    print("\nIzvještaji su spremljeni u:")
    print(REPORT_DIR)


# ---------------------------------------------------------
# GLAVNI PROGRAM
# ---------------------------------------------------------

def main() -> None:
    """
    Glavna funkcija skripte.
    """
    create_output_directory()

    print("Učitavanje dataseta...")
    df = load_dataset()

    print("Analiza nedostajućih vrijednosti...")
    missing_summary = get_missing_value_summary(df)
    missing_rows = get_rows_with_missing_values(df)

    print("Analiza duplikata...")
    duplicate_rows = get_duplicate_rows(df)

    print("Analiza spola i ciljne varijable...")
    gender_distribution = get_gender_distribution(df)
    target_distribution = get_target_distribution(df)

    print("Analiza numeričkih raspona...")
    numeric_range_summary = get_numeric_range_summary(df)
    non_positive_summary = get_non_positive_values_summary(df)

    print("Provjera pragova za ručni pregled...")
    threshold_violations = detect_threshold_violations(df)

    print("IQR analiza outliera...")
    iqr_summary, iqr_outliers = calculate_iqr_outliers(df)

    print("Provjera konzistentnosti hematoloških izračuna...")
    consistency_issues = calculate_derived_consistency_checks(df)

    print("Spremanje CSV izvještaja...")
    iqr_summary.to_csv(
        IQR_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("Spremanje Excel izvještaja...")
    save_excel_reports(
        duplicate_rows=duplicate_rows,
        missing_rows=missing_rows,
        threshold_violations=threshold_violations,
        iqr_outliers=iqr_outliers,
        consistency_issues=consistency_issues,
    )

    print("Spremanje tekstualnog izvještaja...")
    save_text_report(
        df=df,
        missing_summary=missing_summary,
        duplicate_rows=duplicate_rows,
        gender_distribution=gender_distribution,
        target_distribution=target_distribution,
        numeric_range_summary=numeric_range_summary,
        non_positive_summary=non_positive_summary,
        threshold_violations=threshold_violations,
        iqr_summary=iqr_summary,
        consistency_issues=consistency_issues,
    )

    print_console_summary(
        df=df,
        threshold_violations=threshold_violations,
        iqr_summary=iqr_summary,
        consistency_issues=consistency_issues,
    )

    print("\nAnaliza kvalitete podataka uspješno je završena.")


if __name__ == "__main__":
    main()