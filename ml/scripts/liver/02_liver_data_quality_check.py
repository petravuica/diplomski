from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PUTANJE
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parents[1]

DATASET_PATH = (
    ML_DIR
    / "data"
    / "raw"
    / "liver"
    / "liver_raw.csv"
)

REPORT_DIR = (
    ML_DIR
    / "reports"
    / "liver"
    / "data_quality"
)

TEXT_REPORT_PATH = (
    REPORT_DIR
    / "liver_data_quality_report.txt"
)

DUPLICATE_ROWS_PATH = (
    REPORT_DIR
    / "duplicate_rows.xlsx"
)

MISSING_ROWS_PATH = (
    REPORT_DIR
    / "rows_with_missing_values.xlsx"
)

SUSPICIOUS_ROWS_PATH = (
    REPORT_DIR
    / "suspicious_rows.xlsx"
)

IQR_SUMMARY_PATH = (
    REPORT_DIR
    / "iqr_outlier_summary.csv"
)


# ---------------------------------------------------------
# KONFIGURACIJA
# ---------------------------------------------------------

TARGET_COLUMN = "Dataset"

EXPECTED_COLUMNS_RAW = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
    "Dataset",
]

COLUMN_RENAME_MAP = {
    "Alkaline_Phosphotase": "Alkaline_Phosphatase",
    "Alamine_Aminotransferase": "Alanine_Aminotransferase",
    "Total_Protiens": "Total_Proteins",
}

EXPECTED_COLUMNS = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphatase",
    "Alanine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
    "Dataset",
]

NUMERIC_COLUMNS = [
    "Age",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphatase",
    "Alanine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]

ALLOWED_GENDERS = {"Male", "Female"}
ALLOWED_TARGET_VALUES = {1, 2}


# Široke tehničke granice za ručni pregled.
# To nisu normalni laboratorijski referentni intervali.
REVIEW_THRESHOLDS = {
    "Age": {"min": 0, "max": 120},
    "Total_Bilirubin": {"min": 0, "max": 100},
    "Direct_Bilirubin": {"min": 0, "max": 50},
    "Alkaline_Phosphatase": {"min": 0, "max": 3000},
    "Alanine_Aminotransferase": {"min": 0, "max": 3000},
    "Aspartate_Aminotransferase": {"min": 0, "max": 6000},
    "Total_Proteins": {"min": 0, "max": 15},
    "Albumin": {"min": 0, "max": 10},
    "Albumin_and_Globulin_Ratio": {"min": 0, "max": 5},
}


# ---------------------------------------------------------
# UČITAVANJE I STANDARDIZACIJA
# ---------------------------------------------------------

def create_output_directory() -> None:
    """
    Stvara mapu za izvještaje ako ne postoji.
    """
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def standardize_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Uklanja razmake iz naziva stupaca i ispravlja
    poznate tipfelere iz izvornog ILPD dataseta.
    """
    df = df.copy()

    df.columns = df.columns.str.strip()

    df = df.rename(
        columns=COLUMN_RENAME_MAP
    )

    return df


def load_dataset() -> pd.DataFrame:
    """
    Učitava originalni ILPD dataset i provjerava strukturu.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Jetreni dataset nije pronađen na putanji:\n"
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)
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

    return df[EXPECTED_COLUMNS].copy()


# ---------------------------------------------------------
# OSNOVNE PROVJERE
# ---------------------------------------------------------

def get_missing_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vraća broj i postotak nedostajućih vrijednosti.
    """
    return pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percentage": (
                df.isna().mean() * 100
            ).round(2),
        }
    )


def get_rows_with_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vraća retke koji sadrže barem jednu missing vrijednost.
    """
    return df.loc[
        df.isna().any(axis=1)
    ].copy()


def get_duplicate_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vraća sve retke koji pripadaju skupinama duplikata.
    """
    return df.loc[
        df.duplicated(keep=False)
    ].copy()


def get_gender_distribution(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Vraća raspodjelu originalnih oznaka spola.
    """
    return df["Gender"].value_counts(
        dropna=False
    )


def get_target_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vraća broj i postotak zapisa po ciljnoj klasi.
    """
    distribution = (
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
    )

    distribution["percentage"] = (
        distribution["count"]
        / len(df)
        * 100
    ).round(2)

    return distribution


# ---------------------------------------------------------
# NUMERIČKE PROVJERE
# ---------------------------------------------------------

def get_numeric_range_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vraća osnovne statističke raspone numeričkih značajki.
    """
    summary = pd.DataFrame(
        {
            "minimum": df[NUMERIC_COLUMNS].min(),
            "maximum": df[NUMERIC_COLUMNS].max(),
            "mean": df[NUMERIC_COLUMNS].mean(),
            "median": df[NUMERIC_COLUMNS].median(),
            "standard_deviation": (
                df[NUMERIC_COLUMNS].std()
            ),
        }
    )

    return summary.round(4)


def get_non_positive_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi negativne i nulte vrijednosti.
    """
    rows = []

    for column in NUMERIC_COLUMNS:
        rows.append(
            {
                "parameter": column,
                "negative_count": int(
                    (df[column] < 0).sum()
                ),
                "zero_count": int(
                    (df[column] == 0).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def detect_threshold_violations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Označava vrijednosti izvan širokih tehničkih granica.

    Ove vrijednosti nisu automatski pogrešne i neće
    se automatski uklanjati.
    """
    suspicious_records = []

    for column, limits in REVIEW_THRESHOLDS.items():
        lower_limit = limits["min"]
        upper_limit = limits["max"]

        mask = (
            (df[column] < lower_limit)
            | (df[column] > upper_limit)
        )

        matching_rows = df.loc[mask].copy()

        for index, row in matching_rows.iterrows():
            record = row.to_dict()

            record["original_index"] = index
            record["flagged_parameter"] = column
            record["flagged_value"] = row[column]
            record["flag_reason"] = (
                f"{column} izvan raspona za ručni pregled "
                f"[{lower_limit}, {upper_limit}]"
            )

            suspicious_records.append(record)

    if not suspicious_records:
        return pd.DataFrame()

    result = pd.DataFrame(suspicious_records)

    preferred_columns = [
        "original_index",
        "flagged_parameter",
        "flagged_value",
        "flag_reason",
        *EXPECTED_COLUMNS,
    ]

    return result[preferred_columns]


# ---------------------------------------------------------
# IQR OUTLIERI
# ---------------------------------------------------------

def calculate_iqr_outliers(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Izračunava IQR statističke outliere.

    IQR outlier nije automatski pogrešan podatak.
    """
    summary_rows = []
    outlier_rows = []

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

        count = int(mask.sum())

        summary_rows.append(
            {
                "parameter": column,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(
                    lower_bound,
                    4,
                ),
                "upper_bound": round(
                    upper_bound,
                    4,
                ),
                "outlier_count": count,
                "outlier_percentage": round(
                    count / len(df) * 100,
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

            outlier_rows.append(record)

    summary_df = pd.DataFrame(summary_rows)

    if outlier_rows:
        outliers_df = pd.DataFrame(outlier_rows)
    else:
        outliers_df = pd.DataFrame()

    return summary_df, outliers_df


# ---------------------------------------------------------
# KLINIČKO-LOGIČKE PROVJERE
# ---------------------------------------------------------

def check_bilirubin_consistency(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Direktni bilirubin ne bi trebao biti veći
    od ukupnog bilirubina.
    """
    mask = (
        df["Direct_Bilirubin"]
        > df["Total_Bilirubin"]
    )

    result = df.loc[mask].copy()

    if not result.empty:
        result["consistency_reason"] = (
            "Direktni bilirubin veći je od ukupnog bilirubina."
        )

    return result


def check_albumin_consistency(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Albumin ne bi trebao biti veći od ukupnih proteina.
    """
    mask = (
        df["Albumin"]
        > df["Total_Proteins"]
    )

    result = df.loc[mask].copy()

    if not result.empty:
        result["consistency_reason"] = (
            "Albumin je veći od ukupnih proteina."
        )

    return result


def calculate_ag_ratio_consistency(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Provjerava približnu konzistentnost A/G omjera.

    Globulini se približno računaju kao:
        Total proteins - Albumin

    A/G omjer:
        Albumin / Globulini
    """
    result = df.copy()

    result["Calculated_Globulin"] = (
        result["Total_Proteins"]
        - result["Albumin"]
    )

    valid_denominator = (
        result["Calculated_Globulin"] > 0
    )

    result["Calculated_AG_Ratio"] = pd.NA

    result.loc[
        valid_denominator,
        "Calculated_AG_Ratio",
    ] = (
        result.loc[
            valid_denominator,
            "Albumin",
        ]
        / result.loc[
            valid_denominator,
            "Calculated_Globulin",
        ]
    )

    result["AG_Ratio_Absolute_Difference"] = (
        result["Albumin_and_Globulin_Ratio"]
        - result["Calculated_AG_Ratio"]
    ).abs()

    # Omjer u datasetu može biti zaokružen.
    # Za ručni pregled označavamo veće odstupanje od 0.25.
    mask = (
        result[
            "Albumin_and_Globulin_Ratio"
        ].notna()
        & (
            result[
                "AG_Ratio_Absolute_Difference"
            ]
            > 0.25
        )
    )

    selected_columns = [
        *EXPECTED_COLUMNS,
        "Calculated_Globulin",
        "Calculated_AG_Ratio",
        "AG_Ratio_Absolute_Difference",
    ]

    return result.loc[
        mask,
        selected_columns,
    ].copy()


def combine_consistency_issues(
    bilirubin_issues: pd.DataFrame,
    albumin_issues: pd.DataFrame,
) -> pd.DataFrame:
    """
    Spaja osnovne logičke nekonzistentnosti u jednu tablicu.
    """
    frames = []

    if not bilirubin_issues.empty:
        bilirubin_copy = (
            bilirubin_issues.copy()
        )

        bilirubin_copy[
            "issue_type"
        ] = "Bilirubin consistency"

        frames.append(bilirubin_copy)

    if not albumin_issues.empty:
        albumin_copy = (
            albumin_issues.copy()
        )

        albumin_copy[
            "issue_type"
        ] = "Albumin consistency"

        frames.append(albumin_copy)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=False,
    )


# ---------------------------------------------------------
# VALIDACIJA KATEGORIJSKIH VRIJEDNOSTI
# ---------------------------------------------------------

def find_invalid_gender_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi neočekivane vrijednosti spola.
    """
    return df.loc[
        ~df["Gender"].isin(ALLOWED_GENDERS)
    ].copy()


def find_invalid_target_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi neočekivane vrijednosti ciljne klase.
    """
    return df.loc[
        ~df[TARGET_COLUMN].isin(
            ALLOWED_TARGET_VALUES
        )
    ].copy()


# ---------------------------------------------------------
# SPREMANJE IZVJEŠTAJA
# ---------------------------------------------------------

def save_excel_reports(
    duplicate_rows: pd.DataFrame,
    missing_rows: pd.DataFrame,
    threshold_violations: pd.DataFrame,
    iqr_outliers: pd.DataFrame,
    consistency_issues: pd.DataFrame,
    ag_ratio_issues: pd.DataFrame,
    invalid_gender_rows: pd.DataFrame,
    invalid_target_rows: pd.DataFrame,
) -> None:
    """
    Sprema detaljne rezultate u Excel datoteke.
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
            sheet_name="Logical consistency",
            index=True,
        )

        ag_ratio_issues.to_excel(
            writer,
            sheet_name="AG ratio consistency",
            index=True,
        )

        invalid_gender_rows.to_excel(
            writer,
            sheet_name="Invalid gender",
            index=True,
        )

        invalid_target_rows.to_excel(
            writer,
            sheet_name="Invalid target",
            index=True,
        )


def save_text_report(
    df: pd.DataFrame,
    missing_summary: pd.DataFrame,
    duplicate_rows: pd.DataFrame,
    gender_distribution: pd.Series,
    target_distribution: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    non_positive_summary: pd.DataFrame,
    threshold_violations: pd.DataFrame,
    iqr_summary: pd.DataFrame,
    bilirubin_issues: pd.DataFrame,
    albumin_issues: pd.DataFrame,
    ag_ratio_issues: pd.DataFrame,
    invalid_gender_rows: pd.DataFrame,
    invalid_target_rows: pd.DataFrame,
) -> None:
    """
    Sprema detaljan tekstualni izvještaj.
    """
    with TEXT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "DATA QUALITY REPORT – "
            "INDIAN LIVER PATIENT DATASET\n"
        )
        report.write("=" * 80 + "\n\n")

        report.write("1. OSNOVNE INFORMACIJE\n")
        report.write("-" * 80 + "\n")
        report.write(
            f"Broj zapisa: {len(df)}\n"
        )
        report.write(
            f"Broj stupaca: {df.shape[1]}\n"
        )
        report.write(
            f"Broj ponovljenih zapisa prema duplicated(): "
            f"{int(df.duplicated().sum())}\n"
        )
        report.write(
            f"Broj svih redaka uključenih u skupine duplikata: "
            f"{len(duplicate_rows)}\n"
        )
        report.write(
            f"Ukupan broj nedostajućih vrijednosti: "
            f"{int(df.isna().sum().sum())}\n\n"
        )

        report.write(
            "2. NEDOSTAJUĆE VRIJEDNOSTI\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            missing_summary.to_string()
        )
        report.write("\n\n")

        report.write(
            "3. RASPODJELA SPOLA\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            gender_distribution.to_string()
        )
        report.write("\n\n")

        report.write(
            "4. RASPODJELA CILJNE VARIJABLE\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            target_distribution.to_string(
                index=False
            )
        )
        report.write("\n\n")

        report.write(
            "5. RASPONI NUMERIČKIH VARIJABLI\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            numeric_summary.to_string()
        )
        report.write("\n\n")

        report.write(
            "6. NEGATIVNE I NULTE VRIJEDNOSTI\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            non_positive_summary.to_string(
                index=False
            )
        )
        report.write("\n\n")

        report.write(
            "7. VRIJEDNOSTI IZVAN PRAGOVA ZA RUČNI PREGLED\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            f"Broj označenih vrijednosti: "
            f"{len(threshold_violations)}\n"
        )

        if not threshold_violations.empty:
            report.write(
                threshold_violations[
                    [
                        "original_index",
                        "flagged_parameter",
                        "flagged_value",
                        "flag_reason",
                    ]
                ].to_string(index=False)
            )
        else:
            report.write(
                "Nisu pronađene vrijednosti "
                "izvan tehničkih pragova."
            )

        report.write("\n\n")

        report.write(
            "8. IQR OUTLIERI\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            iqr_summary.to_string(index=False)
        )
        report.write("\n\n")

        report.write(
            "9. LOGIČKA KONZISTENTNOST BILIRUBINA\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            "Broj zapisa u kojima je direktni bilirubin "
            "veći od ukupnog bilirubina: "
            f"{len(bilirubin_issues)}\n\n"
        )

        report.write(
            "10. LOGIČKA KONZISTENTNOST ALBUMINA\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            "Broj zapisa u kojima je albumin "
            "veći od ukupnih proteina: "
            f"{len(albumin_issues)}\n\n"
        )

        report.write(
            "11. KONZISTENTNOST A/G OMJERA\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            "Broj zapisa s odstupanjem većim od 0.25: "
            f"{len(ag_ratio_issues)}\n\n"
        )

        report.write(
            "12. NEDOPUŠTENE KATEGORIJSKE VRIJEDNOSTI\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            f"Nedopuštene vrijednosti spola: "
            f"{len(invalid_gender_rows)}\n"
        )
        report.write(
            f"Nedopuštene ciljane klase: "
            f"{len(invalid_target_rows)}\n\n"
        )

        report.write(
            "13. METODOLOŠKA NAPOMENA\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            "Statistički outlieri i visoke patološke "
            "vrijednosti nisu automatski označeni kao "
            "pogreške. Ova skripta ne mijenja dataset, "
            "nego služi za dokumentiranje kvalitete i "
            "donošenje odluka prije čišćenja.\n"
        )


# ---------------------------------------------------------
# ISPIS U TERMINAL
# ---------------------------------------------------------

def print_summary(
    df: pd.DataFrame,
    threshold_violations: pd.DataFrame,
    iqr_summary: pd.DataFrame,
    bilirubin_issues: pd.DataFrame,
    albumin_issues: pd.DataFrame,
    ag_ratio_issues: pd.DataFrame,
) -> None:
    """
    Ispisuje glavne rezultate u terminal.
    """
    print("=" * 80)
    print(
        "DATA QUALITY CHECK – "
        "INDIAN LIVER PATIENT DATASET"
    )
    print("=" * 80)

    print(f"Broj zapisa: {len(df)}")
    print(
        f"Broj duplikata: "
        f"{df.duplicated().sum()}"
    )
    print(
        f"Ukupan broj missing vrijednosti: "
        f"{df.isna().sum().sum()}"
    )

    print("\nMinimalne vrijednosti:")
    print(
        df[NUMERIC_COLUMNS].min()
    )

    print("\nMaksimalne vrijednosti:")
    print(
        df[NUMERIC_COLUMNS].max()
    )

    print(
        "\nVrijednosti izvan pragova "
        "za ručni pregled:"
    )

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
        "\nDirektni bilirubin veći od ukupnog:"
    )
    print(len(bilirubin_issues))

    print(
        "\nAlbumin veći od ukupnih proteina:"
    )
    print(len(albumin_issues))

    print(
        "\nNekonzistentni A/G omjeri:"
    )
    print(len(ag_ratio_issues))

    print("\nIzvještaji su spremljeni u:")
    print(REPORT_DIR)


# ---------------------------------------------------------
# GLAVNI PROGRAM
# ---------------------------------------------------------

def main() -> None:
    create_output_directory()

    print("Učitavanje ILPD dataseta...")
    df = load_dataset()

    print("Provjera nedostajućih vrijednosti...")
    missing_summary = get_missing_summary(df)
    missing_rows = (
        get_rows_with_missing_values(df)
    )

    print("Provjera duplikata...")
    duplicate_rows = get_duplicate_rows(df)

    print("Provjera kategorijskih vrijednosti...")
    gender_distribution = (
        get_gender_distribution(df)
    )
    target_distribution = (
        get_target_distribution(df)
    )
    invalid_gender_rows = (
        find_invalid_gender_rows(df)
    )
    invalid_target_rows = (
        find_invalid_target_rows(df)
    )

    print("Analiza numeričkih raspona...")
    numeric_summary = (
        get_numeric_range_summary(df)
    )
    non_positive_summary = (
        get_non_positive_summary(df)
    )

    print("Provjera tehničkih pragova...")
    threshold_violations = (
        detect_threshold_violations(df)
    )

    print("IQR analiza outliera...")
    (
        iqr_summary,
        iqr_outliers,
    ) = calculate_iqr_outliers(df)

    print("Provjera bilirubina...")
    bilirubin_issues = (
        check_bilirubin_consistency(df)
    )

    print("Provjera albumina...")
    albumin_issues = (
        check_albumin_consistency(df)
    )

    print("Provjera A/G omjera...")
    ag_ratio_issues = (
        calculate_ag_ratio_consistency(df)
    )

    consistency_issues = (
        combine_consistency_issues(
            bilirubin_issues,
            albumin_issues,
        )
    )

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
        threshold_violations=(
            threshold_violations
        ),
        iqr_outliers=iqr_outliers,
        consistency_issues=(
            consistency_issues
        ),
        ag_ratio_issues=ag_ratio_issues,
        invalid_gender_rows=(
            invalid_gender_rows
        ),
        invalid_target_rows=(
            invalid_target_rows
        ),
    )

    print("Spremanje tekstualnog izvještaja...")
    save_text_report(
        df=df,
        missing_summary=missing_summary,
        duplicate_rows=duplicate_rows,
        gender_distribution=(
            gender_distribution
        ),
        target_distribution=(
            target_distribution
        ),
        numeric_summary=numeric_summary,
        non_positive_summary=(
            non_positive_summary
        ),
        threshold_violations=(
            threshold_violations
        ),
        iqr_summary=iqr_summary,
        bilirubin_issues=(
            bilirubin_issues
        ),
        albumin_issues=(
            albumin_issues
        ),
        ag_ratio_issues=(
            ag_ratio_issues
        ),
        invalid_gender_rows=(
            invalid_gender_rows
        ),
        invalid_target_rows=(
            invalid_target_rows
        ),
    )

    print_summary(
        df=df,
        threshold_violations=(
            threshold_violations
        ),
        iqr_summary=iqr_summary,
        bilirubin_issues=(
            bilirubin_issues
        ),
        albumin_issues=(
            albumin_issues
        ),
        ag_ratio_issues=(
            ag_ratio_issues
        ),
    )

    print(
        "\nAnaliza kvalitete jetrenog "
        "dataseta uspješno je završena."
    )


if __name__ == "__main__":
    main()