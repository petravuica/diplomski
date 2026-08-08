from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PUTANJE
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parents[1]

RAW_DATASET_PATH = ML_DIR / "data" / "raw" / "anemia_raw.xlsx"
PROCESSED_DATA_DIR = ML_DIR / "data" / "processed"
REPORT_DIR = ML_DIR / "reports" / "data_quality"

CLEAN_DATASET_PATH = (
    PROCESSED_DATA_DIR / "anemia_dataset_clean.csv"
)

REMOVED_ROWS_PATH = (
    REPORT_DIR / "removed_rows_during_cleaning.xlsx"
)

CLEANING_REPORT_PATH = (
    REPORT_DIR / "cleaning_report.txt"
)


# ---------------------------------------------------------
# KONFIGURACIJA
# ---------------------------------------------------------

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

NUMERIC_COLUMNS = [
    "Age",
    "HGB",
    "RBC",
    "HCT",
    "MCV",
    "MCH",
    "MCHC",
]

ALLOWED_GENDERS = {"F", "M"}
ALLOWED_TARGET_VALUES = {0, 1}


# Pragovi za provjeru unutarnje konzistentnosti.
# Ne predstavljaju laboratorijske referentne intervale.
CONSISTENCY_LIMITS = {
    "MCV_absolute_difference": 10,
    "MCH_absolute_difference": 5,
    "MCHC_absolute_difference": 5,
}


# ---------------------------------------------------------
# PRIPREMA MAPA
# ---------------------------------------------------------

def create_output_directories() -> None:
    """
    Stvara potrebne izlazne mape ako ne postoje.
    """
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------------------------------------------------------
# UČITAVANJE I STANDARDIZACIJA
# ---------------------------------------------------------

def standardize_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardizira nazive stupaca.
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
    Učitava originalni dataset i provjerava strukturu.
    """
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Originalni dataset nije pronađen na putanji:\n"
            f"{RAW_DATASET_PATH}"
        )

    df = pd.read_excel(RAW_DATASET_PATH)
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
# STANDARDIZACIJA VRIJEDNOSTI
# ---------------------------------------------------------

def standardize_gender(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Uklanja suvišne razmake i pretvara oznake spola
    u velika slova.
    """
    df = df.copy()

    df["Gender"] = (
        df["Gender"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return df


def convert_numeric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pretvara numeričke stupce i ciljnu varijablu
    u numeričke tipove.

    Neispravne vrijednosti postaju NaN i kasnije se uklanjaju.
    """
    df = df.copy()

    columns_to_convert = [
        *NUMERIC_COLUMNS,
        "Decision_Class",
    ]

    for column in columns_to_convert:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# ---------------------------------------------------------
# PROVJERE VRIJEDNOSTI
# ---------------------------------------------------------

def find_invalid_gender_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi retke s nedopuštenim vrijednostima spola.
    """
    mask = ~df["Gender"].isin(ALLOWED_GENDERS)

    return df.loc[mask].copy()


def find_invalid_target_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi retke s nedopuštenim vrijednostima ciljne klase.
    """
    mask = ~df["Decision_Class"].isin(
        ALLOWED_TARGET_VALUES
    )

    return df.loc[mask].copy()


def find_non_positive_numeric_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi retke u kojima je barem jedan numerički
    medicinski parametar manji ili jednak nuli.
    """
    mask = (df[NUMERIC_COLUMNS] <= 0).any(axis=1)

    return df.loc[mask].copy()


# ---------------------------------------------------------
# MATEMATIČKA KONZISTENTNOST
# ---------------------------------------------------------

def add_consistency_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Dodaje izračunate hematološke pokazatelje i njihove
    apsolutne razlike u odnosu na izvorne vrijednosti.
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
        result["MCV"]
        - result["Calculated_MCV"]
    ).abs()

    result["MCH_absolute_difference"] = (
        result["MCH"]
        - result["Calculated_MCH"]
    ).abs()

    result["MCHC_absolute_difference"] = (
        result["MCHC"]
        - result["Calculated_MCHC"]
    ).abs()

    return result


def find_inconsistent_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi retke s većim odstupanjem između izvornih
    i matematički izvedenih hematoloških pokazatelja.
    """
    checked_df = add_consistency_columns(df)

    mask = (
        (
            checked_df[
                "MCV_absolute_difference"
            ]
            > CONSISTENCY_LIMITS[
                "MCV_absolute_difference"
            ]
        )
        | (
            checked_df[
                "MCH_absolute_difference"
            ]
            > CONSISTENCY_LIMITS[
                "MCH_absolute_difference"
            ]
        )
        | (
            checked_df[
                "MCHC_absolute_difference"
            ]
            > CONSISTENCY_LIMITS[
                "MCHC_absolute_difference"
            ]
        )
    )

    return checked_df.loc[mask].copy()


# ---------------------------------------------------------
# POMOĆNE FUNKCIJE ZA UKLANJANJE
# ---------------------------------------------------------

def add_removal_reason(
    rows: pd.DataFrame,
    reason: str,
) -> pd.DataFrame:
    """
    Dodaje razlog uklanjanja radi sljedivosti.
    """
    result = rows.copy()
    result["removal_reason"] = reason

    return result


def remove_rows_by_index(
    df: pd.DataFrame,
    rows_to_remove: pd.DataFrame,
) -> pd.DataFrame:
    """
    Uklanja retke prema indeksu.
    """
    return df.drop(
        index=rows_to_remove.index,
        errors="ignore",
    )


# ---------------------------------------------------------
# ČIŠĆENJE PODATAKA
# ---------------------------------------------------------

def clean_dataset(
    original_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    dict[str, pd.DataFrame],
]:
    """
    Provodi čišćenje dataseta korak po korak.

    Returns:
        cleaned_df:
            Konačni očišćeni dataset.

        cleaning_counts:
            Broj uklonjenih zapisa po koraku.

        removed_groups:
            Uklonjeni retci grupirani prema razlogu.
    """
    df = original_df.copy()

    cleaning_counts: dict[str, int] = {}
    removed_groups: dict[str, pd.DataFrame] = {}

    # 1. Standardizacija vrijednosti
    df = standardize_gender(df)
    df = convert_numeric_columns(df)

    # 2. Uklanjanje duplikata
    duplicate_mask = df.duplicated(
        keep="first"
    )

    duplicate_rows = df.loc[
        duplicate_mask
    ].copy()

    removed_groups["Duplicates"] = (
        add_removal_reason(
            duplicate_rows,
            "Potpuno duplicirani zapis",
        )
    )

    cleaning_counts["duplicates_removed"] = (
        len(duplicate_rows)
    )

    df = df.loc[~duplicate_mask].copy()

    # 3. Uklanjanje nedostajućih vrijednosti
    missing_mask = df[
        EXPECTED_COLUMNS
    ].isna().any(axis=1)

    missing_rows = df.loc[
        missing_mask
    ].copy()

    removed_groups["Missing values"] = (
        add_removal_reason(
            missing_rows,
            "Nedostajuća ili nenumerička vrijednost",
        )
    )

    cleaning_counts[
        "missing_rows_removed"
    ] = len(missing_rows)

    df = df.loc[~missing_mask].copy()

    # 4. Nedopuštene oznake spola
    invalid_gender_rows = (
        find_invalid_gender_rows(df)
    )

    removed_groups["Invalid gender"] = (
        add_removal_reason(
            invalid_gender_rows,
            "Nedopuštena vrijednost spola",
        )
    )

    cleaning_counts[
        "invalid_gender_rows_removed"
    ] = len(invalid_gender_rows)

    df = remove_rows_by_index(
        df,
        invalid_gender_rows,
    )

    # 5. Nedopuštene ciljne klase
    invalid_target_rows = (
        find_invalid_target_rows(df)
    )

    removed_groups["Invalid target"] = (
        add_removal_reason(
            invalid_target_rows,
            "Nedopuštena vrijednost ciljne klase",
        )
    )

    cleaning_counts[
        "invalid_target_rows_removed"
    ] = len(invalid_target_rows)

    df = remove_rows_by_index(
        df,
        invalid_target_rows,
    )

    # 6. Nulte ili negativne vrijednosti
    non_positive_rows = (
        find_non_positive_numeric_rows(df)
    )

    removed_groups["Non-positive values"] = (
        add_removal_reason(
            non_positive_rows,
            "Nulta ili negativna numerička vrijednost",
        )
    )

    cleaning_counts[
        "non_positive_rows_removed"
    ] = len(non_positive_rows)

    df = remove_rows_by_index(
        df,
        non_positive_rows,
    )

    # 7. Matematički nekonzistentni zapisi
    inconsistent_rows = (
        find_inconsistent_rows(df)
    )

    removed_groups["Consistency issues"] = (
        add_removal_reason(
            inconsistent_rows,
            (
                "Veće odstupanje između izvornih i "
                "izračunatih hematoloških pokazatelja"
            ),
        )
    )

    cleaning_counts[
        "inconsistent_rows_removed"
    ] = len(inconsistent_rows)

    df = remove_rows_by_index(
        df,
        inconsistent_rows,
    )

    # Konačna priprema
    df = df[EXPECTED_COLUMNS].copy()

    df["Decision_Class"] = (
        df["Decision_Class"].astype(int)
    )

    df = df.reset_index(drop=True)

    cleaning_counts[
        "final_row_count"
    ] = len(df)

    return (
        df,
        cleaning_counts,
        removed_groups,
    )


# ---------------------------------------------------------
# ZAVRŠNA VALIDACIJA
# ---------------------------------------------------------

def validate_clean_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Provjerava zadovoljava li očišćeni dataset
    osnovne uvjete kvalitete.
    """
    if df.empty:
        raise ValueError(
            "Očišćeni dataset je prazan."
        )

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Konačni stupci nisu u očekivanom redoslijedu."
        )

    if df.isna().any().any():
        raise ValueError(
            "Očišćeni dataset još sadrži "
            "nedostajuće vrijednosti."
        )

    if df.duplicated().any():
        raise ValueError(
            "Očišćeni dataset još sadrži duplikate."
        )

    if not set(df["Gender"].unique()).issubset(
        ALLOWED_GENDERS
    ):
        raise ValueError(
            "Očišćeni dataset sadrži "
            "nedopuštenu vrijednost spola."
        )

    if not set(
        df["Decision_Class"].unique()
    ).issubset(ALLOWED_TARGET_VALUES):
        raise ValueError(
            "Očišćeni dataset sadrži "
            "nedopuštenu ciljnu klasu."
        )

    if (df[NUMERIC_COLUMNS] <= 0).any().any():
        raise ValueError(
            "Očišćeni dataset sadrži nulte "
            "ili negativne vrijednosti."
        )


# ---------------------------------------------------------
# SPREMANJE REZULTATA
# ---------------------------------------------------------

def save_clean_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Sprema očišćeni dataset u CSV formatu.
    """
    df.to_csv(
        CLEAN_DATASET_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def save_removed_rows(
    removed_groups: dict[str, pd.DataFrame],
) -> None:
    """
    Sprema uklonjene retke u Excel datoteku,
    svaki razlog na zaseban list.
    """
    with pd.ExcelWriter(
        REMOVED_ROWS_PATH,
        engine="openpyxl",
    ) as writer:
        for sheet_name, rows in (
            removed_groups.items()
        ):
            safe_sheet_name = sheet_name[:31]

            rows.to_excel(
                writer,
                sheet_name=safe_sheet_name,
                index=True,
            )


def save_cleaning_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    cleaning_counts: dict[str, int],
) -> None:
    """
    Sprema tekstualni izvještaj o čišćenju.
    """
    original_target_distribution = (
        original_df["Decision_Class"]
        .value_counts()
        .sort_index()
    )

    cleaned_target_distribution = (
        cleaned_df["Decision_Class"]
        .value_counts()
        .sort_index()
    )

    original_gender_distribution = (
        original_df["Gender"]
        .astype("string")
        .str.strip()
        .str.upper()
        .value_counts()
    )

    cleaned_gender_distribution = (
        cleaned_df["Gender"]
        .value_counts()
    )

    total_removed = (
        len(original_df) - len(cleaned_df)
    )

    removal_percentage = (
        total_removed / len(original_df) * 100
    )

    with CLEANING_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "IZVJEŠTAJ O ČIŠĆENJU – ANEMIA DATASET\n"
        )
        report.write("=" * 70 + "\n\n")

        report.write(
            "1. OSNOVNE INFORMACIJE\n"
        )
        report.write("-" * 70 + "\n")
        report.write(
            f"Početni broj zapisa: "
            f"{len(original_df)}\n"
        )
        report.write(
            f"Konačni broj zapisa: "
            f"{len(cleaned_df)}\n"
        )
        report.write(
            f"Ukupno uklonjeno zapisa: "
            f"{total_removed}\n"
        )
        report.write(
            f"Postotak uklonjenih zapisa: "
            f"{removal_percentage:.2f}%\n\n"
        )

        report.write(
            "2. UKLANJANJE PO KORACIMA\n"
        )
        report.write("-" * 70 + "\n")

        for key, value in cleaning_counts.items():
            report.write(
                f"{key}: {value}\n"
            )

        report.write("\n")

        report.write(
            "3. RASPODJELA CILJNE VARIJABLE PRIJE ČIŠĆENJA\n"
        )
        report.write("-" * 70 + "\n")
        report.write(
            original_target_distribution.to_string()
        )
        report.write("\n\n")

        report.write(
            "4. RASPODJELA CILJNE VARIJABLE NAKON ČIŠĆENJA\n"
        )
        report.write("-" * 70 + "\n")
        report.write(
            cleaned_target_distribution.to_string()
        )
        report.write("\n\n")

        report.write(
            "5. RASPODJELA SPOLA PRIJE ČIŠĆENJA\n"
        )
        report.write("-" * 70 + "\n")
        report.write(
            original_gender_distribution.to_string()
        )
        report.write("\n\n")

        report.write(
            "6. RASPODJELA SPOLA NAKON ČIŠĆENJA\n"
        )
        report.write("-" * 70 + "\n")
        report.write(
            cleaned_gender_distribution.to_string()
        )
        report.write("\n\n")

        report.write(
            "7. ZAVRŠNA PROVJERA\n"
        )
        report.write("-" * 70 + "\n")
        report.write(
            f"Nedostajuće vrijednosti: "
            f"{int(cleaned_df.isna().sum().sum())}\n"
        )
        report.write(
            f"Duplikati: "
            f"{int(cleaned_df.duplicated().sum())}\n"
        )
        report.write(
            "Vrijednosti spola: "
            f"{sorted(cleaned_df['Gender'].unique())}\n"
        )
        report.write(
            "Ciljne klase: "
            f"{sorted(int(value)
                for value in cleaned_df["Decision_Class"].unique())}\n"
        )

        report.write("\n")

        report.write(
            "8. METODOLOŠKA NAPOMENA\n"
        )
        report.write("-" * 70 + "\n")
        report.write(
            "Statistički outlieri nisu automatski uklanjani. "
            "Uklonjeni su samo potpuni duplikati, zapisi s "
            "nedostajućim ili nedopuštenim vrijednostima te "
            "zapisi s većim odstupanjem između izvornih i "
            "matematički izvedenih hematoloških pokazatelja. "
            "Vrijednosti nisu ručno korigirane jer bez pristupa "
            "izvornim laboratorijskim nalazima nije moguće "
            "pouzdano utvrditi koji je parametar pogrešno unesen.\n"
        )


# ---------------------------------------------------------
# ISPIS U TERMINAL
# ---------------------------------------------------------

def print_summary(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    cleaning_counts: dict[str, int],
) -> None:
    """
    Ispisuje sažetak čišćenja u terminal.
    """
    print("=" * 70)
    print("ČIŠĆENJE ANEMIA DATASETA")
    print("=" * 70)

    print(
        f"Početni broj zapisa: "
        f"{len(original_df)}"
    )

    print(
        f"Uklonjeni duplikati: "
        f"{cleaning_counts['duplicates_removed']}"
    )

    print(
        "Uklonjeni zapisi s nedostajućim vrijednostima: "
        f"{cleaning_counts['missing_rows_removed']}"
    )

    print(
        "Uklonjeni zapisi s nedopuštenim spolom: "
        f"{cleaning_counts['invalid_gender_rows_removed']}"
    )

    print(
        "Uklonjeni zapisi s nedopuštenom ciljnom klasom: "
        f"{cleaning_counts['invalid_target_rows_removed']}"
    )

    print(
        "Uklonjeni zapisi s nultim ili negativnim "
        "vrijednostima: "
        f"{cleaning_counts['non_positive_rows_removed']}"
    )

    print(
        "Uklonjeni matematički nekonzistentni zapisi: "
        f"{cleaning_counts['inconsistent_rows_removed']}"
    )

    print(
        f"Konačni broj zapisa: "
        f"{len(cleaned_df)}"
    )

    print("\nRaspodjela ciljne varijable:")
    print(
        cleaned_df[
            "Decision_Class"
        ].value_counts().sort_index()
    )

    print("\nRaspodjela spola:")
    print(
        cleaned_df[
            "Gender"
        ].value_counts().sort_index()
    )

    print("\nOčišćeni dataset spremljen je u:")
    print(CLEAN_DATASET_PATH)

    print("\nIzvještaj o čišćenju spremljen je u:")
    print(CLEANING_REPORT_PATH)

    print("\nUklonjeni zapisi spremljeni su u:")
    print(REMOVED_ROWS_PATH)


# ---------------------------------------------------------
# GLAVNI PROGRAM
# ---------------------------------------------------------

def main() -> None:
    """
    Glavna funkcija skripte.
    """
    create_output_directories()

    print("Učitavanje originalnog dataseta...")
    original_df = load_dataset()

    print("Čišćenje dataseta...")
    (
        cleaned_df,
        cleaning_counts,
        removed_groups,
    ) = clean_dataset(original_df)

    print("Završna validacija...")
    validate_clean_dataset(cleaned_df)

    print("Spremanje očišćenog dataseta...")
    save_clean_dataset(cleaned_df)

    print("Spremanje uklonjenih redaka...")
    save_removed_rows(removed_groups)

    print("Spremanje izvještaja o čišćenju...")
    save_cleaning_report(
        original_df=original_df,
        cleaned_df=cleaned_df,
        cleaning_counts=cleaning_counts,
    )

    print_summary(
        original_df=original_df,
        cleaned_df=cleaned_df,
        cleaning_counts=cleaning_counts,
    )

    print("\nČišćenje je uspješno završeno.")


if __name__ == "__main__":
    main()