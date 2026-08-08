from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PUTANJE
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parents[1]

RAW_DATASET_PATH = (
    ML_DIR
    / "data"
    / "raw"
    / "liver"
    / "liver_raw.csv"
)

PROCESSED_DATA_DIR = (
    ML_DIR
    / "data"
    / "processed"
    / "liver"
)

REPORT_DIR = (
    ML_DIR
    / "reports"
    / "liver"
    / "data_quality"
)

CLEAN_DATASET_PATH = (
    PROCESSED_DATA_DIR
    / "liver_dataset_clean.csv"
)

REMOVED_ROWS_PATH = (
    REPORT_DIR
    / "removed_rows_during_cleaning.xlsx"
)

CLEANING_REPORT_PATH = (
    REPORT_DIR
    / "liver_cleaning_report.txt"
)


# ---------------------------------------------------------
# KONFIGURACIJA
# ---------------------------------------------------------

RAW_TARGET_COLUMN = "Dataset"
TARGET_COLUMN = "Liver_Disease"

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
    RAW_TARGET_COLUMN,
]

FINAL_COLUMNS = [
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
    TARGET_COLUMN,
]

NUMERIC_FEATURE_COLUMNS = [
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
ALLOWED_RAW_TARGET_VALUES = {1, 2}
ALLOWED_FINAL_TARGET_VALUES = {0, 1}

TARGET_MAPPING = {
    1: 1,  # jetreni pacijent
    2: 0,  # osoba bez oznake jetrene bolesti
}


# ---------------------------------------------------------
# PRIPREMA MAPA
# ---------------------------------------------------------

def create_output_directories() -> None:
    """
    Stvara izlazne mape ako ne postoje.
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
    Uklanja suvišne razmake i ispravlja poznate
    tipfelere u nazivima stupaca.
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
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Originalni jetreni dataset nije pronađen:\n"
            f"{RAW_DATASET_PATH}"
        )

    df = pd.read_csv(
        RAW_DATASET_PATH
    )

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


def standardize_gender(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardizira oznake spola na Male i Female.
    """
    df = df.copy()

    normalized = (
        df["Gender"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    gender_mapping = {
        "male": "Male",
        "m": "Male",
        "female": "Female",
        "f": "Female",
    }

    df["Gender"] = normalized.map(
        gender_mapping
    )

    return df


def convert_numeric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pretvara numeričke stupce i ciljnu varijablu
    u numeričke tipove.

    Neispravne vrijednosti pretvaraju se u NaN.
    """
    df = df.copy()

    columns_to_convert = [
        *NUMERIC_FEATURE_COLUMNS,
        RAW_TARGET_COLUMN,
    ]

    for column in columns_to_convert:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# ---------------------------------------------------------
# PRONALAŽENJE PROBLEMATIČNIH REDAKA
# ---------------------------------------------------------

def find_invalid_gender_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi retke s nepoznatom ili nedopuštenom
    oznakom spola.
    """
    mask = ~df["Gender"].isin(
        ALLOWED_GENDERS
    )

    return df.loc[mask].copy()


def find_invalid_target_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi retke s neočekivanom ciljnom klasom.
    """
    mask = ~df[RAW_TARGET_COLUMN].isin(
        ALLOWED_RAW_TARGET_VALUES
    )

    return df.loc[mask].copy()


def find_missing_required_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi retke s nedostajućim obveznim vrijednostima.

    Albumin_and_Globulin_Ratio namjerno nije obvezan jer će
    se kasnije imputirati medijanom unutar ML Pipelinea.
    """
    required_columns = [
        "Age",
        "Gender",
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphatase",
        "Alanine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Proteins",
        "Albumin",
        RAW_TARGET_COLUMN,
    ]

    mask = df[required_columns].isna().any(
        axis=1
    )

    return df.loc[mask].copy()


def find_non_positive_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi retke s nultim ili negativnim vrijednostima.

    Nedostajući A/G omjer ne smatra se greškom.
    """
    columns_to_check = [
        "Age",
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphatase",
        "Alanine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Proteins",
        "Albumin",
    ]

    mask = (
        df[columns_to_check] <= 0
    ).any(axis=1)

    ag_ratio_invalid = (
        df["Albumin_and_Globulin_Ratio"]
        .notna()
        & (
            df["Albumin_and_Globulin_Ratio"]
            <= 0
        )
    )

    mask = mask | ag_ratio_invalid

    return df.loc[mask].copy()


def find_bilirubin_inconsistencies(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pronalazi retke u kojima je direktni bilirubin
    veći od ukupnog bilirubina.
    """
    mask = (
        df["Direct_Bilirubin"]
        > df["Total_Bilirubin"]
    )

    return df.loc[mask].copy()


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
    Uklanja retke prema njihovim izvornim indeksima.
    """
    return df.drop(
        index=rows_to_remove.index,
        errors="ignore",
    )


# ---------------------------------------------------------
# ČIŠĆENJE DATASETA
# ---------------------------------------------------------

def clean_dataset(
    original_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    dict[str, pd.DataFrame],
]:
    """
    Provodi čišćenje jetrenog dataseta korak po korak.
    """
    df = original_df.copy()

    cleaning_counts: dict[str, int] = {}
    removed_groups: dict[
        str,
        pd.DataFrame,
    ] = {}

    # 1. Standardizacija vrijednosti
    df = standardize_gender(df)
    df = convert_numeric_columns(df)

    # 2. Uklanjanje potpunih duplikata
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

    cleaning_counts[
        "duplicates_removed"
    ] = len(duplicate_rows)

    df = df.loc[
        ~duplicate_mask
    ].copy()

    # 3. Nedopuštene oznake spola
    invalid_gender_rows = (
        find_invalid_gender_rows(df)
    )

    removed_groups["Invalid gender"] = (
        add_removal_reason(
            invalid_gender_rows,
            "Nedopuštena ili nepoznata vrijednost spola",
        )
    )

    cleaning_counts[
        "invalid_gender_rows_removed"
    ] = len(invalid_gender_rows)

    df = remove_rows_by_index(
        df,
        invalid_gender_rows,
    )

    # 4. Nedopuštene ciljne klase
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

    # 5. Nedostajuće obvezne vrijednosti
    missing_required_rows = (
        find_missing_required_rows(df)
    )

    removed_groups[
        "Missing required values"
    ] = add_removal_reason(
        missing_required_rows,
        (
            "Nedostaje obvezna vrijednost; "
            "nedostajući A/G omjer nije razlog za uklanjanje"
        ),
    )

    cleaning_counts[
        "missing_required_rows_removed"
    ] = len(missing_required_rows)

    df = remove_rows_by_index(
        df,
        missing_required_rows,
    )

    # 6. Nulte ili negativne vrijednosti
    non_positive_rows = (
        find_non_positive_rows(df)
    )

    removed_groups[
        "Non-positive values"
    ] = add_removal_reason(
        non_positive_rows,
        "Nulta ili negativna numerička vrijednost",
    )

    cleaning_counts[
        "non_positive_rows_removed"
    ] = len(non_positive_rows)

    df = remove_rows_by_index(
        df,
        non_positive_rows,
    )

    # 7. Logička nekonzistentnost bilirubina
    bilirubin_inconsistencies = (
        find_bilirubin_inconsistencies(df)
    )

    removed_groups[
        "Bilirubin inconsistencies"
    ] = add_removal_reason(
        bilirubin_inconsistencies,
        (
            "Direktni bilirubin veći je "
            "od ukupnog bilirubina"
        ),
    )

    cleaning_counts[
        "bilirubin_inconsistent_rows_removed"
    ] = len(bilirubin_inconsistencies)

    df = remove_rows_by_index(
        df,
        bilirubin_inconsistencies,
    )

    # 8. Prekodiranje ciljne varijable
    df[TARGET_COLUMN] = (
        df[RAW_TARGET_COLUMN]
        .map(TARGET_MAPPING)
    )

    df = df.drop(
        columns=[RAW_TARGET_COLUMN]
    )

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN].astype(int)
    )

    # Statistički outlieri i missing A/G vrijednosti ostaju.
    df = df[FINAL_COLUMNS].copy()

    df = df.reset_index(
        drop=True
    )

    cleaning_counts[
        "remaining_missing_ag_ratio"
    ] = int(
        df[
            "Albumin_and_Globulin_Ratio"
        ]
        .isna()
        .sum()
    )

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
    Provjerava osnovne uvjete kvalitete očišćenog dataseta.
    """
    if df.empty:
        raise ValueError(
            "Očišćeni jetreni dataset je prazan."
        )

    if list(df.columns) != FINAL_COLUMNS:
        raise ValueError(
            "Konačni stupci nisu u očekivanom redoslijedu."
        )

    if df.duplicated().any():
        raise ValueError(
            "Očišćeni dataset još sadrži duplikate."
        )

    if not set(
        df["Gender"].dropna().unique()
    ).issubset(ALLOWED_GENDERS):
        raise ValueError(
            "Očišćeni dataset sadrži "
            "nedopuštenu vrijednost spola."
        )

    if not set(
        df[TARGET_COLUMN].unique()
    ).issubset(ALLOWED_FINAL_TARGET_VALUES):
        raise ValueError(
            "Ciljna varijabla nije pravilno prekodirana."
        )

    required_columns = [
        column
        for column in FINAL_COLUMNS
        if column
        != "Albumin_and_Globulin_Ratio"
    ]

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Očišćeni dataset sadrži nedostajuće "
            "vrijednosti u obveznim stupcima."
        )

    invalid_bilirubin = (
        df["Direct_Bilirubin"]
        > df["Total_Bilirubin"]
    ).any()

    if invalid_bilirubin:
        raise ValueError(
            "Očišćeni dataset još sadrži "
            "nelogičan odnos bilirubina."
        )


# ---------------------------------------------------------
# SPREMANJE REZULTATA
# ---------------------------------------------------------

def save_clean_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Sprema očišćeni dataset.
    """
    df.to_csv(
        CLEAN_DATASET_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def save_removed_rows(
    removed_groups: dict[
        str,
        pd.DataFrame,
    ],
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
            safe_sheet_name = (
                sheet_name[:31]
            )

            rows.to_excel(
                writer,
                sheet_name=safe_sheet_name,
                index=True,
            )


def get_distribution(
    series: pd.Series,
) -> pd.DataFrame:
    """
    Vraća broj i postotak zapisa po vrijednosti.
    """
    distribution = (
        series.value_counts(
            dropna=False
        )
        .sort_index()
        .rename_axis("value")
        .reset_index(name="count")
    )

    distribution["percentage"] = (
        distribution["count"]
        / len(series)
        * 100
    ).round(2)

    return distribution


def save_cleaning_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    cleaning_counts: dict[str, int],
) -> None:
    """
    Sprema tekstualni izvještaj o čišćenju.
    """
    original_target_distribution = (
        get_distribution(
            original_df[RAW_TARGET_COLUMN]
        )
    )

    cleaned_target_distribution = (
        get_distribution(
            cleaned_df[TARGET_COLUMN]
        )
    )

    cleaned_gender_distribution = (
        get_distribution(
            cleaned_df["Gender"]
        )
    )

    total_removed = (
        len(original_df)
        - len(cleaned_df)
    )

    removal_percentage = (
        total_removed
        / len(original_df)
        * 100
    )

    with CLEANING_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "IZVJEŠTAJ O ČIŠĆENJU – "
            "INDIAN LIVER PATIENT DATASET\n"
        )
        report.write("=" * 80 + "\n\n")

        report.write(
            "1. OSNOVNE INFORMACIJE\n"
        )
        report.write("-" * 80 + "\n")
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
        report.write("-" * 80 + "\n")

        for key, value in (
            cleaning_counts.items()
        ):
            report.write(
                f"{key}: {value}\n"
            )

        report.write("\n")

        report.write(
            "3. IZVORNA RASPODJELA CILJNE VARIJABLE\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            original_target_distribution.to_string(
                index=False
            )
        )
        report.write("\n\n")

        report.write(
            "4. PREKODIRANJE CILJNE VARIJABLE\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            "Izvorna klasa 1 -> nova klasa 1 "
            "(jetreni pacijent)\n"
        )
        report.write(
            "Izvorna klasa 2 -> nova klasa 0 "
            "(osoba bez oznake jetrene bolesti)\n\n"
        )

        report.write(
            "5. RASPODJELA CILJNE VARIJABLE "
            "NAKON ČIŠĆENJA\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            cleaned_target_distribution.to_string(
                index=False
            )
        )
        report.write("\n\n")

        report.write(
            "6. RASPODJELA SPOLA "
            "NAKON ČIŠĆENJA\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            cleaned_gender_distribution.to_string(
                index=False
            )
        )
        report.write("\n\n")

        report.write(
            "7. ZAVRŠNA PROVJERA\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            f"Duplikati: "
            f"{int(cleaned_df.duplicated().sum())}\n"
        )
        report.write(
            "Nedostajuće A/G vrijednosti: "
            f"{int(cleaned_df['Albumin_and_Globulin_Ratio'].isna().sum())}\n"
        )

        other_missing = (
            cleaned_df.drop(
                columns=[
                    "Albumin_and_Globulin_Ratio"
                ]
            )
            .isna()
            .sum()
            .sum()
        )

        report.write(
            "Nedostajuće vrijednosti u ostalim stupcima: "
            f"{int(other_missing)}\n"
        )
        report.write(
            "Vrijednosti spola: "
            f"{sorted(cleaned_df['Gender'].unique())}\n"
        )
        report.write(
            "Ciljne klase: "
            f"{sorted(int(value) for value in cleaned_df[TARGET_COLUMN].unique())}\n"
        )
        report.write(
            "Zapisi s direktnim bilirubinom većim "
            "od ukupnog: "
            f"{int((cleaned_df['Direct_Bilirubin'] > cleaned_df['Total_Bilirubin']).sum())}\n\n"
        )

        report.write(
            "8. METODOLOŠKA NAPOMENA\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            "Statistički outlieri nisu automatski uklanjani "
            "jer visoke vrijednosti bilirubina, jetrenih "
            "enzima i alkalne fosfataze mogu predstavljati "
            "stvarne patološke nalaze. Četiri zapisa s "
            "nedostajućim A/G omjerom zadržana su, a njihova "
            "će se vrijednost nadomjestiti medijanom isključivo "
            "unutar ML Pipelinea nakon podjele na trening i "
            "testni skup. Tako se izbjegava curenje podataka.\n"
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
    print("=" * 80)
    print(
        "ČIŠĆENJE – "
        "INDIAN LIVER PATIENT DATASET"
    )
    print("=" * 80)

    print(
        f"Početni broj zapisa: "
        f"{len(original_df)}"
    )

    print(
        f"Uklonjeni duplikati: "
        f"{cleaning_counts['duplicates_removed']}"
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
        "Uklonjeni zapisi s nedostajućim obveznim vrijednostima: "
        f"{cleaning_counts['missing_required_rows_removed']}"
    )

    print(
        "Uklonjeni zapisi s nultim ili negativnim vrijednostima: "
        f"{cleaning_counts['non_positive_rows_removed']}"
    )

    print(
        "Uklonjeni zapisi s nekonzistentnim bilirubinom: "
        f"{cleaning_counts['bilirubin_inconsistent_rows_removed']}"
    )

    print(
        f"Konačni broj zapisa: "
        f"{len(cleaned_df)}"
    )

    print(
        "Preostale missing A/G vrijednosti: "
        f"{cleaning_counts['remaining_missing_ag_ratio']}"
    )

    print("\nRaspodjela ciljne varijable:")
    print(
        cleaned_df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    print("\nRaspodjela spola:")
    print(
        cleaned_df[
            "Gender"
        ]
        .value_counts()
        .sort_index()
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

    print("Učitavanje originalnog jetrenog dataseta...")
    original_df = load_dataset()

    print("Čišćenje jetrenog dataseta...")

    (
        cleaned_df,
        cleaning_counts,
        removed_groups,
    ) = clean_dataset(
        original_df
    )

    print("Završna validacija...")
    validate_clean_dataset(
        cleaned_df
    )

    print("Spremanje očišćenog dataseta...")
    save_clean_dataset(
        cleaned_df
    )

    print("Spremanje uklonjenih zapisa...")
    save_removed_rows(
        removed_groups
    )

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

    print(
        "\nČišćenje jetrenog dataseta "
        "uspješno je završeno."
    )


if __name__ == "__main__":
    main()