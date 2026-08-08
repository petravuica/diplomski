from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    / "eda"
)

TEXT_REPORT_PATH = REPORT_DIR / "liver_eda_report.txt"
DESCRIPTIVE_STATS_PATH = REPORT_DIR / "descriptive_statistics.csv"
MISSING_VALUES_PATH = REPORT_DIR / "missing_values.csv"
CLASS_DISTRIBUTION_PATH = REPORT_DIR / "class_distribution.csv"
GENDER_DISTRIBUTION_PATH = REPORT_DIR / "gender_distribution.csv"
CORRELATION_MATRIX_PATH = REPORT_DIR / "correlation_matrix.csv"


# ---------------------------------------------------------
# KONFIGURACIJA
# ---------------------------------------------------------

TARGET_COLUMN = "Dataset"

EXPECTED_COLUMNS = [
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


# ---------------------------------------------------------
# UČITAVANJE
# ---------------------------------------------------------

def create_output_directory() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset nije pronađen na putanji:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    df.columns = (
        df.columns
        .str.strip()
    )

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Nedostaju očekivani stupci: "
            + ", ".join(missing_columns)
        )

    return df[EXPECTED_COLUMNS].copy()


# ---------------------------------------------------------
# IZVJEŠTAJI
# ---------------------------------------------------------

def save_text_report(df: pd.DataFrame) -> None:
    missing_values = df.isna().sum()
    class_distribution = df[TARGET_COLUMN].value_counts(
        dropna=False
    ).sort_index()
    gender_distribution = df["Gender"].value_counts(
        dropna=False
    )

    with TEXT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "EDA IZVJEŠTAJ – INDIAN LIVER PATIENT DATASET\n"
        )
        report.write("=" * 70 + "\n\n")

        report.write("1. OSNOVNE INFORMACIJE\n")
        report.write("-" * 70 + "\n")
        report.write(f"Broj redaka: {len(df)}\n")
        report.write(f"Broj stupaca: {df.shape[1]}\n")
        report.write(
            f"Broj potpunih duplikata: "
            f"{int(df.duplicated().sum())}\n"
        )
        report.write(
            f"Ukupan broj nedostajućih vrijednosti: "
            f"{int(df.isna().sum().sum())}\n\n"
        )

        report.write("Nazivi stupaca:\n")
        for column in df.columns:
            report.write(f"- {column}\n")

        report.write("\n2. TIPOVI PODATAKA\n")
        report.write("-" * 70 + "\n")
        report.write(df.dtypes.to_string())
        report.write("\n\n")

        report.write("3. NEDOSTAJUĆE VRIJEDNOSTI\n")
        report.write("-" * 70 + "\n")
        report.write(missing_values.to_string())
        report.write("\n\n")

        report.write("4. RASPODJELA CILJNE VARIJABLE\n")
        report.write("-" * 70 + "\n")
        report.write(class_distribution.to_string())
        report.write("\n\n")

        report.write("5. RASPODJELA SPOLA\n")
        report.write("-" * 70 + "\n")
        report.write(gender_distribution.to_string())
        report.write("\n\n")

        report.write("6. DESKRIPTIVNA STATISTIKA\n")
        report.write("-" * 70 + "\n")
        report.write(
            df.describe(include="all").to_string()
        )
        report.write("\n")


def save_tabular_reports(df: pd.DataFrame) -> None:
    df.describe(include="all").transpose().to_csv(
        DESCRIPTIVE_STATS_PATH,
        encoding="utf-8-sig",
    )

    missing_df = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percentage": (
                df.isna().mean() * 100
            ).round(2),
        }
    )

    missing_df.to_csv(
        MISSING_VALUES_PATH,
        encoding="utf-8-sig",
    )

    class_distribution = (
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
    )

    class_distribution["percentage"] = (
        class_distribution["count"]
        / len(df)
        * 100
    ).round(2)

    class_distribution.to_csv(
        CLASS_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    gender_distribution = (
        df["Gender"]
        .value_counts(dropna=False)
        .rename_axis("Gender")
        .reset_index(name="count")
    )

    gender_distribution["percentage"] = (
        gender_distribution["count"]
        / len(df)
        * 100
    ).round(2)

    gender_distribution.to_csv(
        GENDER_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )


# ---------------------------------------------------------
# GRAFOVI
# ---------------------------------------------------------

def safe_filename(column: str) -> str:
    return (
        column.lower()
        .replace("/", "_")
        .replace(" ", "_")
    )


def create_class_distribution_chart(
    df: pd.DataFrame,
) -> None:
    counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Raspodjela ciljne varijable")
    plt.xlabel("Klasa")
    plt.ylabel("Broj zapisa")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "class_distribution.png",
        dpi=300,
    )

    plt.close()


def create_gender_distribution_chart(
    df: pd.DataFrame,
) -> None:
    counts = df["Gender"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Raspodjela zapisa prema spolu")
    plt.xlabel("Spol")
    plt.ylabel("Broj zapisa")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "gender_distribution.png",
        dpi=300,
    )

    plt.close()


def create_histograms(
    df: pd.DataFrame,
) -> None:
    numeric_columns = (
        df.select_dtypes(include="number")
        .columns
        .tolist()
    )

    if TARGET_COLUMN in numeric_columns:
        numeric_columns.remove(TARGET_COLUMN)

    for column in numeric_columns:
        plt.figure(figsize=(8, 5))

        plt.hist(
            df[column].dropna(),
            bins=25,
            edgecolor="black",
        )

        plt.title(f"Raspodjela parametra: {column}")
        plt.xlabel(column)
        plt.ylabel("Broj zapisa")
        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / f"histogram_{safe_filename(column)}.png",
            dpi=300,
        )

        plt.close()


def create_boxplots(
    df: pd.DataFrame,
) -> None:
    numeric_columns = (
        df.select_dtypes(include="number")
        .columns
        .tolist()
    )

    if TARGET_COLUMN in numeric_columns:
        numeric_columns.remove(TARGET_COLUMN)

    for column in numeric_columns:
        plt.figure(figsize=(8, 5))

        plt.boxplot(
            df[column].dropna(),
            orientation="vertical",
        )

        plt.title(f"Boxplot parametra: {column}")
        plt.ylabel(column)
        plt.xticks([1], [column])
        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / f"boxplot_{safe_filename(column)}.png",
            dpi=300,
        )

        plt.close()


def create_correlation_matrix(
    df: pd.DataFrame,
) -> None:
    numeric_df = df.select_dtypes(
        include="number"
    )

    correlation_matrix = numeric_df.corr()

    correlation_matrix.to_csv(
        CORRELATION_MATRIX_PATH,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(12, 9))

    image = plt.imshow(
        correlation_matrix,
        interpolation="nearest",
        aspect="auto",
    )

    plt.colorbar(image)

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(correlation_matrix.index)),
        correlation_matrix.index,
    )

    for row_index in range(
        len(correlation_matrix.index)
    ):
        for column_index in range(
            len(correlation_matrix.columns)
        ):
            value = correlation_matrix.iloc[
                row_index,
                column_index,
            ]

            plt.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    plt.title(
        "Korelacijska matrica numeričkih varijabli"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "correlation_matrix.png",
        dpi=300,
    )

    plt.close()


# ---------------------------------------------------------
# TERMINAL
# ---------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("EDA – INDIAN LIVER PATIENT DATASET")
    print("=" * 70)

    print(f"Broj redaka: {len(df)}")
    print(f"Broj stupaca: {df.shape[1]}")
    print(
        f"Broj potpunih duplikata: "
        f"{df.duplicated().sum()}"
    )
    print(
        f"Ukupan broj nedostajućih vrijednosti: "
        f"{df.isna().sum().sum()}"
    )

    print("\nNedostajuće vrijednosti:")
    print(df.isna().sum())

    print("\nRaspodjela ciljne varijable:")
    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nRaspodjela spola:")
    print(
        df["Gender"]
        .value_counts(dropna=False)
    )

    print("\nRezultati se nalaze u:")
    print(REPORT_DIR)


# ---------------------------------------------------------
# GLAVNI PROGRAM
# ---------------------------------------------------------

def main() -> None:
    create_output_directory()

    print("Učitavanje jetrenog dataseta...")
    df = load_dataset()

    print_summary(df)

    print("\nSpremanje tekstualnog izvještaja...")
    save_text_report(df)

    print("Spremanje tabličnih izvještaja...")
    save_tabular_reports(df)

    print("Izrada raspodjele klasa...")
    create_class_distribution_chart(df)

    print("Izrada raspodjele spola...")
    create_gender_distribution_chart(df)

    print("Izrada histograma...")
    create_histograms(df)

    print("Izrada boxplotova...")
    create_boxplots(df)

    print("Izrada korelacijske matrice...")
    create_correlation_matrix(df)

    print("\nEDA analiza jetrenog dataseta uspješno je završena.")


if __name__ == "__main__":
    main()