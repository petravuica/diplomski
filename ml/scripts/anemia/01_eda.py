from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# PUTANJE
# ---------------------------------------------------------

# Lokacija ove skripte:
# blood_analysis/ml/scripts/01_eda.py
SCRIPT_DIR = Path(__file__).resolve().parent

# Glavna ML mapa:
# blood_analysis/ml/
ML_DIR = SCRIPT_DIR.parents[1]

# Ulazni dataset
DATASET_PATH = ML_DIR / "data" / "raw" / "anemia_raw.xlsx"

# Mapa u koju spremamo rezultate EDA analize
REPORT_DIR = ML_DIR / "reports" / "eda"

# Tekstualni izvještaj
REPORT_PATH = REPORT_DIR / "eda_report.txt"

# CSV tablice
DESCRIPTIVE_STATS_PATH = REPORT_DIR / "descriptive_statistics.csv"
MISSING_VALUES_PATH = REPORT_DIR / "missing_values.csv"
CLASS_DISTRIBUTION_PATH = REPORT_DIR / "class_distribution.csv"
GENDER_DISTRIBUTION_PATH = REPORT_DIR / "gender_distribution.csv"
CORRELATION_MATRIX_PATH = REPORT_DIR / "correlation_matrix.csv"


# ---------------------------------------------------------
# POMOĆNE FUNKCIJE
# ---------------------------------------------------------

def create_output_directory() -> None:
    """
    Stvara mapu za EDA rezultate ako ona još ne postoji.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """
    Učitava originalni Excel dataset.

    Raises:
        FileNotFoundError: Ako dataset ne postoji na očekivanoj putanji.

    Returns:
        pd.DataFrame: Učitani dataset.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset nije pronađen na putanji:\n{DATASET_PATH}"
        )

    df = pd.read_excel(DATASET_PATH)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace("HGB(Hemoglobin)", "HGB", regex=False)
        .str.replace("PCV/HCT", "HCT", regex=False)
    )

    return df


def save_text_report(df: pd.DataFrame) -> None:
    """
    Sprema osnovne informacije o datasetu u tekstualnu datoteku.
    """
    duplicate_count = int(df.duplicated().sum())
    missing_values = df.isna().sum()
    total_missing = int(missing_values.sum())

    class_distribution = (
        df["Decision_Class"].value_counts(dropna=False)
        if "Decision_Class" in df.columns
        else pd.Series(dtype="int64")
    )

    gender_distribution = (
        df["Gender"].value_counts(dropna=False)
        if "Gender" in df.columns
        else pd.Series(dtype="int64")
    )

    with REPORT_PATH.open("w", encoding="utf-8") as report:
        report.write("EDA IZVJEŠTAJ – ANEMIA DATASET\n")
        report.write("=" * 50 + "\n\n")

        report.write("1. OSNOVNE INFORMACIJE\n")
        report.write("-" * 50 + "\n")
        report.write(f"Broj redaka: {df.shape[0]}\n")
        report.write(f"Broj stupaca: {df.shape[1]}\n")
        report.write(f"Broj duplikata: {duplicate_count}\n")
        report.write(f"Ukupan broj nedostajućih vrijednosti: {total_missing}\n\n")

        report.write("Nazivi stupaca:\n")
        for column in df.columns:
            report.write(f"- {column}\n")

        report.write("\n2. TIPOVI PODATAKA\n")
        report.write("-" * 50 + "\n")
        report.write(df.dtypes.to_string())
        report.write("\n\n")

        report.write("3. NEDOSTAJUĆE VRIJEDNOSTI\n")
        report.write("-" * 50 + "\n")
        report.write(missing_values.to_string())
        report.write("\n\n")

        report.write("4. RASPODJELA CILJNE VARIJABLE\n")
        report.write("-" * 50 + "\n")
        report.write(class_distribution.to_string())
        report.write("\n\n")

        report.write("5. RASPODJELA SPOLA\n")
        report.write("-" * 50 + "\n")
        report.write(gender_distribution.to_string())
        report.write("\n\n")

        report.write("6. DESKRIPTIVNA STATISTIKA\n")
        report.write("-" * 50 + "\n")
        report.write(df.describe(include="all").to_string())
        report.write("\n")


def save_tabular_reports(df: pd.DataFrame) -> None:
    """
    Sprema tablične rezultate analize u CSV datoteke.
    """
    descriptive_statistics = df.describe(include="all").transpose()
    descriptive_statistics.to_csv(
        DESCRIPTIVE_STATS_PATH,
        encoding="utf-8-sig"
    )

    missing_values = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (df.isna().mean() * 100).round(2)
    })

    missing_values.to_csv(
        MISSING_VALUES_PATH,
        encoding="utf-8-sig"
    )

    if "Decision_Class" in df.columns:
        class_distribution = (
            df["Decision_Class"]
            .value_counts(dropna=False)
            .rename_axis("Decision_Class")
            .reset_index(name="count")
        )

        class_distribution["percentage"] = (
            class_distribution["count"] / len(df) * 100
        ).round(2)

        class_distribution.to_csv(
            CLASS_DISTRIBUTION_PATH,
            index=False,
            encoding="utf-8-sig"
        )

    if "Gender" in df.columns:
        gender_distribution = (
            df["Gender"]
            .value_counts(dropna=False)
            .rename_axis("Gender")
            .reset_index(name="count")
        )

        gender_distribution["percentage"] = (
            gender_distribution["count"] / len(df) * 100
        ).round(2)

        gender_distribution.to_csv(
            GENDER_DISTRIBUTION_PATH,
            index=False,
            encoding="utf-8-sig"
        )


def create_class_distribution_chart(df: pd.DataFrame) -> None:
    """
    Izrađuje stupčasti graf raspodjele ciljne varijable.
    """
    if "Decision_Class" not in df.columns:
        print("Stupac 'Decision_Class' ne postoji. Graf nije izrađen.")
        return

    counts = df["Decision_Class"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Raspodjela ciljne varijable")
    plt.xlabel("Decision_Class")
    plt.ylabel("Broj zapisa")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "class_distribution.png",
        dpi=300
    )

    plt.close()


def create_gender_distribution_chart(df: pd.DataFrame) -> None:
    """
    Izrađuje stupčasti graf raspodjele spola.
    """
    if "Gender" not in df.columns:
        print("Stupac 'Gender' ne postoji. Graf nije izrađen.")
        return

    counts = df["Gender"].astype(str).value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Raspodjela zapisa prema spolu")
    plt.xlabel("Spol")
    plt.ylabel("Broj zapisa")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "gender_distribution.png",
        dpi=300
    )

    plt.close()


def create_age_histogram(df: pd.DataFrame) -> None:
    """
    Izrađuje histogram dobi.
    """
    if "Age" not in df.columns:
        print("Stupac 'Age' ne postoji. Histogram nije izrađen.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(
        df["Age"].dropna(),
        bins=20,
        edgecolor="black"
    )

    plt.title("Raspodjela dobi")
    plt.xlabel("Dob")
    plt.ylabel("Broj zapisa")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "age_distribution.png",
        dpi=300
    )

    plt.close()


def create_numeric_histograms(df: pd.DataFrame) -> None:
    """
    Izrađuje zaseban histogram za svaki numerički stupac,
    osim ciljne varijable.
    """
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if "Decision_Class" in numeric_columns:
        numeric_columns.remove("Decision_Class")

    for column in numeric_columns:
        plt.figure(figsize=(8, 5))

        plt.hist(
            df[column].dropna(),
            bins=20,
            edgecolor="black"
        )

        plt.title(f"Raspodjela parametra: {column}")
        plt.xlabel(column)
        plt.ylabel("Broj zapisa")
        plt.tight_layout()

        safe_column_name = (
            column
            .lower()
            .replace("/", "_")
            .replace(" ", "_")
        )

        plt.savefig(
            REPORT_DIR / f"histogram_{safe_column_name}.png",
            dpi=300
        )

        plt.close()


def create_numeric_boxplots(df: pd.DataFrame) -> None:
    """
    Izrađuje zaseban boxplot za svaki numerički stupac,
    osim ciljne varijable.
    """
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if "Decision_Class" in numeric_columns:
        numeric_columns.remove("Decision_Class")

    for column in numeric_columns:
        plt.figure(figsize=(8, 5))

        plt.boxplot(
            df[column].dropna(),
            orientation="vertical"
        )

        plt.title(f"Boxplot parametra: {column}")
        plt.ylabel(column)
        plt.xticks([1], [column])
        plt.tight_layout()

        safe_column_name = (
            column
            .lower()
            .replace("/", "_")
            .replace(" ", "_")
        )

        plt.savefig(
            REPORT_DIR / f"boxplot_{safe_column_name}.png",
            dpi=300
        )

        plt.close()


def create_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Izračunava i vizualizira korelacijsku matricu numeričkih varijabli.
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print("Nema numeričkih stupaca za korelacijsku matricu.")
        return

    correlation_matrix = numeric_df.corr()

    correlation_matrix.to_csv(
        CORRELATION_MATRIX_PATH,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(10, 8))

    image = plt.imshow(
        correlation_matrix,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar(image)

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(correlation_matrix.index)),
        correlation_matrix.index
    )

    for row_index in range(len(correlation_matrix.index)):
        for column_index in range(len(correlation_matrix.columns)):
            value = correlation_matrix.iloc[row_index, column_index]

            plt.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.title("Korelacijska matrica numeričkih varijabli")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "correlation_matrix.png",
        dpi=300
    )

    plt.close()


def print_console_summary(df: pd.DataFrame) -> None:
    """
    Ispisuje sažetak analize u terminal.
    """
    print("=" * 60)
    print("EDA ANALIZA – ANEMIA DATASET")
    print("=" * 60)

    print(f"Dataset: {DATASET_PATH}")
    print(f"Broj redaka: {df.shape[0]}")
    print(f"Broj stupaca: {df.shape[1]}")
    print(f"Broj duplikata: {df.duplicated().sum()}")
    print(f"Broj nedostajućih vrijednosti: {df.isna().sum().sum()}")

    print("\nStupci:")
    print(df.columns.tolist())

    print("\nNedostajuće vrijednosti po stupcima:")
    print(df.isna().sum())

    if "Decision_Class" in df.columns:
        print("\nRaspodjela ciljne varijable:")
        print(df["Decision_Class"].value_counts(dropna=False))

    if "Gender" in df.columns:
        print("\nRaspodjela spola:")
        print(df["Gender"].value_counts(dropna=False))

    print("\nRezultati su spremljeni u:")
    print(REPORT_DIR)


# ---------------------------------------------------------
# GLAVNI PROGRAM
# ---------------------------------------------------------

def main() -> None:
    """
    Glavna funkcija EDA skripte.
    """
    create_output_directory()

    print("Učitavanje dataseta...")
    df = load_dataset()

    print_console_summary(df)

    print("\nSpremanje tekstualnog izvještaja...")
    save_text_report(df)

    print("Spremanje tabličnih izvještaja...")
    save_tabular_reports(df)

    print("Izrada grafa raspodjele ciljnih klasa...")
    create_class_distribution_chart(df)

    print("Izrada grafa raspodjele spola...")
    create_gender_distribution_chart(df)

    print("Izrada histograma dobi...")
    create_age_histogram(df)

    print("Izrada histograma numeričkih parametara...")
    create_numeric_histograms(df)

    print("Izrada boxplotova numeričkih parametara...")
    create_numeric_boxplots(df)

    print("Izrada korelacijske matrice...")
    create_correlation_matrix(df)

    print("\nEDA analiza je uspješno završena.")
    print(f"Rezultati se nalaze u mapi:\n{REPORT_DIR}")


if __name__ == "__main__":
    main()