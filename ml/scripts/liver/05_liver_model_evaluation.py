from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)


# ---------------------------------------------------------
# PUTANJE
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parents[1]

PROCESSED_DATA_DIR = (
    ML_DIR
    / "data"
    / "processed"
    / "liver"
)

MODELS_DIR = (
    ML_DIR
    / "models"
    / "liver"
)

TRAINING_REPORT_DIR = (
    ML_DIR
    / "reports"
    / "liver"
    / "training"
)

EVALUATION_REPORT_DIR = (
    ML_DIR
    / "reports"
    / "liver"
    / "evaluation"
)

TEST_SPLIT_PATH = (
    PROCESSED_DATA_DIR
    / "liver_test_split.csv"
)

TRAINING_COMPARISON_PATH = (
    TRAINING_REPORT_DIR
    / "model_comparison.csv"
)

DUMMY_MODEL_PATH = (
    MODELS_DIR
    / "dummy_classifier.joblib"
)

LOGISTIC_MODEL_PATH = (
    MODELS_DIR
    / "logistic_regression.joblib"
)

RANDOM_FOREST_MODEL_PATH = (
    MODELS_DIR
    / "random_forest.joblib"
)

BEST_MODEL_PATH = (
    MODELS_DIR
    / "best_liver_model.joblib"
)

TEST_METRICS_PATH = (
    EVALUATION_REPORT_DIR
    / "test_metrics.csv"
)

CV_TEST_COMPARISON_PATH = (
    EVALUATION_REPORT_DIR
    / "cv_test_comparison.csv"
)

PREDICTIONS_PATH = (
    EVALUATION_REPORT_DIR
    / "test_predictions.xlsx"
)

EVALUATION_REPORT_PATH = (
    EVALUATION_REPORT_DIR
    / "evaluation_report.txt"
)

EVALUATION_METADATA_PATH = (
    EVALUATION_REPORT_DIR
    / "evaluation_metadata.json"
)


# ---------------------------------------------------------
# KONFIGURACIJA
# ---------------------------------------------------------

TARGET_COLUMN = "Liver_Disease"

FEATURE_COLUMNS = [
    "Gender",
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

MODEL_PATHS = {
    "Dummy Classifier": DUMMY_MODEL_PATH,
    "Logistic Regression": LOGISTIC_MODEL_PATH,
    "Random Forest": RANDOM_FOREST_MODEL_PATH,
}


# ---------------------------------------------------------
# PRIPREMA MAPA I PODATAKA
# ---------------------------------------------------------

def create_output_directory() -> None:
    """
    Stvara mapu za rezultate završne evaluacije.
    """
    EVALUATION_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_test_dataset() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Učitava prethodno izdvojeni netaknuti testni skup.

    Missing vrijednosti u A/G omjeru dopuštene su jer ih
    spremljeni Pipeline obrađuje medijanom.
    """
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Testni skup nije pronađen na putanji:\n"
            f"{TEST_SPLIT_PATH}\n\n"
            "Prvo pokreni 04_train_liver_models.py."
        )

    df = pd.read_csv(
        TEST_SPLIT_PATH,
        encoding="utf-8-sig",
    )

    expected_columns = [
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "U testnom skupu nedostaju stupci: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError(
            "Testni skup je prazan."
        )

    required_columns = [
        column
        for column in expected_columns
        if column != "Albumin_and_Globulin_Ratio"
    ]

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Testni skup sadrži nedostajuće vrijednosti "
            "izvan dopuštenog A/G omjera."
        )

    if not set(
        df[TARGET_COLUMN].unique()
    ).issubset({0, 1}):
        raise ValueError(
            "Ciljna varijabla mora sadržavati "
            "isključivo klase 0 i 1."
        )

    X_test = df[FEATURE_COLUMNS].copy()
    y_test = df[TARGET_COLUMN].astype(int).copy()

    return X_test, y_test


def load_models() -> dict:
    """
    Učitava spremljene modele iz trening skripte.
    """
    models = {}

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model '{model_name}' nije pronađen:\n"
                f"{model_path}\n\n"
                "Prvo pokreni 04_train_liver_models.py."
            )

        models[model_name] = joblib.load(
            model_path
        )

    return models


# ---------------------------------------------------------
# METRIKE
# ---------------------------------------------------------

def calculate_specificity(
    y_true: pd.Series,
    y_pred,
) -> float:
    """
    Izračunava specifičnost, odnosno recall klase 0.

    specificity = TN / (TN + FP)
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    denominator = tn + fp

    if denominator == 0:
        return 0.0

    return tn / denominator


def calculate_negative_predictive_value(
    y_true: pd.Series,
    y_pred,
) -> float:
    """
    Izračunava negativnu prediktivnu vrijednost.

    NPV = TN / (TN + FN)
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    denominator = tn + fn

    if denominator == 0:
        return 0.0

    return tn / denominator


def get_probability_scores(
    model,
    X_test: pd.DataFrame,
):
    """
    Dohvaća vjerojatnost pozitivne klase 1.

    Ako model nema predict_proba, koristi decision_function.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            X_test
        )

        classes = list(model.classes_)
        positive_index = classes.index(1)

        return probabilities[:, positive_index]

    if hasattr(model, "decision_function"):
        return model.decision_function(
            X_test
        )

    return None


def evaluate_model(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluira jedan model na netaknutom testnom skupu.
    """
    y_pred = model.predict(X_test)

    y_score = get_probability_scores(
        model,
        X_test,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "model_name": model_name,
        "accuracy_test": accuracy_score(
            y_test,
            y_pred,
        ),
        "balanced_accuracy_test": (
            balanced_accuracy_score(
                y_test,
                y_pred,
            )
        ),
        "precision_class_1_test": precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "recall_class_1_test": recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "f1_class_1_test": f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "precision_class_0_test": precision_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
        "recall_class_0_test": recall_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
        "f1_class_0_test": f1_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
        "specificity_test": calculate_specificity(
            y_test,
            y_pred,
        ),
        "negative_predictive_value_test": (
            calculate_negative_predictive_value(
                y_test,
                y_pred,
            )
        ),
        "roc_auc_test": (
            roc_auc_score(
                y_test,
                y_score,
            )
            if y_score is not None
            else None
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    predictions_df = X_test.copy()

    predictions_df[
        "actual_class"
    ] = y_test.to_numpy()

    predictions_df[
        "predicted_class"
    ] = y_pred

    predictions_df[
        "correct_prediction"
    ] = (
        predictions_df["actual_class"]
        == predictions_df["predicted_class"]
    )

    if y_score is not None:
        predictions_df[
            "probability_liver_disease"
        ] = y_score

    predictions_df.insert(
        0,
        "model_name",
        model_name,
    )

    return metrics, predictions_df


# ---------------------------------------------------------
# GRAFIKONI
# ---------------------------------------------------------

def safe_model_name(
    model_name: str,
) -> str:
    """
    Pretvara naziv modela u siguran naziv datoteke.
    """
    return (
        model_name
        .lower()
        .replace(" ", "_")
    )


def save_confusion_matrix_plot(
    model_name: str,
    y_test: pd.Series,
    y_pred,
) -> None:
    """
    Sprema matricu zabune za jedan model.
    """
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=[
            "Bez oznake bolesti",
            "Jetreni pacijent",
        ],
        values_format="d",
    )

    display.ax_.set_title(
        f"Matrica zabune – {model_name}"
    )

    display.ax_.set_xlabel(
        "Predviđena klasa"
    )

    display.ax_.set_ylabel(
        "Stvarna klasa"
    )

    plt.tight_layout()

    output_path = (
        EVALUATION_REPORT_DIR
        / (
            "confusion_matrix_"
            f"{safe_model_name(model_name)}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_normalized_confusion_matrix_plot(
    model_name: str,
    y_test: pd.Series,
    y_pred,
) -> None:
    """
    Sprema matricu zabune normaliziranu po stvarnim klasama.
    """
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=[
            "Bez oznake bolesti",
            "Jetreni pacijent",
        ],
        normalize="true",
        values_format=".2f",
    )

    display.ax_.set_title(
        f"Normalizirana matrica zabune – {model_name}"
    )

    display.ax_.set_xlabel(
        "Predviđena klasa"
    )

    display.ax_.set_ylabel(
        "Stvarna klasa"
    )

    plt.tight_layout()

    output_path = (
        EVALUATION_REPORT_DIR
        / (
            "confusion_matrix_normalized_"
            f"{safe_model_name(model_name)}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_roc_curve_plot(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Sprema ROC krivulje svih modela na zajedničkom grafikonu.
    """
    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    plotted_models = 0

    for model_name, model in models.items():
        y_score = get_probability_scores(
            model,
            X_test,
        )

        if y_score is None:
            continue

        RocCurveDisplay.from_predictions(
            y_test,
            y_score,
            name=model_name,
            ax=axis,
        )

        plotted_models += 1

    if plotted_models == 0:
        plt.close(figure)
        return

    axis.set_title(
        "ROC krivulje jetrenih modela "
        "na testnom skupu"
    )

    axis.set_xlabel(
        "Stopa lažno pozitivnih rezultata"
    )

    axis.set_ylabel(
        "Stopa stvarno pozitivnih rezultata"
    )

    figure.tight_layout()

    figure.savefig(
        EVALUATION_REPORT_DIR
        / "roc_curves_test_set.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ---------------------------------------------------------
# USPOREDBA CV I TESTNIH REZULTATA
# ---------------------------------------------------------

def create_cv_test_comparison(
    test_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Uspoređuje rezultate unakrsne validacije i testnog skupa.
    """
    if not TRAINING_COMPARISON_PATH.exists():
        raise FileNotFoundError(
            "Rezultati treninga nisu pronađeni:\n"
            f"{TRAINING_COMPARISON_PATH}"
        )

    cv_df = pd.read_csv(
        TRAINING_COMPARISON_PATH,
        encoding="utf-8-sig",
    )

    comparison_df = cv_df.merge(
        test_metrics_df,
        on="model_name",
        how="inner",
    )

    comparison_df[
        "balanced_accuracy_difference"
    ] = (
        comparison_df[
            "balanced_accuracy_test"
        ]
        - comparison_df[
            "best_balanced_accuracy_cv"
        ]
    )

    comparison_df[
        "f1_class_1_difference"
    ] = (
        comparison_df[
            "f1_class_1_test"
        ]
        - comparison_df["f1_cv"]
    )

    comparison_df[
        "roc_auc_difference"
    ] = (
        comparison_df["roc_auc_test"]
        - comparison_df["roc_auc_cv"]
    )

    selected_columns = [
        "model_name",
        "best_balanced_accuracy_cv",
        "balanced_accuracy_test",
        "balanced_accuracy_difference",
        "f1_cv",
        "f1_class_1_test",
        "f1_class_1_difference",
        "roc_auc_cv",
        "roc_auc_test",
        "roc_auc_difference",
        "recall_class_0_test",
        "recall_class_1_test",
    ]

    return comparison_df[
        selected_columns
    ].sort_values(
        by="balanced_accuracy_test",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------
# ODABIR SLUŽBENOG MODELA
# ---------------------------------------------------------

def get_training_selected_model_name(
    cv_test_comparison_df: pd.DataFrame,
) -> str:
    """
    Vraća model odabran tijekom treninga prema CV balanced accuracy.

    Testni rezultat ne koristi se za retroaktivnu promjenu modela.
    """
    best_index = (
        cv_test_comparison_df[
            "best_balanced_accuracy_cv"
        ].idxmax()
    )

    return str(
        cv_test_comparison_df.loc[
            best_index,
            "model_name",
        ]
    )


# ---------------------------------------------------------
# SPREMANJE REZULTATA
# ---------------------------------------------------------

def save_predictions(
    predictions_by_model: dict[
        str,
        pd.DataFrame,
    ],
) -> None:
    """
    Sprema predikcije svakog modela u zaseban Excel list.
    """
    with pd.ExcelWriter(
        PREDICTIONS_PATH,
        engine="openpyxl",
    ) as writer:
        for model_name, predictions_df in (
            predictions_by_model.items()
        ):
            predictions_df.to_excel(
                writer,
                sheet_name=model_name[:31],
                index=False,
            )


def save_evaluation_report(
    test_metrics_df: pd.DataFrame,
    cv_test_comparison_df: pd.DataFrame,
    selected_model_name: str,
    selected_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Sprema detaljan tekstualni izvještaj završne evaluacije.
    """
    selected_predictions = (
        selected_model.predict(X_test)
    )

    classification_report_text = (
        classification_report(
            y_test,
            selected_predictions,
            labels=[0, 1],
            target_names=[
                "Klasa 0 – bez oznake bolesti",
                "Klasa 1 – jetreni pacijent",
            ],
            zero_division=0,
            digits=4,
        )
    )

    selected_metrics = test_metrics_df.loc[
        test_metrics_df["model_name"]
        == selected_model_name
    ].iloc[0]

    with EVALUATION_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "ZAVRŠNA EVALUACIJA MODELA – "
            "INDIAN LIVER PATIENT DATASET\n"
        )
        report.write("=" * 95 + "\n\n")

        report.write(
            "1. OSNOVNE INFORMACIJE\n"
        )
        report.write("-" * 95 + "\n")
        report.write(
            f"Broj testnih zapisa: "
            f"{len(X_test)}\n"
        )
        report.write(
            "Testni skup korišten je prvi put "
            "u ovoj evaluaciji.\n"
        )
        report.write(
            "Klasa 1: jetreni pacijent\n"
        )
        report.write(
            "Klasa 0: osoba bez oznake "
            "jetrene bolesti\n"
        )
        report.write(
            "Glavna metrika: balanced accuracy\n\n"
        )

        report.write(
            "2. REZULTATI MODELA NA TESTNOM SKUPU\n"
        )
        report.write("-" * 95 + "\n")

        metric_columns = [
            "model_name",
            "accuracy_test",
            "balanced_accuracy_test",
            "precision_class_1_test",
            "recall_class_1_test",
            "f1_class_1_test",
            "precision_class_0_test",
            "recall_class_0_test",
            "f1_class_0_test",
            "specificity_test",
            "negative_predictive_value_test",
            "roc_auc_test",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ]

        report.write(
            test_metrics_df[
                metric_columns
            ].to_string(index=False)
        )

        report.write("\n\n")

        report.write(
            "3. USPOREDBA CV I TESTNIH REZULTATA\n"
        )
        report.write("-" * 95 + "\n")

        report.write(
            cv_test_comparison_df.to_string(
                index=False
            )
        )

        report.write("\n\n")

        report.write(
            "4. MODEL ODABRAN TIJEKOM TRENINGA\n"
        )
        report.write("-" * 95 + "\n")

        report.write(
            f"Naziv modela: "
            f"{selected_model_name}\n"
        )

        report.write(
            f"Test accuracy: "
            f"{selected_metrics['accuracy_test']:.4f}\n"
        )

        report.write(
            f"Test balanced accuracy: "
            f"{selected_metrics['balanced_accuracy_test']:.4f}\n"
        )

        report.write(
            f"Precision klase 1: "
            f"{selected_metrics['precision_class_1_test']:.4f}\n"
        )

        report.write(
            f"Recall klase 1 / osjetljivost: "
            f"{selected_metrics['recall_class_1_test']:.4f}\n"
        )

        report.write(
            f"F1 klase 1: "
            f"{selected_metrics['f1_class_1_test']:.4f}\n"
        )

        report.write(
            f"Precision klase 0: "
            f"{selected_metrics['precision_class_0_test']:.4f}\n"
        )

        report.write(
            f"Recall klase 0 / specifičnost: "
            f"{selected_metrics['recall_class_0_test']:.4f}\n"
        )

        report.write(
            f"F1 klase 0: "
            f"{selected_metrics['f1_class_0_test']:.4f}\n"
        )

        report.write(
            f"ROC AUC: "
            f"{selected_metrics['roc_auc_test']:.4f}\n"
        )

        report.write(
            f"True negative: "
            f"{int(selected_metrics['true_negative'])}\n"
        )

        report.write(
            f"False positive: "
            f"{int(selected_metrics['false_positive'])}\n"
        )

        report.write(
            f"False negative: "
            f"{int(selected_metrics['false_negative'])}\n"
        )

        report.write(
            f"True positive: "
            f"{int(selected_metrics['true_positive'])}\n\n"
        )

        report.write(
            "5. CLASSIFICATION REPORT ODABRANOG MODELA\n"
        )
        report.write("-" * 95 + "\n")

        report.write(
            classification_report_text
        )

        report.write("\n")

        report.write(
            "6. METODOLOŠKA NAPOMENA\n"
        )
        report.write("-" * 95 + "\n")

        report.write(
            "Završna evaluacija provedena je na prethodno "
            "izdvojenom testnom skupu koji nije korišten "
            "tijekom treniranja, optimizacije hiperparametara "
            "ni odabira modela. Službeni model ostaje model "
            "odabran prema prosječnoj balanced accuracy "
            "vrijednosti iz unakrsne validacije. Rezultat "
            "testnog skupa koristi se kao neovisna procjena "
            "sposobnosti generalizacije, a ne za naknadni "
            "odabir povoljnijeg modela.\n"
        )


def save_evaluation_metadata(
    selected_model_name: str,
    test_metrics_df: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Sprema ključne rezultate evaluacije u JSON formatu.
    """
    selected_metrics = test_metrics_df.loc[
        test_metrics_df["model_name"]
        == selected_model_name
    ].iloc[0]

    metadata = {
        "test_rows": int(len(X_test)),
        "selected_model_name": (
            selected_model_name
        ),
        "accuracy_test": float(
            selected_metrics["accuracy_test"]
        ),
        "balanced_accuracy_test": float(
            selected_metrics[
                "balanced_accuracy_test"
            ]
        ),
        "precision_class_1_test": float(
            selected_metrics[
                "precision_class_1_test"
            ]
        ),
        "recall_class_1_test": float(
            selected_metrics[
                "recall_class_1_test"
            ]
        ),
        "f1_class_1_test": float(
            selected_metrics[
                "f1_class_1_test"
            ]
        ),
        "precision_class_0_test": float(
            selected_metrics[
                "precision_class_0_test"
            ]
        ),
        "recall_class_0_test": float(
            selected_metrics[
                "recall_class_0_test"
            ]
        ),
        "f1_class_0_test": float(
            selected_metrics[
                "f1_class_0_test"
            ]
        ),
        "specificity_test": float(
            selected_metrics[
                "specificity_test"
            ]
        ),
        "negative_predictive_value_test": float(
            selected_metrics[
                "negative_predictive_value_test"
            ]
        ),
        "roc_auc_test": float(
            selected_metrics[
                "roc_auc_test"
            ]
        ),
        "true_negative": int(
            selected_metrics["true_negative"]
        ),
        "false_positive": int(
            selected_metrics["false_positive"]
        ),
        "false_negative": int(
            selected_metrics["false_negative"]
        ),
        "true_positive": int(
            selected_metrics["true_positive"]
        ),
        "test_set_evaluated": True,
    }

    with EVALUATION_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4,
        )


# ---------------------------------------------------------
# ISPIS U TERMINAL
# ---------------------------------------------------------

def print_summary(
    test_metrics_df: pd.DataFrame,
    cv_test_comparison_df: pd.DataFrame,
    selected_model_name: str,
) -> None:
    """
    Ispisuje sažetak završne evaluacije u terminal.
    """
    print("\n" + "=" * 80)
    print("ZAVRŠNA EVALUACIJA JETRENIH MODELA")
    print("=" * 80)

    print("\nRezultati na testnom skupu:")

    print(
        test_metrics_df[
            [
                "model_name",
                "balanced_accuracy_test",
                "recall_class_0_test",
                "recall_class_1_test",
                "f1_class_0_test",
                "f1_class_1_test",
                "roc_auc_test",
            ]
        ].to_string(index=False)
    )

    print(
        "\nUsporedba CV i testnih rezultata:"
    )

    print(
        cv_test_comparison_df[
            [
                "model_name",
                "best_balanced_accuracy_cv",
                "balanced_accuracy_test",
                "balanced_accuracy_difference",
            ]
        ].to_string(index=False)
    )

    print(
        "\nModel odabran tijekom treninga:"
    )

    print(selected_model_name)

    selected_metrics = test_metrics_df.loc[
        test_metrics_df["model_name"]
        == selected_model_name
    ].iloc[0]

    print(
        "Test balanced accuracy: "
        f"{selected_metrics['balanced_accuracy_test']:.4f}"
    )

    print(
        "Recall klase 0 / specifičnost: "
        f"{selected_metrics['recall_class_0_test']:.4f}"
    )

    print(
        "Recall klase 1 / osjetljivost: "
        f"{selected_metrics['recall_class_1_test']:.4f}"
    )

    print(
        "F1 klase 0: "
        f"{selected_metrics['f1_class_0_test']:.4f}"
    )

    print(
        "F1 klase 1: "
        f"{selected_metrics['f1_class_1_test']:.4f}"
    )

    print(
        "ROC AUC: "
        f"{selected_metrics['roc_auc_test']:.4f}"
    )

    print("\nIzvještaj spremljen je u:")
    print(EVALUATION_REPORT_PATH)

    print("\nPredikcije spremljene su u:")
    print(PREDICTIONS_PATH)


# ---------------------------------------------------------
# GLAVNI PROGRAM
# ---------------------------------------------------------

def main() -> None:
    """
    Glavna funkcija završne evaluacije.
    """
    create_output_directory()

    print("Učitavanje testnog skupa...")
    X_test, y_test = load_test_dataset()

    print("Učitavanje modela...")
    models = load_models()

    all_metrics = []
    predictions_by_model = {}

    print("Evaluacija modela...")

    for model_name, model in models.items():
        print("-" * 80)
        print(
            f"Evaluacija modela: {model_name}"
        )

        metrics, predictions_df = (
            evaluate_model(
                model_name=model_name,
                model=model,
                X_test=X_test,
                y_test=y_test,
            )
        )

        all_metrics.append(metrics)

        predictions_by_model[
            model_name
        ] = predictions_df

        y_pred = model.predict(X_test)

        save_confusion_matrix_plot(
            model_name=model_name,
            y_test=y_test,
            y_pred=y_pred,
        )

        save_normalized_confusion_matrix_plot(
            model_name=model_name,
            y_test=y_test,
            y_pred=y_pred,
        )

        print(
            "Balanced accuracy: "
            f"{metrics['balanced_accuracy_test']:.4f}"
        )

        print(
            "Recall klase 0: "
            f"{metrics['recall_class_0_test']:.4f}"
        )

        print(
            "Recall klase 1: "
            f"{metrics['recall_class_1_test']:.4f}"
        )

        print(
            "ROC AUC: "
            f"{metrics['roc_auc_test']:.4f}"
        )

    test_metrics_df = pd.DataFrame(
        all_metrics
    ).sort_values(
        by="balanced_accuracy_test",
        ascending=False,
    ).reset_index(drop=True)

    print("Izrada ROC grafikona...")

    save_roc_curve_plot(
        models=models,
        X_test=X_test,
        y_test=y_test,
    )

    print(
        "Usporedba CV i testnih rezultata..."
    )

    cv_test_comparison_df = (
        create_cv_test_comparison(
            test_metrics_df
        )
    )

    test_metrics_df.to_csv(
        TEST_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    cv_test_comparison_df.to_csv(
        CV_TEST_COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Spremanje pojedinačnih predikcija..."
    )

    save_predictions(
        predictions_by_model
    )

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Najbolji model nije pronađen:\n"
            f"{BEST_MODEL_PATH}"
        )

    selected_model = joblib.load(
        BEST_MODEL_PATH
    )

    selected_model_name = (
        get_training_selected_model_name(
            cv_test_comparison_df
        )
    )

    print(
        "Spremanje završnog izvještaja..."
    )

    save_evaluation_report(
        test_metrics_df=test_metrics_df,
        cv_test_comparison_df=(
            cv_test_comparison_df
        ),
        selected_model_name=(
            selected_model_name
        ),
        selected_model=selected_model,
        X_test=X_test,
        y_test=y_test,
    )

    save_evaluation_metadata(
        selected_model_name=(
            selected_model_name
        ),
        test_metrics_df=test_metrics_df,
        X_test=X_test,
    )

    print_summary(
        test_metrics_df=test_metrics_df,
        cv_test_comparison_df=(
            cv_test_comparison_df
        ),
        selected_model_name=(
            selected_model_name
        ),
    )

    print(
        "\nZavršna evaluacija jetrenih modela "
        "uspješno je završena."
    )


if __name__ == "__main__":
    main()