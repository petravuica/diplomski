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
)

MODELS_DIR = ML_DIR / "models"

EVALUATION_REPORT_DIR = (
    ML_DIR
    / "reports"
    / "evaluation"
)

TEST_SPLIT_PATH = (
    PROCESSED_DATA_DIR
    / "anemia_test_split.csv"
)

TRAINING_COMPARISON_PATH = (
    ML_DIR
    / "reports"
    / "training"
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
    / "best_anemia_model.joblib"
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

TARGET_COLUMN = "Decision_Class"

FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "HGB",
    "RBC",
    "HCT",
    "MCV",
    "MCH",
    "MCHC",
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
    Stvara mapu za rezultate evaluacije.
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
    Učitava netaknuti testni skup.
    """
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Testni skup nije pronađen na putanji:\n"
            f"{TEST_SPLIT_PATH}\n\n"
            "Prvo pokreni skriptu 04_train_models.py."
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

    if df[expected_columns].isna().any().any():
        raise ValueError(
            "Testni skup sadrži nedostajuće vrijednosti."
        )

    X_test = df[FEATURE_COLUMNS].copy()
    y_test = df[TARGET_COLUMN].astype(int).copy()

    return X_test, y_test


def load_models() -> dict:
    """
    Učitava spremljene modele.
    """
    models = {}

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model '{model_name}' nije pronađen:\n"
                f"{model_path}\n\n"
                "Prvo pokreni skriptu 04_train_models.py."
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
    Izračunava specifičnost:

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


def get_probability_scores(
    model,
    X_test: pd.DataFrame,
):
    """
    Dohvaća vjerojatnost pozitivne klase.

    Ako model nema predict_proba, pokušava koristiti
    decision_function.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            X_test
        )

        return probabilities[:, 1]

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
    Evaluira jedan model na testnom skupu.
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
        "precision_test": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "recall_test": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "specificity_test": calculate_specificity(
            y_test,
            y_pred,
        ),
        "f1_test": f1_score(
            y_test,
            y_pred,
            zero_division=0,
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
            "probability_class_1"
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

def save_confusion_matrix_plot(
    model_name: str,
    y_test: pd.Series,
    y_pred,
) -> None:
    """
    Sprema confusion matrix grafikon.
    """
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=[
            "Klasa 0",
            "Klasa 1",
        ],
        values_format="d",
    )

    display.ax_.set_title(
        f"Confusion matrix – {model_name}"
    )

    display.ax_.set_xlabel(
        "Predviđena klasa"
    )

    display.ax_.set_ylabel(
        "Stvarna klasa"
    )

    plt.tight_layout()

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    output_path = (
        EVALUATION_REPORT_DIR
        / f"confusion_matrix_{safe_name}.png"
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
    Sprema ROC krivulje svih modela na jednom grafikonu.
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
        "ROC krivulje modela na testnom skupu"
    )

    axis.set_xlabel(
        "Stopa lažno pozitivnih rezultata"
    )

    axis.set_ylabel(
        "Stopa stvarno pozitivnih rezultata"
    )

    figure.tight_layout()

    output_path = (
        EVALUATION_REPORT_DIR
        / "roc_curves_test_set.png"
    )

    figure.savefig(
        output_path,
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
    Uspoređuje rezultate unakrsne validacije
    i testnog skupa.
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
        "f1_difference"
    ] = (
        comparison_df["f1_test"]
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
        "f1_test",
        "f1_difference",
        "roc_auc_cv",
        "roc_auc_test",
        "roc_auc_difference",
    ]

    return comparison_df[
        selected_columns
    ].sort_values(
        by="balanced_accuracy_test",
        ascending=False,
    ).reset_index(drop=True)


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
    Sprema predikcije svakog modela
    na zaseban Excel list.
    """
    with pd.ExcelWriter(
        PREDICTIONS_PATH,
        engine="openpyxl",
    ) as writer:
        for model_name, predictions_df in (
            predictions_by_model.items()
        ):
            safe_sheet_name = (
                model_name[:31]
            )

            predictions_df.to_excel(
                writer,
                sheet_name=safe_sheet_name,
                index=False,
            )


def save_evaluation_report(
    test_metrics_df: pd.DataFrame,
    cv_test_comparison_df: pd.DataFrame,
    best_model_name: str,
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Sprema tekstualni izvještaj završne evaluacije.
    """
    best_predictions = best_model.predict(
        X_test
    )

    report_text = classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=[
            "Klasa 0",
            "Klasa 1",
        ],
        zero_division=0,
        digits=4,
    )

    best_metrics = test_metrics_df.loc[
        test_metrics_df["model_name"]
        == best_model_name
    ].iloc[0]

    with EVALUATION_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "ZAVRŠNA EVALUACIJA MODELA – "
            "ANEMIA DATASET\n"
        )
        report.write("=" * 80 + "\n\n")

        report.write(
            "1. OSNOVNE INFORMACIJE\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            f"Broj testnih zapisa: "
            f"{len(X_test)}\n"
        )
        report.write(
            "Testni skup korišten je prvi put "
            "u ovoj evaluaciji.\n"
        )
        report.write(
            "Pozitivna klasa: 1\n"
        )
        report.write(
            "Negativna klasa: 0\n\n"
        )

        report.write(
            "2. REZULTATI MODELA NA TESTNOM SKUPU\n"
        )
        report.write("-" * 80 + "\n")

        metric_columns = [
            "model_name",
            "accuracy_test",
            "balanced_accuracy_test",
            "precision_test",
            "recall_test",
            "specificity_test",
            "f1_test",
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
        report.write("-" * 80 + "\n")

        report.write(
            cv_test_comparison_df.to_string(
                index=False
            )
        )

        report.write("\n\n")

        report.write(
            "4. ODABRANI MODEL\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            f"Naziv modela: "
            f"{best_model_name}\n"
        )
        report.write(
            f"Test accuracy: "
            f"{best_metrics['accuracy_test']:.4f}\n"
        )
        report.write(
            f"Test balanced accuracy: "
            f"{best_metrics['balanced_accuracy_test']:.4f}\n"
        )
        report.write(
            f"Test precision: "
            f"{best_metrics['precision_test']:.4f}\n"
        )
        report.write(
            f"Test recall / osjetljivost: "
            f"{best_metrics['recall_test']:.4f}\n"
        )
        report.write(
            f"Test specifičnost: "
            f"{best_metrics['specificity_test']:.4f}\n"
        )
        report.write(
            f"Test F1: "
            f"{best_metrics['f1_test']:.4f}\n"
        )
        report.write(
            f"Test ROC AUC: "
            f"{best_metrics['roc_auc_test']:.4f}\n"
        )
        report.write(
            f"True negative: "
            f"{int(best_metrics['true_negative'])}\n"
        )
        report.write(
            f"False positive: "
            f"{int(best_metrics['false_positive'])}\n"
        )
        report.write(
            f"False negative: "
            f"{int(best_metrics['false_negative'])}\n"
        )
        report.write(
            f"True positive: "
            f"{int(best_metrics['true_positive'])}\n\n"
        )

        report.write(
            "5. CLASSIFICATION REPORT ODABRANOG MODELA\n"
        )
        report.write("-" * 80 + "\n")
        report.write(report_text)
        report.write("\n")

        report.write(
            "6. METODOLOŠKA NAPOMENA\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            "Završna evaluacija provedena je na prethodno "
            "izdvojenom testnom skupu koji nije korišten "
            "tijekom treniranja, optimizacije hiperparametara "
            "ni odabira modela. Na taj način dobivena je "
            "neovisna procjena sposobnosti generalizacije "
            "modela na novim podacima.\n"
        )


def save_evaluation_metadata(
    best_model_name: str,
    test_metrics_df: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Sprema sažetak evaluacije u JSON formatu.
    """
    best_metrics = test_metrics_df.loc[
        test_metrics_df["model_name"]
        == best_model_name
    ].iloc[0]

    metadata = {
        "test_rows": int(len(X_test)),
        "best_model_name": best_model_name,
        "accuracy_test": float(
            best_metrics["accuracy_test"]
        ),
        "balanced_accuracy_test": float(
            best_metrics[
                "balanced_accuracy_test"
            ]
        ),
        "precision_test": float(
            best_metrics["precision_test"]
        ),
        "recall_test": float(
            best_metrics["recall_test"]
        ),
        "specificity_test": float(
            best_metrics["specificity_test"]
        ),
        "f1_test": float(
            best_metrics["f1_test"]
        ),
        "roc_auc_test": float(
            best_metrics["roc_auc_test"]
        ),
        "true_negative": int(
            best_metrics["true_negative"]
        ),
        "false_positive": int(
            best_metrics["false_positive"]
        ),
        "false_negative": int(
            best_metrics["false_negative"]
        ),
        "true_positive": int(
            best_metrics["true_positive"]
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
    best_model_name: str,
) -> None:
    """
    Ispisuje sažetak završne evaluacije.
    """
    print("\n" + "=" * 70)
    print("ZAVRŠNA EVALUACIJA MODELA")
    print("=" * 70)

    print("\nRezultati na testnom skupu:")

    print(
        test_metrics_df[
            [
                "model_name",
                "balanced_accuracy_test",
                "precision_test",
                "recall_test",
                "specificity_test",
                "f1_test",
                "roc_auc_test",
            ]
        ].to_string(index=False)
    )

    print("\nUsporedba CV i testnih rezultata:")

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

    print("\nOdabrani model:")
    print(best_model_name)

    best_metrics = test_metrics_df.loc[
        test_metrics_df["model_name"]
        == best_model_name
    ].iloc[0]

    print(
        "Test balanced accuracy: "
        f"{best_metrics['balanced_accuracy_test']:.4f}"
    )

    print(
        "Test recall / osjetljivost: "
        f"{best_metrics['recall_test']:.4f}"
    )

    print(
        "Test specifičnost: "
        f"{best_metrics['specificity_test']:.4f}"
    )

    print(
        "Test F1: "
        f"{best_metrics['f1_test']:.4f}"
    )

    print(
        "Test ROC AUC: "
        f"{best_metrics['roc_auc_test']:.4f}"
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
    Glavna funkcija skripte.
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
        print("-" * 70)
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

        print(
            "Balanced accuracy: "
            f"{metrics['balanced_accuracy_test']:.4f}"
        )

        print(
            "F1: "
            f"{metrics['f1_test']:.4f}"
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

    print("Usporedba CV i testnih rezultata...")

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

    print("Spremanje pojedinačnih predikcija...")

    save_predictions(
        predictions_by_model
    )

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Najbolji model nije pronađen:\n"
            f"{BEST_MODEL_PATH}"
        )

    best_model = joblib.load(
        BEST_MODEL_PATH
    )

    best_model_name = (
        cv_test_comparison_df.loc[
            cv_test_comparison_df[
                "best_balanced_accuracy_cv"
            ].idxmax(),
            "model_name",
        ]
    )

    print("Spremanje završnog izvještaja...")

    save_evaluation_report(
        test_metrics_df=test_metrics_df,
        cv_test_comparison_df=(
            cv_test_comparison_df
        ),
        best_model_name=best_model_name,
        best_model=best_model,
        X_test=X_test,
        y_test=y_test,
    )

    save_evaluation_metadata(
        best_model_name=best_model_name,
        test_metrics_df=test_metrics_df,
        X_test=X_test,
    )

    print_summary(
        test_metrics_df=test_metrics_df,
        cv_test_comparison_df=(
            cv_test_comparison_df
        ),
        best_model_name=best_model_name,
    )

    print(
        "\nZavršna evaluacija uspješno je završena."
    )


if __name__ == "__main__":
    main()