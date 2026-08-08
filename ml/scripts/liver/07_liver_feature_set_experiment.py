from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    f1_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ---------------------------------------------------------
# PUTANJE
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parents[1]

TRAIN_SPLIT_PATH = (
    ML_DIR
    / "data"
    / "processed"
    / "liver"
    / "liver_train_split.csv"
)

REPORT_DIR = (
    ML_DIR
    / "reports"
    / "liver"
    / "feature_set_experiment"
)

COMPARISON_PATH = (
    REPORT_DIR
    / "feature_set_comparison.csv"
)

DETAILED_RESULTS_PATH = (
    REPORT_DIR
    / "feature_set_cv_results.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "feature_set_report.txt"
)


# ---------------------------------------------------------
# KONFIGURACIJA
# ---------------------------------------------------------

RANDOM_STATE = 42

CV_SPLITS = 5
CV_REPEATS = 5

TARGET_COLUMN = "Liver_Disease"

GENDER_FEATURE = [
    "Gender",
]


# ---------------------------------------------------------
# TRI SKUPA ZNAČAJKI
# ---------------------------------------------------------

FEATURE_SETS = {
    # Originalni ILPD model.
    "A_full_10_features": [
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
    ],

    # Bez A/G omjera.
    "B_without_ag_ratio_9_features": [
        "Gender",
        "Age",
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphatase",
        "Alanine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Proteins",
        "Albumin",
    ],

    # Bez A/G omjera i direktnog bilirubina.
    "C_practical_8_features": [
        "Gender",
        "Age",
        "Total_Bilirubin",
        "Alkaline_Phosphatase",
        "Alanine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Proteins",
        "Albumin",
    ],
}


# ---------------------------------------------------------
# METRIKE
# ---------------------------------------------------------

RECALL_CLASS_0 = make_scorer(
    recall_score,
    pos_label=0,
    zero_division=0,
)

RECALL_CLASS_1 = make_scorer(
    recall_score,
    pos_label=1,
    zero_division=0,
)

F1_MACRO = make_scorer(
    f1_score,
    average="macro",
    zero_division=0,
)

SCORING = {
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
    "recall_class_0": RECALL_CLASS_0,
    "recall_class_1": RECALL_CLASS_1,
    "f1_macro": F1_MACRO,
    "roc_auc": "roc_auc",
}


# ---------------------------------------------------------
# MAPA
# ---------------------------------------------------------

def create_output_directory() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------------------------------------------------------
# PODACI
# ---------------------------------------------------------

def load_training_dataset() -> pd.DataFrame:
    """
    Učitava isključivo prethodno izdvojeni trening skup.

    Testni dataset se u ovoj skripti ne koristi.
    """

    if not TRAIN_SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Trening skup nije pronađen:\n"
            f"{TRAIN_SPLIT_PATH}"
        )

    df = pd.read_csv(
        TRAIN_SPLIT_PATH,
        encoding="utf-8-sig",
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Nedostaje ciljna varijabla {TARGET_COLUMN}."
        )

    return df


# ---------------------------------------------------------
# PRETPROCESIRANJE
# ---------------------------------------------------------

def create_preprocessor(
    feature_columns: list[str],
) -> ColumnTransformer:

    numeric_features = [
        feature
        for feature in feature_columns
        if feature != "Gender"
    ]

    categorical_features = [
        feature
        for feature in feature_columns
        if feature == "Gender"
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


# ---------------------------------------------------------
# MODELI
# ---------------------------------------------------------

def create_model_configurations(
    feature_columns: list[str],
) -> dict:

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    feature_columns
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    feature_columns
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": {
            "pipeline": logistic_pipeline,
            "parameter_grid": {
                "classifier__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ],
                "classifier__class_weight": [
                    None,
                    "balanced",
                ],
            },
        },

        "Random Forest": {
            "pipeline": random_forest_pipeline,
            "parameter_grid": {
                "classifier__n_estimators": [
                    200,
                    400,
                ],
                "classifier__max_depth": [
                    8,
                    15,
                ],
                "classifier__min_samples_leaf": [
                    3,
                    5,
                ],
                "classifier__class_weight": [
                    None,
                    "balanced",
                ],
            },
        },
    }


# ---------------------------------------------------------
# EKSPERIMENT
# ---------------------------------------------------------

def run_feature_set_experiment(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    y_train = (
        df[TARGET_COLUMN]
        .astype(int)
        .copy()
    )

    repeated_cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    comparison_rows = []
    all_cv_results = []

    for (
        feature_set_name,
        feature_columns,
    ) in FEATURE_SETS.items():

        print("\n" + "=" * 80)
        print(
            f"SKUP ZNAČAJKI: {feature_set_name}"
        )
        print("=" * 80)

        missing_columns = [
            column
            for column in feature_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Nedostaju značajke: "
                + ", ".join(missing_columns)
            )

        X_train = df[
            feature_columns
        ].copy()

        configurations = (
            create_model_configurations(
                feature_columns
            )
        )

        for (
            model_name,
            configuration,
        ) in configurations.items():

            print("-" * 80)
            print(
                f"Model: {model_name}"
            )

            grid_search = GridSearchCV(
                estimator=configuration[
                    "pipeline"
                ],
                param_grid=configuration[
                    "parameter_grid"
                ],
                scoring=SCORING,
                refit="balanced_accuracy",
                cv=repeated_cv,
                n_jobs=-1,
                return_train_score=True,
                error_score="raise",
            )

            grid_search.fit(
                X_train,
                y_train,
            )

            results_df = pd.DataFrame(
                grid_search.cv_results_
            )

            results_df.insert(
                0,
                "feature_set",
                feature_set_name,
            )

            results_df.insert(
                1,
                "model_name",
                model_name,
            )

            all_cv_results.append(
                results_df
            )

            best_index = (
                grid_search.best_index_
            )

            comparison_rows.append(
                {
                    "feature_set": (
                        feature_set_name
                    ),
                    "number_of_features": (
                        len(feature_columns)
                    ),
                    "model_name": (
                        model_name
                    ),
                    "balanced_accuracy_mean": (
                        results_df.loc[
                            best_index,
                            "mean_test_balanced_accuracy",
                        ]
                    ),
                    "balanced_accuracy_std": (
                        results_df.loc[
                            best_index,
                            "std_test_balanced_accuracy",
                        ]
                    ),
                    "accuracy_mean": (
                        results_df.loc[
                            best_index,
                            "mean_test_accuracy",
                        ]
                    ),
                    "recall_class_0_mean": (
                        results_df.loc[
                            best_index,
                            "mean_test_recall_class_0",
                        ]
                    ),
                    "recall_class_1_mean": (
                        results_df.loc[
                            best_index,
                            "mean_test_recall_class_1",
                        ]
                    ),
                    "f1_macro_mean": (
                        results_df.loc[
                            best_index,
                            "mean_test_f1_macro",
                        ]
                    ),
                    "roc_auc_mean": (
                        results_df.loc[
                            best_index,
                            "mean_test_roc_auc",
                        ]
                    ),
                    "best_parameters": (
                        json.dumps(
                            grid_search.best_params_,
                            ensure_ascii=False,
                        )
                    ),
                }
            )

            print(
                "Balanced accuracy: "
                f"{grid_search.best_score_:.4f}"
            )

            print(
                "Najbolji parametri:"
            )

            print(
                grid_search.best_params_
            )

    comparison_df = (
        pd.DataFrame(
            comparison_rows
        )
        .sort_values(
            by="balanced_accuracy_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    detailed_results_df = (
        pd.concat(
            all_cv_results,
            ignore_index=True,
        )
    )

    return (
        comparison_df,
        detailed_results_df,
    )


# ---------------------------------------------------------
# IZVJEŠTAJ
# ---------------------------------------------------------

def save_outputs(
    comparison_df: pd.DataFrame,
    detailed_results_df: pd.DataFrame,
) -> None:

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    detailed_results_df.to_csv(
        DETAILED_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def save_text_report(
    df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "USPOREDBA SKUPOVA ZNAČAJKI – "
            "INDIAN LIVER PATIENT DATASET\n"
        )

        report.write(
            "=" * 100 + "\n\n"
        )

        report.write(
            "1. METODOLOGIJA\n"
        )

        report.write(
            "-" * 100 + "\n"
        )

        report.write(
            f"Broj trening zapisa: "
            f"{len(df)}\n"
        )

        report.write(
            f"CV podjele: "
            f"{CV_SPLITS}\n"
        )

        report.write(
            f"Broj ponavljanja: "
            f"{CV_REPEATS}\n"
        )

        report.write(
            "Ukupan broj validacijskih podjela "
            f"po kombinaciji: "
            f"{CV_SPLITS * CV_REPEATS}\n"
        )

        report.write(
            "Testni skup nije korišten.\n\n"
        )

        report.write(
            "2. SKUPOVI ZNAČAJKI\n"
        )

        report.write(
            "-" * 100 + "\n"
        )

        for (
            feature_set_name,
            feature_columns,
        ) in FEATURE_SETS.items():

            report.write(
                f"\n{feature_set_name}\n"
            )

            for feature in feature_columns:
                report.write(
                    f"- {feature}\n"
                )

        report.write("\n")

        report.write(
            "3. REZULTATI\n"
        )

        report.write(
            "-" * 100 + "\n"
        )

        selected_columns = [
            "feature_set",
            "number_of_features",
            "model_name",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "accuracy_mean",
            "recall_class_0_mean",
            "recall_class_1_mean",
            "f1_macro_mean",
            "roc_auc_mean",
            "best_parameters",
        ]

        report.write(
            comparison_df[
                selected_columns
            ].to_string(
                index=False
            )
        )

        report.write("\n\n")

        report.write(
            "4. NAJBOLJI REZULTAT\n"
        )

        report.write(
            "-" * 100 + "\n"
        )

        best_row = (
            comparison_df.iloc[0]
        )

        report.write(
            "Skup značajki: "
            f"{best_row['feature_set']}\n"
        )

        report.write(
            "Model: "
            f"{best_row['model_name']}\n"
        )

        report.write(
            "Balanced accuracy: "
            f"{best_row['balanced_accuracy_mean']:.4f}\n"
        )

        report.write(
            "Recall klase 0: "
            f"{best_row['recall_class_0_mean']:.4f}\n"
        )

        report.write(
            "Recall klase 1: "
            f"{best_row['recall_class_1_mean']:.4f}\n"
        )

        report.write(
            "Macro F1: "
            f"{best_row['f1_macro_mean']:.4f}\n"
        )

        report.write(
            "ROC AUC: "
            f"{best_row['roc_auc_mean']:.4f}\n\n"
        )

        report.write(
            "5. METODOLOŠKA NAPOMENA\n"
        )

        report.write(
            "-" * 100 + "\n"
        )

        report.write(
            "Eksperiment uspoređuje puni ILPD skup "
            "značajki s praktičnijim verzijama koje "
            "uklanjaju parametre koji nisu redovito "
            "dostupni u laboratorijskim PDF nalazima. "
            "Svi modeli evaluirani su na istom trening "
            "skupu uz ponovljenu stratificiranu "
            "unakrsnu validaciju. Testni skup nije "
            "ponovno korišten.\n"
        )


# ---------------------------------------------------------
# TERMINAL
# ---------------------------------------------------------

def print_summary(
    comparison_df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print(
        "USPOREDBA SKUPOVA ZNAČAJKI"
    )
    print("=" * 80)

    print(
        comparison_df[
            [
                "feature_set",
                "model_name",
                "balanced_accuracy_mean",
                "recall_class_0_mean",
                "recall_class_1_mean",
                "f1_macro_mean",
                "roc_auc_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nNajbolja kombinacija:"
    )

    best_row = (
        comparison_df.iloc[0]
    )

    print(
        f"{best_row['feature_set']} "
        f"+ {best_row['model_name']}"
    )

    print(
        "Balanced accuracy: "
        f"{best_row['balanced_accuracy_mean']:.4f}"
    )

    print(
        "\nIzvještaj spremljen je u:"
    )

    print(
        REPORT_PATH
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:

    create_output_directory()

    print(
        "Učitavanje postojećeg trening skupa..."
    )

    df = load_training_dataset()

    print(
        "Pokretanje eksperimenta..."
    )

    (
        comparison_df,
        detailed_results_df,
    ) = run_feature_set_experiment(
        df
    )

    print(
        "Spremanje rezultata..."
    )

    save_outputs(
        comparison_df,
        detailed_results_df,
    )

    save_text_report(
        df,
        comparison_df,
    )

    print_summary(
        comparison_df
    )

    print(
        "\nEksperiment uspješno završen."
    )


if __name__ == "__main__":
    main()