from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    f1_score,
    precision_score,
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
from sklearn.svm import SVC


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
    / "experiments"
)

MODELS_DIR = (
    ML_DIR
    / "models"
    / "liver"
    / "experiments"
)

COMPARISON_PATH = (
    REPORT_DIR
    / "experiment_model_comparison.csv"
)

ALL_RESULTS_PATH = (
    REPORT_DIR
    / "experiment_cv_results.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "experiment_report.txt"
)

METADATA_PATH = (
    REPORT_DIR
    / "experiment_metadata.json"
)

BEST_EXPERIMENT_MODEL_PATH = (
    MODELS_DIR
    / "best_experimental_liver_model.joblib"
)


# ---------------------------------------------------------
# KONFIGURACIJA
# ---------------------------------------------------------

RANDOM_STATE = 42

CV_SPLITS = 5
CV_REPEATS = 5

TARGET_COLUMN = "Liver_Disease"

CATEGORICAL_FEATURES = [
    "Gender",
]

NUMERIC_FEATURES = [
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

FEATURE_COLUMNS = [
    *CATEGORICAL_FEATURES,
    *NUMERIC_FEATURES,
]


# Klasa 1 = jetreni pacijent.
RECALL_CLASS_1_SCORER = make_scorer(
    recall_score,
    pos_label=1,
    zero_division=0,
)

# Klasa 0 = osoba bez oznake jetrene bolesti.
RECALL_CLASS_0_SCORER = make_scorer(
    recall_score,
    pos_label=0,
    zero_division=0,
)

PRECISION_CLASS_1_SCORER = make_scorer(
    precision_score,
    pos_label=1,
    zero_division=0,
)

F1_MACRO_SCORER = make_scorer(
    f1_score,
    average="macro",
    zero_division=0,
)

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "recall_class_0": RECALL_CLASS_0_SCORER,
    "recall_class_1": RECALL_CLASS_1_SCORER,
    "precision_class_1": PRECISION_CLASS_1_SCORER,
    "f1_macro": F1_MACRO_SCORER,
    "roc_auc": "roc_auc",
}


# ---------------------------------------------------------
# PRIPREMA
# ---------------------------------------------------------

def create_output_directories() -> None:
    """
    Stvara mape za izvještaje i eksperimentalne modele.
    """
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_training_dataset() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Učitava samo trening skup.

    Testni skup se namjerno ne otvara niti koristi.
    """
    if not TRAIN_SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Trening skup nije pronađen:\n"
            f"{TRAIN_SPLIT_PATH}\n\n"
            "Prvo pokreni 04_train_liver_models.py."
        )

    df = pd.read_csv(
        TRAIN_SPLIT_PATH,
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
            "U trening skupu nedostaju stupci: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError(
            "Trening skup je prazan."
        )

    required_columns = [
        column
        for column in expected_columns
        if column
        != "Albumin_and_Globulin_Ratio"
    ]

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Trening skup sadrži nedostajuće vrijednosti "
            "izvan dopuštenog A/G omjera."
        )

    X_train = df[FEATURE_COLUMNS].copy()
    y_train = df[TARGET_COLUMN].astype(int).copy()

    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError(
            "Ciljna varijabla mora sadržavati "
            "isključivo klase 0 i 1."
        )

    return X_train, y_train


# ---------------------------------------------------------
# PRETPROCESIRANJE
# ---------------------------------------------------------

def create_scaled_preprocessor() -> ColumnTransformer:
    """
    Pretprocesiranje za modele osjetljive na skalu:
    Logistic Regression i SVM.
    """
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
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def create_tree_preprocessor() -> ColumnTransformer:
    """
    Pretprocesiranje za modele stabala.

    StandardScaler nije nužan za modele temeljene
    na stablima, pa se numeričke vrijednosti samo imputiraju.
    """
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
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
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# ---------------------------------------------------------
# MODELI
# ---------------------------------------------------------

def create_model_configurations() -> dict:
    """
    Definira četiri modela i umjerene hiperparametarske
    mreže prikladne za laptop s ograničenom memorijom.
    """
    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_scaled_preprocessor(),
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
                create_tree_preprocessor(),
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

    svm_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_scaled_preprocessor(),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    gradient_boosting_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_tree_preprocessor(),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
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
        "SVM RBF": {
            "pipeline": svm_pipeline,
            "parameter_grid": {
                "classifier__C": [
                    0.1,
                    1.0,
                    10.0,
                ],
                "classifier__gamma": [
                    "scale",
                    0.01,
                    0.1,
                ],
                "classifier__class_weight": [
                    None,
                    "balanced",
                ],
            },
        },
        "Gradient Boosting": {
            "pipeline": gradient_boosting_pipeline,
            "parameter_grid": {
                "classifier__n_estimators": [
                    50,
                    100,
                ],
                "classifier__learning_rate": [
                    0.03,
                    0.1,
                ],
                "classifier__max_depth": [
                    1,
                    2,
                ],
                "classifier__min_samples_leaf": [
                    3,
                    5,
                ],
            },
        },
    }


# ---------------------------------------------------------
# EKSPERIMENT
# ---------------------------------------------------------

def run_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    dict[str, GridSearchCV],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Izvodi ponovljenu stratificiranu unakrsnu validaciju
    samo na trening skupu.
    """
    repeated_cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    configurations = (
        create_model_configurations()
    )

    trained_searches: dict[
        str,
        GridSearchCV,
    ] = {}

    all_results = []
    comparison_rows = []

    for model_name, configuration in (
        configurations.items()
    ):
        print("-" * 80)
        print(
            f"Eksperimentalni model: {model_name}"
        )

        grid_search = GridSearchCV(
            estimator=configuration["pipeline"],
            param_grid=configuration[
                "parameter_grid"
            ],
            scoring=SCORING,
            refit="balanced_accuracy",
            cv=repeated_cv,
            n_jobs=-1,
            return_train_score=True,
            error_score="raise",
            verbose=0,
        )

        grid_search.fit(
            X_train,
            y_train,
        )

        trained_searches[
            model_name
        ] = grid_search

        results_df = pd.DataFrame(
            grid_search.cv_results_
        )

        results_df.insert(
            0,
            "model_name",
            model_name,
        )

        all_results.append(
            results_df
        )

        best_index = grid_search.best_index_

        comparison_rows.append(
            {
                "model_name": model_name,
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
                "precision_class_1_mean": (
                    results_df.loc[
                        best_index,
                        "mean_test_precision_class_1",
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
                "roc_auc_std": (
                    results_df.loc[
                        best_index,
                        "std_test_roc_auc",
                    ]
                ),
                "best_parameters": json.dumps(
                    grid_search.best_params_,
                    ensure_ascii=False,
                ),
            }
        )

        print(
            "Najbolji repeated-CV balanced accuracy: "
            f"{grid_search.best_score_:.4f}"
        )

        print("Najbolji hiperparametri:")
        print(grid_search.best_params_)

    all_results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    ).sort_values(
        by="balanced_accuracy_mean",
        ascending=False,
    ).reset_index(drop=True)

    return (
        trained_searches,
        all_results_df,
        comparison_df,
    )


def save_best_experimental_model(
    trained_searches: dict[str, GridSearchCV],
    comparison_df: pd.DataFrame,
) -> tuple[str, GridSearchCV]:
    """
    Sprema najbolji eksperimentalni model u zasebnu mapu.

    Ne prepisuje postojeći službeni best_liver_model.joblib.
    """
    best_model_name = str(
        comparison_df.loc[
            0,
            "model_name",
        ]
    )

    best_search = trained_searches[
        best_model_name
    ]

    joblib.dump(
        best_search.best_estimator_,
        BEST_EXPERIMENT_MODEL_PATH,
    )

    return best_model_name, best_search


# ---------------------------------------------------------
# IZVJEŠTAJI
# ---------------------------------------------------------

def save_outputs(
    all_results_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    """
    Sprema tablične rezultate.
    """
    all_results_df.to_csv(
        ALL_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def save_report(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_search: GridSearchCV,
) -> None:
    """
    Sprema tekstualni izvještaj eksperimenta.
    """
    class_distribution = (
        y_train.value_counts()
        .sort_index()
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
    )

    class_distribution["percentage"] = (
        class_distribution["count"]
        / len(y_train)
        * 100
    ).round(2)

    report_columns = [
        "model_name",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "accuracy_mean",
        "recall_class_0_mean",
        "recall_class_1_mean",
        "precision_class_1_mean",
        "f1_macro_mean",
        "roc_auc_mean",
        "roc_auc_std",
        "best_parameters",
    ]

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "DODATNI EKSPERIMENTI – "
            "INDIAN LIVER PATIENT DATASET\n"
        )
        report.write("=" * 95 + "\n\n")

        report.write(
            "1. METODOLOGIJA\n"
        )
        report.write("-" * 95 + "\n")
        report.write(
            f"Broj trening zapisa: {len(X_train)}\n"
        )
        report.write(
            f"Broj podjela: {CV_SPLITS}\n"
        )
        report.write(
            f"Broj ponavljanja: {CV_REPEATS}\n"
        )
        report.write(
            "Ukupan broj validacijskih podjela "
            f"po kombinaciji: {CV_SPLITS * CV_REPEATS}\n"
        )
        report.write(
            "Glavna metrika: balanced accuracy\n"
        )
        report.write(
            "Testni skup nije učitan niti korišten.\n\n"
        )

        report.write(
            "2. RASPODJELA KLASA U TRENING SKUPU\n"
        )
        report.write("-" * 95 + "\n")
        report.write(
            class_distribution.to_string(
                index=False
            )
        )
        report.write("\n\n")

        report.write(
            "3. USPOREDBA MODELA\n"
        )
        report.write("-" * 95 + "\n")
        report.write(
            comparison_df[
                report_columns
            ].to_string(index=False)
        )
        report.write("\n\n")

        report.write(
            "4. NAJBOLJI EKSPERIMENTALNI MODEL\n"
        )
        report.write("-" * 95 + "\n")
        report.write(
            f"Naziv: {best_model_name}\n"
        )
        report.write(
            "Repeated-CV balanced accuracy: "
            f"{best_search.best_score_:.4f}\n"
        )
        report.write(
            "Najbolji hiperparametri:\n"
        )
        report.write(
            json.dumps(
                best_search.best_params_,
                ensure_ascii=False,
                indent=4,
            )
        )
        report.write("\n\n")

        report.write(
            "5. NAPOMENA\n"
        )
        report.write("-" * 95 + "\n")
        report.write(
            "Eksperiment je proveden isključivo na prethodno "
            "izdvojenom trening skupu. Postojeći testni skup "
            "nije ponovno korišten. Eksperimentalni model "
            "spremljen je odvojeno i ne prepisuje službeni "
            "model iz prve faze treniranja.\n"
        )


def save_metadata(
    X_train: pd.DataFrame,
    best_model_name: str,
    best_search: GridSearchCV,
) -> None:
    """
    Sprema metapodatke eksperimenta.
    """
    metadata = {
        "created_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "training_rows": int(len(X_train)),
        "cv_splits": CV_SPLITS,
        "cv_repeats": CV_REPEATS,
        "total_validation_splits": (
            CV_SPLITS * CV_REPEATS
        ),
        "selection_metric": (
            "balanced_accuracy"
        ),
        "models_compared": [
            "Logistic Regression",
            "Random Forest",
            "SVM RBF",
            "Gradient Boosting",
        ],
        "best_model_name": best_model_name,
        "best_balanced_accuracy": float(
            best_search.best_score_
        ),
        "best_parameters": (
            best_search.best_params_
        ),
        "test_dataset_loaded": False,
        "official_model_overwritten": False,
    }

    with METADATA_PATH.open(
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
# ISPIS
# ---------------------------------------------------------

def print_summary(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_search: GridSearchCV,
) -> None:
    """
    Ispisuje završni sažetak.
    """
    print("\n" + "=" * 80)
    print("DODATNI EKSPERIMENTI JETRENIH MODELA")
    print("=" * 80)

    print(
        "\nRezultati ponovljene "
        "unakrsne validacije:"
    )

    print(
        comparison_df[
            [
                "model_name",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
                "recall_class_0_mean",
                "recall_class_1_mean",
                "f1_macro_mean",
                "roc_auc_mean",
            ]
        ].to_string(index=False)
    )

    print("\nNajbolji eksperimentalni model:")
    print(best_model_name)

    print(
        "Repeated-CV balanced accuracy: "
        f"{best_search.best_score_:.4f}"
    )

    print("\nModel je spremljen u:")
    print(BEST_EXPERIMENT_MODEL_PATH)

    print(
        "\nTestni skup nije korišten."
    )


# ---------------------------------------------------------
# GLAVNI PROGRAM
# ---------------------------------------------------------

def main() -> None:
    """
    Glavna funkcija eksperimentalne skripte.
    """
    create_output_directories()

    print(
        "Učitavanje isključivo trening skupa..."
    )

    X_train, y_train = (
        load_training_dataset()
    )

    print(
        "Pokretanje ponovljene "
        "stratificirane unakrsne validacije..."
    )

    (
        trained_searches,
        all_results_df,
        comparison_df,
    ) = run_experiments(
        X_train=X_train,
        y_train=y_train,
    )

    print(
        "Spremanje najboljeg "
        "eksperimentalnog modela..."
    )

    (
        best_model_name,
        best_search,
    ) = save_best_experimental_model(
        trained_searches=trained_searches,
        comparison_df=comparison_df,
    )

    print("Spremanje rezultata...")

    save_outputs(
        all_results_df=all_results_df,
        comparison_df=comparison_df,
    )

    print("Spremanje izvještaja...")

    save_report(
        X_train=X_train,
        y_train=y_train,
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        best_search=best_search,
    )

    save_metadata(
        X_train=X_train,
        best_model_name=best_model_name,
        best_search=best_search,
    )

    print_summary(
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        best_search=best_search,
    )

    print(
        "\nEksperimentalna analiza "
        "uspješno je završena."
    )


if __name__ == "__main__":
    main()