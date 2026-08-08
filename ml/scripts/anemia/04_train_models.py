from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer, precision_score

# ---------------------------------------------------------
# PUTANJE
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parents[1]

CLEAN_DATASET_PATH = (
    ML_DIR
    / "data"
    / "processed"
    / "anemia_dataset_clean.csv"
)

PROCESSED_DATA_DIR = (
    ML_DIR
    / "data"
    / "processed"
)

MODELS_DIR = ML_DIR / "models"

TRAINING_REPORT_DIR = (
    ML_DIR
    / "reports"
    / "training"
)

TRAIN_SPLIT_PATH = (
    PROCESSED_DATA_DIR
    / "anemia_train_split.csv"
)

TEST_SPLIT_PATH = (
    PROCESSED_DATA_DIR
    / "anemia_test_split.csv"
)

CV_RESULTS_PATH = (
    TRAINING_REPORT_DIR
    / "cross_validation_results.csv"
)

MODEL_COMPARISON_PATH = (
    TRAINING_REPORT_DIR
    / "model_comparison.csv"
)

TRAINING_REPORT_PATH = (
    TRAINING_REPORT_DIR
    / "training_report.txt"
)

TRAINING_METADATA_PATH = (
    TRAINING_REPORT_DIR
    / "training_metadata.json"
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


# ---------------------------------------------------------
# KONFIGURACIJA
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_COLUMN = "Decision_Class"

CATEGORICAL_FEATURES = [
    "Gender",
]

NUMERIC_FEATURES = [
    "Age",
    "HGB",
    "RBC",
    "HCT",
    "MCV",
    "MCH",
    "MCHC",
]

FEATURE_COLUMNS = [
    *CATEGORICAL_FEATURES,
    *NUMERIC_FEATURES,
]

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score,zero_division=0,),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


# ---------------------------------------------------------
# PRIPREMA MAPA I PODATAKA
# ---------------------------------------------------------

def create_output_directories() -> None:
    """
    Stvara potrebne izlazne mape.
    """
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRAINING_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_clean_dataset() -> pd.DataFrame:
    """
    Učitava očišćeni dataset i provjerava njegovu strukturu.
    """
    if not CLEAN_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Očišćeni dataset nije pronađen na putanji:\n"
            f"{CLEAN_DATASET_PATH}\n\n"
            "Prvo pokreni skriptu 03_clean_dataset.py."
        )

    df = pd.read_csv(
        CLEAN_DATASET_PATH,
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
            "U očišćenom datasetu nedostaju stupci: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError(
            "Očišćeni dataset ne sadrži zapise."
        )

    if df[expected_columns].isna().any().any():
        raise ValueError(
            "Očišćeni dataset sadrži "
            "nedostajuće vrijednosti."
        )

    if not set(
        df[TARGET_COLUMN].unique()
    ).issubset({0, 1}):
        raise ValueError(
            "Ciljna varijabla mora sadržavati "
            "isključivo klase 0 i 1."
        )

    return df[expected_columns].copy()


def create_train_test_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Izrađuje stratificirani train/test split.

    Stratifikacija održava približno jednaku raspodjelu
    ciljnih klasa u trening i testnom skupu.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def save_train_test_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Sprema točno korištene trening i testne skupove radi
    reproducibilnosti završne evaluacije.
    """
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_df.to_csv(
        TRAIN_SPLIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    test_df.to_csv(
        TEST_SPLIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )


# ---------------------------------------------------------
# PRETPROCESIRANJE
# ---------------------------------------------------------

def create_preprocessor() -> ColumnTransformer:
    """
    Stvara pretprocesor za numeričke i kategorijske značajke.

    Numeričke značajke:
    - sigurnosna imputacija medijanom
    - standardizacija

    Kategorijske značajke:
    - sigurnosna imputacija najčešćom vrijednošću
    - one-hot kodiranje
    """
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
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
                    strategy="most_frequent"
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

    preprocessor = ColumnTransformer(
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

    return preprocessor


# ---------------------------------------------------------
# MODELI I HIPERPARAMETRI
# ---------------------------------------------------------

def create_model_configurations() -> dict:
    """
    Definira modele i hiperparametre koji će se ispitati
    unakrsnom validacijom.
    """
    dummy_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                DummyClassifier(),
            ),
        ]
    )

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
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

    configurations = {
        "Dummy Classifier": {
            "pipeline": dummy_pipeline,
            "parameter_grid": {
                "classifier__strategy": [
                    "most_frequent",
                    "prior",
                ],
            },
            "model_path": DUMMY_MODEL_PATH,
        },
        "Logistic Regression": {
            "pipeline": logistic_pipeline,
            "parameter_grid": {
                "classifier__C": [
                    0.1,
                    1.0,
                    10.0,
                ],
                "classifier__class_weight": [
                    None,
                    "balanced",
                ],
            },
            "model_path": LOGISTIC_MODEL_PATH,
        },
        "Random Forest": {
            "pipeline": random_forest_pipeline,
            "parameter_grid": {
                "classifier__n_estimators": [
                    200,
                    400,
                ],
                "classifier__max_depth": [
                    None,
                    10,
                    20,
                ],
                "classifier__min_samples_leaf": [
                    1,
                    3,
                ],
                "classifier__class_weight": [
                    None,
                    "balanced",
                ],
            },
            "model_path": RANDOM_FOREST_MODEL_PATH,
        },
    }

    return configurations


# ---------------------------------------------------------
# TRENIRANJE I UNAKRSNA VALIDACIJA
# ---------------------------------------------------------

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    dict[str, GridSearchCV],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Trenira sve modele pomoću GridSearchCV postupka.

    Završni testni skup ovdje se ne koristi.
    """
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    configurations = (
        create_model_configurations()
    )

    trained_searches: dict[
        str,
        GridSearchCV,
    ] = {}

    all_cv_results = []
    comparison_rows = []

    for model_name, configuration in (
        configurations.items()
    ):
        print("-" * 70)
        print(f"Treniranje modela: {model_name}")

        grid_search = GridSearchCV(
            estimator=configuration["pipeline"],
            param_grid=configuration[
                "parameter_grid"
            ],
            scoring=SCORING,
            refit="balanced_accuracy",
            cv=cross_validation,
            n_jobs=-1,
            return_train_score=True,
            error_score="raise",
        )

        grid_search.fit(
            X_train,
            y_train,
        )

        trained_searches[model_name] = (
            grid_search
        )

        joblib.dump(
            grid_search.best_estimator_,
            configuration["model_path"],
        )

        model_cv_results = pd.DataFrame(
            grid_search.cv_results_
        )

        model_cv_results.insert(
            0,
            "model_name",
            model_name,
        )

        all_cv_results.append(
            model_cv_results
        )

        best_index = grid_search.best_index_

        comparison_rows.append(
            {
                "model_name": model_name,
                "best_balanced_accuracy_cv": (
                    grid_search.best_score_
                ),
                "accuracy_cv": (
                    model_cv_results.loc[
                        best_index,
                        "mean_test_accuracy",
                    ]
                ),
                "precision_cv": (
                    model_cv_results.loc[
                        best_index,
                        "mean_test_precision",
                    ]
                ),
                "recall_cv": (
                    model_cv_results.loc[
                        best_index,
                        "mean_test_recall",
                    ]
                ),
                "f1_cv": (
                    model_cv_results.loc[
                        best_index,
                        "mean_test_f1",
                    ]
                ),
                "roc_auc_cv": (
                    model_cv_results.loc[
                        best_index,
                        "mean_test_roc_auc",
                    ]
                ),
                "balanced_accuracy_std": (
                    model_cv_results.loc[
                        best_index,
                        "std_test_balanced_accuracy",
                    ]
                ),
                "best_parameters": json.dumps(
                    grid_search.best_params_,
                    ensure_ascii=False,
                ),
            }
        )

        print(
            "Najbolji CV balanced accuracy: "
            f"{grid_search.best_score_:.4f}"
        )

        print(
            "Najbolji hiperparametri:"
        )

        print(
            grid_search.best_params_
        )

    combined_cv_results = pd.concat(
        all_cv_results,
        ignore_index=True,
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    ).sort_values(
        by="best_balanced_accuracy_cv",
        ascending=False,
    )

    comparison_df = comparison_df.reset_index(
        drop=True
    )

    return (
        trained_searches,
        combined_cv_results,
        comparison_df,
    )


def select_and_save_best_model(
    trained_searches: dict[str, GridSearchCV],
    comparison_df: pd.DataFrame,
) -> tuple[str, GridSearchCV]:
    """
    Odabire najbolji model prema prosječnoj balanced accuracy
    vrijednosti iz unakrsne validacije.
    """
    best_model_name = comparison_df.loc[
        0,
        "model_name",
    ]

    best_search = trained_searches[
        best_model_name
    ]

    joblib.dump(
        best_search.best_estimator_,
        BEST_MODEL_PATH,
    )

    return (
        best_model_name,
        best_search,
    )


# ---------------------------------------------------------
# SPREMANJE IZVJEŠTAJA
# ---------------------------------------------------------

def get_class_distribution(
    y: pd.Series,
) -> pd.DataFrame:
    """
    Vraća broj i postotak zapisa po klasi.
    """
    distribution = (
        y.value_counts()
        .sort_index()
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
    )

    distribution["percentage"] = (
        distribution["count"]
        / len(y)
        * 100
    ).round(2)

    return distribution


def save_training_outputs(
    combined_cv_results: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    """
    Sprema rezultate unakrsne validacije i usporedbu modela.
    """
    combined_cv_results.to_csv(
        CV_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    comparison_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def save_training_report(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_search: GridSearchCV,
) -> None:
    """
    Sprema tekstualni izvještaj o treniranju.
    """
    full_distribution = (
        get_class_distribution(
            df[TARGET_COLUMN]
        )
    )

    train_distribution = (
        get_class_distribution(y_train)
    )

    test_distribution = (
        get_class_distribution(y_test)
    )

    with TRAINING_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "IZVJEŠTAJ O TRENIRANJU MODELA – "
            "ANEMIA DATASET\n"
        )
        report.write("=" * 75 + "\n\n")

        report.write(
            "1. OSNOVNE INFORMACIJE\n"
        )
        report.write("-" * 75 + "\n")
        report.write(
            f"Ukupan broj zapisa: {len(df)}\n"
        )
        report.write(
            f"Broj ulaznih značajki: "
            f"{len(FEATURE_COLUMNS)}\n"
        )
        report.write(
            f"Trening zapisi: {len(X_train)}\n"
        )
        report.write(
            f"Testni zapisi: {len(X_test)}\n"
        )
        report.write(
            f"Omjer testnog skupa: "
            f"{TEST_SIZE:.0%}\n"
        )
        report.write(
            f"Random state: {RANDOM_STATE}\n"
        )
        report.write(
            f"Broj CV podjela: {CV_FOLDS}\n"
        )
        report.write(
            "Metrika za odabir modela: "
            "balanced accuracy\n\n"
        )

        report.write(
            "2. ULAZNE ZNAČAJKE\n"
        )
        report.write("-" * 75 + "\n")

        for feature in FEATURE_COLUMNS:
            report.write(
                f"- {feature}\n"
            )

        report.write("\n")

        report.write(
            "3. RASPODJELA KLASA U CIJELOM DATASETU\n"
        )
        report.write("-" * 75 + "\n")
        report.write(
            full_distribution.to_string(
                index=False
            )
        )
        report.write("\n\n")

        report.write(
            "4. RASPODJELA KLASA U TRENING SKUPU\n"
        )
        report.write("-" * 75 + "\n")
        report.write(
            train_distribution.to_string(
                index=False
            )
        )
        report.write("\n\n")

        report.write(
            "5. RASPODJELA KLASA U TESTNOM SKUPU\n"
        )
        report.write("-" * 75 + "\n")
        report.write(
            test_distribution.to_string(
                index=False
            )
        )
        report.write("\n\n")

        report.write(
            "6. USPOREDBA MODELA NA TRENING SKUPU\n"
        )
        report.write("-" * 75 + "\n")

        report_columns = [
            "model_name",
            "best_balanced_accuracy_cv",
            "accuracy_cv",
            "precision_cv",
            "recall_cv",
            "f1_cv",
            "roc_auc_cv",
            "balanced_accuracy_std",
            "best_parameters",
        ]

        report.write(
            comparison_df[
                report_columns
            ].to_string(index=False)
        )
        report.write("\n\n")

        report.write(
            "7. ODABRANI MODEL\n"
        )
        report.write("-" * 75 + "\n")
        report.write(
            f"Naziv modela: {best_model_name}\n"
        )
        report.write(
            "Najbolji CV balanced accuracy: "
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
            "8. METODOLOŠKA NAPOMENA\n"
        )
        report.write("-" * 75 + "\n")
        report.write(
            "Podaci su podijeljeni stratificirano na trening "
            "i testni skup. Optimizacija hiperparametara i "
            "usporedba modela provedene su isključivo na "
            "trening skupu pomoću peterostruke stratificirane "
            "unakrsne validacije. Testni skup nije korišten "
            "tijekom odabira modela i ostavljen je za zasebnu "
            "završnu evaluaciju.\n"
        )


def save_training_metadata(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    best_model_name: str,
    best_search: GridSearchCV,
) -> None:
    """
    Sprema ključne informacije o treningu u JSON formatu.
    """
    metadata = {
        "created_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "dataset_path": str(
            CLEAN_DATASET_PATH
        ),
        "total_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cross_validation_folds": CV_FOLDS,
        "selection_metric": (
            "balanced_accuracy"
        ),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "best_model_name": best_model_name,
        "best_cv_balanced_accuracy": float(
            best_search.best_score_
        ),
        "best_parameters": (
            best_search.best_params_
        ),
        "test_set_evaluated": False,
    }

    with TRAINING_METADATA_PATH.open(
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
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_search: GridSearchCV,
) -> None:
    """
    Ispisuje sažetak treniranja u terminal.
    """
    print("\n" + "=" * 70)
    print("SAŽETAK TRENIRANJA MODELA")
    print("=" * 70)

    print(
        f"Ukupan broj zapisa: {len(df)}"
    )

    print(
        f"Trening skup: {len(X_train)}"
    )

    print(
        f"Testni skup: {len(X_test)}"
    )

    print("\nRaspodjela klasa u trening skupu:")
    print(
        y_train.value_counts().sort_index()
    )

    print("\nRaspodjela klasa u testnom skupu:")
    print(
        y_test.value_counts().sort_index()
    )

    print("\nUsporedba modela:")
    print(
        comparison_df[
            [
                "model_name",
                "best_balanced_accuracy_cv",
                "f1_cv",
                "roc_auc_cv",
            ]
        ].to_string(index=False)
    )

    print("\nOdabrani model:")
    print(best_model_name)

    print(
        "Najbolji CV balanced accuracy: "
        f"{best_search.best_score_:.4f}"
    )

    print("\nNajbolji hiperparametri:")
    print(
        best_search.best_params_
    )

    print("\nNajbolji model spremljen je u:")
    print(BEST_MODEL_PATH)

    print("\nTestni skup još nije evaluiran.")
    print(
        "Bit će korišten u skripti "
        "05_model_evaluation.py."
    )


# ---------------------------------------------------------
# GLAVNI PROGRAM
# ---------------------------------------------------------

def main() -> None:
    """
    Glavna funkcija skripte.
    """
    create_output_directories()

    print("Učitavanje očišćenog dataseta...")
    df = load_clean_dataset()

    print("Izrada stratificiranog train/test splita...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = create_train_test_split(df)

    print("Spremanje train/test splita...")
    save_train_test_split(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print("Treniranje i optimizacija modela...")
    (
        trained_searches,
        combined_cv_results,
        comparison_df,
    ) = train_models(
        X_train=X_train,
        y_train=y_train,
    )

    print("Odabir najboljeg modela...")
    (
        best_model_name,
        best_search,
    ) = select_and_save_best_model(
        trained_searches=trained_searches,
        comparison_df=comparison_df,
    )

    print("Spremanje rezultata treninga...")
    save_training_outputs(
        combined_cv_results=combined_cv_results,
        comparison_df=comparison_df,
    )

    print("Spremanje izvještaja o treningu...")
    save_training_report(
        df=df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        best_search=best_search,
    )

    print("Spremanje metapodataka...")
    save_training_metadata(
        df=df,
        X_train=X_train,
        X_test=X_test,
        best_model_name=best_model_name,
        best_search=best_search,
    )

    print_summary(
        df=df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        best_search=best_search,
    )

    print("\nTreniranje modela uspješno je završeno.")


if __name__ == "__main__":
    main()