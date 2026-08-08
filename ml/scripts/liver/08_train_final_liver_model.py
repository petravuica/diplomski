from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parents[1]

TRAIN_SPLIT_PATH = (
    ML_DIR
    / "data"
    / "processed"
    / "liver"
    / "liver_train_split.csv"
)

MODELS_DIR = (
    ML_DIR
    / "models"
    / "liver"
)

REPORT_DIR = (
    ML_DIR
    / "reports"
    / "liver"
    / "final_model"
)

FINAL_MODEL_PATH = (
    MODELS_DIR
    / "final_liver_model.joblib"
)

METADATA_PATH = (
    REPORT_DIR
    / "final_liver_model_metadata.json"
)

TARGET_COLUMN = "Liver_Disease"

CATEGORICAL_FEATURES = [
    "Gender",
]

NUMERIC_FEATURES = [
    "Age",
    "Total_Bilirubin",
    "Alkaline_Phosphatase",
    "Alanine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
]

FEATURE_COLUMNS = [
    *CATEGORICAL_FEATURES,
    *NUMERIC_FEATURES,
]


def create_output_directories():
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_training_data():
    if not TRAIN_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Trening skup nije pronađen:\n{TRAIN_SPLIT_PATH}"
        )

    df = pd.read_csv(
        TRAIN_SPLIT_PATH,
        encoding="utf-8-sig",
    )

    missing_columns = [
        column
        for column in [
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Nedostaju stupci: "
            + ", ".join(missing_columns)
        )

    X_train = df[
        FEATURE_COLUMNS
    ].copy()

    y_train = (
        df[TARGET_COLUMN]
        .astype(int)
        .copy()
    )

    return X_train, y_train


def create_pipeline():
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

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                model,
            ),
        ]
    )

    return pipeline


def save_metadata(
    X_train,
):
    metadata = {
        "model_name": "Random Forest",
        "model_purpose": (
            "Informativna procjena obrasca "
            "povezanog s jetrenom bolešću"
        ),
        "feature_set": "C_practical_8_features",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "negative_class": 0,
        "training_rows": len(X_train),
        "hyperparameters": {
            "n_estimators": 400,
            "max_depth": 8,
            "min_samples_leaf": 3,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 1,
        },
        "selection_reason": (
            "Random Forest je odabran zbog uravnoteženijeg "
            "prepoznavanja obje klase i višeg recall-a pozitivne "
            "klase u odnosu na logističku regresiju, uz prihvatljiv "
            "pad balanced accuracy."
        ),
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


def main():
    create_output_directories()

    print(
        "Učitavanje trening skupa..."
    )

    X_train, y_train = (
        load_training_data()
    )

    print(
        "Izrada finalnog Random Forest pipelinea..."
    )

    model = create_pipeline()

    print(
        "Treniranje finalnog jetrenog modela..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Spremanje modela..."
    )

    joblib.dump(
        model,
        FINAL_MODEL_PATH,
    )

    save_metadata(
        X_train
    )

    print("=" * 70)
    print(
        "FINALNI JETRENI MODEL USPJEŠNO TRENIRAN"
    )
    print("=" * 70)

    print(
        "Model: Random Forest"
    )

    print(
        f"Broj trening zapisa: "
        f"{len(X_train)}"
    )

    print(
        "Značajke:"
    )

    for feature in FEATURE_COLUMNS:
        print(
            f"- {feature}"
        )

    print(
        "\nModel spremljen u:"
    )

    print(
        FINAL_MODEL_PATH
    )


if __name__ == "__main__":
    main()