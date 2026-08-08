from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "liver"
    / "final_liver_model.joblib"
)


FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Total_Bilirubin",
    "Alkaline_Phosphatase",
    "Alanine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
]


_model = None


def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Jetreni model nije pronađen: {MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def normalize_gender(gender):
    if gender is None:
        raise ValueError("Spol je obvezan za jetrenu ML analizu.")

    value = str(gender).strip().lower()

    mapping = {
        "male": "Male",
        "m": "Male",
        "muško": "Male",
        "musko": "Male",

        "female": "Female",
        "f": "Female",
        "žensko": "Female",
        "zensko": "Female",
    }

    if value not in mapping:
        raise ValueError(
            f"Neprepoznata vrijednost spola: {gender}"
        )

    return mapping[value]


def to_float(value, name):
    if value is None:
        raise ValueError(
            f"Nedostaje vrijednost parametra: {name}"
        )

    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Neispravna vrijednost parametra {name}: {value}"
        )

def bilirubin_umol_l_to_mg_dl(value):
    value = to_float(value, "Total_Bilirubin")
    return value / 17.104


def protein_g_l_to_g_dl(value, name):
    value = to_float(value, name)
    return value / 10

def predict_liver(
    gender,
    age,
    total_bilirubin,
    alkaline_phosphatase,
    alt,
    ast,
    total_proteins,
    albumin,
):
    """
    Pokreće finalni jetreni ML model.

    Klasa 1:
        obrazac povezan s jetrenom bolešću

    Klasa 0:
        obrazac bez oznake jetrene bolesti

    Funkcija ne postavlja dijagnozu.
    """

    model = get_model()

    input_data = {
        "Gender": normalize_gender(gender),
        "Age": to_float(age, "Age"),
        "Total_Bilirubin": bilirubin_umol_l_to_mg_dl(
         total_bilirubin
        ),
        "Alkaline_Phosphatase": to_float(
            alkaline_phosphatase,
            "Alkaline_Phosphatase",
        ),
        "Alanine_Aminotransferase": to_float(
            alt,
            "Alanine_Aminotransferase",
        ),
        "Aspartate_Aminotransferase": to_float(
            ast,
            "Aspartate_Aminotransferase",
        ),
        "Total_Proteins": protein_g_l_to_g_dl(
            total_proteins,
            "Total_Proteins",
        ),
        "Albumin": protein_g_l_to_g_dl(
            albumin,
            "Albumin",
        ),
    }

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS,
    )

    prediction = int(
        model.predict(input_df)[0]
    )

    probability_array = (
        model.predict_proba(input_df)[0]
    )

    classes = list(model.classes_)

    positive_index = classes.index(1)

    liver_probability = float(
        probability_array[positive_index]
    )

    predicted_class_index = (
        classes.index(prediction)
    )

    prediction_probability = float(
        probability_array[
            predicted_class_index
        ]
    )

    if prediction == 1:
        label = (
            "Prepoznat obrazac povezan "
            "s odstupanjem jetrenih parametara"
        )
    else:
        label = (
            "Nije prepoznat izražen obrazac "
            "povezan s odstupanjem jetrenih parametara"
        )

    return {
        "prediction": prediction,
        "label": label,
        "probability": round(
            prediction_probability,
            4,
        ),
        "probability_percent": round(
            prediction_probability * 100,
            2,
        ),
        "liver_probability": round(
            liver_probability,
            4,
        ),
        "liver_probability_percent": round(
            liver_probability * 100,
            2,
        ),
    }