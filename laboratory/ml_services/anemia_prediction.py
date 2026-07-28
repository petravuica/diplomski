from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings


MODEL_PATH = (
    Path(settings.BASE_DIR)
    / "ml"
    / "models"
    / "best_anemia_model.joblib"
)


@lru_cache(maxsize=1)
def load_anemia_model():
    """
    Učitava i privremeno sprema model za klasifikaciju anemije.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model nije pronađen na lokaciji: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def normalize_gender(gender):
    """
    Pretvara vrijednost spola iz Django aplikacije
    u oznaku korištenu prilikom treniranja modela.
    """

    normalized_gender = str(gender).strip().lower()

    gender_mapping = {
        "female": "F",
        "f": "F",
        "ženski": "F",
        "zenski": "F",
        "male": "M",
        "m": "M",
        "muški": "M",
        "muski": "M",
    }

    if normalized_gender not in gender_mapping:
        raise ValueError(
            "Model anemije trenutačno podržava samo ženski ili muški spol."
        )

    return gender_mapping[normalized_gender]


def predict_anemia(
    gender,
    age,
    hgb,
    rbc,
    hct,
    mcv,
    mch,
    mchc,
):
    """
    Izrađuje predikciju anemije na temelju laboratorijskih
    i demografskih parametara.

    Očekivane jedinice iz aplikacije:
    HGB  = g/L
    RBC  = 10^12/L
    HCT  = %
    MCV  = fL
    MCH  = pg
    MCHC = g/L

    Model koristi:
    HGB i MCHC u g/dL.

    Decision_Class:
    0 = nema anemije
    1 = anemija
    """

    model = load_anemia_model()

    normalized_gender = normalize_gender(gender)

    # Pretvaranje iz g/L u g/dL.
    normalized_hgb = float(hgb) / 10
    normalized_mchc = float(mchc) / 10

    input_data = pd.DataFrame(
        [
            {
                "Gender": normalized_gender,
                "Age": float(age),
                "HGB": normalized_hgb,
                "RBC": float(rbc),
                "HCT": float(hct),
                "MCV": float(mcv),
                "MCH": float(mch),
                "MCHC": normalized_mchc,
            }
        ]
    )

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]
    model_classes = list(model.classes_)

    anemia_index = model_classes.index(1)
    no_anemia_index = model_classes.index(0)

    anemia_probability = float(probabilities[anemia_index])
    no_anemia_probability = float(probabilities[no_anemia_index])

    if prediction == 1:
        label = "Prepoznati pokazatelji anemije"
        predicted_probability = anemia_probability
    else:
        label = "Nisu prepoznati pokazatelji anemije"
        predicted_probability = no_anemia_probability

    return {
        "prediction": prediction,
        "label": label,
        "probability": round(predicted_probability, 4),
        "probability_percent": round(
            predicted_probability * 100,
            2,
        ),
        "anemia_probability": round(anemia_probability, 4),
        "anemia_probability_percent": round(
            anemia_probability * 100,
            2,
        ),
    }