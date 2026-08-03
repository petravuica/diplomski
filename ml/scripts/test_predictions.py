import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from laboratory.ml_services.anemia_prediction import predict_anemia

test_cases = [
    {
        "name": "Uredan nalaz",
        "data": {
            "gender": "female",
            "age": 30,
            "hgb": 135,
            "rbc": 4.5,
            "hct": 41,
            "mcv": 91,
            "mch": 30,
            "mchc": 330,
        },
    },
    {
        "name": "Blaga anemija",
        "data": {
            "gender": "female",
            "age": 30,
            "hgb": 112,
            "rbc": 4.0,
            "hct": 35,
            "mcv": 87,
            "mch": 28,
            "mchc": 320,
        },
    },
    {
        "name": "Izraženija anemija",
        "data": {
            "gender": "female",
            "age": 30,
            "hgb": 90,
            "rbc": 3.4,
            "hct": 28,
            "mcv": 82,
            "mch": 26,
            "mchc": 310,
        },
    },
    {
        "name": "Mikrocitna anemija",
        "data": {
            "gender": "female",
            "age": 30,
            "hgb": 85,
            "rbc": 3.8,
            "hct": 29,
            "mcv": 72,
            "mch": 22,
            "mchc": 295,
        },
    },
    {
        "name": "Makrocitna anemija",
        "data": {
            "gender": "female",
            "age": 30,
            "hgb": 95,
            "rbc": 3.0,
            "hct": 31,
            "mcv": 104,
            "mch": 33,
            "mchc": 315,
        },
    },
]

for case in test_cases:
    result = predict_anemia(**case["data"])

    print("=" * 60)
    print(case["name"])
    print(result)