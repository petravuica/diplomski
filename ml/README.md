# Machine Learning modul – BloodLab AI

Ova mapa sadrži kod, podatke, izvještaje i spremljene modele
za razvoj modula strojnog učenja u aplikaciji BloodLab AI.

## Cilj prvog modela

Prvi model klasificira hematološki nalaz u jednu od dviju skupina:

- obrazac bez znakova anemije
- obrazac povezan s anemijom

Rezultat modela predstavlja procjenu laboratorijskog obrasca i ne
predstavlja medicinsku dijagnozu.

## Struktura

- `data/raw/` – izvorni, nepromijenjeni skup podataka
- `data/processed/` – očišćeni i pripremljeni podaci
- `scripts/` – skripte za analizu, čišćenje i treniranje
- `notebooks/` – eksperimentalne analize
- `reports/eda/` – rezultati eksplorativne analize
- `reports/evaluation/` – rezultati evaluacije modela
- `models/` – spremljeni ML modeli i pipelinei

## Dataset

Za prvi hematološki model koristi se Raw Hematological Dataset for
Anemia Analysis and Classification.

Dataset sadrži hematološke podatke 1004 anonimizirana zapisa i
ciljnu varijablu `Decision_Class`.

Izvorni dataset čuva se nepromijenjen u mapi `data/raw/`.