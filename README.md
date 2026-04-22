# Healthcare NLP ADR Classification

This repository implements a runnable baseline for **adverse drug reaction (ADR)** text classification aligned with your research outline:

- Multi-domain support (`clinical`, `social`)
- Classical baselines (TF-IDF + Logistic Regression, SVM, Random Forest, Naive Bayes)
- 70/15/15 split and comparative evaluation
- Domain transfer evaluation (train one domain, test the other)
- Transformer fine-tuning starter plan (stub)
- Real benchmark dataset support: ADE Corpus V2 (Hugging Face)

## 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Prepare Real Dataset (ADE Corpus V2)

```powershell
$env:PYTHONPATH = "src"
python -m healthcare_nlp.cli prepare-ade-dataset --output data/processed/ade_corpus_v2_classification.csv
```

## 3) Run Baseline

```powershell
$env:PYTHONPATH = "src"
python -m healthcare_nlp.cli run-baseline --data data/processed/ade_corpus_v2_classification.csv --output outputs
```

Artifacts:

- `outputs/leaderboard.json`
- `outputs/report.json`
- `outputs/best_model_<model>.joblib`
- `outputs/domain_transfer_results.json`

## 4) Transformer Plan

```powershell
$env:PYTHONPATH = "src"
python -m healthcare_nlp.cli transformer-plan
```

## 5) Dataset Format

CSV columns required:

- `text`: input text
- `label`: `1` (ADR present), `0` (no ADR)
- `domain`: `clinical` or `social`

## Notes

- The default workflow now uses ADE Corpus V2, a widely used ADR benchmark.
- A small sample dataset is still included at `data/sample/public_adr_sample.csv` for smoke tests.
- SHAP/LIME dependencies are included for future explainability integration.
