# Healthcare NLP ADR Classification

This repository implements a runnable baseline for **adverse drug reaction (ADR)** text classification aligned with your research outline:

- Multi-domain support (`clinical`, `social`)
- Classical baselines (TF-IDF + Logistic Regression, SVM, Random Forest, Naive Bayes)
- 70/15/15 split and comparative evaluation
- Domain transfer evaluation (train one domain, test the other)
- Transformer fine-tuning starter plan (stub)

## 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run Baseline

```powershell
$env:PYTHONPATH = "src"
python -m healthcare_nlp.cli run-baseline --data data/sample/public_adr_sample.csv --output outputs
```

Artifacts:

- `outputs/leaderboard.json`
- `outputs/report.json`
- `outputs/best_model_<model>.joblib`
- `outputs/domain_transfer_results.json`

## 3) Transformer Plan

```powershell
$env:PYTHONPATH = "src"
python -m healthcare_nlp.cli transformer-plan
```

## 4) Dataset Format

CSV columns required:

- `text`: input text
- `label`: `1` (ADR present), `0` (no ADR)
- `domain`: `clinical` or `social`

## Notes

- This baseline uses a public sample dataset for immediate reproducibility.
- Replace `data/sample/public_adr_sample.csv` with your real dataset when available.
- SHAP/LIME dependencies are included for future explainability integration.
