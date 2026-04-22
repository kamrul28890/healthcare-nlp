# Healthcare NLP ADR Classification

This repository implements a runnable baseline for **adverse drug reaction (ADR)** text classification aligned with your research outline:

- Multi-domain support (`clinical`, `social`)
- Classical baselines (TF-IDF + Logistic Regression, SVM, Random Forest, Naive Bayes)
- 70/15/15 split and comparative evaluation
- Domain transfer evaluation (train one domain, test the other)
- BioClinicalBERT fine-tuning pipeline
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

## 4) Fine-Tune BioClinicalBERT

```powershell
$env:PYTHONPATH = "src"
python -m healthcare_nlp.cli run-bioclinicalbert --data data/processed/ade_corpus_v2_classification.csv --output outputs --epochs 1 --train-sample-size 5000 --eval-sample-size 2000
```

## 5) Write Final Report

```powershell
$env:PYTHONPATH = "src"
python -m healthcare_nlp.cli write-report --baseline-summary reports/ade_corpus_v2_baseline_summary.json --bioclinicalbert-results outputs/bioclinicalbert_results.json --output reports/final_project_report.md
```

## 6) Transformer Plan

```powershell
$env:PYTHONPATH = "src"
python -m healthcare_nlp.cli transformer-plan
```

## 7) Dataset Format

CSV columns required:

- `text`: input text
- `label`: `1` (ADR present), `0` (no ADR)
- `domain`: `clinical` or `social`

## Notes

- The default workflow now uses ADE Corpus V2, a widely used ADR benchmark.
- A small sample dataset is still included at `data/sample/public_adr_sample.csv` for smoke tests.
- SHAP/LIME dependencies are included for future explainability integration.
