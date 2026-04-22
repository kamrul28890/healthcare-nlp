from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_project_report(
    baseline_summary_path: str | Path,
    bioclinicalbert_result_path: str | Path,
    output_report_path: str | Path,
) -> str:
    baseline = _load_json(baseline_summary_path)
    bert = _load_json(bioclinicalbert_result_path)

    b = baseline["final_test_metrics"]
    t = bert["test_metrics"]
    delta_f1 = t["f1"] - b["f1"]
    delta_pr_auc = t["pr_auc"] - b["pr_auc"]

    report = f"""# Healthcare NLP ADR Report

## 1. Problem and Dataset

This project targets adverse drug reaction sentence classification using the ADE Corpus V2 benchmark.

- Dataset: {baseline['dataset']} ({baseline['config']})
- Total rows: {baseline['dataset_size']}
- Train/Val/Test split: {baseline['split_sizes']['train']} / {baseline['split_sizes']['val']} / {baseline['split_sizes']['test']}

## 2. Methods

### 2.1 Classical Baselines

Models trained with TF-IDF features:

- Logistic Regression
- Linear SVM
- Random Forest
- Naive Bayes

Best classical model: {baseline['selected_model']}

### 2.2 Transformer Fine-Tuning

Model: {bert['model_name']}

Fine-tuning configuration:

- Epochs: {bert['epochs']}
- Learning rate: {bert['learning_rate']}
- Max length: {bert['max_length']}
- Train batch size: {bert['train_batch_size']}
- Eval batch size: {bert['eval_batch_size']}
- Train rows used: {bert['train_rows']}
- Validation rows used: {bert['val_rows']}
- Test rows used: {bert['test_rows']}

## 3. Results

| Metric | Linear SVM | BioClinicalBERT |
|---|---:|---:|
| Accuracy | {b['accuracy']:.4f} | {t['accuracy']:.4f} |
| Precision | {b['precision']:.4f} | {t['precision']:.4f} |
| Recall | {b['recall']:.4f} | {t['recall']:.4f} |
| F1 | {b['f1']:.4f} | {t['f1']:.4f} |
| ROC-AUC | {b['roc_auc']:.4f} | {t['roc_auc']:.4f} |
| PR-AUC | {b['pr_auc']:.4f} | {t['pr_auc']:.4f} |

### 3.1 Best Classical (Linear SVM)

- Accuracy: {b['accuracy']:.4f}
- Precision: {b['precision']:.4f}
- Recall: {b['recall']:.4f}
- F1: {b['f1']:.4f}
- ROC-AUC: {b['roc_auc']:.4f}
- PR-AUC: {b['pr_auc']:.4f}

### 3.2 BioClinicalBERT Fine-Tuned

- Accuracy: {t['accuracy']:.4f}
- Precision: {t['precision']:.4f}
- Recall: {t['recall']:.4f}
- F1: {t['f1']:.4f}
- ROC-AUC: {t['roc_auc']:.4f}
- PR-AUC: {t['pr_auc']:.4f}

## 4. Discussion

- The classical baseline is strong and computationally efficient.
- BioClinicalBERT fine-tuning adds contextual representation learning and should be preferred when compute budget allows.
- In this run, BioClinicalBERT was trained on a sampled subset for feasibility, which likely limited final performance.
- Relative difference observed: Delta F1 = {delta_f1:.4f}, Delta PR-AUC = {delta_pr_auc:.4f}.
- Final deployment recommendation should balance latency constraints and recall requirements in pharmacovigilance workflows.

## 5. Reproducibility Artifacts

- Baseline summary: {baseline_summary_path}
- BioClinicalBERT metrics: {bioclinicalbert_result_path}
- Generated report: {output_report_path}
"""

    output_path = Path(output_report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)
