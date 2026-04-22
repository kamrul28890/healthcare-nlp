# Healthcare NLP ADR Report

## 1. Problem and Dataset

This project targets adverse drug reaction sentence classification using the ADE Corpus V2 benchmark.

- Dataset: ade-benchmark-corpus/ade_corpus_v2 (Ade_corpus_v2_classification)
- Total rows: 23516
- Train/Val/Test split: 16461 / 3527 / 3528

## 2. Methods

### 2.1 Classical Baselines

Models trained with TF-IDF features:

- Logistic Regression
- Linear SVM
- Random Forest
- Naive Bayes

Best classical model: linear_svm

### 2.2 Transformer Fine-Tuning

Model: emilyalsentzer/Bio_ClinicalBERT

Fine-tuning configuration:

- Epochs: 1
- Learning rate: 2e-05
- Max length: 256
- Train batch size: 8
- Eval batch size: 16
- Train rows used: 3000
- Validation rows used: 1000
- Test rows used: 1000

## 3. Results

| Metric | Linear SVM | BioClinicalBERT |
|---|---:|---:|
| Accuracy | 0.9059 | 0.9000 |
| Precision | 0.8127 | 0.8116 |
| Recall | 0.8778 | 0.8235 |
| F1 | 0.8440 | 0.8175 |
| ROC-AUC | 0.9600 | 0.9536 |
| PR-AUC | 0.9201 | 0.8953 |

### 3.1 Best Classical (Linear SVM)

- Accuracy: 0.9059
- Precision: 0.8127
- Recall: 0.8778
- F1: 0.8440
- ROC-AUC: 0.9600
- PR-AUC: 0.9201

### 3.2 BioClinicalBERT Fine-Tuned

- Accuracy: 0.9000
- Precision: 0.8116
- Recall: 0.8235
- F1: 0.8175
- ROC-AUC: 0.9536
- PR-AUC: 0.8953

## 4. Discussion

- The classical baseline is strong and computationally efficient.
- BioClinicalBERT fine-tuning adds contextual representation learning and should be preferred when compute budget allows.
- In this run, BioClinicalBERT was trained on a sampled subset for feasibility, which likely limited final performance.
- Relative difference observed: Delta F1 = -0.0265, Delta PR-AUC = -0.0248.
- Final deployment recommendation should balance latency constraints and recall requirements in pharmacovigilance workflows.

## 5. Reproducibility Artifacts

- Baseline summary: reports/ade_corpus_v2_baseline_summary.json
- BioClinicalBERT metrics: outputs/bioclinicalbert_results.json
- Generated report: reports/final_project_report.md
