from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformerStubConfig:
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    num_labels: int = 2
    max_length: int = 256
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 16


def get_transformer_plan(config: TransformerStubConfig | None = None) -> str:
    config = config or TransformerStubConfig()
    return (
        "Transformer fine-tuning stub:\n"
        f"- Base model: {config.model_name}\n"
        f"- Labels: {config.num_labels}\n"
        f"- Max length: {config.max_length}\n"
        f"- Learning rate: {config.learning_rate}\n"
        f"- Epochs: {config.epochs}\n"
        f"- Batch size: {config.batch_size}\n"
        "\nNext steps:\n"
        "1. Install requirements including transformers and torch.\n"
        "2. Convert the dataset to Hugging Face Dataset format.\n"
        "3. Fine-tune AutoModelForSequenceClassification.\n"
        "4. Compare metrics with classical baselines.\n"
        "5. Add SHAP/LIME explainability for model outputs."
    )
