from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from healthcare_nlp.data import load_dataset, split_train_val_test


def _compute_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, probs[:, 1])),
        "pr_auc": float(average_precision_score(labels, probs[:, 1])),
    }


def _hf_compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    return _compute_metrics_from_logits(np.asarray(logits), np.asarray(labels))


def _df_to_hf_dataset(df) -> Dataset:
    slim = df[["text", "label"]].copy()
    return Dataset.from_pandas(slim, preserve_index=False)


def finetune_bioclinicalbert(
    data_path: str | Path,
    output_dir: str | Path,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    epochs: int = 1,
    learning_rate: float = 2e-5,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    max_length: int = 256,
    train_sample_size: int | None = 5000,
    eval_sample_size: int | None = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    train_df, val_df, test_df = split_train_val_test(df, random_state=seed)

    if train_sample_size is not None:
        train_df = train_df.sample(n=min(train_sample_size, len(train_df)), random_state=seed)
    if eval_sample_size is not None:
        val_df = val_df.sample(n=min(eval_sample_size, len(val_df)), random_state=seed)
        test_df = test_df.sample(n=min(eval_sample_size, len(test_df)), random_state=seed)

    train_ds = _df_to_hf_dataset(train_df)
    val_ds = _df_to_hf_dataset(val_df)
    test_ds = _df_to_hf_dataset(test_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_tok = train_ds.map(tokenize_fn, batched=True)
    val_tok = val_ds.map(tokenize_fn, batched=True)
    test_tok = test_ds.map(tokenize_fn, batched=True)

    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("label", "labels")
    test_tok = test_tok.rename_column("label", "labels")

    keep_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_tok.column_names:
        keep_cols.append("token_type_ids")

    train_tok.set_format(type="torch", columns=keep_cols)
    val_tok.set_format(type="torch", columns=keep_cols)
    test_tok.set_format(type="torch", columns=keep_cols)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_path / "bioclinicalbert_checkpoints"),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=50,
            report_to="none",
            seed=seed,
            fp16=False,
            dataloader_num_workers=0,
        ),
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=_hf_compute_metrics,
    )

    trainer.train()
    validation_metrics = trainer.evaluate(eval_dataset=val_tok)
    test_predictions = trainer.predict(test_tok)
    test_metrics = _compute_metrics_from_logits(
        logits=np.asarray(test_predictions.predictions),
        labels=np.asarray(test_predictions.label_ids),
    )

    model_path = output_path / "bioclinicalbert_model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    summary = {
        "model_name": model_name,
        "data_path": str(data_path),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_sample_size": train_sample_size,
        "eval_sample_size": eval_sample_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "validation_metrics": {
            "accuracy": float(validation_metrics.get("eval_accuracy", 0.0)),
            "precision": float(validation_metrics.get("eval_precision", 0.0)),
            "recall": float(validation_metrics.get("eval_recall", 0.0)),
            "f1": float(validation_metrics.get("eval_f1", 0.0)),
            "roc_auc": float(validation_metrics.get("eval_roc_auc", 0.0)),
            "pr_auc": float(validation_metrics.get("eval_pr_auc", 0.0)),
        },
        "test_metrics": test_metrics,
        "artifacts": {
            "model_dir": str(model_path),
            "result_json": str(output_path / "bioclinicalbert_results.json"),
        },
    }

    (output_path / "bioclinicalbert_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
