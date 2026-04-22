from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from healthcare_nlp.data import load_dataset, split_train_val_test
from healthcare_nlp.dataset_sources import prepare_ade_corpus_v2
from healthcare_nlp.evaluation import evaluate_binary_model
from healthcare_nlp.models import build_model_specs, tune_model
from healthcare_nlp.preprocess import preprocess_series
from healthcare_nlp.transformers_stub import get_transformer_plan


def _domain_transfer_eval(best_model, df, output_dir: Path) -> dict:
    results = {}
    domains = sorted(df["domain"].unique().tolist())
    if len(domains) < 2:
        return {"message": "Domain transfer skipped: only one domain available."}

    for train_domain in domains:
        for test_domain in domains:
            if train_domain == test_domain:
                continue

            train_df = df[df["domain"] == train_domain]
            test_df = df[df["domain"] == test_domain]
            if train_df.empty or test_df.empty:
                continue

            x_train = preprocess_series(train_df["text"])
            y_train = train_df["label"]
            x_test = preprocess_series(test_df["text"])
            y_test = test_df["label"]

            best_model.fit(x_train, y_train)
            eval_key = f"train_{train_domain}_test_{test_domain}"
            results[eval_key] = evaluate_binary_model(best_model, x_test, y_test)

    transfer_path = output_dir / "domain_transfer_results.json"
    transfer_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_baseline(data_path: str, output_dir: str, random_state: int = 42) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    train_df, val_df, test_df = split_train_val_test(df, random_state=random_state)

    x_train = preprocess_series(train_df["text"])
    y_train = train_df["label"]
    x_val = preprocess_series(val_df["text"])
    y_val = val_df["label"]
    x_test = preprocess_series(test_df["text"])
    y_test = test_df["label"]

    model_specs = build_model_specs(random_state=random_state)
    leaderboard = []
    fitted_models = {}

    for spec in model_specs:
        search = tune_model(spec, x_train, y_train)
        fitted_models[spec.name] = search.best_estimator_

        val_metrics = evaluate_binary_model(search.best_estimator_, x_val, y_val)
        test_metrics = evaluate_binary_model(search.best_estimator_, x_test, y_test)

        leaderboard.append(
            {
                "model": spec.name,
                "best_params": search.best_params_,
                "val_f1": val_metrics["f1"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_roc_auc": test_metrics.get("roc_auc"),
                "test_pr_auc": test_metrics.get("pr_auc"),
            }
        )

    leaderboard.sort(key=lambda x: x["val_f1"], reverse=True)
    best_name = leaderboard[0]["model"]
    best_model = fitted_models[best_name]

    # Retrain best model on train+val to maximize final test-time generalization.
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    best_model.fit(preprocess_series(combined_df["text"]), combined_df["label"])

    final_test_metrics = evaluate_binary_model(best_model, x_test, y_test)
    domain_transfer = _domain_transfer_eval(best_model, df, output_path)

    model_path = output_path / f"best_model_{best_name}.joblib"
    joblib.dump(best_model, model_path)

    report = {
        "data_path": data_path,
        "dataset_size": int(len(df)),
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "leaderboard": leaderboard,
        "selected_model": best_name,
        "final_test_metrics": final_test_metrics,
        "domain_transfer": domain_transfer,
        "artifacts": {
            "best_model": str(model_path),
            "leaderboard": str(output_path / "leaderboard.json"),
            "report": str(output_path / "report.json"),
        },
    }

    (output_path / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    (output_path / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Best model: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Report: {output_path / 'report.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Healthcare NLP ADR classification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_ade = subparsers.add_parser(
        "prepare-ade-dataset",
        help="Download and prepare ADE Corpus V2 classification dataset",
    )
    prepare_ade.add_argument(
        "--output",
        default="data/processed/ade_corpus_v2_classification.csv",
        help="Output CSV path for prepared ADE dataset",
    )
    prepare_ade.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row count sample for quick experiments",
    )
    prepare_ade.add_argument("--seed", type=int, default=42, help="Random seed")

    baseline = subparsers.add_parser("run-baseline", help="Run baseline classical ML pipeline")
    baseline.add_argument(
        "--data",
        default="data/processed/ade_corpus_v2_classification.csv",
        help="Path to CSV dataset",
    )
    baseline.add_argument("--output", default="outputs", help="Output directory")
    baseline.add_argument("--seed", type=int, default=42, help="Random seed")

    transformer = subparsers.add_parser("transformer-plan", help="Show transformer fine-tuning plan")
    transformer.add_argument("--model-name", default="emilyalsentzer/Bio_ClinicalBERT")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-ade-dataset":
        summary = prepare_ade_corpus_v2(
            output_csv_path=args.output,
            sample_size=args.sample_size,
            random_state=args.seed,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "run-baseline":
        run_baseline(data_path=args.data, output_dir=args.output, random_state=args.seed)
    elif args.command == "transformer-plan":
        print(get_transformer_plan())
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
