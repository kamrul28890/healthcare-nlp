from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed" / "ade_corpus_v2_classification.csv"
    baseline_path = root / "reports" / "ade_corpus_v2_baseline_summary.json"
    bert_path = root / "reports" / "bioclinicalbert_results_summary.json"
    out_dir = root / "reports" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    bert = json.loads(bert_path.read_text(encoding="utf-8"))

    # Basic derived columns
    df["char_len"] = df["text"].astype(str).str.len()
    df["word_len"] = df["text"].astype(str).str.split().str.len()

    # 1) Label distribution
    label_counts = df["label"].value_counts().sort_index()
    label_map = {0: "No ADR", 1: "ADR"}
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar([label_map[i] for i in label_counts.index], label_counts.values, color=["#4C72B0", "#DD8452"])
    plt.title("Class Distribution in ADE Corpus V2")
    plt.ylabel("Sentence Count")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 100, f"{int(h):,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "fig01_label_distribution.png", dpi=180)
    plt.close()

    # 2) Word length histogram by class
    plt.figure(figsize=(8, 4.8))
    for label, color in [(0, "#4C72B0"), (1, "#DD8452")]:
        subset = df[df["label"] == label]["word_len"]
        plt.hist(subset, bins=35, alpha=0.5, label=label_map[label], color=color)
    plt.title("Sentence Length Distribution by Class (Word Count)")
    plt.xlabel("Words per sentence")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig02_word_length_hist.png", dpi=180)
    plt.close()

    # 3) Boxplot of word lengths
    plt.figure(figsize=(7, 4.5))
    box_data = [df[df["label"] == 0]["word_len"], df[df["label"] == 1]["word_len"]]
    plt.boxplot(box_data, tick_labels=["No ADR", "ADR"], patch_artist=True)
    plt.title("Word Length Spread by Class")
    plt.ylabel("Words per sentence")
    plt.tight_layout()
    plt.savefig(out_dir / "fig03_word_length_boxplot.png", dpi=180)
    plt.close()

    # 4) Top unigrams by class (count-based)
    vectorizer = CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 1))
    x = vectorizer.fit_transform(df["text"].astype(str))
    vocab = np.array(vectorizer.get_feature_names_out())

    class_top = {}
    for label in [0, 1]:
        mask = (df["label"].to_numpy() == label)
        sums = np.asarray(x[mask].sum(axis=0)).ravel()
        idx = np.argsort(sums)[-20:][::-1]
        class_top[label] = pd.DataFrame({"term": vocab[idx], "count": sums[idx]})

    class_top[0].to_csv(out_dir / "table_top_terms_no_adr.csv", index=False)
    class_top[1].to_csv(out_dir / "table_top_terms_adr.csv", index=False)

    # 5) Top discriminative terms via TF-IDF difference
    tfidf = TfidfVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2))
    tx = tfidf.fit_transform(df["text"].astype(str))
    terms = np.array(tfidf.get_feature_names_out())

    adr_mask = (df["label"].to_numpy() == 1)
    no_adr_mask = (df["label"].to_numpy() == 0)
    adr_mean = np.asarray(tx[adr_mask].mean(axis=0)).ravel()
    no_adr_mean = np.asarray(tx[no_adr_mask].mean(axis=0)).ravel()
    delta = adr_mean - no_adr_mean

    top_adr_idx = np.argsort(delta)[-25:][::-1]
    top_no_adr_idx = np.argsort(delta)[:25]

    discrim = pd.DataFrame(
        {
            "top_adr_terms": terms[top_adr_idx],
            "adr_minus_no_adr": delta[top_adr_idx],
            "top_no_adr_terms": terms[top_no_adr_idx],
            "no_adr_minus_adr": -delta[top_no_adr_idx],
        }
    )
    discrim.to_csv(out_dir / "table_discriminative_tfidf_terms.csv", index=False)

    # 6) Model comparison chart
    leaderboard = pd.DataFrame(baseline["leaderboard"])
    bert_row = {
        "model": "bioclinicalbert",
        "test_f1": bert["test_metrics"]["f1"],
        "test_precision": bert["test_metrics"]["precision"],
        "test_recall": bert["test_metrics"]["recall"],
        "test_roc_auc": bert["test_metrics"]["roc_auc"],
        "test_pr_auc": bert["test_metrics"]["pr_auc"],
    }
    all_models = pd.concat([leaderboard, pd.DataFrame([bert_row])], ignore_index=True)
    all_models.to_csv(out_dir / "table_model_comparison_full.csv", index=False)

    plot_df = all_models[["model", "test_f1", "test_pr_auc", "test_roc_auc"]].copy()
    plot_df = plot_df.sort_values("test_f1", ascending=False)

    x_pos = np.arange(len(plot_df))
    width = 0.25
    plt.figure(figsize=(11, 5.5))
    plt.bar(x_pos - width, plot_df["test_f1"], width=width, label="F1")
    plt.bar(x_pos, plot_df["test_pr_auc"], width=width, label="PR-AUC")
    plt.bar(x_pos + width, plot_df["test_roc_auc"], width=width, label="ROC-AUC")
    plt.xticks(x_pos, plot_df["model"], rotation=20)
    plt.ylim(0.65, 1.0)
    plt.title("Model Comparison on ADE Classification")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig04_model_metric_comparison.png", dpi=180)
    plt.close()

    # 7) Confusion matrix heatmap for best baseline
    cm = np.array(baseline["final_test_metrics"]["confusion_matrix"])
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Best Baseline: Linear SVM)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["No ADR", "ADR"])
    plt.yticks([0, 1], ["No ADR", "ADR"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(out_dir / "fig05_confusion_matrix_baseline.png", dpi=180)
    plt.close()

    # 8) Summary statistics table
    stats = {
        "rows": int(len(df)),
        "adr_rows": int((df["label"] == 1).sum()),
        "no_adr_rows": int((df["label"] == 0).sum()),
        "adr_ratio": float((df["label"] == 1).mean()),
        "mean_words": float(df["word_len"].mean()),
        "median_words": float(df["word_len"].median()),
        "std_words": float(df["word_len"].std()),
        "q1_words": float(df["word_len"].quantile(0.25)),
        "q3_words": float(df["word_len"].quantile(0.75)),
        "max_words": int(df["word_len"].max()),
        "mean_words_adr": float(df[df["label"] == 1]["word_len"].mean()),
        "mean_words_no_adr": float(df[df["label"] == 0]["word_len"].mean()),
    }
    (out_dir / "table_dataset_summary_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Artifacts generated in", out_dir)


if __name__ == "__main__":
    main()
