"""
Multi-Class ROC Curve and AUC Analysis Engine
=============================================
Computes One-vs-Rest (OvR) ROC curves, per-class AUC, and micro/macro-average
AUC for traditional ML baselines and fine-tuned Transformer models on AG News.
Exports publication-quality figures to results/figures/ and metrics to results/.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
import utils


class TextDataset(Dataset):
    """Simple PyTorch dataset for tokenizing batches of text."""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def collate_fn_factory(tokenizer, max_length: int = 128):
    def collate_fn(batch_texts):
        return tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    return collate_fn


def get_transformer_probabilities(model_key: str, dataset_name: str, model_dir: Path, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Run batch inference for a transformer model or load cached probabilities."""
    cache_path = config.RESULTS_DIR / f"probs_{model_key}_{dataset_name}.npy"
    if cache_path.exists():
        print(f"  [CACHE] Loaded cached probabilities for {model_key} from {cache_path.name}")
        return np.load(cache_path)

    print(f"  Loading transformer from {model_dir.name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(config.DEVICE)
    model.eval()

    dataset = TextDataset(texts, tokenizer, max_length=config.MAX_SEQ_LENGTH)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_factory(tokenizer, config.MAX_SEQ_LENGTH),
        num_workers=config.NUM_WORKERS
    )

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stacked = np.vstack(all_probs)
    np.save(cache_path, stacked)
    print(f"  [CACHE] Cached probabilities to {cache_path.name}")
    return stacked


def compute_ovr_roc(y_true_bin: np.ndarray, y_score: np.ndarray, label_names: List[str]) -> Dict:
    """
    Compute One-vs-Rest ROC curve and AUC for each class, plus micro and macro averages.
    """
    n_classes = len(label_names)
    fpr = {}
    tpr = {}
    roc_auc = {}

    # Per-class ROC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[label_names[i]] = float(auc(fpr[i], tpr[i]))

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = float(auc(fpr["micro"], tpr["micro"]))

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = float(auc(fpr["macro"], tpr["macro"]))

    curves = {
        "per_class": {
            label_names[i]: {"fpr": fpr[i].tolist(), "tpr": tpr[i].tolist(), "auc": roc_auc[label_names[i]]}
            for i in range(n_classes)
        },
        "micro": {"fpr": fpr["micro"].tolist(), "tpr": tpr["micro"].tolist(), "auc": roc_auc["micro"]},
        "macro": {"fpr": fpr["macro"].tolist(), "tpr": tpr["macro"].tolist(), "auc": roc_auc["macro"]},
    }

    return {"curves": curves, "auc_scores": roc_auc}


def plot_combined_roc(model_curves: Dict[str, Dict], save_path: Path):
    """
    Plot Macro-average ROC curves of all models together on one plot.
    """
    utils.setup_plot_style()
    plt.figure(figsize=(9, 7))

    palette = sns.color_palette("tab10", len(model_curves))
    for idx, (model_name, data) in enumerate(model_curves.items()):
        macro = data["curves"]["macro"]
        plt.plot(
            macro["fpr"],
            macro["tpr"],
            label=f"{model_name} (Macro AUC = {macro['auc']:.4f})",
            color=palette[idx],
            linewidth=2.2,
        )

    # Reference line for random guessing
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random Guess (AUC = 0.5000)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Multi-Class ROC Curves Comparison (Macro-Average, One-vs-Rest)\nAG News Benchmark", fontsize=13, pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=config.FIGURE_DPI)
    plt.close()
    print(f"  [SAVED] Combined ROC curve: {save_path.name}")


def plot_detailed_grid_roc(model_curves: Dict[str, Dict], label_names: List[str], save_path: Path):
    """
    Create a multi-panel subplot grid showing per-class ROC curves for each model.
    """
    utils.setup_plot_style()
    n_models = len(model_curves)
    cols = 3 if n_models >= 3 else n_models
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows), sharex=True, sharey=True)
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    class_colors = sns.color_palette("Set1", len(label_names))

    for idx, (model_name, data) in enumerate(model_curves.items()):
        ax = axes[idx]
        curves = data["curves"]

        # Plot per-class curves
        for c_idx, c_name in enumerate(label_names):
            class_data = curves["per_class"][c_name]
            ax.plot(
                class_data["fpr"],
                class_data["tpr"],
                label=f"{c_name} (AUC={class_data['auc']:.3f})",
                color=class_colors[c_idx],
                lw=1.5,
            )

        # Plot micro & macro
        ax.plot(
            curves["macro"]["fpr"],
            curves["macro"]["tpr"],
            label=f"Macro-avg (AUC={curves['macro']['auc']:.3f})",
            color="black",
            linestyle=":",
            lw=2,
        )

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_title(f"{model_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", fontsize=8, frameon=True)
        ax.grid(True, alpha=0.25)

    # Hide extra unused subplots
    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-Class One-vs-Rest ROC Curves Across Models (AG News)", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    print(f"  [SAVED] Detailed ROC grid: {save_path.name}")


def run_roc_analysis(dataset_name: str = "ag_news"):
    print(f"\n{'='*70}")
    print(f"  Starting Multi-Class ROC & AUC Analysis: {dataset_name}")
    print(f"{'='*70}")

    test_path = config.DATA_DIR / f"{dataset_name}_test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Please run data_prep.py first.")

    test_df = pd.read_csv(test_path)
    label_names = config.DATASETS[dataset_name]["label_names"]
    n_classes = len(label_names)

    texts = test_df["text"].fillna("").tolist()
    y_true = test_df["label"].to_numpy()
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    model_curves = {}
    auc_summary = {}

    # ── 1. Traditional Baselines ─────────────────────────────────────────
    tfidf_path = config.MODELS_DIR / f"{dataset_name}_tfidf_vectorizer.joblib"
    if tfidf_path.exists():
        print("\nLoading TF-IDF vectorizer and transforming test texts...")
        vectorizer = joblib.load(tfidf_path)
        X_test_tfidf = vectorizer.transform(texts)

        # Logistic Regression
        lr_path = config.MODELS_DIR / f"{dataset_name}_logistic_regression.joblib"
        if lr_path.exists():
            print("  Evaluating Logistic Regression...")
            lr_model = joblib.load(lr_path)
            y_probs = lr_model.predict_proba(X_test_tfidf)
            roc_res = compute_ovr_roc(y_true_bin, y_probs, label_names)
            model_curves["Logistic Regression"] = roc_res
            auc_summary["Logistic Regression"] = roc_res["auc_scores"]

        # Naive Bayes
        nb_path = config.MODELS_DIR / f"{dataset_name}_naive_bayes.joblib"
        if nb_path.exists():
            print("  Evaluating Naive Bayes...")
            nb_model = joblib.load(nb_path)
            y_probs = nb_model.predict_proba(X_test_tfidf)
            roc_res = compute_ovr_roc(y_true_bin, y_probs, label_names)
            model_curves["Naive Bayes"] = roc_res
            auc_summary["Naive Bayes"] = roc_res["auc_scores"]

        # Linear SVC (Decision Function -> Softmax for probability approximation)
        svc_path = config.MODELS_DIR / f"{dataset_name}_svm_linearsvc.joblib"
        if svc_path.exists():
            print("  Evaluating Linear SVC (Calibrated Softmax)...")
            svc_model = joblib.load(svc_path)
            decision = svc_model.decision_function(X_test_tfidf)
            # Softmax calibration over decision function
            exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            y_probs = exp_d / np.sum(exp_d, axis=1, keepdims=True)
            roc_res = compute_ovr_roc(y_true_bin, y_probs, label_names)
            model_curves["Linear SVC"] = roc_res
            auc_summary["Linear SVC"] = roc_res["auc_scores"]

    # ── 2. Fine-Tuned Transformers ───────────────────────────────────────
    for model_key in ["DistilBERT", "BERT", "RoBERTa"]:
        model_dir = config.MODELS_DIR / f"{model_key}_{dataset_name}"
        if model_dir.exists():
            print(f"\nEvaluating Transformer: {model_key}...")
            y_probs = get_transformer_probabilities(model_key, dataset_name, model_dir, texts, batch_size=64)
            roc_res = compute_ovr_roc(y_true_bin, y_probs, label_names)
            model_curves[model_key] = roc_res
            auc_summary[model_key] = roc_res["auc_scores"]
        else:
            print(f"  Model directory {model_dir} not found. Skipping.")

    # ── 3. Plot and Export ───────────────────────────────────────────────
    figures_dir = config.FIGURES_DIR
    combined_plot_path = figures_dir / f"{dataset_name}_roc_curves_combined.png"
    detailed_plot_path = figures_dir / f"{dataset_name}_roc_curves_detailed.png"

    print("\nGenerating ROC visual assets...")
    plot_combined_roc(model_curves, combined_plot_path)
    plot_detailed_grid_roc(model_curves, label_names, detailed_plot_path)

    # Export metrics JSON
    metrics_path = config.RESULTS_DIR / f"{dataset_name}_roc_auc_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(auc_summary, f, indent=2)
    print(f"  [SAVED] ROC AUC metrics to {metrics_path.name}")

    # Summary table
    print("\n" + "="*70)
    print(f"{'Model':<22} | {'Macro AUC':<10} | {'Micro AUC':<10} | {'World':<7} | {'Sports':<7} | {'Business':<8} | {'Sci/Tech':<8}")
    print("-" * 75)
    for model_name, scores in auc_summary.items():
        print(f"{model_name:<22} | {scores['macro']:<10.4f} | {scores['micro']:<10.4f} | {scores['World']:<7.4f} | {scores['Sports']:<7.4f} | {scores['Business']:<8.4f} | {scores['Sci/Tech']:<8.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_roc_analysis("ag_news")
