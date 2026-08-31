"""
Shared Utilities for Document Categorization Project
=====================================================
Common functions for metrics, plotting, timing, and model analysis
used across all project scripts.
"""

import time
import json
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

import config


# ============================================
# Metrics Computation
# ============================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        label_names: Human-readable label names.

    Returns:
        Dictionary with accuracy, F1 scores, per-class metrics, and classification report.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro")),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
        "per_class_f1": {
            name: float(f1)
            for name, f1 in zip(
                label_names,
                f1_score(y_true, y_pred, average=None),
            )
        },
        "per_class_precision": {
            name: float(p)
            for name, p in zip(
                label_names,
                precision_score(y_true, y_pred, average=None),
            )
        },
        "per_class_recall": {
            name: float(r)
            for name, r in zip(
                label_names,
                recall_score(y_true, y_pred, average=None),
            )
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True
        ),
    }
    return metrics


def print_metrics(metrics: Dict[str, Any], model_name: str) -> None:
    """Pretty-print metrics for a model."""
    print(f"\n{'='*60}")
    print(f"  Results for: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:    {metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision:{metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:   {metrics['macro_recall']:.4f}")
    print(f"\n  Per-Class F1:")
    for cls_name, f1_val in metrics["per_class_f1"].items():
        print(f"    {cls_name:20s}: {f1_val:.4f}")
    print(f"{'='*60}\n")


# ============================================
# Timing Utilities
# ============================================

class Timer:
    """Context manager for measuring execution time."""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"  ⏱ {self.description}: {self.elapsed:.2f}s")


def measure_inference_latency(
    predict_fn,
    inputs: List[str],
    num_samples: int = None,
    warmup_runs: int = None,
) -> Dict[str, float]:
    """
    Measure inference latency statistics.

    Args:
        predict_fn: Callable that takes a single text string and returns prediction.
        inputs: List of text inputs.
        num_samples: Number of samples to measure (defaults to config).
        warmup_runs: Number of warmup runs (defaults to config).

    Returns:
        Dictionary with avg, median, p95, p99 latency in milliseconds.
    """
    if num_samples is None:
        num_samples = config.BENCHMARK_NUM_SAMPLES
    if warmup_runs is None:
        warmup_runs = config.BENCHMARK_WARMUP_RUNS

    samples = inputs[:num_samples]

    # Warmup
    for text in samples[:warmup_runs]:
        predict_fn(text)

    # Measure
    latencies = []
    for text in samples:
        start = time.perf_counter()
        predict_fn(text)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        latencies.append(elapsed)

    latencies = np.array(latencies)
    return {
        "avg_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "total_time_s": float(np.sum(latencies) / 1000),
    }


# ============================================
# Model Analysis
# ============================================

def count_parameters(model) -> Dict[str, int]:
    """Count trainable and total parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "total_parameters_millions": round(total / 1e6, 2),
    }


def get_model_size_mb(model_path: Path) -> float:
    """Calculate total size of saved model files in MB."""
    total_size = 0
    if model_path.is_dir():
        for f in model_path.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
    elif model_path.is_file():
        total_size = model_path.stat().st_size
    return round(total_size / (1024 * 1024), 2)


# ============================================
# Plotting Helpers
# ============================================

def setup_plot_style():
    """Configure matplotlib with consistent styling."""
    try:
        plt.style.use(config.PLOT_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")
    sns.set_palette(config.COLOR_PALETTE)
    plt.rcParams.update({
        "figure.dpi": config.FIGURE_DPI,
        "savefig.dpi": config.FIGURE_DPI,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    normalize: bool = True,
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    setup_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format=config.FIGURE_FORMAT)
        print(f"  📊 Saved confusion matrix: {save_path}")
    plt.close(fig)


def plot_bar_comparison(
    data: Dict[str, float],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[Path] = None,
    horizontal: bool = False,
    color_palette: str = None,
) -> None:
    """Plot a bar chart comparing values across models."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(data.keys())
    values = list(data.values())
    colors = sns.color_palette(color_palette or config.COLOR_PALETTE, len(names))

    if horizontal:
        bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=10)
    else:
        bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format=config.FIGURE_FORMAT)
        print(f"  📊 Saved chart: {save_path}")
    plt.close(fig)


def plot_scatter_comparison(
    x_data: Dict[str, float],
    y_data: Dict[str, float],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot a scatter chart comparing two metrics across models."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    names = list(x_data.keys())
    x_vals = [x_data[n] for n in names]
    y_vals = [y_data[n] for n in names]
    colors = sns.color_palette(config.COLOR_PALETTE, len(names))

    for i, name in enumerate(names):
        ax.scatter(x_vals[i], y_vals[i], s=150, c=[colors[i]], edgecolor="black",
                   linewidth=1.5, zorder=5)
        ax.annotate(name, (x_vals[i], y_vals[i]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=10, fontweight="bold")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format=config.FIGURE_FORMAT)
        print(f"  📊 Saved scatter plot: {save_path}")
    plt.close(fig)


def plot_training_history(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Loss",
    save_path: Optional[Path] = None,
) -> None:
    """Plot training (and optionally validation) loss curves."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", label="Training Loss", linewidth=2, markersize=6)

    if val_losses:
        ax.plot(epochs, val_losses, "r-s", label="Validation Loss", linewidth=2, markersize=6)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format=config.FIGURE_FORMAT)
        print(f"  📊 Saved training history: {save_path}")
    plt.close(fig)


# ============================================
# File I/O Helpers
# ============================================

def save_results(results: Dict, filepath: Path) -> None:
    """Save results dictionary as JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"  💾 Saved results: {filepath}")


def load_results(filepath: Path) -> Dict:
    """Load results from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
