"""
Unified Benchmarking Engine
============================
Loads all saved results from baselines and transformer training,
generates comprehensive comparative visualizations, and exports
a unified comparison table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Benchmarking Engine")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ag_news"],
        default="ag_news",
        help="Dataset to benchmark",
    )
    return parser.parse_args()


def load_all_results(dataset_name):
    """
    Load results for all models on a given dataset.

    Handles the actual file naming conventions:
      - baselines.py saves: {dataset}_baselines_results.json  (dict of model_name -> {metrics, latency})
      - train_transformer.py saves: results_{ModelKey}_{dataset}.json  (flat metrics dict + latency)
      - train_transformer.py saves: results_zero-shot_{dataset}.json
    """
    results = {}

    # ── 1. Baseline results ──────────────────────────────────────────────
    baseline_path = config.RESULTS_DIR / f"{dataset_name}_baselines_results.json"
    if baseline_path.exists():
        data = utils.load_results(baseline_path)
        for model_name, model_data in data.items():
            results[model_name] = model_data
        print(f"  ✅ Loaded baseline results ({len(data)} models)")
    else:
        print(f"  ⚠️  Baseline results not found: {baseline_path.name}")

    # ── 2. Transformer results (one file per model) ──────────────────────
    for model_key in config.TRANSFORMER_MODELS:
        result_path = config.RESULTS_DIR / f"results_{model_key}_{dataset_name}.json"
        if result_path.exists():
            data = utils.load_results(result_path)
            # train_transformer.py saves metrics at top level, wrap to match baseline format
            results[model_key] = {
                "metrics": {
                    "accuracy": data.get("accuracy", np.nan),
                    "macro_f1": data.get("macro_f1", np.nan),
                    "weighted_f1": data.get("weighted_f1", np.nan),
                    "per_class_f1": data.get("per_class_f1", {}),
                },
                "latency": data.get("latency", {}),
            }
            print(f"  ✅ Loaded results for {model_key}")
        else:
            print(f"  ⚠️  Results not found: {result_path.name}")

    # ── 3. Zero-shot results ─────────────────────────────────────────────
    zs_path = config.RESULTS_DIR / f"results_zero-shot_{dataset_name}.json"
    if zs_path.exists():
        data = utils.load_results(zs_path)
        results["Zero-Shot (BART)"] = {
            "metrics": {
                "accuracy": data.get("accuracy", np.nan),
                "macro_f1": data.get("macro_f1", np.nan),
                "weighted_f1": data.get("weighted_f1", np.nan),
                "per_class_f1": data.get("per_class_f1", {}),
            },
            "latency": data.get("latency", {}),
        }
        print(f"  ✅ Loaded zero-shot results")
    else:
        print(f"  ⚠️  Zero-shot results not found: {zs_path.name}")

    return results


def create_comparison_dataframe(all_dataset_results):
    """Create a unified DataFrame from loaded results."""
    records = []

    for dataset, models in all_dataset_results.items():
        for model_name, data in models.items():
            metrics = data.get("metrics", {})
            latency = data.get("latency", {})
            record = {
                "Model": model_name,
                "Dataset": dataset,
                "Accuracy": metrics.get("accuracy", np.nan),
                "Macro_F1": metrics.get("macro_f1", np.nan),
                "Weighted_F1": metrics.get("weighted_f1", np.nan),
                "Avg_Latency_ms": latency.get("avg_latency_ms", np.nan),
                "P95_Latency_ms": latency.get("p95_latency_ms", np.nan),
                "per_class_f1": metrics.get("per_class_f1", {}),
            }
            records.append(record)

    return pd.DataFrame(records)


def generate_visualizations(df):
    """Generate all comparative visualizations, grouped per dataset."""
    utils.setup_plot_style()

    for dataset in df["Dataset"].unique():
        ds_df = df[df["Dataset"] == dataset].copy().sort_values("Macro_F1", ascending=False)
        if ds_df.empty:
            continue

        # ── 1. Accuracy comparison ───────────────────────────────────────
        utils.plot_bar_comparison(
            dict(zip(ds_df["Model"], ds_df["Accuracy"])),
            title=f"Accuracy Comparison — {dataset}",
            xlabel="Model",
            ylabel="Accuracy",
            save_path=config.FIGURES_DIR / f"{dataset}_bench_accuracy.{config.FIGURE_FORMAT}",
        )

        # ── 2. Macro F1 comparison ───────────────────────────────────────
        utils.plot_bar_comparison(
            dict(zip(ds_df["Model"], ds_df["Macro_F1"])),
            title=f"Macro F1 Comparison — {dataset}",
            xlabel="Model",
            ylabel="Macro F1",
            save_path=config.FIGURES_DIR / f"{dataset}_bench_macro_f1.{config.FIGURE_FORMAT}",
        )

        # ── 3. F1 vs Inference Latency scatter ───────────────────────────
        valid = ds_df.dropna(subset=["Avg_Latency_ms", "Macro_F1"])
        if len(valid) >= 2:
            utils.plot_scatter_comparison(
                x_data=dict(zip(valid["Model"], valid["Avg_Latency_ms"])),
                y_data=dict(zip(valid["Model"], valid["Macro_F1"])),
                title=f"Macro F1 vs Inference Latency — {dataset}",
                xlabel="Avg Latency (ms)",
                ylabel="Macro F1",
                save_path=config.FIGURES_DIR / f"{dataset}_bench_f1_vs_latency.{config.FIGURE_FORMAT}",
            )

        # ── 4. Per-class F1 heatmap ──────────────────────────────────────
        f1_rows = []
        model_names = []
        for _, row in ds_df.iterrows():
            pcf1 = row.get("per_class_f1", {})
            if pcf1:
                f1_rows.append(pcf1)
                model_names.append(row["Model"])
        if f1_rows:
            f1_df = pd.DataFrame(f1_rows, index=model_names)
            fig, ax = plt.subplots(figsize=(max(10, len(f1_df.columns) * 2), max(5, len(f1_df) * 0.8)))
            sns.heatmap(f1_df, annot=True, cmap="YlGnBu", fmt=".3f", ax=ax, linewidths=0.5)
            ax.set_title(f"Per-Class F1 Heatmap — {dataset}")
            ax.set_ylabel("Model")
            ax.set_xlabel("Class")
            plt.tight_layout()
            fig.savefig(config.FIGURES_DIR / f"{dataset}_bench_per_class_f1.{config.FIGURE_FORMAT}",
                        dpi=config.FIGURE_DPI)
            plt.close(fig)
            print(f"  📊 Saved per-class F1 heatmap: {dataset}")


def run_benchmark(dataset_arg):
    print("\n" + "=" * 70)
    print("  UNIFIED PERFORMANCE BENCHMARKING ENGINE")
    print("=" * 70)

    datasets_to_process = (
        list(config.DATASETS.keys()) if dataset_arg == "all" else [dataset_arg]
    )

    all_dataset_results = {}
    for ds in datasets_to_process:
        print(f"\n📂 Loading results for: {ds}")
        ds_results = load_all_results(ds)
        if ds_results:
            all_dataset_results[ds] = ds_results
        else:
            print(f"  ⚠️  No results found for {ds}.")

    if not all_dataset_results:
        print("\n❌ No results loaded. Run baselines.py and/or train_transformer.py first.")
        return

    # Build unified comparison table
    df = create_comparison_dataframe(all_dataset_results)

    # Generate visualizations
    print("\n🎨 Generating comparative visualizations...")
    generate_visualizations(df)

    # Export CSV (without the nested per_class_f1 column)
    export_cols = ["Model", "Dataset", "Accuracy", "Macro_F1", "Weighted_F1",
                   "Avg_Latency_ms", "P95_Latency_ms"]
    export_df = df[export_cols].copy()
    csv_path = config.RESULTS_DIR / "benchmark_comparison.csv"
    export_df.to_csv(csv_path, index=False)
    print(f"\n💾 Exported comparison table: {csv_path}")

    # Console output
    print("\n" + "=" * 70)
    print("  UNIFIED COMPARISON TABLE")
    print("=" * 70)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(export_df.to_string(index=False))

    # Summary per dataset
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for ds in export_df["Dataset"].unique():
        ds_df = export_df[export_df["Dataset"] == ds]
        best_acc_row = ds_df.loc[ds_df["Accuracy"].idxmax()]
        best_f1_row = ds_df.loc[ds_df["Macro_F1"].idxmax()]
        fastest_row = ds_df.loc[ds_df["Avg_Latency_ms"].idxmin()]
        print(f"\n  Dataset: {ds}")
        print(f"    🏆 Best Accuracy:  {best_acc_row['Model']} ({best_acc_row['Accuracy']:.4f})")
        print(f"    🏆 Best Macro F1:  {best_f1_row['Model']} ({best_f1_row['Macro_F1']:.4f})")
        print(f"    ⚡ Fastest Inference: {fastest_row['Model']} ({fastest_row['Avg_Latency_ms']:.2f} ms)")

    print("\n✅ Benchmarking complete.\n")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args.dataset)
