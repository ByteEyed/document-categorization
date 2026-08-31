"""
Exploratory Data Analysis Script
================================
Generates visualizations and statistics for the processed datasets.
"""

import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import config
import utils

def load_data(dataset_name: str) -> pd.DataFrame:
    """Load train, val, test splits and combine them for EDA."""
    dfs = []
    for split in ["train", "val", "test"]:
        path = config.DATA_DIR / f"{dataset_name}_{split}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['split'] = split
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CSVs found for dataset: {dataset_name}. Run data_prep.py first.")
    return pd.concat(dfs, ignore_index=True)

def plot_class_distribution(df: pd.DataFrame, dataset_name: str):
    """Plot bar chart of class distribution."""
    utils.setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    order = df['label_name'].value_counts().index
    sns.countplot(data=df, x='label_name', order=order, palette=config.COLOR_PALETTE, ax=ax)
    
    ax.set_title(f"Class Distribution: {dataset_name}")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = config.FIGURES_DIR / f"{dataset_name}_class_dist.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved class distribution to {save_path.name}")

def plot_text_length_histogram(df: pd.DataFrame, dataset_name: str):
    """Plot histogram of text lengths."""
    utils.setup_plot_style()
    df['text_len'] = df['text'].astype(str).apply(lambda x: len(x.split()))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='text_len', bins=50, kde=True, color='skyblue', ax=ax)
    
    ax.set_title(f"Text Length Distribution: {dataset_name}")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    
    save_path = config.FIGURES_DIR / f"{dataset_name}_length_hist.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved text length histogram to {save_path.name}")

def plot_text_length_boxplot(df: pd.DataFrame, dataset_name: str):
    """Plot boxplots of text lengths by category."""
    utils.setup_plot_style()
    if 'text_len' not in df.columns:
        df['text_len'] = df['text'].astype(str).apply(lambda x: len(x.split()))
        
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='label_name', y='text_len', palette=config.COLOR_PALETTE, ax=ax)
    
    ax.set_title(f"Text Length by Category: {dataset_name}")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Words")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = config.FIGURES_DIR / f"{dataset_name}_length_box.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved text length boxplot to {save_path.name}")

def plot_word_clouds(df: pd.DataFrame, dataset_name: str):
    """Plot word clouds for each category in a grid."""
    labels = df['label_name'].unique()
    n_labels = len(labels)
    cols = 2
    rows = (n_labels + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    for i, label in enumerate(labels):
        text_data = " ".join(df[df['label_name'] == label]['text'].astype(str).tolist())
        wordcloud = WordCloud(
            width=800, height=400, 
            background_color='white',
            stopwords=ENGLISH_STOP_WORDS
        ).generate(text_data)
        
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f"Word Cloud: {label}")
        axes[i].axis("off")
        
    # Hide any unused axes
    for j in range(len(labels), len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    save_path = config.FIGURES_DIR / f"{dataset_name}_wordclouds.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved word clouds to {save_path.name}")

def plot_top_words(df: pd.DataFrame, dataset_name: str, top_n: int = 20):
    """Plot top N most frequent words for each category."""
    labels = df['label_name'].unique()
    n_labels = len(labels)
    cols = 2
    rows = (n_labels + 1) // 2
    
    utils.setup_plot_style()
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    axes = axes.flatten()
    
    for i, label in enumerate(labels):
        texts = df[df['label_name'] == label]['text'].astype(str)
        all_words = " ".join(texts).split()
        
        # Filter out stopwords and short words
        filtered_words = [w for w in all_words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
        
        counter = Counter(filtered_words)
        common = counter.most_common(top_n)
        
        if common:
            words, counts = zip(*common)
            sns.barplot(x=list(counts), y=list(words), ax=axes[i], palette="viridis")
        
        axes[i].set_title(f"Top {top_n} Words: {label}")
        axes[i].set_xlabel("Frequency")
        axes[i].set_ylabel("Word")
        
    # Hide any unused axes
    for j in range(len(labels), len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    save_path = config.FIGURES_DIR / f"{dataset_name}_top_words.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved top words to {save_path.name}")

def run_eda(dataset_name: str):
    """Execute the EDA pipeline for a given dataset."""
    print(f"\n{'='*40}")
    print(f"Running EDA for dataset: {dataset_name}")
    print(f"{'='*40}")
    
    try:
        df = load_data(dataset_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    print(f"Total combined samples: {len(df)}")
    print("Generating visualizations...")
    
    plot_class_distribution(df, dataset_name)
    plot_text_length_histogram(df, dataset_name)
    plot_text_length_boxplot(df, dataset_name)
    plot_word_clouds(df, dataset_name)
    plot_top_words(df, dataset_name)
    
    print(f"EDA complete for {dataset_name}.\n")

def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis")
    parser.add_argument("--dataset", choices=["ag_news"], default="ag_news", 
                        help="Dataset to analyze (default: ag_news)")
    args = parser.parse_args()
    
    run_eda("ag_news")

if __name__ == "__main__":
    main()
