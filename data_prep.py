"""
Data Preparation Script
=======================
Handles dataset loading, cleaning, splitting, and saving for AG News.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

import config
import utils

def clean_text(text: str) -> str:
    """
    Clean text by lowercasing, removing special characters (keeping alphanumeric + spaces),
    and stripping extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Strip extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def print_stats(name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print dataset statistics including size, class distribution, and length."""
    print(f"\n{'='*40}")
    print(f"Dataset Statistics: {name}")
    print(f"{'='*40}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    
    # Class distribution (using train_df as representative)
    print("\nClass Distribution (Train):")
    dist = train_df['label_name'].value_counts()
    for lbl, count in dist.items():
        print(f"  {lbl}: {count} ({count/len(train_df)*100:.1f}%)")
    
    # Text length stats
    all_texts = pd.concat([train_df['text'], val_df['text'], test_df['text']])
    lengths = all_texts.apply(lambda x: len(str(x).split()))
    print("\nText Length (Words):")
    print(f"  Average: {lengths.mean():.1f}")
    print(f"  Min:     {lengths.min()}")
    print(f"  Max:     {lengths.max()}")
    print(f"{'='*40}\n")

def process_ag_news():
    """Load, process, and split the AG News dataset."""
    print("Loading AG News dataset...")
    dataset = load_dataset(config.DATASETS["ag_news"]["hf_name"])
    
    # Convert to pandas
    train_full = dataset["train"].to_pandas()
    test_full = dataset["test"].to_pandas()
    
    # Standardize columns and add label_name
    label_names = config.DATASETS["ag_news"]["label_names"]
    train_full['label_name'] = train_full['label'].map(lambda x: label_names[x])
    test_full['label_name'] = test_full['label'].map(lambda x: label_names[x])
    
    # Apply text cleaning
    print("Cleaning text...")
    train_full['text'] = train_full['text'].apply(clean_text)
    test_full['text'] = test_full['text'].apply(clean_text)
    
    # Create train and val splits from train_full using config sizes
    train_df, val_df = train_test_split(
        train_full, 
        train_size=config.AG_NEWS_TRAIN_SUBSET, 
        test_size=config.AG_NEWS_VAL_SUBSET, 
        stratify=train_full['label'], 
        random_state=config.RANDOM_SEED
    )
    
    # For test, sample if test_full is larger than config limit, else keep full
    if len(test_full) > config.AG_NEWS_TEST_SUBSET:
        test_df = test_full.sample(n=config.AG_NEWS_TEST_SUBSET, random_state=config.RANDOM_SEED)
    else:
        test_df = test_full
    
    # Ensure standard columns
    cols = ['text', 'label', 'label_name']
    train_df = train_df[cols]
    val_df = val_df[cols]
    test_df = test_df[cols]
    
    # Print stats
    print_stats("AG News", train_df, val_df, test_df)
    
    # Save CSVs
    print("Saving AG News CSVs...")
    train_df.to_csv(config.DATA_DIR / "ag_news_train.csv", index=False)
    val_df.to_csv(config.DATA_DIR / "ag_news_val.csv", index=False)
    test_df.to_csv(config.DATA_DIR / "ag_news_test.csv", index=False)
    print("AG News processing complete.")

def main():
    process_ag_news()

if __name__ == "__main__":
    main()
