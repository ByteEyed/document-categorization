"""
Centralized Configuration for Document Categorization Project
==============================================================
All hyperparameters, paths, model names, and constants in one place.
Optimized for RTX 2050 (4GB VRAM) — uses small batch sizes, fp16, and
gradient accumulation to fit within memory constraints.
"""

import os
from pathlib import Path

# ============================================
# Project Paths
# ============================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

# Create directories
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================
# Dataset Configuration
# ============================================
DATASETS = {
    "ag_news": {
        "hf_name": "fancyzhx/ag_news",
        "hf_subset": None,
        "text_column": "text",
        "label_column": "label",
        "label_names": ["World", "Sports", "Business", "Sci/Tech"],
        "num_classes": 4,
    },
}

# ============================================
# Text Preprocessing
# ============================================
MAX_SEQ_LENGTH = 128          # Keep short for 4GB VRAM
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)   # Unigrams + Bigrams

# ============================================
# Traditional ML Baselines
# ============================================
BASELINE_MODELS = {
    "Logistic Regression": {
        "model": "LogisticRegression",
        "params": {"max_iter": 1000, "C": 1.0, "solver": "lbfgs"},
    },
    "Naive Bayes": {
        "model": "MultinomialNB",
        "params": {"alpha": 1.0},
    },
    "SVM (LinearSVC)": {
        "model": "LinearSVC",
        "params": {"max_iter": 2000, "C": 1.0},
    },
}

# ============================================
# Transformer Configuration (optimized for RTX 2050 4GB VRAM)
# ============================================
TRANSFORMER_MODELS = {
    "BERT": "bert-base-uncased",
    "DistilBERT": "distilbert-base-uncased",
    "RoBERTa": "roberta-base",
}

ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

# Training Hyperparameters (memory-optimized)
TRAIN_BATCH_SIZE = 8          # Small batch for 4GB VRAM
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4   # Effective batch size = 8 * 4 = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
FP16 = True                  # Mixed precision for memory savings
MAX_GRAD_NORM = 1.0

# Early stopping
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.01

# AG News subset for faster training (full dataset is 120K)
AG_NEWS_TRAIN_SUBSET = 20000  # Use 20K samples for training (manageable on 4GB VRAM)
AG_NEWS_VAL_SUBSET = 2000
AG_NEWS_TEST_SUBSET = 7600    # Full test set

# ============================================
# Benchmarking
# ============================================
BENCHMARK_NUM_SAMPLES = 500   # Number of samples for latency measurement
BENCHMARK_WARMUP_RUNS = 10    # Warmup runs before timing

# ============================================
# Visualization
# ============================================
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
COLOR_PALETTE = "Set2"

# ============================================
# Random Seed
# ============================================
RANDOM_SEED = 42

# ============================================
# Device Configuration
# ============================================
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 if os.name == "nt" else 2  # Windows doesn't handle multiprocessing well with DataLoader
