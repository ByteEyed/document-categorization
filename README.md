# Intelligent Document Categorization using Hugging Face Transformers with Performance Benchmarking

**Subject:** Natural Language Processing and Text Analysis  
**Team Size:** 2

---

## Overview

An end-to-end document categorization system comparing traditional ML baselines against fine-tuned Transformer models on the **AG News** dataset (4-class news classification: World, Sports, Business, Sci/Tech).

## Models

| Type | Models |
|------|--------|
| **Traditional ML** | TF-IDF + Logistic Regression, Naive Bayes, SVM |
| **Transformers** | BERT, DistilBERT, RoBERTa (fine-tuned) |
| **Zero-Shot** | BART-large-MNLI (no training) |

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

> Requires Python 3.10+ and NVIDIA GPU with CUDA for transformer training.

## Usage

Run scripts in order:

```bash
python data_prep.py                              # 1. Download & prepare AG News
python eda.py                                    # 2. Generate EDA visualizations
python baselines.py                              # 3. Train TF-IDF baselines
python train_transformer.py --model distilbert   # 4. Fine-tune transformers (one at a time)
python train_transformer.py --model bert
python train_transformer.py --model roberta
python train_transformer.py --model zero-shot    # 5. Zero-shot inference
python benchmark.py                              # 6. Generate benchmark comparison
python app.py                                    # 7. Launch Gradio demo
```

## Project Structure

```
├── config.py               # Hyperparameters & paths
├── utils.py                # Shared metrics, plotting, timing
├── data_prep.py            # Dataset loading & preprocessing
├── eda.py                  # Exploratory Data Analysis
├── baselines.py            # TF-IDF + ML classifiers
├── train_transformer.py    # Transformer fine-tuning & zero-shot
├── benchmark.py            # Unified performance benchmarking
├── app.py                  # Gradio web demo
├── report/report.tex       # LaTeX project report
├── data/                   # Processed CSVs
├── models/                 # Saved checkpoints
└── results/figures/        # Charts & plots
```

## References

1. Devlin et al. (2019) — BERT
2. Sanh et al. (2019) — DistilBERT
3. Liu et al. (2019) — RoBERTa
4. Lewis et al. (2020) — BART
5. Wolf et al. (2020) — Hugging Face Transformers
