# Intelligent Document Categorization using Hugging Face Transformers with Performance Benchmarking

**Subject:** Natural Language Processing and Text Analysis  
**Team Size:** 2

---

## Overview

An end-to-end intelligent document categorization system comparing traditional machine learning baselines against fine-tuned Transformer models on the **AG News** dataset (4-class news classification: World, Sports, Business, Sci/Tech).

The project features:
- **Comprehensive Benchmarking:** Trade-off analysis between contextual deep learning (BERT, DistilBERT, RoBERTa) and fast classical ML (TF-IDF + Naive Bayes, Logistic Regression, Linear SVM).
- **Interactive Streamlit Dashboard:** Multi-page analytics app with real-time model selection, live GPU/CPU inference latency tracking, class confidence distributions, and multi-model comparison.
- **ROC-AUC & Error Analysis:** Multi-class One-vs-Rest (OvR) ROC analysis, calibration curves, confusion matrices, and EDA.

---

## Benchmark Results

Evaluated on the full 7,600-sample AG News test set:

| Model | Architecture | Accuracy | Macro F1 | Latency (ms) | Hardware |
|---|---|---|---|---|---|
| **BERT-base** | Transformer (110M) | **92.29%** | **0.9227** | 34.96 ms | GPU (CUDA) |
| **DistilBERT** | Transformer (66M) | **92.28%** | **0.9227** | **12.55 ms** | GPU (CUDA) |
| **RoBERTa-base** | Transformer (125M) | 92.05% | 0.9204 | 19.46 ms | GPU (CUDA) |
| **SVM (LinearSVC)** | TF-IDF + Linear SVM | 90.11% | 0.9008 | 0.42 ms | CPU |
| **Logistic Regression** | TF-IDF + L2 Logistic | 89.32% | 0.8928 | 0.40 ms | CPU |
| **Naive Bayes** | TF-IDF + Multinomial NB | 89.09% | 0.8905 | 1.13 ms | CPU |
| **BART Zero-Shot** | Pretrained Zero-Shot (407M) | 71.40% | 0.6961 | 186.72 ms | GPU (CUDA) |

> **Key Takeaway:** DistilBERT matches BERT's accuracy (92.28% vs. 92.29%) while providing **2.8× faster inference** with only 60% of the parameter count. Traditional baselines offer sub-millisecond CPU inference, making them compelling for low-latency, resource-constrained environments.

---

## Streamlit Analytics & Prediction Dashboard

Launch the interactive dashboard with:
```bash
streamlit run streamlit_app.py
```

### Dashboard Capabilities:
- **Model Selection at Will:** Interactively choose between **DistilBERT**, **BERT**, **RoBERTa**, and **Naive Bayes** to classify text in real time.
- **Live Performance Metrics:** Measures actual live inference execution time (ms), confidence percentages, and lead margin over runner-up classes.
- **Side-by-Side Model Comparison:** Evaluate custom text across all models simultaneously to inspect differing class confidences.
- **Exploratory Data Analysis (EDA):** Class distributions, word length histograms, and category word frequency visualizers.
- **Model Comparison Suite:** Interactive scatter plots (Latency vs. Accuracy Pareto frontier) and metric leaderboards.

---

## Setup & Installation

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

> **Requirements:** Python 3.10+ and an NVIDIA GPU with CUDA (recommended for transformer training and fast inference).

---

## Usage Pipeline

Run the pipeline scripts in order:

```bash
python data_prep.py                              # 1. Download & preprocess AG News dataset
python eda.py                                    # 2. Generate Exploratory Data Analysis figures
python baselines.py                              # 3. Train TF-IDF baselines (LR, NB, SVM)
python train_transformer.py --model distilbert   # 4. Fine-tune DistilBERT
python train_transformer.py --model bert         #    Fine-tune BERT
python train_transformer.py --model roberta      #    Fine-tune RoBERTa
python train_transformer.py --model zero-shot    # 5. Evaluate BART Zero-Shot classifier
python benchmark.py                              # 6. Generate unified benchmark tables & charts
python roc_analysis.py                           # 7. Multi-Class ROC curves & AUC analysis
streamlit run streamlit_app.py                   # 8. Launch Streamlit Analytical Dashboard
python app.py                                    # 9. (Optional) Launch Gradio Demo
```

---

## Project Structure

```
├── config.py               # Centralized hyperparameters, paths & device config
├── utils.py                # Shared metrics, timing, plotting & persistence helpers
├── data_prep.py            # Dataset download, cleaning & train/val/test splits
├── eda.py                  # Exploratory Data Analysis & wordclouds
├── baselines.py            # TF-IDF feature extraction & classical ML training
├── train_transformer.py    # Transformer fine-tune (Trainer API) & zero-shot
├── benchmark.py            # Unified latency, accuracy & resource benchmarking
├── roc_analysis.py         # Multi-class OvR ROC curves & AUC analysis
├── streamlit_app.py        # Streamlit Analytical Dashboard with Model Selection
├── app.py                  # Gradio web demo
├── report/report.tex       # Academic LaTeX project report
├── data/                   # Processed dataset CSVs (ag_news_train/val/test)
├── models/                 # Saved checkpoints (.joblib and Hugging Face models)
└── results/                # Benchmark JSON/CSV results and figures
```

## References

1. Devlin et al. (2019) — BERT
2. Sanh et al. (2019) — DistilBERT
3. Liu et al. (2019) — RoBERTa
4. Lewis et al. (2020) — BART
5. Wolf et al. (2020) — Hugging Face Transformers
