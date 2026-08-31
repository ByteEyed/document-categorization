# Intelligent Document Categorization using Hugging Face Transformers with Performance Benchmarking

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Subject:** Natural Language Processing and Text Analysis  
**Team Size:** 2 Members

---

## 📋 Abstract

This project implements an end-to-end intelligent document categorization system that compares traditional machine learning baselines (TF-IDF with Logistic Regression, Naive Bayes, SVM) against fine-tuned Transformer models (BERT, DistilBERT, RoBERTa) and a zero-shot classification approach (BART-large-MNLI). The system is evaluated on two standard benchmark datasets — AG News (4-class, 120K+ samples) and BBC News (5-class, 2.2K samples) — with comprehensive performance benchmarking including accuracy, F1 score, inference latency, model size, and memory usage.

---

## 🏗️ Project Structure

```
document-categorization/
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies
├── config.py                    # Centralized hyperparameters & paths
├── data_prep.py                 # Dataset loading, cleaning, splitting
├── eda.py                       # Exploratory Data Analysis & visualizations
├── baselines.py                 # TF-IDF + traditional ML classifiers
├── train_transformer.py         # Fine-tuning HF Transformer models
├── benchmark.py                 # Performance benchmarking engine
├── app.py                       # Gradio demo for live inference
├── utils.py                     # Shared utilities (metrics, plotting)
├── data/                        # Cached processed datasets
├── models/                      # Saved model checkpoints
├── results/                     # Benchmark results, plots, tables
│   └── figures/                 # Generated visualization charts
└── report/                      # LaTeX project report
    ├── report.tex               # Main LaTeX source
    └── report.pdf               # Compiled report
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (tested on RTX 2050 4GB VRAM)
- Git

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/<username>/document-categorization.git
cd document-categorization

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 📖 How to Run

Execute the scripts in the following order:

### Step 1: Data Preparation
```bash
# Prepare both datasets
python data_prep.py --dataset all

# Or prepare individually
python data_prep.py --dataset ag_news
python data_prep.py --dataset bbc_news
```

### Step 2: Exploratory Data Analysis
```bash
# Generate all EDA visualizations
python eda.py --dataset all
```
Outputs saved to `results/figures/`.

### Step 3: Train Baseline Models
```bash
# Train all baselines on both datasets
python baselines.py --dataset all
```

### Step 4: Fine-Tune Transformer Models
```bash
# Fine-tune all transformer models (BERT, DistilBERT, RoBERTa) + zero-shot
python train_transformer.py --model all --dataset all

# Or train individually (recommended for 4GB VRAM — one at a time)
python train_transformer.py --model distilbert --dataset ag_news
python train_transformer.py --model bert --dataset ag_news
python train_transformer.py --model roberta --dataset ag_news
python train_transformer.py --model zero-shot --dataset ag_news
```

> ⚠️ **GPU Memory Note:** With 4GB VRAM, train one model at a time. The config uses batch_size=8, gradient_accumulation=4, fp16=True, and max_seq_length=128 to fit within memory constraints.

### Step 5: Run Benchmarks
```bash
# Generate all comparative benchmarks
python benchmark.py --dataset all
```
Results saved to `results/` as CSV tables and charts.

### Step 6: Launch Demo App
```bash
python app.py
```
Opens a Gradio web interface at `http://localhost:7860` for live document categorization.

---

## 📊 Datasets

| Dataset | Samples | Classes | Source |
|---------|---------|---------|--------|
| **AG News** | 127,600 (120K train / 7.6K test) | 4: World, Sports, Business, Sci/Tech | [Hugging Face](https://huggingface.co/datasets/ag_news) |
| **BBC News** | 2,225 | 5: Business, Entertainment, Politics, Sport, Tech | [Hugging Face](https://huggingface.co/datasets/SetFit/bbc-news) |

---

## 🤖 Models

### Traditional ML Baselines
| Model | Feature Extraction | Key Parameters |
|-------|--------------------|----------------|
| Logistic Regression | TF-IDF (50K features, 1-2 grams) | C=1.0, max_iter=1000 |
| Multinomial Naive Bayes | TF-IDF | alpha=1.0 |
| LinearSVC (SVM) | TF-IDF | C=1.0, max_iter=2000 |

### Transformer Models
| Model | Parameters | Hugging Face ID |
|-------|-----------|-----------------|
| BERT-base | 110M | `bert-base-uncased` |
| DistilBERT | 66M | `distilbert-base-uncased` |
| RoBERTa-base | 125M | `roberta-base` |

### Zero-Shot Baseline
| Model | Hugging Face ID |
|-------|-----------------|
| BART-large-MNLI | `facebook/bart-large-mnli` |

---

## 📈 Benchmarking Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification correctness |
| Macro F1 | Class-balanced F1 (important for imbalanced data) |
| Weighted F1 | F1 weighted by class support |
| Inference Latency | Average milliseconds per sample |
| Model Parameters | Total parameter count |
| Disk Size | Saved model size in MB |
| Training Time | Total fine-tuning time |
| Peak Memory | GPU/RAM usage during inference |

---

## 🛠️ Configuration

All hyperparameters are centralized in [`config.py`](config.py). Key settings:

| Parameter | Value | Reason |
|-----------|-------|--------|
| MAX_SEQ_LENGTH | 128 | Fits 4GB VRAM |
| TRAIN_BATCH_SIZE | 8 | Memory-optimized |
| GRADIENT_ACCUMULATION | 4 | Effective batch = 32 |
| FP16 | True | Half-precision for memory savings |
| LEARNING_RATE | 2e-5 | Standard for BERT fine-tuning |
| NUM_EPOCHS | 3 | Sufficient for convergence |
| AG_NEWS_TRAIN_SUBSET | 20,000 | Manageable on limited VRAM |

---

## 📚 References

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL-HLT*.
2. Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." *NeurIPS Workshop*.
3. Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." *arXiv:1907.11692*.
4. Lewis, M., et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." *ACL*.
5. Yin, W., et al. (2019). "Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach." *EMNLP*.
6. Wolf, T., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." *EMNLP (Demo)*.

---

## 📜 License

This project is for academic purposes as part of the NLP and Text Analysis course.
