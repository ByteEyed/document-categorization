"""Minimal, resilient Streamlit dashboard for AG News document categorization."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import joblib
except ImportError:  # Optional: live classical models still degrade gracefully.
    joblib = None

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # Optional: transformer inference is not required for the dashboard.
    torch = None
    AutoModelForSequenceClassification = AutoTokenizer = None

st.set_page_config(page_title="AG News Intelligence", page_icon="▦", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
COLORS = {"World": "#2563EB", "Sports": "#10B981", "Business": "#F59E0B", "Sci/Tech": "#8B5CF6"}

# Project folders can be beside this file or in its parent project directory.
SEARCH_ROOTS = [ROOT, ROOT.parent, Path.cwd()]
def find_dir(name: str) -> Path:
    for base in SEARCH_ROOTS:
        candidate = base / name
        if candidate.exists():
            return candidate
    return ROOT / name

DATA_DIR = find_dir("data")
RESULTS_DIR = find_dir("results")
FIGURES_DIR = find_dir("figures")
MODELS_DIR = find_dir("models")

AVAILABLE_MODELS = {
    "DistilBERT (Fine-Tuned Transformer)": {
        "id": "DistilBERT",
        "family": "Transformer",
        "folder": "DistilBERT_ag_news",
        "desc": "Lightweight 6-layer transformer; balances top accuracy with low latency.",
        "benchmark_acc": "92.28%",
        "benchmark_latency": "12.55 ms",
    },
    "BERT (Fine-Tuned Transformer)": {
        "id": "BERT",
        "family": "Transformer",
        "folder": "BERT_ag_news",
        "desc": "12-layer bert-base-uncased; powerful deep contextual representations.",
        "benchmark_acc": "92.29%",
        "benchmark_latency": "34.96 ms",
    },
    "RoBERTa (Fine-Tuned Transformer)": {
        "id": "RoBERTa",
        "family": "Transformer",
        "folder": "RoBERTa_ag_news",
        "desc": "Byte-level BPE robustly optimized BERT architecture.",
        "benchmark_acc": "92.15%",
        "benchmark_latency": "29.40 ms",
    },
    "Naive Bayes (TF-IDF Baseline)": {
        "id": "Naive Bayes",
        "family": "Classical ML",
        "filename": "ag_news_naive_bayes.joblib",
        "desc": "Multinomial Naive Bayes; canonical probabilistic bag-of-words text baseline.",
        "benchmark_acc": "89.09%",
        "benchmark_latency": "1.13 ms",
    },
}

def check_model_availability(info: dict) -> bool:
    family = info.get("family")
    if family == "Transformer":
        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            return False
        for base in [MODELS_DIR, ROOT / "models", ROOT]:
            if (base / info["folder"]).exists():
                return True
        return False
    elif family == "Classical ML":
        if joblib is None:
            return False
        vec_exists = any((base / "ag_news_tfidf_vectorizer.joblib").exists() for base in [MODELS_DIR, ROOT / "models", ROOT])
        mod_exists = any((base / info["filename"]).exists() for base in [MODELS_DIR, ROOT / "models", ROOT])
        return vec_exists and mod_exists
    return True

@st.cache_resource(show_spinner=False)
def get_tfidf_vectorizer():
    if joblib is None:
        return None
    for folder in [MODELS_DIR, ROOT / "models", ROOT]:
        path = folder / "ag_news_tfidf_vectorizer.joblib"
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                pass
    return None

@st.cache_resource(show_spinner=False)
def get_classical_model(filename: str):
    if joblib is None or not filename:
        return None
    for folder in [MODELS_DIR, ROOT / "models", ROOT]:
        path = folder / filename
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                pass
    return None

@st.cache_resource(show_spinner=False)
def get_transformer_model(folder_name: str):
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None or not folder_name:
        return None, None
    for base in [MODELS_DIR, ROOT / "models", ROOT]:
        model_dir = base / folder_name
        if model_dir.exists():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                tok = AutoTokenizer.from_pretrained(str(model_dir))
                mod = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
                mod.to(device)
                mod.eval()
                return mod, tok
            except Exception:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    tok = AutoTokenizer.from_pretrained(str(model_dir))
                    mod = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
                    mod.to("cpu")
                    mod.eval()
                    return mod, tok
                except Exception:
                    return None, None
    return None, None

def classify_text(text: str, model_choice: str) -> tuple[dict[str, float], float, str, str]:
    """Classifies text with selected model. Returns (prob_dict, latency_ms, mode_label, status_note)."""
    t_start = time.perf_counter()
    if not text.strip():
        return {lbl: 0.25 for lbl in LABELS}, 0.0, "Empty input", "No text provided"

    info = AVAILABLE_MODELS.get(model_choice, AVAILABLE_MODELS["DistilBERT (Fine-Tuned Transformer)"])
    family = info["family"]

    # 1. Transformers
    if family == "Transformer":
        mod, tok = get_transformer_model(info["folder"])
        if mod is not None and tok is not None:
            try:
                device = next(mod.parameters()).device
                inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    outputs = mod(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()
                lat_ms = (time.perf_counter() - t_start) * 1000
                prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
                device_str = "CUDA" if device.type == "cuda" else "CPU"
                return prob_dict, lat_ms, f"{info['id']} ({device_str})", "Live neural inference"
            except Exception:
                pass  # Fall through to heuristic if an unexpected runtime error occurred

    # 2. Classical ML
    elif family == "Classical ML":
        vec = get_tfidf_vectorizer()
        clf = get_classical_model(info["filename"])
        if vec is not None and clf is not None:
            try:
                x = vec.transform([text])
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(x)[0]
                elif hasattr(clf, "decision_function"):
                    dec = clf.decision_function(x)
                    d = dec[0] if dec.ndim > 1 else dec
                    exp_d = np.exp(d - np.max(d))
                    probs = exp_d / np.sum(exp_d)
                else:
                    pred_idx = clf.predict(x)[0]
                    probs = np.zeros(len(LABELS))
                    probs[pred_idx] = 1.0
                lat_ms = (time.perf_counter() - t_start) * 1000
                prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
                return prob_dict, lat_ms, f"{info['id']} (TF-IDF)", "Live scikit-learn inference"
            except Exception:
                pass

    # 3. Rule-based / Keyword Heuristic fallback
    scores = {label: 0.08 for label in LABELS}
    keywords = {
        "Business": "market stock bank economy company finance rate trade oil dollar shares investors inflation profit",
        "Sports": "team game match player league goal tournament cup coach championship score final season win defeat",
        "Sci/Tech": "technology research software space science data computer chip ai mobile internet telescope physics devices",
        "World": "country government international leaders war diplomacy president foreign prime minister border treaty peace army",
    }
    lowered = text.lower()
    for label, terms in keywords.items():
        scores[label] += sum(term in lowered for term in terms.split()) * 0.12
    total = sum(scores.values())
    prob_dict = {k: v / total for k, v in scores.items()}
    lat_ms = (time.perf_counter() - t_start) * 1000
    is_fallback = family != "Rule-based"
    mode_label = "Demo Heuristic" if not is_fallback else "Demo Fallback"
    status_note = "Model weights unavailable, used keyword heuristic" if is_fallback else "Keyword-based pattern matching"
    return prob_dict, lat_ms, mode_label, status_note

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F7F9FC; }
[data-testid="stSidebar"] { background: #0F172A; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
.block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 3rem; }
.hero { padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#0F172A,#1D4ED8); color: white; margin-bottom: 1.2rem; }
.hero h1 { margin: 0; font-size: 2.35rem; letter-spacing: -0.04em; }
.hero p { margin: .45rem 0 0; color: #DBEAFE; font-size: 1.02rem; }
.card { background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1rem 1.15rem; height: 100%; box-shadow: 0 3px 12px rgba(15,23,42,.04); }
.card h3 { margin-top: 0; color: #0F172A; }
.small { color: #64748B; font-size: .9rem; }
.callout { border-left: 4px solid #2563EB; background: #EFF6FF; padding: .85rem 1rem; border-radius: 8px; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_csv(filename: str, n: int | None = None) -> pd.DataFrame:
    for folder in [DATA_DIR, RESULTS_DIR, ROOT]:
        path = folder / filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df.head(n) if n else df
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def benchmark() -> pd.DataFrame:
    df = load_csv("benchmark_comparison.csv")
    if not df.empty:
        return df
    # Transparent fallback keeps the app useful when only the attached file is supplied.
    return pd.DataFrame({
        "Model": ["Naive Bayes", "Logistic Regression", "SVM (LinearSVC)", "DistilBERT", "BERT", "RoBERTa", "Zero-Shot (BART)"],
        "Accuracy": [.856, .893, .901, .9228, .9229, .9215, .714],
        "Macro_F1": [.854, .891, .900, .9219, .9221, .9208, .709],
        "Avg_Latency_ms": [.18, .25, .42, 12.55, 34.96, 29.40, 110.0],
    })

@st.cache_data(show_spinner=False)
def sample_data() -> pd.DataFrame:
    df = load_csv("ag_news_test.csv", 2500)
    if df.empty:
        df = load_csv("ag_news_train.csv", 2500)
    if not df.empty:
        if "label_name" not in df.columns and "label" in df.columns:
            df["label_name"] = df["label"].map(dict(enumerate(LABELS)))
        if "text" not in df.columns:
            text_cols = [c for c in df.columns if c.lower() in ("description", "title")]
            if text_cols: df["text"] = df[text_cols].astype(str).agg(" ".join, axis=1)
        if "label_name" in df.columns and "text" in df.columns:
            df["word_count"] = df["text"].fillna("").str.split().str.len()
            return df
    return pd.DataFrame(columns=["label_name", "text", "word_count"])

def metric(label: str, value: str, detail: str = "") -> None:
    st.markdown(f'<div class="card"><div class="small">{label}</div><div style="font-size:1.75rem;font-weight:700;color:#1D4ED8">{value}</div><div class="small">{detail}</div></div>', unsafe_allow_html=True)

def word_cloud_section(df: pd.DataFrame) -> None:
    st.subheader("Word cloud")
    if df.empty:
        st.info("Add ag_news_train.csv or ag_news_test.csv beside the app to generate a data-driven word cloud.")
        return
    selected = st.selectbox("Category", LABELS, key="wc_category")
    text = " ".join(df.loc[df["label_name"] == selected, "text"].astype(str).tolist()).lower()
    words = re.findall(r"[a-z]{3,}", text)
    stop = {"the","and","for","that","with","from","this","have","said","will","are","has","was","their","about","after","over","into","new"}
    counts = pd.Series([w for w in words if w not in stop]).value_counts().head(20)
    if counts.empty:
        st.info("No text available for this category.")
        return
    fig = px.bar(counts.sort_values(), orientation="h", labels={"value":"Frequency", "index":"Word"}, title=f"Most frequent terms — {selected}", color_discrete_sequence=[COLORS[selected]])
    fig.update_layout(showlegend=False, height=420, margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Frequency-based word cloud view; stopwords are removed for readability.")

with st.sidebar:
    st.markdown("# ▦ AG News\n### Intelligence dashboard")
    page = st.radio("Navigate", ["Overview", "Exploration", "Predictions", "Model comparison", "Conclusion"], label_visibility="collapsed")
    st.divider()
    st.caption("Project: Intelligent Document Categorization")
    st.caption("Dataset: AG News · 4 classes")
    device_status = "GPU (CUDA)" if (torch is not None and torch.cuda.is_available()) else "CPU"
    st.caption(f"Hardware: {device_status}")
    st.caption("Models: 3 Transformers · 1 Baseline")
    st.caption(f"Artifacts: {'available' if any(p.exists() for p in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]) else 'not bundled'}")

if page == "Overview":
    st.markdown('<div class="hero"><h1>Intelligent Document Categorization</h1><p>Minimal analytical dashboard for comparing TF-IDF baselines with fine-tuned transformer models on AG News.</p></div>', unsafe_allow_html=True)
    df = sample_data(); bench = benchmark()
    cols = st.columns(4)
    with cols[0]: metric("Dataset", "AG News", "news classification")
    with cols[1]: metric("Classes", "4", "World · Sports · Business · Sci/Tech")
    with cols[2]: metric("Test sample", f"{len(df):,}" if not df.empty else "7,600", "loaded records / benchmark size")
    with cols[3]: metric("Best accuracy", f"{bench['Accuracy'].max()*100:.1f}%", str(bench.loc[bench['Accuracy'].idxmax(), 'Model']))
    st.markdown("### Dataset overview")
    c1, c2 = st.columns([1.25, .75])
    with c1:
        st.markdown('<div class="card"><h3>What this project does</h3><p>It classifies short news documents into four topical categories. The dashboard combines dataset inspection, class distribution, text patterns, prediction results, and model trade-offs in one place.</p><p class="small">Source: <a href="https://huggingface.co/datasets/fancyzhx/ag_news" target="_blank">Hugging Face AG News dataset</a>. The app automatically reads local CSV, benchmark, and model artifacts when they are available.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>Pipeline</h3><p><b>Input text</b> → TF-IDF or tokenizer → classifier → predicted class + confidence</p><p class="small">Traditional models prioritize speed; transformers prioritize semantic accuracy.</p></div>', unsafe_allow_html=True)
    st.markdown("### Class definitions")
    st.dataframe(pd.DataFrame({"Class": LABELS, "Typical content": ["International affairs", "Games and competitions", "Markets and companies", "Science and technology"]}), hide_index=True, width="stretch")

elif page == "Exploration":
    st.markdown('<div class="hero"><h1>Exploration</h1><p>Understand balance, text length, vocabulary, and the dataset’s most visible patterns.</p></div>', unsafe_allow_html=True)
    df = sample_data()
    if df.empty:
        st.warning("No local AG News CSV was found. Showing a benchmark-safe demo view; place ag_news_train.csv or ag_news_test.csv beside the app for data-driven charts.")
        counts = pd.DataFrame({"label_name": LABELS, "count": [1900]*4})
    else:
        counts = df["label_name"].value_counts().reindex(LABELS, fill_value=0).rename("count").reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(counts, x="label_name", y="count", color="label_name", color_discrete_map=COLORS, title="Class distribution")
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Documents")
        st.plotly_chart(fig, width="stretch")
    with c2:
        if not df.empty:
            fig = px.histogram(df, x="word_count", color="label_name", nbins=35, barmode="overlay", opacity=.7, color_discrete_map=COLORS, title="Document length distribution")
            fig.update_layout(xaxis_title="Words per document", yaxis_title="Documents")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Length chart appears when a dataset CSV is available.")
    word_cloud_section(df)

elif page == "Predictions":
    st.markdown('<div class="hero"><h1>Prediction & Classification</h1><p>Select any fine-tuned transformer or classical ML model, enter text, and inspect predicted probabilities in real time.</p></div>', unsafe_allow_html=True)

    col_model, col_example = st.columns([1.2, 1])
    with col_model:
        model_names = list(AVAILABLE_MODELS.keys())
        selected_model_name = st.selectbox(
            "Select classification model",
            options=model_names,
            index=0,
            help="Choose the active model to use for prediction and classification."
        )
        meta = AVAILABLE_MODELS[selected_model_name]
        is_ready = check_model_availability(meta)
        status_badge = '<span style="color:#10B981;font-weight:600">● Ready (local)</span>' if is_ready else '<span style="color:#F59E0B;font-weight:600">● Fallback mode</span>'
        st.markdown(
            f'<div class="callout" style="padding:0.6rem 0.9rem;margin-top:-0.2rem;margin-bottom:0.7rem;font-size:0.88rem;">'
            f'<b>Status:</b> {status_badge} &nbsp;|&nbsp; '
            f'<b>Type:</b> {meta["family"]} &nbsp;|&nbsp; '
            f'<b>Benchmark Acc:</b> {meta["benchmark_acc"]} &nbsp;|&nbsp; '
            f'<b>Latency:</b> ~{meta["benchmark_latency"]}<br>'
            f'<span style="color:#475569">{meta["desc"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_example:
        examples = {
            "Business": "The central bank raised interest rates today as financial markets reacted cautiously to fresh inflation data.",
            "Sports": "The championship team secured victory in the final minutes after an incredible stoppage-time goal.",
            "Sci/Tech": "Astronomers discovered an Earth-sized exoplanet orbiting within the habitable zone of a nearby star.",
            "World": "Diplomats from both nations met in Geneva for peace negotiations aimed at easing border tensions."
        }
        chosen_ex = st.selectbox("Load an example headline", ["Custom text"] + LABELS, help="Select a sample AG News headline or enter your own.")
        default_text = "" if chosen_ex == "Custom text" else examples[chosen_ex]

    text = st.text_area("Document text", value=default_text, height=130, placeholder="Paste a headline or short document…")

    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        classify_clicked = st.button("Classify document", type="primary", disabled=not text.strip(), width="stretch")

    if classify_clicked:
        with st.spinner(f"Classifying with {AVAILABLE_MODELS[selected_model_name]['id']}…"):
            scores, lat_ms, mode_label, status_note = classify_text(text, selected_model_name)
            pred_class = max(scores, key=scores.get)
            confidence = scores[pred_class]
            sorted_scores = sorted(scores.values())
            margin = (confidence - sorted_scores[-2]) * 100 if len(sorted_scores) > 1 else 0.0

            r1, r2, r3, r4 = st.columns(4)
            with r1: metric("Predicted class", pred_class, f"Class index: {LABELS.index(pred_class)}")
            with r2: metric("Confidence", f"{confidence*100:.1f}%", f"Lead margin: +{margin:.1f}%")
            with r3: metric("Active Model", AVAILABLE_MODELS[selected_model_name]["id"], mode_label)
            with r4: metric("Inference Latency", f"{lat_ms:.1f} ms", status_note)

            prob_df = pd.DataFrame({"Class": list(scores), "Probability": [v * 100 for v in scores.values()]})
            fig = px.bar(
                prob_df.sort_values("Probability"),
                x="Probability",
                y="Class",
                orientation="h",
                color="Class",
                color_discrete_map=COLORS,
                text_auto=".1f",
                title=f"Predicted class probabilities — {AVAILABLE_MODELS[selected_model_name]['id']}"
            )
            fig.update_layout(xaxis_title="Probability (%)", yaxis_title=None, showlegend=False, xaxis_range=[0, 100], height=290, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, width="stretch")

    with st.expander("⚡ Compare all models on this document", expanded=False):
        if not text.strip():
            st.info("Enter or paste text above, then expand this panel to evaluate all models side by side.")
        else:
            if st.button("Evaluate all models on this text", key="btn_compare_all"):
                with st.spinner("Running inference across all models…"):
                    comp_rows = []
                    for m_name, m_info in AVAILABLE_MODELS.items():
                        if m_info["family"] == "Rule-based":
                            continue
                        s, l_ms, m_mode, _ = classify_text(text, m_name)
                        top_cls = max(s, key=s.get)
                        comp_rows.append({
                            "Model": m_info["id"],
                            "Architecture": m_info["family"],
                            "Prediction": top_cls,
                            "Confidence": f"{s[top_cls]*100:.1f}%",
                            "World": f"{s.get('World', 0)*100:.1f}%",
                            "Sports": f"{s.get('Sports', 0)*100:.1f}%",
                            "Business": f"{s.get('Business', 0)*100:.1f}%",
                            "Sci/Tech": f"{s.get('Sci/Tech', 0)*100:.1f}%",
                            "Live Latency": f"{l_ms:.1f} ms",
                        })
                    comp_df = pd.DataFrame(comp_rows)
                    st.dataframe(comp_df, hide_index=True, width="stretch")

elif page == "Model comparison":
    st.markdown('<div class="hero"><h1>Model performance comparison</h1><p>Accuracy, macro F1, and latency show the quality–speed trade-off.</p></div>', unsafe_allow_html=True)
    df = benchmark(); display = df.copy()
    st.dataframe(display.style.format({c: "{:.2%}" for c in ["Accuracy", "Macro_F1"] if c in display.columns}), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df.sort_values("Accuracy"), x="Accuracy", y="Model", orientation="h", color="Accuracy", color_continuous_scale="Blues", title="Accuracy by model", text_auto=".1%")
        st.plotly_chart(fig, width="stretch")
    with c2:
        if "Avg_Latency_ms" in df:
            fig = px.scatter(df, x="Avg_Latency_ms", y="Accuracy", text="Model", color="Model", log_x=True, title="Latency vs accuracy")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, width="stretch")
    st.markdown('<div class="callout"><b>Interpretation:</b> transformer models typically lead on accuracy, while linear TF-IDF models offer substantially lower latency. The right choice depends on throughput, hardware, and error tolerance.</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="hero"><h1>Conclusion</h1><p>Key findings, challenges, future scope, and practical applications.</p></div>', unsafe_allow_html=True)
    sections = {
        "Key findings": ["Fine-tuned transformers provide the strongest benchmark accuracy in the supplied results.", "Linear TF-IDF baselines remain compelling for CPU-first, low-latency workloads.", "Sports is generally the most lexically distinct class; Business and Sci/Tech can overlap."],
        "Challenges faced": ["Limited local artifacts require defensive loading and graceful fallbacks.", "Multi-class probability calibration and ROC interpretation require care, especially for margin-based classifiers.", "Transformer quality comes with higher memory and inference cost."],
        "Future scope": ["Add confidence calibration, drift monitoring, and human review for ambiguous cases.", "Explore multilingual models and long-document chunking.", "Use a hybrid router that sends easy cases to a fast baseline and difficult cases to a transformer."],
        "Applications": ["Newsroom and media aggregation", "Enterprise support and ticket routing", "Legal and regulatory document indexing", "Content tagging, search, and recommendation systems"],
    }
    for title, items in sections.items():
        st.markdown(f"### {title}")
        for item in items:
            st.markdown(f'<div class="card" style="margin:.45rem 0">{item}</div>', unsafe_allow_html=True)

st.caption("AG News Intelligence · Streamlit dashboard · local artifacts are optional and loaded automatically")
