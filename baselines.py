import argparse
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

import config
import utils

# Mapping from config string to actual sklearn classes
MODEL_CLASSES = {
    "LogisticRegression": LogisticRegression,
    "MultinomialNB": MultinomialNB,
    "LinearSVC": LinearSVC,
}

def train_and_evaluate(dataset_name: str):
    print(f"\n{'='*80}")
    print(f"Running Traditional Baselines for Dataset: {dataset_name}")
    print(f"{'='*80}\n")

    dataset_config = config.DATASETS[dataset_name]
    label_names = dataset_config["label_names"]
    
    # 1. Load processed CSVs
    train_path = config.DATA_DIR / f"{dataset_name}_train.csv"
    val_path = config.DATA_DIR / f"{dataset_name}_val.csv"
    test_path = config.DATA_DIR / f"{dataset_name}_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        print(f"Error: Processed data files not found for {dataset_name}. Please run data_prep.py first.")
        return

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train_text = train_df[dataset_config["text_column"]].fillna("").tolist()
    y_train = train_df[dataset_config["label_column"]].tolist()
    
    X_test_text = test_df[dataset_config["text_column"]].fillna("").tolist()
    y_test = test_df[dataset_config["label_column"]].tolist()

    # 2. Create TF-IDF vectorizer
    print("\nFitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES, 
        ngram_range=config.TFIDF_NGRAM_RANGE
    )
    
    with utils.Timer("TF-IDF Vectorizer Fitting and Transform"):
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
    
    # Save vectorizer
    vectorizer_path = config.MODELS_DIR / f"{dataset_name}_tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"  💾 Saved TF-IDF vectorizer to {vectorizer_path}")

    all_results = {}
    accuracy_scores = {}
    f1_scores = {}

    # 3 & 4. Train and evaluate models
    for model_name, model_info in config.BASELINE_MODELS.items():
        print(f"\n--- Training {model_name} ---")
        
        model_class_name = model_info["model"]
        model_params = model_info["params"]
        
        if model_class_name not in MODEL_CLASSES:
            print(f"Warning: Model class {model_class_name} not supported. Skipping.")
            continue
            
        model = MODEL_CLASSES[model_class_name](**model_params)
        
        with utils.Timer(f"Training {model_name}"):
            model.fit(X_train, y_train)
            
        # 5. Save trained model
        safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        model_save_path = config.MODELS_DIR / f"{dataset_name}_{safe_model_name}.joblib"
        joblib.dump(model, model_save_path)
        print(f"  💾 Saved model to {model_save_path}")

        # Evaluation
        print("Evaluating...")
        y_pred = model.predict(X_test)
        
        # Compute and print metrics
        metrics = utils.compute_metrics(y_test, y_pred, label_names)
        utils.print_metrics(metrics, model_name)
        
        print("\nClassification Report (sklearn):")
        print(classification_report(y_test, y_pred, target_names=label_names))

        # Confusion matrix
        cm_path = config.FIGURES_DIR / f"{dataset_name}_{safe_model_name}_cm.png"
        utils.plot_confusion_matrix(
            y_test, 
            y_pred, 
            label_names, 
            title=f"Confusion Matrix - {model_name} ({dataset_name})",
            save_path=cm_path
        )
        
        # Measure inference latency
        def predict_fn(text: str):
            vec = vectorizer.transform([text])
            return model.predict(vec)[0]
            
        latency_stats = utils.measure_inference_latency(predict_fn, X_test_text)
        print(f"  ⏱ Inference Latency: Avg={latency_stats['avg_latency_ms']:.2f}ms, p95={latency_stats['p95_latency_ms']:.2f}ms")
        
        all_results[model_name] = {
            "metrics": metrics,
            "latency": latency_stats
        }
        
        accuracy_scores[model_name] = metrics["accuracy"]
        f1_scores[model_name] = metrics["macro_f1"]

    # 7. Save all results
    results_path = config.RESULTS_DIR / f"{dataset_name}_baselines_results.json"
    utils.save_results(all_results, results_path)
    
    # 8. Generate comparison charts
    acc_chart_path = config.FIGURES_DIR / f"{dataset_name}_baselines_accuracy.png"
    utils.plot_bar_comparison(
        accuracy_scores,
        title=f"Baseline Models Accuracy ({dataset_name})",
        xlabel="Model",
        ylabel="Accuracy",
        save_path=acc_chart_path
    )
    
    f1_chart_path = config.FIGURES_DIR / f"{dataset_name}_baselines_f1.png"
    utils.plot_bar_comparison(
        f1_scores,
        title=f"Baseline Models Macro F1 ({dataset_name})",
        xlabel="Model",
        ylabel="Macro F1 Score",
        save_path=f1_chart_path
    )


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate traditional ML baselines")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["ag_news"], 
        default="ag_news",
        help="Dataset to train baselines on"
    )
    args = parser.parse_args()
    train_and_evaluate("ag_news")

if __name__ == "__main__":
    main()
