import argparse
import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    EvalPrediction
)

import config
import utils

def compute_metrics_hf(eval_pred: EvalPrediction):
    """Callback for Hugging Face Trainer to compute metrics during evaluation."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    
    # We use a dummy label names list here since we only need aggregate metrics
    dummy_names = [str(i) for i in range(predictions.max() + 1)]
    metrics = utils.compute_metrics(labels, predictions, dummy_names)
    
    return {
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1']
    }

def run_fine_tuning(model_key: str, dataset_key: str):
    """Fine-tune a transformer model on a dataset."""
    print(f"\n{'='*60}")
    print(f"Fine-tuning {model_key} on {dataset_key}")
    print(f"{'='*60}")
    
    # Clean cache
    torch.cuda.empty_cache()
    
    model_name_or_path = config.TRANSFORMER_MODELS[model_key]
    dataset_cfg = config.DATASETS[dataset_key]
    num_classes = dataset_cfg["num_classes"]
    label_names = dataset_cfg["label_names"]
    
    # 1. Load Data
    print("Loading data...")
    train_df = pd.read_csv(config.DATA_DIR / f"{dataset_key}_train.csv")
    val_df = pd.read_csv(config.DATA_DIR / f"{dataset_key}_val.csv")
    test_df = pd.read_csv(config.DATA_DIR / f"{dataset_key}_test.csv")
    
    # Subsample if necessary (e.g. AG News)
    if dataset_key == "ag_news":
        train_df = train_df.head(config.AG_NEWS_TRAIN_SUBSET)
        val_df = val_df.head(config.AG_NEWS_VAL_SUBSET)
        test_df = test_df.head(config.AG_NEWS_TEST_SUBSET)
    
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)
    
    # 2. Tokenization
    print(f"Tokenizing with {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH
        )
    
    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized = val_ds.map(tokenize_function, batched=True)
    test_tokenized = test_ds.map(tokenize_function, batched=True)
    
    # Ensure correct column formats
    train_tokenized = train_tokenized.rename_column("label", "labels")
    val_tokenized = val_tokenized.rename_column("label", "labels")
    test_tokenized = test_tokenized.rename_column("label", "labels")
    
    train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # 3. Model Setup
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_classes
    )
    
    # 4. Training Arguments
    output_dir = config.MODELS_DIR / f"{model_key}_{dataset_key}"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        fp16=config.FP16,
        warmup_steps=100,
        max_grad_norm=config.MAX_GRAD_NORM,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        dataloader_num_workers=config.NUM_WORKERS,
        seed=config.RANDOM_SEED,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics_hf
    )
    
    # 5. Training
    print("Starting training...")
    with utils.Timer(f"Training {model_key}"):
        trainer.train()
    
    # Save the best model
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Plot training history
    log_history = trainer.state.log_history
    train_losses = [log['loss'] for log in log_history if 'loss' in log]
    val_losses = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
    
    if train_losses:
        # Plot train loss curve (per step) — don't pass val_losses since they have different x-axis
        utils.plot_training_history(
            train_losses,
            val_losses=None,
            title=f"Training Loss: {model_key} on {dataset_key}",
            save_path=config.FIGURES_DIR / f"train_loss_{model_key}_{dataset_key}.{config.FIGURE_FORMAT}"
        )
    if val_losses:
        utils.plot_training_history(
            val_losses,
            val_losses=None,
            title=f"Validation Loss: {model_key} on {dataset_key} (per epoch)",
            save_path=config.FIGURES_DIR / f"val_loss_{model_key}_{dataset_key}.{config.FIGURE_FORMAT}"
        )
    
    # 6. Evaluation
    print("Evaluating on test set...")
    predictions_output = trainer.predict(test_tokenized)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids
    
    metrics = utils.compute_metrics(labels, preds, label_names)
    utils.print_metrics(metrics, f"{model_key} - {dataset_key}")
    
    # Confusion Matrix
    utils.plot_confusion_matrix(
        labels, preds, label_names,
        title=f"Confusion Matrix: {model_key} on {dataset_key}",
        save_path=config.FIGURES_DIR / f"cm_{model_key}_{dataset_key}.{config.FIGURE_FORMAT}"
    )
    
    # 7. Inference Latency
    print("Measuring inference latency...")
    def predict_fn(text):
        inputs = tokenizer(text, return_tensors="pt", max_length=config.MAX_SEQ_LENGTH, truncation=True, padding="max_length").to(config.DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    
    latency_stats = utils.measure_inference_latency(predict_fn, test_df["text"].tolist())
    metrics["latency"] = latency_stats
    
    # Save results
    results_path = config.RESULTS_DIR / f"results_{model_key}_{dataset_key}.json"
    utils.save_results(metrics, results_path)
    
    print(f"Finished {model_key} on {dataset_key}\n")

def run_zero_shot(dataset_key: str):
    """Run zero-shot classification using BART-MNLI."""
    print(f"\n{'='*60}")
    print(f"Zero-shot classification on {dataset_key}")
    print(f"{'='*60}")
    
    torch.cuda.empty_cache()
    
    dataset_cfg = config.DATASETS[dataset_key]
    label_names = dataset_cfg["label_names"]
    num_classes = dataset_cfg["num_classes"]
    
    print("Loading data...")
    test_df = pd.read_csv(config.DATA_DIR / f"{dataset_key}_test.csv")
    
    # Subsample test set for zero-shot (it's slow)
    num_samples = config.BENCHMARK_NUM_SAMPLES
    test_df = test_df.head(num_samples)
    
    print(f"Initializing zero-shot pipeline with {config.ZERO_SHOT_MODEL}...")
    device_id = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model=config.ZERO_SHOT_MODEL, device=device_id)
    
    labels = test_df["label"].tolist()
    texts = test_df["text"].tolist()
    preds = []
    
    print(f"Running inference on {len(texts)} samples...")
    with utils.Timer("Zero-shot Inference"):
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(texts)} samples...")
            
            # Predict
            result = classifier(text, candidate_labels=label_names)
            
            # Get top predicted label and its index
            top_label_str = result["labels"][0]
            pred_idx = label_names.index(top_label_str)
            preds.append(pred_idx)
            
    # Compute metrics
    metrics = utils.compute_metrics(labels, preds, label_names)
    utils.print_metrics(metrics, f"Zero-Shot - {dataset_key}")
    
    # Confusion Matrix
    utils.plot_confusion_matrix(
        labels, preds, label_names,
        title=f"Confusion Matrix: Zero-Shot on {dataset_key}",
        save_path=config.FIGURES_DIR / f"cm_zero-shot_{dataset_key}.{config.FIGURE_FORMAT}"
    )
    
    # Inference Latency
    print("Measuring inference latency...")
    def predict_fn(text):
        _ = classifier(text, candidate_labels=label_names)
        
    latency_stats = utils.measure_inference_latency(predict_fn, texts, num_samples=min(100, len(texts)), warmup_runs=2)
    metrics["latency"] = latency_stats
    
    # Save results
    results_path = config.RESULTS_DIR / f"results_zero-shot_{dataset_key}.json"
    utils.save_results(metrics, results_path)
    print(f"Finished Zero-Shot on {dataset_key}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer models for document categorization.")
    parser.add_argument("--model", type=str, default="all", choices=["bert", "distilbert", "roberta", "zero-shot", "all"],
                        help="Model to run.")
    parser.add_argument("--dataset", type=str, default="ag_news", choices=["ag_news"],
                        help="Dataset to use.")
    
    args = parser.parse_args()
    
    models_to_run = []
    if args.model == "all":
        models_to_run = ["BERT", "DistilBERT", "RoBERTa", "zero-shot"]
    else:
        # Match case for config dictionary keys
        if args.model == "bert": models_to_run.append("BERT")
        elif args.model == "distilbert": models_to_run.append("DistilBERT")
        elif args.model == "roberta": models_to_run.append("RoBERTa")
        elif args.model == "zero-shot": models_to_run.append("zero-shot")
        
    for model_key in models_to_run:
        if model_key == "zero-shot":
            run_zero_shot("ag_news")
        else:
            run_fine_tuning(model_key, "ag_news")
