import os
import gradio as gr
import torch
import config
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import numpy as np

def get_available_models():
    """Scan models directory for trained transformer models."""
    available = []
    if config.MODELS_DIR.exists():
        for d in config.MODELS_DIR.iterdir():
            if d.is_dir() and "best_model" in d.name:
                available.append(d.name)
            # Some models might just be saved by their name
            elif d.is_dir() and d.name in [f"{dataset}_{model}" for dataset in config.DATASETS for model in config.TRANSFORMER_MODELS.keys()]:
                available.append(d.name)
    return available

def load_model(dataset_name, model_name):
    """Load a specific model and tokenizer."""
    # train_transformer.py saves as {ModelKey}_{dataset_name}
    model_dir = config.MODELS_DIR / f"{model_name}_{dataset_name}"
    
    if not model_dir.exists():
        # Try alternative naming: {dataset}_{model}
        model_dir = config.MODELS_DIR / f"{dataset_name}_{model_name}"
        
    if not model_dir.exists():
        return None, None
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(config.DEVICE)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict(text, dataset_name, model_name):
    if not text.strip():
        return "Please enter some text.", {}
        
    model, tokenizer = load_model(dataset_name, model_name)
    if model is None:
        return f"Model '{model_name}' for dataset '{dataset_name}' not found. Please train it first.", {}
        
    labels = config.DATASETS[dataset_name]["label_names"]
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=config.MAX_SEQ_LENGTH,
        padding=True
    ).to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        
    pred_idx = np.argmax(probs)
    pred_label = labels[pred_idx]
    confidence = probs[pred_idx]
    
    prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    
    return f"{pred_label} (Confidence: {confidence:.2%})", prob_dict

def update_model_choices(dataset_name):
    # Depending on dataset, the models we might expect
    expected_models = list(config.TRANSFORMER_MODELS.keys())
    return gr.Dropdown(choices=expected_models, value="DistilBERT")

def get_examples():
    return [
        ["The stock market saw a massive drop today due to inflation fears.", "ag_news", "DistilBERT"],
        ["The football team won the championship after a thrilling penalty shootout.", "ag_news", "DistilBERT"],
        ["Scientists have discovered a new exoplanet that could potentially support life.", "ag_news", "DistilBERT"],
        ["The president announced new trade agreements with neighboring countries.", "ag_news", "DistilBERT"],
    ]

def create_app():
    with gr.Blocks(title="Document Categorization Demo") as demo:
        gr.Markdown("# 📄 Intelligent Document Categorization")
        gr.Markdown("Test out fine-tuned Transformer models for document classification.")
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    lines=10, 
                    placeholder="Enter or paste document text here...", 
                    label="Document Text"
                )
                
                with gr.Row():
                    dataset_dropdown = gr.Dropdown(
                        choices=list(config.DATASETS.keys()), 
                        value="ag_news", 
                        label="Dataset (Label Space)"
                    )
                    model_dropdown = gr.Dropdown(
                        choices=list(config.TRANSFORMER_MODELS.keys()), 
                        value="DistilBERT", 
                        label="Model"
                    )
                    
                dataset_dropdown.change(
                    fn=update_model_choices, 
                    inputs=dataset_dropdown, 
                    outputs=model_dropdown
                )
                
                submit_btn = gr.Button("Categorize Document", variant="primary")
                
            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Prediction")
                output_label = gr.Label(label="Class Probabilities")
                
        gr.Examples(
            examples=get_examples(),
            inputs=[text_input, dataset_dropdown, model_dropdown],
        )
        
        submit_btn.click(
            fn=predict, 
            inputs=[text_input, dataset_dropdown, model_dropdown], 
            outputs=[output_text, output_label]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="127.0.0.1", share=False)
