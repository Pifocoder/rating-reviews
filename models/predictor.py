import os
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from typing import Dict
import joblib

np.random.seed(42)
torch.manual_seed(42)

class ModelForClassification(torch.nn.Module):
    def __init__(self, model_path: str, config: Dict, freeze_bert_layers=False, use_classifier=False):
        super().__init__()
        self.model_name = model_path
        self.config = config
        self.n_classes = config['num_classes']
        self.dropout_rate = config['dropout_rate']
        self.bert = AutoModel.from_pretrained(model_path)
        if model_path == "cointegrated/rubert-tiny2":
            for name, param in self.bert.named_parameters():
              if 'encoder.layer.2' not in name and 'encoder.layer.1' not in name:
                  param.requires_grad = False
        if model_path == "RussianNLP/ruRoBERTa-large-rucola":
            for name, param in self.bert.named_parameters():
              if 'encoder.layer.22' not in name and 'encoder.layer.23' not in name:
                  param.requires_grad = False
        if freeze_bert_layers:
            for param in self.bert.parameters():
                param.requires_grad = False
        total_params = sum(p.numel() for p in self.bert.parameters())
        trainable_params = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        print(f"Всего параметров: {total_params:,}")
        print(f"Обучаемых параметров: {trainable_params:,} ({trainable_params/total_params:.1%})")
        bert_output_size = self.bert.config.hidden_size
        print(bert_output_size)
        self.use_classifier = use_classifier
        
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(bert_output_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 192),
            torch.nn.LayerNorm(192),
        )
        
        for layer in self.projection_head:
          if isinstance(layer, torch.nn.Linear):
              torch.nn.init.kaiming_normal_(layer.weight)
        if self.use_classifier:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(192, 128),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Linear(128, self.n_classes),
            )
            for layer in self.classifier:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        projection = self.projection_head(hidden_state)
        # out = torch.nn.functional.normalize(projection, p=2, dim=1)
        if self.use_classifier:
            return self.classifier(projection)
        return projection
    

def load_inference_components(
    checkpoint_path: str,
    tokenizer_name_or_path: str,
    label_encoder_path: str = None,
    device: str = None
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading inference components to device: {device}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return None, None, None
    
    model_config_from_ckpt = checkpoint.get('config')
    base_bert_model_name = checkpoint.get('model_name')

    if model_config_from_ckpt is None:
        print("Error: Model configuration not found in checkpoint. Please ensure 'config' or 'model_config' is saved.")
        return None, None, None


    model = ModelForClassification(
        model_path=base_bert_model_name, # "cointegrated/rubert-tiny2"
        config=model_config_from_ckpt,
        use_classifier=True
    )
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        print("Error: 'model_state_dict' not found in checkpoint.")
        return None, None, None
    except RuntimeError as e:
        print(f"Error loading state_dict, possibly due to mismatched keys or model architecture: {e}")
        print("Ensure the loaded model architecture matches the saved state_dict.")
        return None, None, None
        
    model.to(device)
    model.eval()
    print(f"Model '{base_bert_model_name}' loaded successfully from {checkpoint_path}.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        print(f"Tokenizer '{tokenizer_name_or_path}' loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name_or_path}': {e}")
        return model, None, None

    label_encoder = None
    if label_encoder_path:
        try:
            label_encoder = joblib.load(label_encoder_path)
            print(f"LabelEncoder loaded successfully from {label_encoder_path}.")
            if not hasattr(label_encoder, 'classes_'):
                print(f"Warning: Loaded LabelEncoder from {label_encoder_path} does not have 'classes_' attribute. Inverse transform might fail.")
        except FileNotFoundError:
            print(f"Warning: LabelEncoder file not found at {label_encoder_path}. Predictions will be numeric.")
            return model, tokenizer, label_encoder
        except Exception as e:
            print(f"Warning: Error loading LabelEncoder from {label_encoder_path}: {e}. Predictions will be numeric.")
            return model, tokenizer, label_encoder
            
    return model, tokenizer, label_encoder

def predict_rating(
    text: str,
    model: ModelForClassification,
    tokenizer: AutoTokenizer,
    label_encoder: object = None,
    max_len: int = 512,
    device: str = None
):

    if model is None or tokenizer is None:
        return "Error: Model or Tokenizer not loaded."

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    encoded_review = tokenizer.encode_plus(
        text,
        max_length=max_len,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_class_idx = torch.argmax(logits, dim=1).item()

    if label_encoder is not None and hasattr(label_encoder, 'classes_'):
        try:
            predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
        except IndexError:
            print(f"Warning: Predicted class index {predicted_class_idx} is out of bounds for LabelEncoder classes: {label_encoder.classes_}")
            predicted_label = f"Raw_Index_{predicted_class_idx}"
        except Exception as e:
            print(f"Error during inverse_transform with LabelEncoder: {e}")
            predicted_label = f"Raw_Index_{predicted_class_idx}"

    else:
        predicted_label = predicted_class_idx

    # return predicted_label, probabilities
    return predicted_label
