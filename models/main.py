from fastapi import FastAPI
from models import TextRequest, PredictionResponse, KeywordsResponse
from keyword_extractor import extract_keywords
from noun_filter import filter_nouns
from predictor import load_inference_components, predict_rating
import torch

app = FastAPI()

MODEL_CHECKPOINT_PATH = "./best_model_val_acc_phase2_CrossEntropyLoss.pt"
TOKENIZER_NAME = "cointegrated/rubert-tiny2"
LABEL_ENCODER_PATH = "./label_encoder.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model, loaded_tokenizer, loaded_label_encoder = load_inference_components(
        MODEL_CHECKPOINT_PATH,
        TOKENIZER_NAME,
        LABEL_ENCODER_PATH,
        device=DEVICE
    )

@app.post("/keywords", response_model=KeywordsResponse)
async def process_text(request: TextRequest):
    text = request.text
    
    # Step 1: Extract keywords
    keywords = extract_keywords(text)
    
    # Step 2: Filter nouns
    nouns = filter_nouns(text, keywords)
    
    return {
        "keywords": nouns,
    }

@app.post("/predict", response_model=PredictionResponse)
async def process_text(request: TextRequest):
    text = request.text
    
    prediction = predict_rating(text, loaded_model, loaded_tokenizer, loaded_label_encoder, device=DEVICE)
    
    return {
        "prediction": prediction
    }
