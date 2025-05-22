from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: float

class KeywordsResponse(BaseModel):
    keywords: list[str]