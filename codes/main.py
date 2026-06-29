from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import numpy as np

from utils import encode_text, pad_sequence_to_length, MODEL_REGISTRY, ATTENTION_REGISTRY
from model_loader import get_model, WEIGHT_PATHS
from config import MAX_LEN

app = FastAPI(title="Sentiment Analysis API")

# ----------------------------
# CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Device  (CPU-only — ONNX Runtime handles its own execution provider)
# ----------------------------
device = "cpu"

# ----------------------------
# Load shared assets once on startup
# ----------------------------
word2idx = torch.load("word2idx.pt", map_location="cpu", weights_only=True)
idx2word = {idx: word for word, idx in word2idx.items()}


# ----------------------------
# Request / Response Schemas
# ----------------------------
class PredictRequest(BaseModel):
    text: str
    model_name: Optional[str] = "vanilla_lstm"
    attention_name: Optional[str] = "luong_general"


class PredictResponse(BaseModel):
    model_name: str
    attention_name: str
    prediction: int
    sentiment: str
    confidence: float
    tokens: list[str]
    attention_weights: list[float]


# ----------------------------
# Inference helper
# ----------------------------
def predict_with_attention(
    text: str,
    model_name: str = "vanilla_lstm",
    attention_name: str = "luong_general",
    max_len: int = MAX_LEN,
) -> dict:

    # Validate names
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. "
                   f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    if attention_name not in ATTENTION_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown attention '{attention_name}'. "
                   f"Choose from: {list(ATTENTION_REGISTRY.keys())}"
        )

    # Load (or retrieve cached) ONNX session
    try:
        session = get_model(model_name, attention_name)  # embedding baked into ONNX graph
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Encode & pad
    tokens = encode_text(text, word2idx)
    padded_tokens = pad_sequence_to_length(tokens, max_len, word2idx["<PAD>"])

    input_np = np.array([padded_tokens], dtype=np.int64)   # shape [1, max_len]

    logits_np, attn_np = session.run(None, {"input": input_np})

    # logits_np: [1, 2] float32  —  attn_np: [1, max_len] float32
    probs = np.exp(logits_np) / np.exp(logits_np).sum(axis=1, keepdims=True)  # softmax
    prediction = int(np.argmax(probs, axis=1)[0])
    confidence = float(probs[0][prediction])

    # Strip padding from token list
    words = [
        idx2word.get(idx, "<UNK>")
        for idx in padded_tokens
        if idx != word2idx["<PAD>"]
    ]
    attn_weights = attn_np[0, :len(words)].tolist()  # strip padding, numpy → list

    return {
        "model_name": model_name,
        "attention_name": attention_name,
        "prediction": prediction,
        "sentiment": "Positive" if prediction == 1 else "Negative",
        "confidence": round(confidence, 4),
        "tokens": words,
        "attention_weights": attn_weights,
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API running"}


@app.get("/models")
def list_models():
    """List all available (model, attention) combinations that have trained weights."""
    return {
        "available_combinations": [
            {"model_name": m, "attention_name": a}
            for m, a in WEIGHT_PATHS.keys()
        ],
        "all_models": list(MODEL_REGISTRY.keys()),
        "all_attention": list(ATTENTION_REGISTRY.keys()),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return predict_with_attention(
        text=request.text,
        model_name=request.model_name,
        attention_name=request.attention_name,
    )