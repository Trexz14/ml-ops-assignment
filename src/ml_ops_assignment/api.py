from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from typing import Any
from ml_ops_assignment.model import load_model, load_config, TextClassificationModel

# Dictionary to store model and tokenizer
model_assets: dict[str, Any] = {}


class PredictRequest(BaseModel):
    text: str  # What we expect input to be


class PredictResponse(BaseModel):
    # expected types of output
    text: str  # inputtet text
    label: int  # predicted label
    class_name: str  # predicted class name
    status_code: int  # HTTP status code


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Loads the model and tokenizer on startup and cleans up on shutdown.
    """
    # Define paths
    checkpoint_path = Path("models/model_final.pt")
    config_path = Path("configs/experiments/default.yaml")

    # Load configuration
    config = load_config(config_path)
    model_assets["config"] = config

    print(f"Loading model from {checkpoint_path}...")

    # Check if checkpoint exists, otherwise load dummy for initial development/testing
    if checkpoint_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(checkpoint_path, config_path=config_path, device=device)
        model_assets["model"] = model
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Running in mock mode.")
        model_assets["model"] = None

    # Load tokenizer from config
    tokenizer_name = config["model"]["model_name"]
    print(f"Loading tokenizer {tokenizer_name}...")
    try:
        model_assets["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"WARNING: Failed to load tokenizer {tokenizer_name}: {e}")
        model_assets["tokenizer"] = None

    yield
    # Cleanup
    model_assets.clear()


app = FastAPI(
    title="MLOps Text Classification API",
    description="API for predicting text quality on a 0-2 scale.",
    version="0.0.1",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict the class of the input text.
    """
    model = model_assets.get("model")
    tokenizer = model_assets.get("tokenizer")

    if model is None or tokenizer is None:
        # Mock prediction for development/testing when assets are missing
        return PredictResponse(
            text=request.text,
            label=0,
            class_name="Elementary",
            status_code=HTTPStatus.OK,
        )

    # Tokenize
    config = model_assets.get("config")
    max_length = config["model"].get("max_length", 512) if config else 512

    # Help mypy understand types
    assert tokenizer is not None
    assert model is not None

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        prediction = int(torch.argmax(logits, dim=-1).item())

    # Domain mapping (0: Elementary, 1: Intermediate, 2: Advance)
    class_names = ["Elementary", "Intermediate", "Advance"]

    return PredictResponse(
        text=request.text,
        label=prediction,
        class_name=class_names[prediction] if prediction < len(class_names) else "unknown",
        status_code=HTTPStatus.OK,
    )
