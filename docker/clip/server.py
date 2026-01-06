#!/usr/bin/env python3
"""
CLIP embedding server using HuggingFace Transformers.
Provides OpenAI-compatible /v1/embeddings endpoint for both text and images.
"""
import base64
import io
import logging
import os
import time
from typing import List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import CLIPModel, CLIPProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "openai/clip-vit-large-patch14")

# Require CUDA - this server is designed for DGX Spark
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required but not available")

DEVICE = "cuda"

app = FastAPI(title="CLIP Embedding API")

# Global model instances
model: Optional[CLIPModel] = None
processor: Optional[CLIPProcessor] = None


class EmbeddingInput(BaseModel):
    """Single embedding input - can be text or image."""
    text: Optional[str] = None
    image: Optional[str] = None  # Base64 encoded image


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    model: str = Field(..., description="Model name")
    input: Union[str, List[str], List[EmbeddingInput], EmbeddingInput] = Field(
        ..., description="Text string, list of texts, or list of {text/image} objects"
    )
    encoding_format: str = Field("float", description="Encoding format: 'float' or 'base64'")


class EmbeddingData(BaseModel):
    """Single embedding result."""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage info."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode a base64 string to PIL Image."""
    # Handle data URL format
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]

    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def get_text_embedding(texts: List[str]) -> torch.Tensor:
    """Get embeddings for text inputs."""
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    return text_features


def get_image_embedding(images: List[Image.Image]) -> torch.Tensor:
    """Get embeddings for image inputs."""
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    return image_features


@app.on_event("startup")
async def load_model():
    """Load the CLIP model on startup."""
    global model, processor

    logger.info(f"Loading CLIP model: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")

    processor = CLIPProcessor.from_pretrained(MODEL_PATH)
    model = CLIPModel.from_pretrained(MODEL_PATH)
    model = model.to(DEVICE)
    model.eval()

    logger.info("Model loaded successfully")
    logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "device": DEVICE,
        "model_loaded": model is not None
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_PATH,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai"
            }
        ]
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for text or images (OpenAI-compatible endpoint).

    Supports:
    - Single text string: {"input": "hello world"}
    - List of texts: {"input": ["hello", "world"]}
    - Single image: {"input": [{"image": "<base64>"}]}
    - Mixed: {"input": [{"text": "hello"}, {"image": "<base64>"}]}
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    embeddings = []
    token_count = 0

    # Normalize input to list of items
    inputs = request.input
    if isinstance(inputs, str):
        inputs = [{"text": inputs}]
    elif isinstance(inputs, EmbeddingInput):
        inputs = [inputs.model_dump()]
    elif isinstance(inputs, list):
        normalized = []
        for item in inputs:
            if isinstance(item, str):
                normalized.append({"text": item})
            elif isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, EmbeddingInput):
                normalized.append(item.model_dump())
            else:
                raise HTTPException(status_code=400, detail=f"Invalid input type: {type(item)}")
        inputs = normalized

    # Process each input
    for idx, item in enumerate(inputs):
        if isinstance(item, dict):
            text = item.get("text")
            image = item.get("image")
        else:
            text = None
            image = None

        if image:
            logger.debug(f"Processing image input {idx}")
            pil_image = decode_base64_image(image)
            embedding = get_image_embedding([pil_image])
            token_count += 257  # CLIP uses 257 tokens for images
        elif text:
            logger.debug(f"Processing text input {idx}: {text[:50]}...")
            embedding = get_text_embedding([text])
            token_count += len(text.split())
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Input {idx} must have either 'text' or 'image' field"
            )

        embeddings.append(EmbeddingData(
            embedding=embedding[0].cpu().tolist(),
            index=idx
        ))

    elapsed = time.time() - start_time
    logger.info(f"Generated {len(embeddings)} embedding(s) in {elapsed:.3f}s")

    return EmbeddingResponse(
        data=embeddings,
        model=request.model,
        usage=EmbeddingUsage(
            prompt_tokens=token_count,
            total_tokens=token_count
        )
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
