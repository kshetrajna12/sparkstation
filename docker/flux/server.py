#!/usr/bin/env python3
"""
Simple FLUX.1-dev image generation server using Diffusers.
Provides OpenAI-compatible /v1/images/generations endpoint.
"""
import base64
import io
import logging
import os
import time
from typing import Optional

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

app = FastAPI(title="FLUX Image Generation API")

# Global pipeline instance
pipeline: Optional[FluxPipeline] = None


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the image to generate")
    model: Optional[str] = Field(None, description="Model name (ignored, always uses FLUX.1-dev)")
    n: int = Field(1, ge=1, le=4, description="Number of images to generate (1-4)")
    size: str = Field("1024x1024", description="Image size (e.g., '1024x1024')")
    response_format: str = Field("b64_json", description="Response format: 'b64_json' or 'url'")
    quality: str = Field("standard", description="Quality: 'standard' or 'hd' (ignored)")
    steps: int = Field(50, ge=1, le=100, description="Number of inference steps")
    guidance_scale: float = Field(3.5, ge=1.0, le=20.0, description="Guidance scale")


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]


@app.on_event("startup")
async def load_model():
    """Load the FLUX model on startup."""
    global pipeline

    logger.info(f"Loading FLUX model: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}, Dtype: {DTYPE}")

    try:
        pipeline = FluxPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=DTYPE,
            use_auth_token=os.getenv("HF_TOKEN")
        )
        pipeline = pipeline.to(DEVICE)

        logger.info("Model loaded successfully")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "device": DEVICE,
        "model_loaded": pipeline is not None
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "black-forest-labs/FLUX.1-dev",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "black-forest-labs"
            }
        ]
    }


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_images(request: ImageGenerationRequest):
    """
    Generate images from text prompt (OpenAI-compatible endpoint).
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Generating {request.n} image(s) for prompt: {request.prompt[:100]}...")

    try:
        # Parse image size
        width, height = map(int, request.size.split("x"))

        # Generate images
        start_time = time.time()

        images = []
        for i in range(request.n):
            logger.info(f"Generating image {i+1}/{request.n}...")

            result = pipeline(
                prompt=request.prompt,
                height=height,
                width=width,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                generator=torch.Generator(device=DEVICE).manual_seed(int(time.time() * 1000) + i)
            )

            images.append(result.images[0])

        elapsed = time.time() - start_time
        logger.info(f"Generated {request.n} image(s) in {elapsed:.2f}s ({elapsed/request.n:.2f}s per image)")

        # Convert images to base64
        data = []
        for image in images:
            if request.response_format == "b64_json":
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                data.append(ImageData(b64_json=b64_string))
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Only 'b64_json' response format is supported"
                )

        return ImageGenerationResponse(
            created=int(time.time()),
            data=data
        )

    except Exception as e:
        logger.error(f"Error generating images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
