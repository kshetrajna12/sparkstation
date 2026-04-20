#!/usr/bin/env python3
"""
Face detection + recognition server using InsightFace.

Pipeline:
  1. SCRFD — detect faces with bounding boxes + landmarks
  2. ArcFace — extract 512-dim face embeddings
  3. Gallery matching — cosine similarity against known identities

Provides custom /v1/detections endpoint for the OpenClaw WhatsApp agent.
"""
import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GALLERY_PATH = os.getenv("GALLERY_PATH", "/gallery/gallery.json")
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.35"))
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.4"))

app = FastAPI(title="Face Recognition API")

# Global model instance
face_app: Optional[FaceAnalysis] = None

# Gallery: list of {"name": str, "embeddings": list[list[float]]}
gallery: list[dict] = []
gallery_matrix: Optional[np.ndarray] = None  # (N, 512) precomputed for fast matching
gallery_names: list[str] = []  # name per row in gallery_matrix


# ── Request/Response models ─────────────────────────────────────────────────

class DetectionRequest(BaseModel):
    """Request to detect and recognize faces in an image."""
    image: str = Field(..., description="Base64-encoded image (with or without data: prefix)")
    detection_threshold: float = Field(DETECTION_THRESHOLD, ge=0.0, le=1.0)
    recognition_threshold: float = Field(RECOGNITION_THRESHOLD, ge=0.0, le=1.0)


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class FaceMatch(BaseModel):
    name: str
    confidence: float


class FaceDetection(BaseModel):
    category: str = "person"
    confidence: float
    bbox: BoundingBox
    match: Optional[FaceMatch] = None
    age: Optional[int] = None
    gender: Optional[str] = None


class DetectionResponse(BaseModel):
    num_detections: int
    num_recognized: int
    num_unknown: int
    detections: list[FaceDetection]
    processing_time_s: float


# ── Image decoding ──────────────────────────────────────────────────────────

def decode_image(b64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV BGR array."""
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]
    image_data = base64.b64decode(b64_string)
    pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── Gallery matching ────────────────────────────────────────────────────────

def match_face(embedding: np.ndarray, threshold: float) -> Optional[FaceMatch]:
    """Match a face embedding against the gallery using cosine similarity."""
    if gallery_matrix is None or len(gallery_names) == 0:
        return None

    # Cosine similarity (embeddings are already L2-normalized by InsightFace)
    sims = gallery_matrix @ embedding
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim >= threshold:
        return FaceMatch(name=gallery_names[best_idx], confidence=round(best_sim, 4))
    return None


def load_gallery():
    """Load precomputed gallery from JSON."""
    global gallery, gallery_matrix, gallery_names

    path = Path(GALLERY_PATH)
    if not path.exists():
        logger.warning(f"Gallery not found at {GALLERY_PATH} — recognition disabled until gallery is built")
        return

    with open(path) as f:
        gallery = json.load(f)

    rows = []
    names = []
    for person in gallery:
        name = person["name"]
        for emb in person["embeddings"]:
            vec = np.array(emb, dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            rows.append(vec)
            names.append(name)

    if rows:
        gallery_matrix = np.stack(rows)
        gallery_names = names
        unique = set(names)
        logger.info(f"Gallery loaded: {len(unique)} people, {len(rows)} total embeddings")
    else:
        logger.warning("Gallery file exists but contains no embeddings")


# ── Model loading ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_models():
    """Load InsightFace models on startup."""
    global face_app

    logger.info("Loading InsightFace (SCRFD + ArcFace)...")
    t0 = time.time()

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=DETECTION_THRESHOLD)

    logger.info(f"InsightFace loaded in {time.time() - t0:.1f}s")

    load_gallery()

    if torch.cuda.is_available():
        logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")


# ── API endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "models": {
            "insightface": face_app is not None,
        },
        "gallery_loaded": gallery_matrix is not None,
        "gallery_people": len(set(gallery_names)),
        "gallery_embeddings": len(gallery_names),
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible pattern)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "face-recognition",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "sparkstation",
                "description": "InsightFace SCRFD + ArcFace face detection and recognition",
            }
        ],
    }


@app.post("/v1/detections", response_model=DetectionResponse)
async def detect_faces(request: DetectionRequest):
    """
    Detect and recognize faces in an image.

    Pipeline:
    1. SCRFD detects face bounding boxes + landmarks
    2. ArcFace extracts 512-dim embedding per face
    3. Cosine similarity match against gallery
    """
    if face_app is None:
        raise HTTPException(status_code=503, detail="InsightFace not loaded")

    t0 = time.time()

    try:
        img = decode_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    faces = face_app.get(img)
    img_h, img_w = img.shape[:2]

    detections = []
    num_recognized = 0
    num_unknown = 0

    for face in faces:
        det_conf = float(face.det_score)
        if det_conf < request.detection_threshold:
            continue

        # bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = face.bbox
        bbox = BoundingBox(
            x=round(float(x1) / img_w, 4),
            y=round(float(y1) / img_h, 4),
            width=round(float(x2 - x1) / img_w, 4),
            height=round(float(y2 - y1) / img_h, 4),
        )

        # Match against gallery
        match = None
        if face.embedding is not None:
            emb = face.embedding / np.linalg.norm(face.embedding)
            match = match_face(emb, request.recognition_threshold)

        if match:
            num_recognized += 1
        else:
            num_unknown += 1

        detection = FaceDetection(
            confidence=round(det_conf, 4),
            bbox=bbox,
            match=match,
            age=int(face.age) if hasattr(face, "age") and face.age is not None else None,
            gender="male" if hasattr(face, "gender") and face.gender == 1 else "female" if hasattr(face, "gender") and face.gender == 0 else None,
        )
        detections.append(detection)

    elapsed = time.time() - t0
    logger.info(
        f"Detected {len(detections)} faces ({num_recognized} recognized, {num_unknown} unknown) "
        f"in {elapsed:.2f}s"
    )

    return DetectionResponse(
        num_detections=len(detections),
        num_recognized=num_recognized,
        num_unknown=num_unknown,
        detections=detections,
        processing_time_s=round(elapsed, 3),
    )


@app.post("/v1/gallery/reload")
async def reload_gallery():
    """Hot-reload the gallery without restarting the server."""
    load_gallery()
    return {
        "status": "ok",
        "gallery_people": len(set(gallery_names)),
        "gallery_embeddings": len(gallery_names),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting Face Recognition server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
