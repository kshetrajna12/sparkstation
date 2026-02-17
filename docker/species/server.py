#!/usr/bin/env python3
"""
Species detection + classification server for wildlife photography.

Ensemble pipeline:
  1. MegaDetector v5a — detect animals, people, vehicles (bounding boxes)
  2. SpeciesNet classifier — camera-trap-trained species ID (best for African mammals)
  3. iNat21 ViT-L/14 classifier — iNaturalist-trained species ID (best for birds/reptiles)
  4. Ensemble logic — pick best prediction based on confidence and model strengths

Provides custom /v1/detections endpoint for the image indexing pipeline.
"""
import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import timm
from timm.data import resolve_data_config, create_transform
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Species Detection API")

# ── Global model instances ──────────────────────────────────────────────────

megadetector = None
speciesnet_model = None
inat_model = None
inat_transform = None
inat_label_names = None
inat_label_descs = None

# MegaDetector category mapping
MD_CATEGORIES = {"1": "animal", "2": "person", "3": "vehicle"}


# ── Request/Response models ─────────────────────────────────────────────────

class DetectionRequest(BaseModel):
    """Request to detect and classify species in an image."""
    image: str = Field(..., description="Base64-encoded image (with or without data: prefix)")
    country: Optional[str] = Field(None, description="ISO 3166-1 alpha-3 country code for geofencing (e.g. ZAF, BWA)")
    detection_threshold: float = Field(0.2, ge=0.0, le=1.0, description="MegaDetector confidence threshold")
    classification_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum species confidence to report")


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class SpeciesClassification(BaseModel):
    species: str
    scientific_name: Optional[str] = None
    confidence: float
    taxonomy: Optional[dict] = None
    source: str  # "speciesnet", "inat21", "ensemble"


class Detection(BaseModel):
    category: str  # "animal", "person", "vehicle"
    confidence: float
    bbox: BoundingBox
    classifications: list[SpeciesClassification] = []
    best_species: Optional[str] = None
    best_confidence: float = 0.0
    best_source: Optional[str] = None


class DetectionResponse(BaseModel):
    num_detections: int
    num_animals: int
    num_people: int
    num_vehicles: int
    detections: list[Detection]
    processing_time_s: float


# ── Image decoding ──────────────────────────────────────────────────────────

def decode_image(b64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


# ── SpeciesNet parsing ──────────────────────────────────────────────────────

def parse_speciesnet_taxonomy(label: str) -> dict:
    """Parse SpeciesNet label like 'uuid;class;order;family;genus;species;common_name'."""
    parts = label.split(";")
    if len(parts) >= 7:
        return {
            "class": parts[1] or None,
            "order": parts[2] or None,
            "family": parts[3] or None,
            "genus": parts[4] or None,
            "species_epithet": parts[5] or None,
            "common_name": parts[6] or None,
        }
    return {"raw": label}


def speciesnet_common_name(label: str) -> str:
    """Extract common name from SpeciesNet taxonomy string."""
    parts = label.split(";")
    if len(parts) >= 7 and parts[6]:
        return parts[6]
    if len(parts) >= 4 and parts[3]:
        return f"{parts[3]} family"
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return label


def speciesnet_scientific(label: str) -> Optional[str]:
    """Extract scientific name from SpeciesNet taxonomy string."""
    parts = label.split(";")
    if len(parts) >= 6 and parts[4] and parts[5]:
        return f"{parts[4].capitalize()} {parts[5]}"
    if len(parts) >= 5 and parts[4]:
        return parts[4].capitalize()
    return None


def speciesnet_is_species_level(label: str) -> bool:
    """Check if SpeciesNet prediction is at species level (not family/order)."""
    parts = label.split(";")
    return len(parts) >= 6 and bool(parts[5])


# ── Classification functions ────────────────────────────────────────────────

def classify_speciesnet(crop: Image.Image, country: Optional[str] = None) -> list[SpeciesClassification]:
    """Classify a crop using SpeciesNet."""
    if speciesnet_model is None:
        return []

    try:
        # Save temp file for SpeciesNet (it requires file paths)
        tmp_path = "/tmp/species_crop.jpg"
        crop.save(tmp_path, format="JPEG", quality=90)

        result = speciesnet_model.predict(
            filepaths=[tmp_path],
            country=country,
            run_mode="single_thread",
            batch_size=1,
        )

        preds = result.get("predictions", []) if result else []
        if not preds:
            return []

        pred = preds[0]
        classifications = []

        # Get top-5 from classifier output
        cls_data = pred.get("classifications", {})
        classes = cls_data.get("classes", [])
        scores = cls_data.get("scores", [])

        for label, score in zip(classes[:5], scores[:5]):
            if score < 0.01:
                continue
            classifications.append(SpeciesClassification(
                species=speciesnet_common_name(label),
                scientific_name=speciesnet_scientific(label),
                confidence=float(score),
                taxonomy=parse_speciesnet_taxonomy(label),
                source="speciesnet",
            ))

        return classifications

    except Exception as e:
        logger.error(f"SpeciesNet classification error: {e}")
        return []


def classify_inat21(crop: Image.Image) -> list[SpeciesClassification]:
    """Classify a crop using iNat21 ViT."""
    if inat_model is None:
        return []

    try:
        img_tensor = inat_transform(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = inat_model(img_tensor)
            probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(5)

        classifications = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            if prob < 0.01:
                continue
            name = inat_label_names[idx]
            desc = inat_label_descs.get(name, "")
            common = desc.split(",")[0] if desc else name

            # Extract taxonomy from description
            taxonomy = {}
            if desc and "," in desc:
                parts = desc.split(",")
                taxonomy["common_name"] = parts[0].strip()
                if len(parts) > 1:
                    taxonomy["class"] = parts[1].strip()

            classifications.append(SpeciesClassification(
                species=common,
                scientific_name=name,
                confidence=float(prob),
                taxonomy=taxonomy,
                source="inat21",
            ))

        return classifications

    except Exception as e:
        logger.error(f"iNat21 classification error: {e}")
        return []


def ensemble_classify(
    crop: Image.Image,
    country: Optional[str] = None,
    threshold: float = 0.1,
) -> tuple[list[SpeciesClassification], Optional[str], float, Optional[str]]:
    """
    Run both classifiers and ensemble the results.

    Returns:
        (all_classifications, best_species, best_confidence, best_source)
    """
    snet_results = classify_speciesnet(crop, country=country)
    inat_results = classify_inat21(crop)

    all_classifications = snet_results + inat_results

    # Ensemble logic:
    # 1. If SpeciesNet gives species-level ID with high confidence (>0.7), trust it
    # 2. If SpeciesNet only gives family/order level, prefer iNat21
    # 3. If both give species-level, prefer higher confidence
    # 4. Special case: iNat21 is better for birds, reptiles, amphibians

    best_species = None
    best_confidence = 0.0
    best_source = None

    snet_top = snet_results[0] if snet_results else None
    inat_top = inat_results[0] if inat_results else None

    if snet_top and inat_top:
        snet_is_species = snet_top.taxonomy and snet_top.taxonomy.get("species_epithet")
        inat_is_bird = inat_top.taxonomy and inat_top.taxonomy.get("class") in ("Bird", "Reptile", "Amphibian")

        if inat_is_bird and inat_top.confidence > 0.3:
            # iNat21 is much better for birds/reptiles
            best_species = inat_top.species
            best_confidence = inat_top.confidence
            best_source = "inat21"
        elif snet_is_species and snet_top.confidence > 0.7:
            # SpeciesNet has high-confidence species-level ID
            best_species = snet_top.species
            best_confidence = snet_top.confidence
            best_source = "speciesnet"
        elif snet_is_species and not snet_is_species:
            # SpeciesNet has species, iNat doesn't match well
            best_species = snet_top.species
            best_confidence = snet_top.confidence
            best_source = "speciesnet"
        elif inat_top.confidence > snet_top.confidence:
            best_species = inat_top.species
            best_confidence = inat_top.confidence
            best_source = "inat21"
        else:
            best_species = snet_top.species
            best_confidence = snet_top.confidence
            best_source = "speciesnet"

    elif snet_top:
        best_species = snet_top.species
        best_confidence = snet_top.confidence
        best_source = "speciesnet"
    elif inat_top:
        best_species = inat_top.species
        best_confidence = inat_top.confidence
        best_source = "inat21"

    if best_confidence < threshold:
        best_species = None
        best_confidence = 0.0
        best_source = None

    return all_classifications, best_species, best_confidence, best_source


# ── Model loading ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_models():
    """Load all models on startup."""
    global megadetector, speciesnet_model
    global inat_model, inat_transform, inat_label_names, inat_label_descs

    # 1. Load MegaDetector via SpeciesNet (it bundles it)
    logger.info("Loading SpeciesNet (includes MegaDetector)...")
    t0 = time.time()
    try:
        from speciesnet import SpeciesNet
        speciesnet_model = SpeciesNet(
            model_name=os.getenv("SPECIESNET_MODEL", "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"),
        )
        logger.info(f"SpeciesNet loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.error(f"Failed to load SpeciesNet: {e}")

    # Also load standalone MegaDetector for the detection step
    logger.info("Loading MegaDetector v5a...")
    t0 = time.time()
    try:
        from megadetector.detection.pytorch_detector import PTDetector
        md_path = os.getenv("MEGADETECTOR_PATH", "models/md_v5a.0.0.pt")
        megadetector = PTDetector(md_path)
        logger.info(f"MegaDetector loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.error(f"Failed to load MegaDetector: {e}")

    # 2. Load iNat21 ViT
    logger.info("Loading iNat21 ViT-L/14 classifier...")
    t0 = time.time()
    try:
        inat_model = timm.create_model(
            "hf_hub:timm/vit_large_patch14_clip_336.datacompxl_ft_inat21",
            pretrained=True,
        ).to(DEVICE).eval()

        cfg = inat_model.pretrained_cfg
        inat_label_names = cfg["label_names"]
        inat_label_descs = cfg["label_descriptions"]

        data_cfg = resolve_data_config(cfg)
        inat_transform = create_transform(**data_cfg, is_training=False)

        logger.info(f"iNat21 loaded in {time.time() - t0:.1f}s ({len(inat_label_names)} classes)")
    except Exception as e:
        logger.error(f"Failed to load iNat21: {e}")

    logger.info("All models loaded.")
    if torch.cuda.is_available():
        logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")


# ── API endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "models": {
            "megadetector": megadetector is not None,
            "speciesnet": speciesnet_model is not None,
            "inat21": inat_model is not None,
        },
        "device": DEVICE,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible pattern)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "species-ensemble",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "sparkstation",
                "description": "MegaDetector + SpeciesNet + iNat21 ensemble",
            }
        ],
    }


@app.post("/v1/detections", response_model=DetectionResponse)
async def detect_species(request: DetectionRequest):
    """
    Detect animals and classify species in an image.

    Pipeline:
    1. MegaDetector detects bounding boxes (animal/person/vehicle)
    2. For each animal crop, run SpeciesNet + iNat21 ensemble
    3. Return all detections with species classifications
    """
    if megadetector is None:
        raise HTTPException(status_code=503, detail="MegaDetector not loaded")

    t0 = time.time()

    # Decode image
    try:
        img = decode_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Run MegaDetector
    md_result = megadetector.generate_detections_one_image(
        img, image_id="request", detection_threshold=request.detection_threshold
    )

    img_w, img_h = img.size
    detections = []
    num_animals = 0
    num_people = 0
    num_vehicles = 0

    for det in md_result.get("detections", []):
        category = MD_CATEGORIES.get(det["category"], f"class_{det['category']}")
        conf = det["conf"]
        if conf < request.detection_threshold:
            continue

        # MegaDetector bbox: [x_min, y_min, width, height] normalized
        x, y, w, h = det["bbox"]
        bbox = BoundingBox(x=x, y=y, width=w, height=h)

        detection = Detection(
            category=category,
            confidence=float(conf),
            bbox=bbox,
        )

        if category == "animal":
            num_animals += 1

            # Crop with padding for classification
            pad_x = w * img_w * 0.05
            pad_y = h * img_h * 0.05
            crop = img.crop((
                max(0, x * img_w - pad_x),
                max(0, y * img_h - pad_y),
                min(img_w, (x + w) * img_w + pad_x),
                min(img_h, (y + h) * img_h + pad_y),
            ))

            # Ensemble classification
            all_cls, best_sp, best_conf, best_src = ensemble_classify(
                crop,
                country=request.country,
                threshold=request.classification_threshold,
            )

            detection.classifications = all_cls
            detection.best_species = best_sp
            detection.best_confidence = best_conf
            detection.best_source = best_src

        elif category == "person":
            num_people += 1
        elif category == "vehicle":
            num_vehicles += 1

        detections.append(detection)

    elapsed = time.time() - t0
    logger.info(
        f"Detected {num_animals} animals, {num_people} people, {num_vehicles} vehicles "
        f"in {elapsed:.2f}s"
    )

    return DetectionResponse(
        num_detections=len(detections),
        num_animals=num_animals,
        num_people=num_people,
        num_vehicles=num_vehicles,
        detections=detections,
        processing_time_s=round(elapsed, 3),
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting Species Detection server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
