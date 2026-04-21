#!/usr/bin/env python3
"""
Build a face recognition gallery from reference photos.

Usage:
    python scripts/build_face_gallery.py /path/to/photos /path/to/gallery.json

Directory structure expected:
    /path/to/photos/
        alice/
            photo1.jpg
            photo2.jpg
        bob/
            photo1.jpg
            photo2.png
        ...

Each subdirectory name becomes the person's display name.
Outputs gallery.json with per-face ArcFace embeddings (512-dim).
"""
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}


def build_gallery(photos_dir: Path, output_path: Path):
    """Build gallery.json from a directory of person folders."""

    logger.info("Loading InsightFace (SCRFD + ArcFace)...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

    gallery = []
    person_dirs = sorted([d for d in photos_dir.iterdir() if d.is_dir()])

    if not person_dirs:
        logger.error(f"No subdirectories found in {photos_dir}")
        sys.exit(1)

    logger.info(f"Found {len(person_dirs)} people: {[d.name for d in person_dirs]}")

    for person_dir in person_dirs:
        name = person_dir.name
        images = [f for f in person_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

        if not images:
            logger.warning(f"  {name}: no images found, skipping")
            continue

        logger.info(f"  {name}: processing {len(images)} images...")

        embeddings = []
        skipped = 0

        for img_path in sorted(images):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"    {img_path.name}: failed to read, skipping")
                skipped += 1
                continue

            faces = app.get(img)

            if len(faces) == 0:
                logger.warning(f"    {img_path.name}: no face detected, skipping")
                skipped += 1
                continue

            if len(faces) > 1:
                # Pick the largest face (highest bbox area)
                faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                logger.info(f"    {img_path.name}: {len(faces)} faces, using largest")

            face = faces[0]
            if face.embedding is None:
                logger.warning(f"    {img_path.name}: no embedding extracted, skipping")
                skipped += 1
                continue

            emb = face.embedding / np.linalg.norm(face.embedding)
            embeddings.append(emb.tolist())

        if embeddings:
            gallery.append({
                "name": name,
                "embeddings": embeddings,
            })
            logger.info(f"  {name}: {len(embeddings)} embeddings ({skipped} skipped)")
        else:
            logger.warning(f"  {name}: no valid embeddings extracted!")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(gallery, f)

    total_embeddings = sum(len(p["embeddings"]) for p in gallery)
    logger.info(f"Gallery saved to {output_path}: {len(gallery)} people, {total_embeddings} embeddings")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <photos_dir> <output_json>")
        print(f"Example: {sys.argv[0]} ~/face-gallery/ ~/.openclaw/workspace-whatsapp-schoolfriends-group/gallery.json")
        sys.exit(1)

    photos_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not photos_dir.exists():
        logger.error(f"Photos directory not found: {photos_dir}")
        sys.exit(1)

    build_gallery(photos_dir, output_path)
