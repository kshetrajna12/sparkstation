# Species Detection + Classification Server
# Ensemble: MegaDetector v5a + SpeciesNet + iNat21 ViT-L/14
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Install system deps (libGL for OpenCV used by megadetector)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install setuptools <70 (needed for pkg_resources used by yolov5)
RUN pip install --no-cache-dir "setuptools<70"

# Install uv for fast, reliable dependency resolution
RUN pip install --no-cache-dir uv

# Install all deps directly into system python, but exclude torch/onnx
# (already in base image as CUDA-enabled builds — must not be overwritten)
RUN uv pip install --system \
    --no-deps megadetector speciesnet

RUN uv pip install --system \
    --exclude-newer 2026-02-18 \
    fastapi \
    "uvicorn[standard]" \
    pillow \
    timm \
    httpx \
    humanfriendly \
    cloudpathlib \
    kagglehub \
    yolov5 \
    sahi \
    supervision \
    jsonpickle \
    pyqtree \
    seaborn \
    scikit-learn \
    python-dateutil \
    reverse_geocoder \
    absl-py \
    onnx2torch \
    opencv-python-headless

# Copy server code
WORKDIR /app
COPY server.py /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Run server
CMD ["python3", "server.py"]
