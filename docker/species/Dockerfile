# Species Detection + Classification Server
# Ensemble: MegaDetector v5a + SpeciesNet + iNat21 ViT-L/14
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Install system deps (libGL for OpenCV used by megadetector)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for reliable dependency resolution
RUN pip install --no-cache-dir uv

# Create a venv that inherits base image packages (torch, onnx, numpy, etc.)
RUN uv venv /opt/species --python $(which python3) --seed
ENV VIRTUAL_ENV=/opt/species
ENV PATH="/opt/species/bin:$PATH"

# Install setuptools <70 into venv (needed for pkg_resources used by yolov5)
RUN uv pip install "setuptools<70"

# Install all deps into the venv — uv resolves cleanly and uses
# existing system packages where compatible
RUN uv pip install \
    megadetector \
    speciesnet \
    fastapi \
    "uvicorn[standard]" \
    pillow \
    timm \
    httpx

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
