# Species Detection + Classification Server
# Ensemble: MegaDetector v5a + SpeciesNet + iNat21 ViT-L/14
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Install system deps (libGL for OpenCV used by megadetector)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install setuptools <70 (needed for pkg_resources used by yolov5)
# Pin onnx to version already in base image to prevent source rebuild
RUN pip install --no-cache-dir "setuptools<70" && \
    pip install --no-cache-dir "onnx==1.18.0"

# Install megadetector and speciesnet without upgrading onnx
RUN pip install --no-cache-dir \
    --no-deps \
    megadetector speciesnet

# Install their actual runtime deps (minus onnx which we already have)
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.31.0 \
    pillow>=10.4.0 \
    timm>=1.0.0 \
    httpx \
    yolov5 \
    sahi \
    supervision \
    kagglehub \
    humanfriendly

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
