# Species Detection + Classification Server
# Ensemble: MegaDetector v5a + SpeciesNet + iNat21 ViT-L/14
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Fix cmake (broken shim in this base image) and install build deps for onnx
RUN pip install --no-cache-dir cmake

# Install setuptools first (needed for pkg_resources used by yolov5)
RUN pip install --no-cache-dir "setuptools<70"

# Install onnx first (avoids build issues when pulled as transitive dep)
RUN pip install --no-cache-dir onnx

# Install main dependencies
RUN pip install --no-cache-dir \
    --upgrade \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.31.0 \
    pillow>=10.4.0 \
    timm>=1.0.0 \
    httpx \
    megadetector \
    speciesnet

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
