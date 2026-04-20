# Face Recognition Server
# InsightFace: SCRFD detection + ArcFace embeddings
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Install system deps (libGL for OpenCV used by InsightFace)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir \
    insightface>=0.7.3 \
    onnxruntime>=1.18.0 \
    opencv-python-headless>=4.9.0 \
    fastapi>=0.115.0 \
    "uvicorn[standard]>=0.31.0" \
    pillow>=10.4.0

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
