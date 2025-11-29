# FLUX.1-dev Image Generation Server
# Uses Diffusers library for maximum compatibility
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Install dependencies
RUN pip install --no-cache-dir \
    --upgrade \
    torchao \
    diffusers>=0.30.0 \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.31.0 \
    pillow>=10.4.0 \
    sentencepiece>=0.2.0 \
    protobuf>=5.28.0

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
