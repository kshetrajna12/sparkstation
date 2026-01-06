# CLIP Embedding Server
# Uses HuggingFace Transformers for proper image+text embedding support
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Install dependencies
RUN pip install --no-cache-dir \
    --upgrade \
    transformers>=4.44.0 \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.31.0 \
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
