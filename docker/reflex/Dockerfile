# reflex decision-model server — Jev-style "System One" typed judgments
# (POST /v1/systemone) on a local Qwen3.5 checkpoint.
#
# Unlike the other custom servers (clip/face/species) this does NOT build on
# the NGC PyTorch base: NGC 25.11 ships transformers 4.57, which predates the
# Qwen3.5 hybrid architecture reflex defaults to. Instead it reproduces the
# reflex repo's own uv-locked environment (Python 3.12, torch cu130 with the
# pip-packaged CUDA libs, transformers 5.x, flash-linear-attention) — the
# exact set the reflex calibration/LoRA results were measured with on GB10.
#
# Build (on the host that will run it — DGX Spark is linux/arm64):
#   cd docker/reflex
#   docker build --platform linux/arm64 -t reflex-server:latest .
#   # pin a reflex commit:  --build-arg REFLEX_REF=<sha or branch>
#
# The full (non-slim) python image is deliberate: Triton JIT-compiles the
# linear-attention kernels at first use and needs a C compiler + Python
# headers at runtime.
ARG PYTHON_IMAGE=python:3.12-bookworm
FROM ${PYTHON_IMAGE}

ARG REFLEX_REPO=https://github.com/kshetrajna12/reflex.git
ARG REFLEX_REF=main

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
# init+fetch (not `clone --branch`) so REFLEX_REF may be a branch OR a sha.
RUN git init -q reflex \
    && cd reflex \
    && git remote add origin "${REFLEX_REPO}" \
    && git fetch -q --depth 1 origin "${REFLEX_REF}" \
    && git checkout -q FETCH_HEAD \
    && git rev-parse HEAD > /app/REFLEX_COMMIT \
    && uv sync --frozen --no-dev --no-editable --python /usr/local/bin/python3.12 \
    && rm -rf .git /root/.cache/uv   # the uv wheel cache alone is 5.5 GB

COPY server.py /app/server.py

ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH=Qwen/Qwen3.5-4B \
    SERVED_MODEL_NAME=reflex

EXPOSE 8000
CMD ["/app/reflex/.venv/bin/python", "/app/server.py"]
