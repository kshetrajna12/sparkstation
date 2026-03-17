FROM vllm/vllm-openai:cu130-nightly

ENTRYPOINT []
CMD []

# Downgrade vLLM from nightly to 0.17.0 for MXFP4 patch compatibility
RUN pip install --no-cache-dir --force-reinstall --no-deps 'vllm==0.17.0' \
    --extra-index-url https://wheels.vllm.ai/0.17.0/cu130 \
    2>&1 | tail -5

# Deps per namake-taro instructions
RUN pip install --no-cache-dir 'nvidia-nccl-cu13>=2.29.2' 2>&1 | tail -3 || true
RUN pip install --no-cache-dir 'transformers==5.3.0' 'huggingface_hub==1.5.0' fastsafetensors 2>&1 | tail -3 || true

# Apply MXFP4 patches
COPY vllm-custom/patches/vllm_all.patch /tmp/vllm_all.patch
COPY vllm-custom/patches/flashinfer_cutlass_sfb_layout_fix.patch /tmp/flashinfer_fix.patch
RUN cd /usr/local/lib/python3.12/dist-packages && \
    patch -p0 < /tmp/vllm_all.patch && \
    patch -p0 < /tmp/flashinfer_fix.patch

# Fix import path bug in patch
COPY fix_import.sh /tmp/fix_import.sh
RUN bash /tmp/fix_import.sh

# Verify
RUN python3 -c "import vllm; print('vLLM:', vllm.__version__)"
