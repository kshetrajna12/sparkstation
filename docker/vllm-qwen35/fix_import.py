#!/usr/bin/env python3
"""Fix cutlass_fp4_supported import path in patched mxfp4.py"""
import glob

for path in glob.glob("/opt/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/mxfp4.py"):
    with open(path) as f:
        content = f.read()
    content = content.replace(
        "from vllm.model_executor.layers.quantization.utils.quant_utils import (\n                cutlass_fp4_supported,",
        "from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (\n                cutlass_fp4_supported,"
    )
    with open(path, 'w') as f:
        f.write(content)
    print(f"Fixed import in {path}")
