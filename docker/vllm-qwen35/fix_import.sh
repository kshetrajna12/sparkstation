#!/bin/bash
# Fix cutlass_fp4_supported import in patched mxfp4.py
FILE="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py"
python3 -c "
p='$FILE'
t=open(p).read()
old='from vllm.model_executor.layers.quantization.utils.quant_utils import (\n                cutlass_fp4_supported,'
new='from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (\n                cutlass_fp4_supported,'
t=t.replace(old, new)
open(p,'w').write(t)
print('Fixed cutlass_fp4_supported import in mxfp4.py')
"
