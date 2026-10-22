# GLM-5.3-Flash image admission failures (2026-10-21)

The deployed MiaAI recipe's rank-0 container could lose GPU device access on a
systemd cgroup rebuild. Explicit device declarations now protect both ranks.
The engine image, weights, quantization, memory budget and TensorFold version
remain pinned to the existing deployment.

## Evidence and confidence

- Worker1 uses Docker's systemd cgroup driver. NVIDIA CDI refresh was disabled,
  `/var/run/cdi/nvidia.yaml` was absent, and the serving container's systemd
  `DeviceAllow` omitted NVIDIA devices. Docker also logged unresolved CDI device
  injection and fell back to the legacy GPU hook. Worker2 already had CDI enabled.
- The failed container ran from October 12 to October 21. On October 16 at
  13:05:22 and 13:05:25 PDT, snapd requested systemd daemon reloads on worker1.
  Both precede the retained October 20 vision failures. The archived rank-0 log
  also contains a Triton `operation not permitted` failure on October 21 at
  10:30:25 UTC, followed by the permanent rank-desynchronization latch.
- Someone had restarted both ranks at about 10:43 UTC on October 21 before this
  investigation. The original 320x200 fixtures then worked without changing the
  image or configuration. A six-request check through gateway `default` passed:
  text, one image, three images, one image, three images, text. Image answers
  correctly named the shapes, colours and digits; all image prompts had unique
  system prefixes and zero cached prompt tokens.
- In a disposable container using the **same engine image and real vision
  weights**, warm the tower with its normal 56x56 startup picture. Then
  `docker update --cpus 1 <probe>` rebuilds only that container's device filter.
  NVML immediately fails. A 336x224 canvas fails at attention with CUDA error 800,
  and a 644x1400 canvas fails in the attention output projection with both
  `cublasLtMatmul` and fallback `cublasGemmEx` internal errors. CUDA logging
  reports `Failed to create CPU mapping` and `CUDA_ERROR_NOT_PERMITTED` from
  `cuKernelGetFunction`. An isolated small linear can still succeed: losing
  device access does not imply that every existing GPU operation stops.
- With the explicit device arguments in this patch, the same forced update
  preserves NVML access and **both complete tower encodes pass**, producing
  `[96, 4096]` and `[1150, 4096]` features respectively.

This reproduces a concrete cause of vision/cuBLAS failure and verifies its
prevention. The historical qkv failure at `glm.py:403` was not reproduced at that
exact line: the disposable tower's failing sites were `:407` and `:409`. The
original failed process is gone, and its NVML state was never recorded. Thus
device-access loss is strongly supported as the historical cause, rather than
proven by direct inspection of that process. Independent allocator or cuBLAS
faults are not exhaustively excluded. No memory-limit workaround or changes to
`empty_cache()` are justified by these results.

NVIDIA documents this failure mode and the explicit-device workaround in its
[Container Toolkit troubleshooting guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error).
The related upstream report is
[MiaAI recipe issue #150](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/150)
(see the October 15 CDI/cgroup diagnosis).

## Applied fix

`glm53-nvidia-devices.patch` adds the GPU and control devices to the recipe's
shared `RUN_ARGS`, used by both head and worker. This makes runc/systemd aware of
the allowed devices instead of relying on the legacy NVIDIA hook's out-of-band
cgroup edits. It is the scoped alternative to enabling host-wide CDI refresh.

The patch applies to recipe `b5808ac620c795ec9c0640f3ca2a2f7be21aaa63` (the
deployed checkout) and `3fd60df` (upstream main checked October 21). Apply in the
recipe checkout on worker1, then restart **both** ranks with the usual
SparkStation deployment wrapper:

```sh
git apply --check /path/to/sparkstation/patches/glm53-nvidia-devices.patch
git apply /path/to/sparkstation/patches/glm53-nvidia-devices.patch
bash -n start.sh
```

These are single-GPU DGX Sparks with all listed device nodes present. A missing
device will cause Docker to refuse the launch. When upgrading the recipe, keep
the patch until the upstream launch arguments or host CDI configuration provide
equivalent protection. Reverse it with `git apply -R` and restart both ranks.

## Upstream updates

MiaAI main `3fd60df` differs from our `b5808ac` only by eight README lines about
precision/quality trade-offs. It still pins TensorFold v0.6.0 and image digest
`sha256:14b5072e8e9918f1080153f57321415a387bc7d78cf9c9b44a7aeae2d4399ccf`.
There is no newer recipe engine fix to pull.

TensorFold itself has released v1.1.2 (October 21). Its native Zig serving engine
is a migration from this Python/PyTorch recipe and its 107 patches, not a
drop-in package update. The upstream GLM CUDA vision implementation is also
different. Upgrading it is a separate qualification task; it would not repair
the host/container device-policy problem demonstrated here.

## Repeat verification

The patched recipe was deployed on worker1 and both ranks restarted using the
normal SparkStation wrapper. Startup, greedy and sampled text, and the wrapper's
6,947-, 39,306- and 141,639-token prompt/follow-up warm-ups all passed. Both live
containers expose the explicit devices; worker1's systemd policy now includes
the NVIDIA GPU, UVM and capability devices.

The original-fixture six-request check passed after restart. Then, on **each
live rank**, `docker update --cpus 0 glm53-flash-tf` rebuilt the container policy
while retaining its existing unlimited CPU setting (`NanoCpus=0` before and
after). NVML still passed on both ranks, and another original-fixture six-request
check passed through gateway `default`. Across those two checks all 12 requests
returned HTTP 200, all eight image answers named the expected shapes/colours/
digits, and all four text answers were correct. Cached prompt tokens were zero.
The final stack health check passed and the new engine log contained no cuBLAS,
request, permission or rank-desynchronization errors.

`scripts/glm53_vision_smoke.py` uses only the standard library, draws the same
320x200 RGBA shape/digit fixtures, and verifies image answers plus text requests
before and after them. `--fixture-dir` accepts the original Photon-encoded PNGs.
The default route is gateway `:8000`, alias `default`; it reads the API key from
`GATEWAY_LOCAL_KEY` without printing it. Each request uses a fresh system prefix
so a saved image prompt cannot bypass the encoder. An engine endpoint can be
supplied with `--base-url` and its served name with `--model`.
Pillow decoding independently confirmed pixel-for-pixel equivalence of all three
generated PNGs with the original Photon fixtures. Python compilation, Ruff,
patch application against the pinned and current recipe, Bash syntax, and diff
whitespace checks passed.

Diagnostic scripts, original synthetic PNGs, CUDA logs, and before/after receipts
are retained under
`/home/kshetrajna/.bb/thread-storage/glm53-vision-fix-20261021/`.
