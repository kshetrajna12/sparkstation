# GLM 1104 native engine deployment (2026-11-07)

SparkStation's `glm5.3-flash` entry moves from TensorFold Python v0.6.0 to
[MiaAI GLM 1104](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104)
at `13f2464ee84ab2b5dabf0bb264d1144b02cf5fb7`: TensorFold native v1.1.9,
`f350cd3ea253a042e205746928271f9d8f71a82d`, plus 110 engine patches.
The old Python recipe remains at `~/glm-tf` on worker1; the native recipe is
at `~/glm53-1104`. Both use worker1 as head and worker2 as follower over CX7.

The supervisor still calls the primary's `~/glm53-tf-deploy/{start,stop,health}.sh`.
The new start and stop wrappers invoke the native recipe on worker1; the health
probe checks both containers. The wrapper accepts the launcher's `--host` and
`--port` arguments, validates their pinned values, and keeps them out of the
recipe's argument list (which accepts only `restart`). The native engine compiles
kernels into the image, so the Python recipe's Triton-JIT warm-up is removed.

## Pins and site configuration

Weights: `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` at
`76c0b5173166d2795dd48860f45d8224817f894c`.
Drafter: `incoai/GLM-5.3-Flash-DFlash2` at
`bf582e4eacc1810f76656d1811693ff6c6737d2a`, 4-bit.
The Hugging Face cache downloads missing files at these exact revisions;
resolved files are hard-linked into `~/.cache/glm53-1104/{model,dflash2}`.
Existing cache snapshots and the previous image are retained.

Worker1's gitignored `scripts/local.sh` sets the worker's direct-link SSH target
and these settings:

```sh
SERVED_NAME=glm-5.3-flash
PORT=8009
CONTAINER_NAME=glm53-flash-tf
DATA_DIR=/home/kshetrajna/.cache/glm53-1104
WORKER_WEIGHTS=rsync
MODEL_SOURCE=download
PARALLEL=4
CONTEXT=auto
KV=fp8
DENSE=q8
HEAD=q8
MEMORY_RESERVE_GIB=12
CACHE_FIRST=1
PROMPT_CACHE_GIB=8
PROMPT_CACHE_MIN_GIB=2
LEARN_GIB=30
HOST_CACHE_GIB=0
KV_SHARD=0
THINKING=1
REASONING_EFFORT=xhigh
WITH_VIDEO=1
VISION=1
SERVE_EXTRA_ARGS='--top-k 20'
```

`CONTEXT=auto` tries 1,048,576 tokens first. The cache minimum prevents a start
that silently leaves too little prompt cache. Disk conversation states live in
`DATA_DIR/zig-home/learn` on the head and are capped at 30 GiB. KV sharding is off.
The new q8 dense/head format is symmetric grouped integer quantization; it is
different from the old Python recipe's FP8 dense weights.

## Local launch patch

Apply [glm53-native-launch.patch](glm53-native-launch.patch) in the native recipe
checkout before starting it:

```sh
git apply --check /path/to/sparkstation/patches/glm53-native-launch.patch
git apply /path/to/sparkstation/patches/glm53-native-launch.patch
bash scripts/tests/run.sh
```

It carries the explicit NVIDIA devices on both ranks from our
[device-policy fix](glm53-nvidia-devices.md). It also removes rsync's `--inplace`:
the destination may be hard-linked into the retained Hugging Face cache, so
updates must replace files rather than overwrite the old snapshot's bytes.
These changes affect launch and weight synchronization, not engine compilation.

## Rollback

The original deployment wrappers and old recipe diff are backed up at
`/home/kshetrajna/.bb/thread-storage/glm53-native-upgrade-20261107/rollback-wrapper/`
and `rollback-recipe.patch`. To roll back, stop GLM through the supervisor first,
restore the original wrapper scripts to `~/glm53-tf-deploy`, and start the same
`glm5.3-flash` alias through the supervisor. The restored wrapper invokes
`~/glm-tf`; it uses the retained old image, weights, and site configuration.
Restore the recipe comments in `models.yaml` to match. No download or reverse
engine patch is needed, and both ranks must be stopped and restarted together.

The thread-storage directory also contains the dry-run plan, launch request,
and verification scripts and receipts.

## Deployment verification

The built image is `tensorfold-glm53-spark:cbca6e5808f3`; both ranks run
`tensorfold-native 1.1.9`. The head loaded in 171.4 seconds. Its startup line
funded the full 1,048,576-token request window, four streams, an 8.0 GiB device
prompt cache, and 3,142 pages (1,608,704 shared KV positions). RoCE all-gather is
active on the direct-link HCA; NCCL also reports NET/IB transport.

Passed on the deployed service:

- Recipe script suite, Bash syntax, patch reverse-application check, YAML parse,
  and diff whitespace checks. Shellcheck was unavailable on worker1.
- Whole-pair health probe and supervisor RUNNING/healthy registration, retaining
  `glm5.3-flash` as the default and vision model.
- Direct API and gateway `default`: text, thinking off and low/high/max, SSE
  completion, a forced function call and tool-result round trip, and prompt reuse.
- Gateway top-level `reasoning: false`, `reasoning_effort: high|max`, server-default
  thinking, and four concurrent requests.
- All six cases of `scripts/glm53_vision_smoke.py` through gateway `default`:
  text before and after images, single images, and three images together. Every
  shape, colour and digit matched, with zero cached prompt tokens.
- A synthetic three-colour MP4 through the gateway: "Red, green, blue" in order.
- The 12.4K-token gateway conversation resumed 12,288 tokens: 0.79 seconds for the
  follow-up versus 12.94 seconds for the cold turn. These are functional cache
  checks, not throughput benchmarks.
- Disk-state writes were observed (about 1.7 GiB after the cache tests); restore
  across a restart was not exercised during this deployment.
- NVIDIA devices are explicitly declared on the new containers, and the head
  exports the existing `tensorfold:prompt_tokens_total`, cached-token and
  generation-token metric names.

The configured 1M limit was confirmed in startup; the longest validation prompt
was about 12.4K tokens. Full-window stress, model-quality comparisons and a
long-running soak were not part of this upgrade.

## Speed check after deployment

Measured on November 7 against the direct worker1 backend from the primary,
using streamed completion token counts and the interval between the first and
last generated token. DFlash2 is enabled. Single-stream runs generated 512
completion tokens, four-stream runs 384 per request. Thinking-on rates include
reasoning tokens. GPU clocks were observed around 2.17-2.19 GHz; clocks were not
changed or locked for this test.

| Workload | Observed decode rate |
| --- | --- |
| Prose, greedy, thinking off, one stream, two runs | 37.4-38.0 tok/s |
| Prose, T=0.7, high effort, one stream, two runs | 36.8-40.5 tok/s |
| Code, greedy, thinking off, one stream, two runs | 49.5-50.9 tok/s |
| Four prose streams, greedy, thinking off, two runs | 70.9-72.1 tok/s combined |
| Four prose streams, T=0.7, high effort, one run | 71.4 tok/s combined |

The combined rate divides all streams' generated tokens (minus each first token)
by the interval from the earliest first token to the latest last token. Summing
individually timed stream rates instead gives 73.5-74.8 tok/s for greedy and
75.1 for sampled thinking, but those intervals differ and overstate the common
window's throughput. Single-stream short-prompt TTFT was 0.16-0.18 seconds;
four-stream maximum TTFT was 0.86-0.93 seconds.

Cold prefill used synthetic harbor records with a fresh prefix on every request;
all three requests reported zero cached prompt tokens. Times below include
prefill through first token, and the production disk conversation tier stayed on.

| Actual prompt tokens | First token | Effective prompt tok/s |
| --- | --- | --- |
| 9,791 | 12.07 s | 811 |
| 39,770 | 38.64 s | 1,029 |
| 80,561 | 76.79 s | 1,049 |

No matching benchmark was captured on the previous Python engine before the
upgrade. Its archived container logs contained startup diagnostics but no usable
per-request throughput baseline. The recipe author's old Python v0.6.0 result
(29.6 tok/s sampled prose at one stream, 68-71 at four) is external context, not a
measurement of our previous deployment, and cannot establish our percentage gain.

Harness and raw results are in the thread-storage directory above as `bench.py`,
`speed-20261107.jsonl` and `speed-run.log`. This was a short operational benchmark,
not a full-window stress or model-quality comparison.
