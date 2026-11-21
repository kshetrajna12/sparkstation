# GLM 5.3 Flash prompt-end checkpoints

This candidate uses the [GLM 1104 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/tree/13f2464ee84ab2b5dabf0bb264d1144b02cf5fb7) at
`13f2464ee84ab2b5dabf0bb264d1144b02cf5fb7` with TensorFold 1.1.9 at
`f350cd3ea253a042e205746928271f9d8f71a82d`.

The engine uses the 4,096-token compute grid for fresh plain prompts
and can save a checkpoint at the natural history boundary just before
the generation suffix. Tool turns retain the 2,048-token checkpoint grid.
When another request is decoding, prompt work yields after at most 2,048
tokens. The follow-up keeps plain histories shorter than 2,048 tokens on
the original path, avoiding the extra pass observed on very short replies.

## Build a candidate image

In a separate recipe checkout pinned to the revision above, copy the two
patches into `patches/engine/` after the recipe's existing 110 patches.
Build on the primary Spark before staging the resulting image on both ranks.

```bash
cd /path/to/pinned-glm-recipe
cp /path/to/sparkstation/patches/glm53-prompt-end/*.patch patches/engine/
(
  export REPO_ROOT="$PWD" DRY_RUN=0 WITH_VIDEO=1
  source scripts/config.sh
  source scripts/prepare.sh
  set_image
  prepare_image
)
```

Use a separate state directory for the candidate. Run the managed
SparkStation trial on both ranks with an 8 GiB prompt cache, q8 dense and
head weights, FP8 KV pages, and the 4-bit DFlash2 drafter. The [native
recipe notes](../glm53-native-upgrade.md) identify the model and drafter
revisions and the managed launch wrappers.

## Qualification status

The on-worker trial before the short-prompt fix processed a fresh 27K
prompt in 26.08 seconds versus 35.05 seconds on the serving build. A
repeated plain prompt reached identical output in 0.539 seconds versus
1.809 seconds. The tool-result response began in 1.448 seconds. The
candidate increased first-text latency for uncached short requests, so
SparkStation restored the serving build.

The short-prompt follow-up passed the native compile and scheduling host
checks. A rebuilt image still needs the full on-worker gate, including
uncached short first-text latency, cold 27K and 68K retrieval, plain
replay, the tool loop, four-stream decode, images and video. Merge records
the candidate source. Activate it through the managed trial after these
checks pass.
