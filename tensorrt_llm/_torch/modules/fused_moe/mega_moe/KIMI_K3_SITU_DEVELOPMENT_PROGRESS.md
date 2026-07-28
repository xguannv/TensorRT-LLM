<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kimi K3 SiTU on DeepGEMM MegaMoE: Development Progress

Last updated: 2026-07-28

## Objective

Enable Kimi K3 routed experts to use DeepGEMM's fused FP8xFP4 MegaMoE
backend on Blackwell while preserving the existing TRTLLM-Gen path and the
standard DeepGEMM SwiGLU path.

The target is not a SiTU-only MegaMoE implementation. The DeepGEMM API keeps
`activation="swiglu"` as its default and adds an independently specialized
`activation="situ"` path. TRT-LLM selects SiTU only for models whose
pretrained configuration supplies the Kimi SiTU parameters.

The checkpoint used for this work is:

```text
/mnt/s3fs/jet-artifacts/model/moonshotai_kimi-k3/hf/hf-301be1b_orig
```

Relevant Kimi K3 routed-MoE configuration:

| Parameter | Value |
| --- | ---: |
| Model hidden size | 7168 |
| Routed-expert latent hidden size | 3584 |
| Routed-expert intermediate size | 3072 |
| Routed experts | 896 |
| Experts selected per token | 16 |
| Shared experts | 2 |
| SiTU gate beta | 4.0 |
| SiTU linear beta | 25.0 |

This work applies MegaMoE to the routed expert bank. The two shared experts
remain on the existing replicated Kimi shared-MLP path.

## Implementation Completed

### DeepGEMM

The TRT-LLM integration is pinned to the following development fork and
commit:

```text
Repository: https://github.com/longlee0622/DeepGEMM
Branch:     feat/kimi-k3-situ-mega-moe
Commit:     cc832217b65bbe693a9eaa64f1077f65daaee67c
```

The DeepGEMM change:

- Adds `activation="situ"` to `fp8_fp4_mega_moe` while retaining SwiGLU.
- Adds `situ_beta` and `situ_linear_beta` API arguments.
- Specializes the activation at JIT compile time; it does not add a runtime
  branch to the inner activation loop.
- Implements the Kimi activation:

  ```text
  gate = beta * tanh(gate / beta) * sigmoid(gate)
  up   = linear_beta * tanh(up / linear_beta)
  out  = gate * up
  ```

- Keeps the DeepGEMM gate/up packing convention explicit in its weight
  transform and tests.
- Adds DeepGEMM tests for both SwiGLU and SiTU.

### TensorRT-LLM dependency pin

`3rdparty/fetch_content.json` now fetches the fork commit above. The
dependency metadata and file-to-dependency attribution keys were updated to
the same commit.

This fork pin is appropriate for development and validation. Before an
upstream TRT-LLM merge, the preferred dependency source should be revisited:
either merge the DeepGEMM change upstream and pin that commit, or agree on an
acceptable long-lived source for the dependency.

### TensorRT-LLM MegaMoE backend

`MegaMoEDeepGemm` now:

- Treats generic `ActivationType.Swiglu` as the gated FC1 tensor geometry
  shared by SwiGLU and SiTU.
- Infers SiTU when `activation_situ_beta` is present in the pretrained
  configuration; otherwise it defaults to SwiGLU.
- Reads `activation_situ_beta` and `activation_situ_linear_beta`, validates
  that both are positive, and forwards them to DeepGEMM.
- Rejects SiTU combined with the SwiGLU activation clamp.
- Rejects invalid activation names or incomplete SiTU configuration.

No global activation enum or ABI was changed.

### Kimi K3 routed-MoE selection

The Kimi K3 runtime previously forced the routed expert backend to `TRTLLM`
even when the user explicitly requested `MEGAMOE_DEEPGEMM`. It now:

- Preserves the historical TRTLLM-Gen default.
- Honors an explicit `MEGAMOE_DEEPGEMM` selection.
- Passes TRTLLM-Gen-specific SiTU arguments only to the TRTLLM-Gen backend.
- Lets MegaMoE obtain its SiTU parameters from the pretrained configuration.
- Fails loudly if an explicit MegaMoE request resolves to a different
  backend, rather than silently running a mathematically incorrect fallback.

## Validation Completed

### Build and runtime environment

Validation used:

```text
Host:      umbriel-b200-027
Container: 525b84e86366 (tensorrt_llm/devel:latest-jonasl)
Wheel:     /code/tensorrt_llm/build/
           tensorrt_llm-1.3.0rc21-cp312-cp312-linux_x86_64.whl
GPU:       NVIDIA B200
```

The installed wheel exposes the expected bundled API:

```text
fp8_fp4_mega_moe(..., activation='swiglu', activation_clamp=None,
                 fast_math=True, situ_beta=None,
                 situ_linear_beta=None)
```

### Configuration and wiring tests

Four targeted tests pass:

- Kimi preserves an explicit `MEGAMOE_DEEPGEMM` backend.
- Kimi retains TRTLLM-Gen as its default routed backend.
- MegaMoE infers SiTU with beta 4.0 and linear beta 25.0.
- MegaMoE defaults to SwiGLU without SiTU configuration.

### TRTLLM-Gen versus MegaMoE SiTU parity

A backend-level parity test was added using:

- The real K3 routed-expert GEMM dimensions: hidden 3584 and intermediate
  3072.
- Identical group-32 packed MXFP4 weights and scales.
- Identical BF16 inputs, router logits, selected experts, and routing
  weights.
- SiTU beta 4.0 and linear beta 25.0.
- Token counts 1, 16, and 128.
- Top-1, top-2, and top-16 routing. The top-16 case uses 32 experts to cover
  the production top-k without allocating the complete 896-expert bank.

All nine cases pass the committed criteria of cosine similarity greater than
0.998 and relative L2 error below 6%.

| Routing | Tokens | Cosine similarity | Relative L2 |
| --- | ---: | ---: | ---: |
| 8 experts, top-1 | 1-128 | 0.99965-0.99967 | 2.64%-2.72% |
| 8 experts, top-2 | 1-128 | 0.99900-0.99909 | 4.31%-4.58% |
| 32 experts, top-16 | 1-128 | 0.99897-0.99908 | 4.34%-4.61% |

The comparison intentionally uses semantic rather than elementwise parity.
MegaMoE folds routing weights into the FC1 activation before its MXFP8
requantization, while TRTLLM-Gen combines expert results at a different point
in the quantized graph. The stable cosine and relative-L2 results, including
at top-16 and increasing token counts, support equivalent SiTU semantics
without hiding the expected quantization-order difference.

Setting DeepGEMM `fast_math=False` produced the same reported metrics, so the
observed difference is not caused by the fast reciprocal used in sigmoid.

### Production-scale and real-checkpoint parity

Follow-up validation on `umbriel-b200-027` completed the two remaining
single-GPU parity milestones:

- Full production routed bank: 896 experts, top-16, 128 tokens, hidden 3584,
  and intermediate 3072. The result was cosine similarity 0.99908459 and
  relative L2 error 4.326262%.
- Real Kimi K3 layer-1 expert weights: 896 experts, top-16, and 128 tokens.
  The routed-kernel result was cosine similarity 0.99947089 and relative L2
  error 3.310431%.
- Real layer-1 routed latent path: the checkpoint router, 7168-to-3584 down
  projection, 896-expert top-16 MoE, routed RMSNorm, and 3584-to-7168 up
  projection. The result was cosine similarity 0.99968600, relative L2 error
  2.506052%, p99 absolute error 0.00073242, and maximum absolute error
  0.00177956. The MegaMoE output contained no NaN or Inf values.

The 896-expert runs bounded the MegaMoE symmetric workspace to 128 tokens.
The 16 GiB checkpoint shard was staged from S3FS to node-local storage before
the real-weight runs to avoid random page-fault stalls.

### Initial EP8 validation

Eight-rank validation also completed on the same B200 node:

- An NCCL all-reduce and barrier smoke test passed on all eight GPUs. GPU 2,
  whose `nvidia-smi` utilization counter remained at 100% without a process or
  memory allocation, participated normally.
- A full fused-versus-reference EP8 parity case passed with 32 experts,
  top-16 routing, hidden 3584, and intermediate 3072. This exercises the
  DeepGEMM fused dispatch, local expert compute, and combine path with the K3
  GEMM geometry.
- A production-size 896-expert, top-16 EP8 fused-forward smoke test passed
  with 112 experts per rank. Every rank returned a finite, nonzero `[8, 3584]`
  output, and the output sum, norm, and maximum matched across all ranks.

The generic test's production-size numerical reference remains impractical:
it expands the complete 896-expert BF16 reference bank on every rank and
estimates 220.5 GiB per GPU, exceeding the B200's 178.3 GiB. The fused backend
itself stayed well within memory limits. Completing the same numerical case on
B200 would require a sharded or CPU/offloaded reference; the B300 run below
instead provides enough device memory for the existing full reference.

### Full production EP8 SiTU parity on B300

The memory-limited numerical case was subsequently completed on Slurm job
`3317909`, node `umb-b300-004`, using eight NVIDIA B300 GPUs with 275040 MiB
per GPU. The validation container was
`jonasl-kimi-k3-megamoe-ep8`, based on
`tensorrt_llm/devel:latest`, with the current locally built TRT-LLM wheel.

The complete fused-versus-reference case used:

- EP8 with 112 local experts per rank.
- 896 routed experts and top-16 routing.
- Eight tokens per rank.
- Hidden size 3584 and intermediate size 3072.
- BF16 inputs, MXFP4 weights, and MXFP8 activation quantization.
- DeepSeek-V3-style routing with FP32 router logits.
- Kimi SiTU with gate beta 4.0 and linear beta 25.0.

The harness asserted that every fused backend instance resolved its activation
configuration to `("situ", 4.0, 25.0)`. Its reference used the same FP32 SiTU
math as the Kimi HF implementation and retained MegaMoE's pre-L2 routing-weight
placement. All eight ranks passed the existing MXFP4/MXFP8 numerical accuracy
criteria, and the process exited successfully with:

```text
B300_EP8_896E_TOP16_KIMI_SITU_FULL_PARITY_PASS
```

This closes the production-dimension synthetic-weight EP8 numerical milestone.
The earlier generic run on the same B300 node also passed but used the test
framework's default SwiGLU activation; it is only a communication and baseline
regression result, not evidence for Kimi SiTU. The explicit activation
assertion above prevents that distinction from being lost.

## Remaining Work

1. **EP8 reference and routing-stress coverage**
   - Repeat the production EP8 comparison with real Kimi checkpoint weights.
   - Compare MegaMoE fused dispatch/combine numerically against the TRTLLM-Gen
     EP path.
   - Cover empty experts, uniform routing, and deliberately hot experts.
   - Treat shared-node results as correctness-only; interference invalidates
     performance measurements.

2. **Accuracy and performance smoke tests**
   - After backend and EP parity pass, run a small fixed-prompt or golden
     output set through Kimi K3.
   - Full-model execution is a final integration and accuracy check, not a
     prerequisite for backend-level SiTU validation.
   - Benchmark latency and throughput only after correctness is established.

3. **Upstreaming**
   - Upstream or otherwise stabilize the DeepGEMM dependency commit.
   - Re-run the TRT-LLM MoE test matrix and relevant B200 CI stages.

## Current Backend Constraints

The DeepGEMM MegaMoE path remains intentionally narrow:

- SM100-family GPUs.
- BF16 activations.
- `W4A8_MXFP4_MXFP8` quantization.
- Hidden and intermediate dimensions divisible by 512.
- MoE expert parallelism only; MoE tensor parallelism is unsupported.
- Multi-rank Kimi use requires attention DP with EP spanning the parallel
  group.
- Kimi packed-checkpoint streaming does not yet support dynamic EPLB or
  replicated expert slots.
