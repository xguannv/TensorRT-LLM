# Kimi-K3 NVFP4 AgentX 最新优化配方（2026-10-13）

## 代码

| 项 | 值 |
|---|---|
| TRT-LLM 分支 | gitlab `ftp/tekit` `user/dongxuy/k3_agentx_opt`。本配方对应快照 s44 = `b71f9788d3`（b ≤ 16 的点用 s43 = `6158a944b6`，两者差别只在融合 Helix a2a 的修复，b ≤ 16 不开 a2a 融合）。分支头之后的 `185efc7b0f`（只改测试）和 `310160b62f`（CuTe q absorb 给 RoPE kernel 留 SMEM，输出不变，A/B 待测）可直接带上 |
| 起点 | `b0d8f6484a`（rubin-advance 9-23，0924 参考）→ 快进 rubin-advance `9a51c77ed5` + cherry-pick |
| 运行镜像 | taizhongw 的 b0d8 镜像（sm103）。分支需在该镜像里编译 sm103（`scripts/build_wheel.py --cuda_architectures "103-real" --skip_building_wheel --linking_install_binary`；有 C++ 改动：Helix a2a fence 修复、MNNVL AR one-shot 上限），然后把 `tensorrt_llm/`（含编译出的库）+ `triton_kernels/` 复制成快照目录，放在 `PYTHONPATH` 最前面 |
| 快照 | jhb `/home/dongxuy/work_dir/k3_opt_snaps/s44`、`s43`（每个目录的 `COMMIT` 文件记录 commit，工作区干净） |
| harness | `bench-trtllm-disagg-opt` 分支 `dongxuy/k3-agentx-opt`（jhb，未推远端）；yaml：`opt/jhb_yamls/jhb_s44f_*.yaml`、`jhb_s43f_*.yaml`（各开关全在 yaml 里，以下各节就是这些 yaml 的内容） |

所有优化都在环境变量开关后面，默认关闭；不设开关就是继承代码（S1）的行为。

## 公共配置（所有点）

ctx_config: `{tp_size: 1, max_batch: 16, max_num_tokens: 16384, enable_attention_dp: true, gpu_memory_fraction: 0.7}`，16 个 CTX（DWDP16）。

```yaml
extra_ctx_config:
  moe_config:
    max_num_tokens: 16384          # MoE 一个 16k 分块（原 8192）
extra_ctx_worker_env:
  K3_TOKENIZER_FASTPATH: "1"
  TLLM_V2_CTX_DEMOTE_UNCACHED_TOKENS: "16384"
  TLLM_V2_CTX_DEMOTE_MAX_ITERS: "8"
  TLLM_ROUTING_DISABLE_COOP_LAUNCH: "1"
  TRTLLM_K3_DEEPGEMM_SM_RESERVE: "30"   # 给 DWDP gather 留 30 个 SM
  TRTLLM_K3_MLA_CTX_FP8_EXPAND: "2"     # 融合 cached-KV 展开 + FlashInfer FP8 ragged FMHA
  FLASHINFER_WORKSPACE_BASE: <共享目录>
extra_server_env:
  K3_TOKENIZER_FASTPATH: "1"
extra_gen_config:
  speculative_config:
    advanced_sampling_mode: no_topk_no_topp
server_config_extra:
  gen_strip_message_history: true
  gen_tokids_ctxbytes: true
  num_workers: 4
```

GEN 公共开关：

```yaml
  # 投机解码 / DSpark
  TLLM_SPEC_VOCAB_PARALLEL: "1"
  TLLM_SPEC_VOCAB_PARALLEL_SAMPLING: "1"
  TLLM_SPEC_TAIL_FUSION: "1"
  TLLM_DSPARK_DRAFT_FUSION: "1"
  TLLM_DSPARK_FUSED_MARKOV: "1"
  TLLM_DRAFT_KV_GPU_BUDGET_GIB: "24"
  TRTLLM_DEFER_DISAGG_GEN_RECV: "1"
  # MLA / Helix
  TRTLLM_HELIX_V_UP_BEFORE_A2A: "1"
  TRTLLM_K3_MLA_FP8_Q_ABSORB: "1"
  TRTLLM_K3_MLA_FP8_Q_ABSORB_CUTE: "1"
  TRTLLM_MLA_RESCALE_THRESHOLD: "8"
  TRTLLM_K3_MLA_GATE_FP8: "1"
  # GEMM
  TRTLLM_K3_FP8W_LOWM_GEMM: "1"
  TRTLLM_K3_FP8W_LOWM_FUSED_A: "1"
  TRTLLM_K3_FP8_MID_GEMM: "lu,o,qb,qa,g,qkvg"
  TRTLLM_K3_BF16_LOWM_GEMV: "1"
  # KDA
  TRTLLM_K3_KDA_FAB_FUSED: "1"
  TRTLLM_K3_KDA_DECODE_FUSED_FAB: "1"
  # MoE
  TRTLLM_K3_FUSE_MOE_RESIDUAL: "1"
  TRTLLM_K3_ROUTE_TAIL: "1"
  TRTLLM_K3_ROUTE_TAIL_MAX_M: "256"
  TRTLLM_K3_LU_AR_QUANT: "1"
  TRTLLM_MOE_WEIGHT_EVICT_FIRST: "1"
  TRTLLM_K3_FUSED_ROUTER: "0"
```

## 按 batch 的差异

### 低 batch b1 / b4（MoE TP4×EP4）——c18 b1、c84 b4（16p3d）

```yaml
extra_gen_config: {moe_expert_parallel_size: 4, moe_tensor_parallel_size: 4}
  KIMI_K3_FP8_WEIGHT_READ_SHARED_TP: "1"
  TRTLLM_K3_ATTN_RES_PDL: "2"
  TRTLLM_LOW_M_GEMM_BACKEND: auto
  TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES: "2097152"
```

### b8 / b16（MoE TP16 + swap-AB）——c105 b8、c171 b16（16p3d）、c116 b16（16p2d）

```yaml
extra_gen_config: {moe_expert_parallel_size: 1, moe_tensor_parallel_size: 16}
  KIMI_K3_FP8_WEIGHT_READ_SHARED_TP: "1"
  TRTLLM_K3_ATTN_RES_PDL: "2"
  TRTLLM_K3_ATTN_RES_PDL_MAX_M: "128"        # 只有 b16
  TRTLLM_LOW_M_GEMM_BACKEND: auto
  TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES: "2097152"
  TRTLLM_MOE_FINALIZE_SWAPAB: "1"
  TRTLLM_MOE_FINALIZE_SWAPAB_SPLIT_ROWS: "1"
  TRTLLM_MOE_FINALIZE_SWAPAB_EARLY_WEIGHTS: "1"
```

### b ≥ 32（MoE EP16）——c264 b32、16p2d c302 / c360 b64、c220 b128、c300 b192、c400 b256

```yaml
  TRTLLM_HELIX_A2A_FUSED: "1"           # 需快照 s44（b71f9788d3 修复）
  TRTLLM_K3_KDA_DECODE_2CTA: "1"
  TRTLLM_K3_KDA_DECODE_FP8_OUT: "1"
  TRTLLM_K3_LATENT_STREAM_PRIORITY: "-1"
  TRTLLM_K3_LD_FUSED: "1"               # c264 b32 用 "2"
  TRTLLM_K3_LU_FUSED: "1"               # c264 b32 用 "2"
  TRTLLM_NVLS_AR: "1"
  TRTLLM_NVLS_AR_V2: "1"
```

## 拓扑（gen_configs）

`[ctx_num, gen_num, gen_cp, batch, max_num_tokens, helix, gpu_mem, mtp, eplb, 每个 GEN 的并发]`

| 点 | 行 |
|---|---|
| c18 b1 | `[16, 3, 16, 1, 7, helix, 0.8, 6, 0, 6]` |
| c84 b4 | `[16, 3, 16, 4, 32, helix, 0.8, 7, 0, 28]` |
| c105 b8 | `[16, 3, 16, 8, 56, helix, 0.8, 6, 0, 35]` |
| c171 b16 | `[16, 3, 16, 16, 128, helix, 0.8, 7, 0, 57]` |
| 16p2d c116 b16 | `[16, 2, 16, 16, 96, helix, 0.8, 5, 0, 58]` |
| c264 b32 | `[16, 3, 16, 32, 256, helix, 0.8, 7, 0, 88]` |
| 16p2d c302 b64 | `[16, 2, 16, 64, 384, helix, 0.8, 5, 0, 151]` |
| 16p2d c360 b64 | `[16, 2, 16, 64, 384, helix, 0.8, 5, 0, 180]` |
| c220 b128 | `[16, 1, 16, 128, 896, helix, 0.8, 6, 0, 220]` |
| c300 b192 | `[16, 1, 16, 192, 1344, helix, 0.8, 6, 0, 300]` |
| c400 b256 | `[16, 1, 16, 256, 1792, helix, 0.8, 6, 0, 400]` |

segment 16（不用 segment 8）。jhb 集群配置需加 OccupiedIdleGPUsJobReaper 豁免（多 GEN 预热会空闲超过 30 min）。

## 不要打开的开关

| 开关 | 原因 |
|---|---|
| `TRTLLM_K3_SHARED_FIRST`、M > 256 多 CTA 路由（s34+） | shared expert 饿死，b128 变慢 |
| `TRTLLM_K3_FUSED_ROUTER=sort` | 同上（低 batch 在重新评估） |
| `TRTLLM_NVLS_AR` 用于 b ≤ 16（MoE TP16） | 收益约 0.2%，曾在预热时崩溃 |
| `TRTLLM_HELIX_A2A_FUSED` 用于 b ≤ 16 | 持平；且必须用 s44 及以后，否则 GEN 启动段错误 |
| CTX `block_reuse_config.max_num_turns` 1 / 16 | 都比默认 5 差 |
| 精度测试 | 去掉 `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`（harness 性能跑注入的固定接受长度） |

## 结果状态

- 本配方（s43f / s44f）的 11 个正式点 e2e 正在 jhb 上排队，尚无 e2e 数字。
- gen-only：b128 53.39 ms（s32 55.43）、b16 20.94 ms（s33 21.89）。新增开关全部逐位 / 逐字节一致；GSM8K b16 s38 0.9613。
- 目前已测得的最好 e2e 点来自更早的快照（s28 / s30 / s33，开关是本配方的子集），见网页正式点表。
