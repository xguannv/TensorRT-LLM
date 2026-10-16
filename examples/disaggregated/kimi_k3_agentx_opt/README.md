# Kimi-K3 NVFP4 AgentX 最新优化配方（2026-10-15，快照 s53）

## 代码

| 项 | 值 |
|---|---|
| TRT-LLM 分支 | gitlab `ftp/tekit` `user/dongxuy/k3_agentx_opt`。本配方对应快照 s53 = `91e96bff7c`。分支头之后的 `fa56986d3a`（测试）、`4de7004ab3`（文档）、`1e68f87f10`（可选的 MoE tactic 固定开关，默认关）不改变默认行为，可直接带上 |
| 起点 | `b0d8f6484a`（rubin-advance 9-23，0924 参考）→ 快进 rubin-advance `9a51c77ed5` + cherry-pick |
| 运行镜像 | taizhongw 的 b0d8 镜像（sm103）。分支需在该镜像里编译 sm103（`scripts/build_wheel.py --cuda_architectures "103-real" --skip_building_wheel --linking_install_binary`；有 C++ 改动），然后把 `tensorrt_llm/`（含编译出的库）+ `triton_kernels/` 复制成快照目录，放在 `PYTHONPATH` 最前面 |
| 快照 | jhb `/home/dongxuy/work_dir/k3_opt_snaps/s53`（目录里的 `COMMIT` 文件记录 commit，工作区干净） |
| harness | `bench-trtllm-disagg-opt` 分支 `dongxuy/k3-agentx-opt`（jhb，未推远端）；yaml：`opt/jhb_yamls/jhb_s53f_*.yaml`（各开关全在 yaml 里，以下各节就是这些 yaml 的内容） |
| 专家重排表 | 本目录 `placements/k3_ep16_e2e_c88b32_v2.json`（c264 b32）、`placements/k3_ep16_e2e_b128like_v2.json`（c220 b128） |

所有优化都在环境变量开关后面，默认关闭（s48 的低 batch MLA decode、s49 的 B ≥ 32 MLA decode 流水、s50 的 KDA fab 分级 tail 除外：这三项逐位一致、默认开）。

## 公共配置（所有点）

ctx_config: `{tp_size: 1, max_batch: 16, max_num_tokens: 32768, enable_attention_dp: true, gpu_memory_fraction: 0.67}`，16 个 CTX（DWDP16）。

```yaml
extra_ctx_config:
  moe_config:
    max_num_tokens: 32768                       # 一次迭代 32k token，摊薄每次约 2 s 的 DWDP gather
  kv_cache_config:
    block_reuse_config:
      max_num_turns: 3
extra_ctx_worker_env:
  K3_TOKENIZER_FASTPATH: "1"
  TLLM_V2_CTX_DEMOTE_UNCACHED_TOKENS: "16384"
  TLLM_V2_CTX_DEMOTE_MAX_ITERS: "8"
  TLLM_ROUTING_DISABLE_COOP_LAUNCH: "1"
  TRTLLM_K3_DEEPGEMM_SM_RESERVE: "16"           # 给 DWDP gather 留 16 个 SM
  TRTLLM_CUTEDSL_SM_RESERVE: "16"
  TRTLLM_DWDP_GATHER_ROWS: "8"                  # gather 8 行 × 16 块
  TRTLLM_DWDP_GATHER_BLOCKS: "16"
  TRTLLM_K3_KDA_PREFILL_SM_RESERVE: "1"         # KDA prefill grid 让出 gather 的 SM
  TRTLLM_K3_MLA_CTX_FP8_EXPAND: "2"             # 融合 cached-KV 展开
  TRTLLM_K3_MLA_CTX_FMHA: "cute"                # CuTe DSL FP8 MLA 上下文 FMHA（非逐位一致，GSM8K 已过）
  TRTLLM_MLA_CHUNKED_PREFILL_MAX_TOKENS: "65536" # 分块 prefill 分区上限（否则 KV cache v2 扩容时 OOM）
  TLLM_V2_LOW_PRIORITY_SNAPSHOT_PROMPT_LEN: "16384" # 16k 以下 prompt 的 KDA 快照最先逐出
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

### 低 batch b1 / b2 / b4（MoE TP4×EP4）——c18 b1、c36 b2、c84 b4（16p3d）

```yaml
extra_gen_config: {moe_expert_parallel_size: 4, moe_tensor_parallel_size: 4}
  KIMI_K3_FP8_WEIGHT_READ_SHARED_TP: "1"
  TRTLLM_K3_ATTN_RES_PDL: "2"
  TRTLLM_LOW_M_GEMM_BACKEND: auto
  TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES: "2097152"
  TRTLLM_MLA_PARTIAL_BF16: "1"                # split-KV partial O 存 BF16（有损，GSM8K 0.9606）
```

### b8 / b16（MoE TP16 + swap-AB）——c105 b8、c171 b16（16p3d）、c116 b16（16p2d）

```yaml
extra_gen_config: {moe_expert_parallel_size: 1, moe_tensor_parallel_size: 16}
  KIMI_K3_FP8_WEIGHT_READ_SHARED_TP: "1"
  TRTLLM_K3_ATTN_RES_PDL: "2"
  TRTLLM_K3_ATTN_RES_PDL_MAX_M: "128"        # 只有 b16
  TRTLLM_LOW_M_GEMM_BACKEND: auto
  TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES: "2097152"
  TRTLLM_MLA_PARTIAL_BF16: "1"
  TRTLLM_MOE_FINALIZE_SWAPAB: "1"
  TRTLLM_MOE_FINALIZE_SWAPAB_SPLIT_ROWS: "1"
  TRTLLM_MOE_FINALIZE_SWAPAB_EARLY_WEIGHTS: "1"
```

### b ≥ 32（MoE EP16）——c264 b32、16p2d c302 / c360 b64、c220 b128、c300 b192、c400 b256

```yaml
  TRTLLM_HELIX_A2A_FUSED: "1"
  TRTLLM_K3_KDA_DECODE_2CTA: "1"
  TRTLLM_K3_KDA_DECODE_FP8_OUT: "1"
  TRTLLM_K3_LATENT_STREAM_PRIORITY: "-1"
  TRTLLM_K3_LD_FUSED: "1"               # c264 b32 用 "2"
  TRTLLM_K3_LU_FUSED: "1"               # c264 b32 用 "2"
  TRTLLM_NVLS_AR: "1"
  TRTLLM_NVLS_AR_V2: "1"
  TRTLLM_NVLS_AR_PROTOCOL: "ptx"        # 符合 PTX 内存模型的到达协议（每次调用一次 sys fence），结果与 gpu 协议逐位相同
  TRTLLM_NVLS_AR_ZERO_COPY: "1"         # o_proj / shared down 的 GEMM 直接写 NVLS 对称缓冲
  # 只有 c264 b32 和 c220 b128：EP16 静态专家重排
  TRTLLM_K3_EXPERT_PLACEMENT: <本目录>/placements/k3_ep16_e2e_c88b32_v2.json     # c264 b32
  TRTLLM_K3_EXPERT_PLACEMENT: <本目录>/placements/k3_ep16_e2e_b128like_v2.json   # c220 b128
```

## 拓扑（gen_configs）

`[ctx_num, gen_num, gen_cp, batch, max_num_tokens, helix, gpu_mem, mtp, eplb, 每个 GEN 的并发]`

| 点 | 行 |
|---|---|
| c18 b1 | `[16, 3, 16, 1, 7, helix, 0.8, 6, 0, 6]` |
| c36 b2 | `[16, 3, 16, 2, 14, helix, 0.8, 6, 0, 12]` |
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
| `TRTLLM_K3_FUSED_ROUTER=sort` | 同上 |
| `TRTLLM_NVLS_AR` 用于 b ≤ 16（MoE TP16） | 收益约 0.2%，曾在预热时崩溃 |
| `TRTLLM_HELIX_A2A_FUSED` 用于 b ≤ 16 | 持平 |
| `TRTLLM_NVLS_AR_FENCE_SYS=1` | 只作用于 v1，b128 +2.3%；用 `TRTLLM_NVLS_AR_PROTOCOL=ptx` |
| `TRTLLM_NVLS_AR_ROWS=4` | b128 shared AR 上关键路径，变慢 |
| `TRTLLM_K3_KDA_FAB_MAX_ROWS=384`（b64） | 非逐位一致，未过 GSM8K |
| b128 mtp5 | 步长 −2.2%，但每个提交 token 慢 1.4% |
| CTX `block_reuse_config.max_num_turns` 1 / 16 | 都比 3 / 5 差 |
| 精度测试 | 去掉 `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`（harness 性能跑注入的固定接受长度） |

## 结果（AgentX e2e 3600 s，jhb GB300；tps/user = 1000 / p90 ITL，tps/GPU 按全部 GPU）

| 点 | 快照 | tps/user | out tps/GPU | p90 TTFT | 0924 参考 |
|---|---|---|---|---|---|
| c18 b1 | s53f | 374.53 | 11.37 | 12.8 s | 256.41 / 9.44 |
| c36 b2 | s53f | 328.95 | 22.59 | 9.5 s | 16p6d-helix8 c36 b1 226.76 / 17.46 |
| c171 b16 | s53f | 166.39 | 103.83 | 7.9 s | 106.27 / 78.36 |
| 16p2d c360 b64 | s53f | 84.96 | 219.75 | 15.7 s | — |
| c220 b128 | s52f | 57.94 | 196.73 | 8.6 s | 38.28 / 133.05 |
| c300 b192 | s53f | 39.82 | 223.83 | 11.4 s | — |
| c400 b256 | s53f | 31.5 | 243.4 | 19.2 s | — |

其余点（c84 b4、c105 b8、16p2d c116 b16、c264 b32、16p2d c302 b64、c220 b128 + 专家重排）的 s53f 在跑或排队。
GSM8K：s53 b16 + 全套 CTX 设置 0.9644 / 0.9651；s53 c220 GEN 配置 + 专家重排 0.9621 / 0.9629（参考 0.9613 / 0.9621）。
