# Kimi-K3 NVFP4 AgentX 低延迟配方（2026-10-20，快照 ll5）

本目录是 AgentX 低延迟段（GEN batch 1 / 2 / 4，helix16，MTP6 / 7）的复现说明。高吞吐段与公共 CTX 设置见
`examples/disaggregated/kimi_k3_agentx_opt/README.md`（快照 s53 / s55）。

## 代码

| 项 | 值 |
|---|---|
| TRT-LLM 分支 | gitlab `ftp/tekit` `user/dongxuy/k3_lowlat`。本配方 = 快照 ll5，代码树与内部整合分支 `k3ll_int5`（`a5177ab03f`）相同；分支上另有本 README 和专家重排表 |
| 起点 | `f4150ffa9e`（`user/dongxuy/k3_agentx_opt` 的 s55 代码）之上 120 个代码提交（ll1 → ll5 五轮） |
| C++ | 相对起点没有 C++ 改动（`git diff f4150ffa9e HEAD -- cpp` 为空），新增的都是 Python / CuTe DSL / Triton，运行时 JIT 编译 |
| 运行镜像 | 与 s55 相同：taizhongw 的 b0d8 镜像（sm103），`/lustre/fsw/portfolios/coreai/users/taizhongw/images/trtllm_ra_b0d8f6484a_ctm_sm100_103.sqsh` |
| 快照 | 做法与 s53 配方相同：在镜像里编译本分支的 sm103（`scripts/build_wheel.py --cuda_architectures "103-real" --skip_building_wheel --linking_install_binary`），把 `tensorrt_llm/`（含编译出的库）+ `triton_kernels/` 复制成快照目录，放在 `PYTHONPATH` 最前面。因为没有 C++ 改动，也可以用 s55 的快照目录，再用本分支覆盖 `git diff --name-only f4150ffa9e HEAD -- tensorrt_llm` 列出的文件（jhb 上即 `/home/dongxuy/work_dir/k3_opt_snaps/ll5`） |
| harness | `bench-trtllm-disagg-opt` 分支 `dongxuy/k3-agentx-opt`（jhb），集群配置 `server-ocjhb-gb300-dongxuy-opt-short.config`（DWDP CTX，QOS short，2 h）；正式点 yaml 的写法见下文 |
| 专家重排表 | 本目录 `placements/k3_ep4_e2eR_c16b2_v1.json`（EP4，92 个 MoE 层，按 e2e c16 b2 的真实路由训练，b1 / b2 使用；md5 `85b43df4b2c456b375a9ae20a92295c3`） |

所有新开关都在环境变量后面，默认关闭。

## 结果（AgentX e2e 3600 s，jhb GB300，16p3d：16 个 CTX + 3 个 helix16 GEN，64 GPU）

指标：tps/user = 1000 / p90 full-response ITL；tps/GPU = 输出 token / s / 全部 64 个 GPU（CTX + GEN）。

| 点 | 代码 | tps/user | out tps/GPU | p90 TTFT | GSM8K strict |
|---|---|---|---|---|---|
| c48 b2（每 GEN 16 条 lane） | **ll5** | **429.18** | **31.28** | 8.7 s | 0.9591 |
| c48 b2 | ll4 | 414.94 | 30.8 | 9.3 s | 0.9598 |
| c48 b2 | ll3 | 392.16 | 30.34 | 10.7 s | 0.9621 |
| c48 b2 | 本轮起点 s53 | 326.8 | 27.22 | 17.5 s | — |
| c24 b1（每 GEN 8 条 lane） | ll4 | 473.93 | 14.8 | 12.7 s | 0.9598 |
| c24 b1 | 本轮起点 s53 | 375.94 | 13.87 | 19.9 s | — |
| c84 b4（每 GEN 28 条 lane，MTP7） | ll2 | 304.88 | 55.17 | 20.6 s | 0.9591 |
| c84 b4 | 本轮起点 s53 | 276.24 | 50.85 | 25.2 s | — |

- ll5 c48 b2：6777 个请求、0 错误；p90 full-response ITL 2.33 ms，mean TPOT 2.22 ms。相对起点 +31.3% / +14.9%，相对 ll4 +3.43% / +1.56%
  （gen-only 回放预测 b2 步长 −3.68%）。
- ll4 c24 b1：3323 个请求、0 错误；p90 full-response ITL 2.11 ms。相对起点 +26.1% / +6.7%。
- GSM8K 参考值 0.9613；门槛 0.95。GSM8K 用 b2 开关组 + `TRTLLM_K3_LOWM_SF_SLOTS=1`，GEN max batch 4（M ≤ 28），不强制接受。
- ll5 的 b1 / b4 组只有 gen-only 回放结果（同一 token 流）：b1 7.860 → 7.509 ms（−4.47%），b4 12.291 → 12.175 ms（−0.94%，见下）。

## 公共配置

yaml 头部（数据集、client、CTX 配置、`extra_ctx_worker_env`、`extra_ctx_config`、`server_config_extra`、
`speculative_config.advanced_sampling_mode`）与 s53 配方相同；低延迟点另外：

```yaml
benchmark_env_var:
  AGENTX_DURATION: 3600
  AGENTX_MAX_CTX: 996579
  AGENTX_WARMUP_PER_LANE: 3
extra_gen_config:
  moe_expert_parallel_size: 4      # b1 / b2 / b4 都用 MoE TP4 x EP4
  moe_tensor_parallel_size: 4
slurm:
  job_time: '02:00:00'             # QOS short；一次 3600 s 正式跑约 1:35-1:45
```

## GEN 开关

### 公共（b1 / b2 / b4 都开）

```yaml
extra_gen_worker_env:
  # 投机解码 / DSpark 尾部
  TLLM_SPEC_VOCAB_PARALLEL: "1"
  TLLM_SPEC_VOCAB_PARALLEL_SAMPLING: "1"
  TLLM_SPEC_TAIL_FUSION: "1"
  TLLM_SPEC_HEAD_FUSION: "1"
  TLLM_DSPARK_DRAFT_FUSION: "1"
  TLLM_DSPARK_FUSED_MARKOV: "1"
  TLLM_DSPARK_MARKOV_V2: "2"                   # ll5 的 b1 / b2 改成 "3"（见下）
  TLLM_DSPARK_TAIL_V2: "1"
  TLLM_DSPARK_DRAFT_GEMM: "1"
  TLLM_DSPARK_CTX_SIDE_STREAM: "1"
  TLLM_DRAFT_KV_GPU_BUDGET_GIB: "24"
  # 步间准备 / 调度
  TRTLLM_DEFER_DISAGG_GEN_RECV: "1"
  TRTLLM_DISAGG_GEN_FAST_SYNC: "1"
  TRTLLM_BATCHED_INPUT_COMMIT: "1"
  TRTLLM_SPEC_SAMPLER_D2H_STREAM: "1"
  TRTLLM_SPEC_SAMPLER_FUSED_SCATTER: "1"
  TRTLLM_STATE_INDEX_COPY_KERNEL: "1"
  TRTLLM_KV_BLOCK_OFFSETS_TRITON: "2"
  TRTLLM_SLICED_HELIX_PARTITION: "1"
  # 小 M GEMM
  TRTLLM_K3_FP8W_LOWM_GEMM: "1"
  TRTLLM_K3_FP8W_LOWM_FUSED_A: "1"
  TRTLLM_K3_FP8_MID_GEMM: "lu,o,qb,qa,g,qkvg"
  TRTLLM_K3_BF16_LOWM_GEMV: "1"
  TRTLLM_LOW_M_GEMM_BACKEND: auto
  TRTLLM_FP8_QUANT_1X128_TACTIC: "cuda"        # 各 rank 固定同一个 1x128 量化 kernel（autotune 会让 rank 选得不一样）
  KIMI_K3_FP8_WEIGHT_READ_SHARED_TP: "1"
  # KDA 注意力 / attn_res
  TRTLLM_K3_KDA_FAB_FUSED: "1"
  TRTLLM_K3_KDA_DECODE_FUSED_FAB: "1"
  TRTLLM_K3_KDA_DECODE_EARLY_TRIGGER: "1"
  TRTLLM_K3_ATTN_RES_PDL: "2"
  TRTLLM_K3_ATTN_RES_EARLY_TRIGGER: "1"
  # MLA 注意力 / Helix
  TRTLLM_HELIX_V_UP_BEFORE_A2A: "1"
  TRTLLM_MLA_PARTIAL_BF16: "1"                 # split-KV partial O 存 BF16（有损，已过 GSM8K）
  TRTLLM_MLA_RESCALE_THRESHOLD: "8"
  TRTLLM_K3_MLA_FP8_Q_ABSORB: "1"
  TRTLLM_K3_MLA_FP8_Q_ABSORB_CUTE: "1"
  TRTLLM_K3_MLA_GATE_FP8: "1"
  TRTLLM_K3_MLA_FUSED_V_UP: "1"
  TRTLLM_K3_MLA_GATE_SIDE: "1"
  TRTLLM_K3_MLA_Q_NORM_PDL: "1"
  TRTLLM_K3_MLA_KV_SIDE: "1"
  TRTLLM_K3_MLA_KV_SIDE_PRIO: "1"
  # all-reduce
  TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES: "2097152"
  TRTLLM_K3_AR_PUSH: "1"                       # o_proj 的 GEMM 尾声直接把行推进 owner rank 的 Lamport 槽
  TRTLLM_K3_AR_PUSH_AMID: "1"                  # A_mid 原地读 all-reduce 结果
  # MoE
  TRTLLM_K3_FUSE_MOE_RESIDUAL: "1"
  TRTLLM_K3_FUSED_ROUTER: "0"
  TRTLLM_K3_ROUTE_TAIL: "1"
  TRTLLM_K3_ROUTE_TAIL_MAX_M: "256"
  TRTLLM_K3_LU_AR_QUANT: "1"
  TRTLLM_MOE_LOWM: "1"
  TRTLLM_MOE_WEIGHT_EVICT_FIRST: "1"
```

### ll5：b1 / b2 都加（b4 回放也确认有效，见下）

```yaml
  # decode 尾部（lltail2，全部逐位一致，DSPARK_MLA_FUSED 只改 draft 的数值）
  TLLM_DSPARK_MARKOV_V2: "3"                   # Markov chain 一个 CuTe kernel（cluster 内归约的跨 rank 交换）
  TLLM_DSPARK_LMHEAD_ALL_ROWS: "1"             # draft LM head 对所有 block 行算，去掉 gather
  TLLM_DSPARK_ROPE_PDL: "1"
  TRTLLM_MLA_FP16_PDL: "1"                     # drafter 的 BF16 MLA decode（split-KV + reduction）走 PDL
  TLLM_DSPARK_NORM_EARLY_TRIGGER: "1"
  TLLM_DSPARK_MLA_FUSED: "1"                   # drafter 每层一个 clustered CuTe 注意力 kernel
  # MLA 块（llmla2，逐位一致）
  TRTLLM_K3_MLA_KV_FUSED: "1"                  # kv_a RMSNorm + latent 行 + FP8 KV 写入 + decode 缓冲合成一个 kernel
  TRTLLM_K3_MLA_KV_JOIN_EARLY: "1"
  TRTLLM_HELIX_A2A_FUSED: "1"                  # Helix all-to-all + 后处理融合
  TRTLLM_MLA_KV_EVICT_FIRST: "v"
  # KDA / MoE 交接（llkda2，逐位一致）
  TRTLLM_K3_SHARED_JOIN_AFTER_LU: "1"          # shared expert 在 latent_up 之后才 join
```

### b2（c48 b2 正式点）另外

```yaml
  TRTLLM_K3_KDA_L2_PREFETCH: "1"
  TRTLLM_K3_ATTN_RES_MXFP8: "1"
  TRTLLM_K3_MX_SF_EARLY_RELEASE: "1"           # ll5：MXFP8 小 M GEMM 每个 K tile 的 scale 折完就释放
  TRTLLM_K3_LD_FP4: "1"
  TRTLLM_K3_ROUTER_FIRST: "1"
  TRTLLM_K3_ROUTER_EXACT: "kda"                # 精确 router 只用在 KDA 层（MLA 层会拖慢 route tail）
  TRTLLM_K3_LATENT_AR_OWN_WS: "1"
  TRTLLM_K3_LATENT_AR_CUTE: "1"
  TRTLLM_K3_LATENT_AR_ROW: "1"                 # latent all-reduce 每行一个 CTA 的 row kernel
  TRTLLM_K3_LATENT_AR_SIDE_CLEAR: "ld"         # ll5：Lamport 清零放 latent_down 流，按 token 数自动选位置并显式 join
  TRTLLM_K3_LU_MXFP8: "1"
  TRTLLM_K3_LU_MX_STAGES: "12:8"
  TRTLLM_MOE_LOWM_FC1_CTAS: "8:128,16:120"
  TRTLLM_MOE_LOWM_FC2_CTAS: "8:120,16:128"
  TRTLLM_MOE_LOWM_FC1_TRIGGER: "8:last,16:last"
  TRTLLM_MOE_LOWM_DYN: "1"                     # 9-16 token 时 FC1 + FC2 一个动态调度的融合 grid
  TRTLLM_MOE_LOWM_DYN_MIN_TOKENS: "9"
  TRTLLM_MOE_LOWM_DYN_MAX_TOKENS: "16"
  TRTLLM_MOE_LOWM_DYN_CTAS: "16:136"
  TRTLLM_K3_SHARED_TAIL_PRIO: "-1"
  TRTLLM_K3_SHARED_TAIL_MIN_M: "9"
  TRTLLM_K3_SHARED_TAIL_MAX_M: "16"
  TRTLLM_K3_EXPERT_PLACEMENT: <本目录>/placements/k3_ep4_e2eR_c16b2_v1.json   # EP4 静态专家重排
```

### b1（c24 b1）另外

```yaml
  TRTLLM_K3_KDA_L2_PREFETCH: "1"
  TRTLLM_K3_LOWM_SF_SLOTS: "1"
  TRTLLM_K3_LD_FP4: "1"
  TRTLLM_K3_ROUTER_FIRST: "1"
  TRTLLM_K3_ROUTER_EXACT: "kda"
  TRTLLM_K3_LATENT_AR_OWN_WS: "1"
  TRTLLM_K3_LATENT_AR_CUTE: "1"
  TRTLLM_K3_LATENT_AR_ROW: "1"
  TRTLLM_K3_LATENT_AR_SIDE_CLEAR: "ld"
  TRTLLM_K3_LU_MXFP8: "1"
  TRTLLM_K3_LU_MX_STAGES: "12:8"
  TRTLLM_MOE_LOWM_FC1_CTAS: "8:128,16:120"
  TRTLLM_MOE_LOWM_FC2_CTAS: "8:120,16:128"
  TRTLLM_MOE_LOWM_FC1_TRIGGER: "8:last,16:last"
  TRTLLM_K3_EXPERT_PLACEMENT: <本目录>/placements/k3_ep4_e2eR_c16b2_v1.json
```

### b4（c84 b4，MTP7）

正式点仍是 ll2 的组（公共开关）。ll5 在 b4 的回放 A/B（同一 token 流）：公共 12.291 → + 上面 lltail2 尾部的 6 个开关 12.243（−48 us）
→ 再 + MLA 四开关 12.175 ms（累计 −116 us，−0.94%）；`TRTLLM_K3_ATTN_RES_MXFP8` + `TRTLLM_K3_MX_SF_EARLY_RELEASE` 在 b4 无收益。
MoE 前后段开关在 b4 的回放还在测（下一版）。b4 上 `TRTLLM_K3_ROUTER_EXACT`、`LATENT_AR_CUTE` / `_ROW`、`LU_MXFP8`、FC1/FC2 CTA 表、
`SHARED_TAIL_*`、`MOE_LOWM_DYN`、`KDA_L2_PREFETCH` 都不生效（M = 32 超出它们的范围）。

## 拓扑（gen_configs）

`[ctx_num, gen_num, gen_cp, batch, max_num_tokens, helix, gpu_mem, mtp, eplb, 每个 GEN 的并发]`

| 点 | 行 |
|---|---|
| c24 b1 | `[16, 3, 16, 1, 7, helix, 0.8, 6, 0, 8]` |
| c48 b2 | `[16, 3, 16, 2, 14, helix, 0.8, 6, 0, 16]` |
| c84 b4 | `[16, 3, 16, 4, 32, helix, 0.8, 7, 0, 28]` |
| 16p4d c80 b2（在测） | `[16, 4, 16, 2, 14, helix, 0.8, 6, 0, 20]`（20 节点） |
| 16p4d c48 b1（在测） | `[16, 4, 16, 1, 7, helix, 0.8, 6, 0, 12]`（20 节点） |

AgentX 的 lane 是闭环的：tps/user 由 batch 上限和步长决定，lane 只影响排队（TTFT）和 GEN slot 利用率。正式点里每个 GEN 的
slot 利用率：c24 b1 约 65%，c48 b2 约 74–76%，c84 b4 约 94%；CTX（DWDP16）每个请求 prefill 约 2 s、整体只忙约 12%。
所以 b1 / b2 加 lane、多挂 GEN（16p4d 摊薄 CTX 的 16 张卡）可以在 tps/user 不变时提高 tps/GPU（上面两行在测）。
b2 上 MTP7 与 MTP6 持平（e2e 代理 393.7 vs 393.7），所以用 MTP6。

## ll5 新增的优化（相对 ll4；gen-only 回放，同一 token 流）

| 开关 | 内容 | 数值 | 收益 |
|---|---|---|---|
| `TRTLLM_K3_SHARED_JOIN_AFTER_LU` + `TRTLLM_K3_MX_SF_EARLY_RELEASE`（b2） | shared expert 的流在 latent_up 之后才 join（标记 kernel）；MXFP8 小 M GEMM 每个 K tile 的 scale 折完就释放 stage | 逐位一致 | b2 −165 us/步（b1 只用前者） |
| lltail2 尾部 6 个开关 | CuTe Markov chain、draft LM head 全行、RoPE / MLA / norm 的 PDL、drafter 融合注意力 | 前 5 个逐位一致；`TLLM_DSPARK_MLA_FUSED` 只改 draft 数值（与原路径误差同级） | b2 −47 us/步（前 5 个）再 −34（MLA_FUSED）；b1 −21 us（MLA_FUSED）；b4 合计 −48 |
| MLA 四开关 | KV 半边一个 kernel、KV 提前 join、融合 Helix a2a、decode 读 V 时 EVICT_FIRST | 逐位一致 | b2 −86 us/步，b1 −100 us/步（−1.31%），b4 −68 us/步 |
| `TRTLLM_K3_LATENT_AR_SIDE_CLEAR=ld` 修复 | 清零位置按调用的 token 数选（融合 kernel 范围内用 1，否则 ld），用 ld 时在 latent AR 前显式 join | 逐位一致 | 让 b2 也能用 ld |
| `TLLM_SPEC_DECODE_FORCE_REPLAY`（测量） | teacher-forced 回放一次真实运行录下的 token 流，同一配置步长复现到 0.02% | 仅 gen-only 测量用，永远不要用于 GSM8K / 正式点 | — |

合计（回放）：b2 9.076 → 8.742 ms（−3.68%），b1 7.860 → 7.509 ms（−4.47%）。

另有默认关、本版未启用的：`TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS`（batch 满时提前接收下一个请求做 KV 传输，修好了 seq slot 溢出，e2e 待测）、
`TLLM_SPEC_DECODE_FORCE_ACCEPT_KEY=state` / `TLLM_SPEC_DECODE_STREAM_LOG`（gen-only 测量用，见下）。

## 不要打开的开关

| 开关 | 原因 |
|---|---|
| `TRTLLM_MOE_LOWM_DYN` 用于 b1 / b4 | b1 持平（token 范围已排除 M = 7）；b4 +5 us/层 |
| 去掉 `TRTLLM_MOE_LOWM_FC1_CTAS` / `_FC2_CTAS` / `_FC1_TRIGGER`（b1 / b2） | 同 token 流对比 +1.5% 步长；b4（M = 32）则不要这些上限 |
| `TRTLLM_K3_ROUTER_EXACT=1`（所有层） | MLA 层 route tail 被 latent_down + shared gate_up 的突发拖慢，+1.87 us/层；用 `kda` |
| `TRTLLM_K3_MOE_FRONT_PREFETCH`、`TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA`、`TLLM_DSPARK_L2_PREFETCH` | 把 HBM 读挪进注意力 / AR 阶段，在 situ 净变慢（replica 不带邻居的 HBM 流量时会误判为变快） |
| `TLLM_DSPARK_AR_NORM`（以及只和它一起测过的 `TLLM_DSPARK_DRAFT_EXACT_SMEM`、`TLLM_DSPARK_DRAFT_PREFETCH`） | drafter 的融合 AR + norm 在 TP16 下更慢（暴露 11.0–11.7 vs 7.6–7.9 us） |
| `TRTLLM_K3_ATTN_RES_MXFP8` + `TRTLLM_K3_MX_SF_EARLY_RELEASE`（b4） | qkvg −226 us 但 kda_fab 变关键 +191、A_pre 的 MX 版 +0.47 us/层，净 0 |
| `TRTLLM_K3_MLA_GATE_EARLY` | 与 a2a 抢资源，+2.1 us/层 |
| `TRTLLM_MLA_KV_EVICT_FIRST=k` / `kv`（b4） | K tile 的 L2 复用丢失，b4 decode +1.2…+2.0 us；b2 上 `kv` 与 `v` 相同，用 `v` |
| o_proj / shared AR 改 one-shot（`TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES` 4 MiB）、shared + latent AR 合并、b2 的 MoE TP16 | b2 +7% / +4.4% / +1.8% |
| 精度测试 | 去掉 `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`（harness 的 gen-only 性能跑注入固定接受长度），也不要带 `TLLM_SPEC_DECODE_FORCE_REPLAY` |

## 精度（GSM8K）

在 b2 开关组上加 `TRTLLM_K3_LOWM_SF_SLOTS: "1"`，`gsm8k_eval: true`，gen_configs `[16, 1, 16, 4, 28, helix, 0.8, 6, 0, 16]`
（GEN max batch 4，覆盖 M ≤ 28），集群配置用 DWDP 的 `opt*` 配置（`-dep` 配置下 CTX 会 OOM）。不要带任何强制接受 / 回放的变量。

## gen-only A/B 的测量方法

- gen-only 跑法（`opt-dep-short` 集群配置，CTX 1 + GEN 1，nsys 窗口 100-120 次迭代）用强制接受（2.75 个 draft / 步）。
  MoE FC2 + finalize 的 BF16 原子求和运行间不逐位确定，强制接受会把一个近似平票的 token 翻转放大成另一个退化 token 循环：
  同一配置两次运行的步长可差 2-3.5%，而且退化循环碰到的专家比真实文本少（gen-only 比 e2e 乐观）。
- ll5 起用 teacher-forced 回放：GEN 环境加 `TLLM_SPEC_DECODE_FORCE_ACCEPT_KEY: state` 和
  `TLLM_SPEC_DECODE_FORCE_REPLAY: <参考流 jsonl>`。参考流来自一次不强制接受的真实运行（GEN 环境
  `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS: "0"` + `TLLM_SPEC_DECODE_STREAM_LOG: <dir>`，用 `<dir>/rank0.jsonl`，按数据集的最大并发录，
  例如 b4 MTP7 录一次可供 b1 / b2 / b4 用）。回放下同一配置的步长复现到 0.02%，各 case 走同一 token 流，路由接近真实文本
  （b2 步长比旧的强制接受高约 5%：基线要重测，不要跨口径比较）。
- 以 GEN 迭代日志里 nsys 窗口外的步长 p50 为准；`nsys --cuda-graph-trace=node` 会把窗口内的 PDL 边变成完整边，对不同流拓扑的配置
  代价不同（b1 实测 A +129、B +47 us/步）。
