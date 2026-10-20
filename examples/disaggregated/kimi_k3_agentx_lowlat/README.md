# Kimi-K3 NVFP4 AgentX 低延迟配方（2026-10-20，快照 ll4）

本目录是 AgentX 低延迟段（GEN batch 1 / 2 / 4，helix16，MTP6 / 7）的复现说明。高吞吐段与公共 CTX 设置见
`examples/disaggregated/kimi_k3_agentx_opt/README.md`（快照 s53 / s55）。

## 代码

| 项 | 值 |
|---|---|
| TRT-LLM 分支 | gitlab `ftp/tekit` `user/dongxuy/k3_lowlat`。本配方 = 快照 ll4，代码树与内部整合分支 `k3ll_int4`（`65f3409047`）相同；分支头是 ll4 的代码加本 README |
| 起点 | `f4150ffa9e`（`user/dongxuy/k3_agentx_opt` 的 s55 代码）之上 103 个提交（ll1 → ll4 四轮） |
| C++ | 相对起点没有 C++ 改动（`git diff f4150ffa9e HEAD -- cpp` 为空），新增的都是 Python / CuTe DSL / Triton，运行时 JIT 编译 |
| 运行镜像 | 与 s55 相同：taizhongw 的 b0d8 镜像（sm103），`/lustre/fsw/portfolios/coreai/users/taizhongw/images/trtllm_ra_b0d8f6484a_ctm_sm100_103.sqsh` |
| 快照 | 做法与 s53 配方相同：在镜像里编译本分支的 sm103（`scripts/build_wheel.py --cuda_architectures "103-real" --skip_building_wheel --linking_install_binary`），把 `tensorrt_llm/`（含编译出的库）+ `triton_kernels/` 复制成快照目录，放在 `PYTHONPATH` 最前面。因为没有 C++ 改动，也可以用 s55 的快照目录，再用本分支覆盖 `git diff --name-only f4150ffa9e HEAD -- tensorrt_llm` 列出的文件（jhb 上即 `/home/dongxuy/work_dir/k3_opt_snaps/ll4`） |
| harness | `bench-trtllm-disagg-opt` 分支 `dongxuy/k3-agentx-opt`（jhb），集群配置 `server-ocjhb-gb300-dongxuy-opt-short.config`（DWDP CTX，QOS short，2 h）；正式点 yaml 的写法见下文 |
| 专家重排表 | 本目录 `placements/k3_ep4_e2eR_c16b2_v1.json`（EP4，92 个 MoE 层，按 e2e c16 b2 的真实路由训练，b1 / b2 使用；md5 `85b43df4b2c456b375a9ae20a92295c3`） |

所有新开关都在环境变量后面，默认关闭。

## 结果（AgentX e2e 3600 s，jhb GB300，16p3d：16 个 CTX + 3 个 helix16 GEN，64 GPU）

指标：tps/user = 1000 / p90 full-response ITL；tps/GPU = 输出 token / s / 全部 64 个 GPU（CTX + GEN）。

| 点 | 代码 | tps/user | out tps/GPU | p90 TTFT | GSM8K strict |
|---|---|---|---|---|---|
| c48 b2（每 GEN 16 条 lane） | **ll4** | **414.94** | **30.8** | 9.3 s | 0.9598 |
| c48 b2 | ll3 | 392.16 | 30.34 | 10.7 s | 0.9621 |
| c48 b2 | ll2 | 381.68 | 29.84 | 11.0 s | 0.9591 |
| c48 b2 | 本轮起点 s53 | 326.8 | 27.22 | 17.5 s | — |
| c84 b4（每 GEN 28 条 lane，MTP7） | ll2 | 304.88 | 55.17 | 20.6 s | 0.9591 |
| c84 b4 | 本轮起点 s53 | 276.24 | 50.85 | 25.2 s | — |
| c24 b1（每 GEN 8 条 lane） | 本轮起点 s53 | 375.94 | 13.87 | 19.9 s | — |

- ll4 c48 b2：6698 个请求、0 错误；p90 full-response ITL 2.41 ms，mean TPOT 2.31 ms。相对起点 +27.0% / +13.2%，相对 ll3 +5.8% / +1.5%。
  c48 b2 在两个指标上都已压过起点的 c24 b1。
- GSM8K 参考值 0.9613；门槛 0.95。ll4 GSM8K 用 b2 开关组 + `TRTLLM_K3_LOWM_SF_SLOTS=1`，GEN max batch 4（M ≤ 28），不强制接受。

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

### 公共（b1 / b2 / b4 都开，b4 只开这些）

```yaml
extra_gen_worker_env:
  # 投机解码 / DSpark 尾部
  TLLM_SPEC_VOCAB_PARALLEL: "1"
  TLLM_SPEC_VOCAB_PARALLEL_SAMPLING: "1"
  TLLM_SPEC_TAIL_FUSION: "1"
  TLLM_SPEC_HEAD_FUSION: "1"
  TLLM_DSPARK_DRAFT_FUSION: "1"
  TLLM_DSPARK_FUSED_MARKOV: "1"
  TLLM_DSPARK_MARKOV_V2: "2"
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

### b2（c48 b2 正式点）在公共之上

```yaml
  TRTLLM_K3_KDA_L2_PREFETCH: "1"
  TRTLLM_K3_ATTN_RES_MXFP8: "1"
  TRTLLM_K3_LD_FP4: "1"
  TRTLLM_K3_ROUTER_FIRST: "1"
  TRTLLM_K3_ROUTER_EXACT: "kda"                # ll4：精确 router 只用在 KDA 层（MLA 层会拖慢 route tail）
  TRTLLM_K3_LATENT_AR_OWN_WS: "1"
  TRTLLM_K3_LATENT_AR_CUTE: "1"
  TRTLLM_K3_LATENT_AR_ROW: "1"                 # ll4：latent all-reduce 每行一个 CTA 的 row kernel
  TRTLLM_K3_LATENT_AR_SIDE_CLEAR: "1"          # ll4：Lamport 清零放到 latent_down 流（b2 必须用 1，见下）
  TRTLLM_K3_LU_MXFP8: "1"                      # ll4：latent_up 走 MXFP8 小 M GEMM
  TRTLLM_K3_LU_MX_STAGES: "12:8"
  TRTLLM_MOE_LOWM_FC1_CTAS: "8:128,16:120"
  TRTLLM_MOE_LOWM_FC2_CTAS: "8:120,16:128"
  TRTLLM_MOE_LOWM_FC1_TRIGGER: "8:last,16:last"
  TRTLLM_MOE_LOWM_DYN: "1"                     # ll4：9-16 token 时 FC1 + FC2 一个动态调度的融合 grid
  TRTLLM_MOE_LOWM_DYN_MIN_TOKENS: "9"
  TRTLLM_MOE_LOWM_DYN_MAX_TOKENS: "16"
  TRTLLM_MOE_LOWM_DYN_CTAS: "16:136"
  TRTLLM_K3_SHARED_TAIL_PRIO: "-1"
  TRTLLM_K3_SHARED_TAIL_MIN_M: "9"
  TRTLLM_K3_SHARED_TAIL_MAX_M: "16"
  TRTLLM_K3_EXPERT_PLACEMENT: <本目录>/placements/k3_ep4_e2eR_c16b2_v1.json   # ll4：EP4 静态专家重排
```

### b1（c24 b1）在公共之上

```yaml
  TRTLLM_K3_KDA_L2_PREFETCH: "1"
  TRTLLM_K3_LOWM_SF_SLOTS: "1"
  TRTLLM_K3_LD_FP4: "1"
  TRTLLM_K3_ROUTER_FIRST: "1"
  TRTLLM_K3_ROUTER_EXACT: "kda"
  TRTLLM_K3_LATENT_AR_OWN_WS: "1"
  TRTLLM_K3_LATENT_AR_CUTE: "1"
  TRTLLM_K3_LATENT_AR_ROW: "1"
  TRTLLM_K3_LATENT_AR_SIDE_CLEAR: "ld"         # b1（M = 7）不走融合 FC1+FC2，ld 放置安全
  TRTLLM_K3_LU_MXFP8: "1"
  TRTLLM_K3_LU_MX_STAGES: "12:8"
  TRTLLM_MOE_LOWM_FC1_CTAS: "8:128,16:120"
  TRTLLM_MOE_LOWM_FC2_CTAS: "8:120,16:128"
  TRTLLM_MOE_LOWM_FC1_TRIGGER: "8:last,16:last"
  TRTLLM_K3_EXPERT_PLACEMENT: <本目录>/placements/k3_ep4_e2eR_c16b2_v1.json
```

## 拓扑（gen_configs）

`[ctx_num, gen_num, gen_cp, batch, max_num_tokens, helix, gpu_mem, mtp, eplb, 每个 GEN 的并发]`

| 点 | 行 |
|---|---|
| c24 b1 | `[16, 3, 16, 1, 7, helix, 0.8, 6, 0, 8]` |
| c48 b2 | `[16, 3, 16, 2, 14, helix, 0.8, 6, 0, 16]` |
| c60 b2（lane 扫描） | `[16, 3, 16, 2, 14, helix, 0.8, 6, 0, 20]` |
| c84 b4 | `[16, 3, 16, 4, 32, helix, 0.8, 7, 0, 28]` |

AgentX 的 lane 是闭环的，tps/GPU 主要由每个 GEN 的 lane 数决定；b2 上 MTP7 与 MTP6 持平（e2e 代理 393.7 vs 393.7），所以用 MTP6。

## ll4 新增的优化（相对 ll3）

| 开关 | 内容 | 数值 | b2 收益 |
|---|---|---|---|
| `TRTLLM_MOE_LOWM_DYN` | 小 batch 路由专家 FC1 + FC2 用一个持久 grid（136 CTA）动态调度，FC2 tile 等 FC1 释放；自己清零输出，去掉 memset 和流 join | FC1 行、单专家部分和逐位一致；只有 BF16 原子求和顺序变 | 同 token 流对比关掉它慢 191 us/步 |
| `TRTLLM_K3_EXPERT_PLACEMENT`（EP4） | 静态专家重排，降低最忙 EP 组碰到的专家数 | 选中的专家不变，跨 rank 求和顺序变 | e2e 代理 +0.4% tps/user；EP 组间差距 −0.4…−0.7 us/层 |
| `TRTLLM_K3_ROUTER_EXACT=kda` | 与 `dsv3_router_gemm_op` 逐位一致的 router kernel，只在 KDA 层用 | 逐位一致 | 同 token 流对比关掉它慢 107 us/步 |
| `TRTLLM_K3_LATENT_AR_ROW` | latent all-reduce + RMSNorm：每行一个 512 线程 CTA，本 rank 槽从寄存器求和，只重轮询缺的槽 | 与 C++ one-shot 逐位一致到 38 行（生产里最多 32 行；下一版把上限从 64 收到 38） | 16 rank replica −1.4 us/层（含下面两项） |
| `TRTLLM_K3_LATENT_AR_SIDE_CLEAR` | Lamport 缓冲的清零放到 latent_down 流 | 逐位一致 | 同上 |
| `TRTLLM_K3_LU_MXFP8` + `_STAGES=12:8` | latent_up 走 MXFP8 小 M GEMM | 逐位一致 | AR 结束 → latent_up 结束 −0.68 us/层 |

另有默认关、本版未启用的：`TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS`（batch 满时提前接收下一个请求做 KV 传输，修好了 seq slot 溢出，e2e 待测）、
`TLLM_SPEC_DECODE_FORCE_ACCEPT_KEY=state` / `TLLM_SPEC_DECODE_STREAM_LOG`（gen-only 测量用，见下）、
`TRTLLM_K3_MLA_KV_FUSED` / `_KV_JOIN_EARLY` / `TRTLLM_MLA_KV_EVICT_FIRST`（MLA 块，下一版评估）、MNNVL all-reduce workspace 建立失败时
不再重复尝试（修复，无开关）。

## 不要打开的开关

| 开关 | 原因 |
|---|---|
| `TRTLLM_K3_LATENT_AR_SIDE_CLEAR=ld` 与 `TRTLLM_MOE_LOWM_DYN` 同时用 | 融合 FC1+FC2 跳过输出 memset 的流 join，ld 的清零顺序失去保证；ll4 的 b2 用 `1`（下一版按调用的 token 数自动选择并显式 join） |
| `TRTLLM_MOE_LOWM_DYN` 用于 b1 / b4 | b1 持平（token 范围已排除 M = 7）；b4 +5 us/层 |
| 去掉 `TRTLLM_MOE_LOWM_FC1_CTAS` / `_FC2_CTAS` / `_FC1_TRIGGER`（b1 / b2） | 同 token 流对比 +1.5% 步长；b4（M = 32）则不要这些上限 |
| `TRTLLM_K3_ROUTER_EXACT=1`（所有层） | MLA 层 route tail 被 latent_down + shared gate_up 的突发拖慢，+1.87 us/层；用 `kda` |
| `TRTLLM_K3_MOE_FRONT_PREFETCH`、`TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA`、`TLLM_DSPARK_L2_PREFETCH` | 把 HBM 读挪进注意力 / AR 阶段，在 situ 净变慢（replica 不带邻居的 HBM 流量时会误判为变快） |
| `TRTLLM_K3_MLA_GATE_EARLY` | 与 a2a 抢资源，+2.1 us/层 |
| `TRTLLM_MLA_KV_EVICT_FIRST=k` / `kv`（b4） | K tile 的 L2 复用丢失，b4 decode +1.2…+2.0 us |
| o_proj / shared AR 改 one-shot（`TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES` 4 MiB）、shared + latent AR 合并、b2 的 MoE TP16 | b2 +7% / +4.4% / +1.8% |
| 精度测试 | 去掉 `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`（harness 的 gen-only 性能跑注入固定接受长度） |

## 精度（GSM8K）

在 b2 开关组上加 `TRTLLM_K3_LOWM_SF_SLOTS: "1"`，`gsm8k_eval: true`，gen_configs `[16, 1, 16, 4, 28, helix, 0.8, 6, 0, 16]`
（GEN max batch 4，覆盖 M ≤ 28），集群配置用 DWDP 的 `opt*` 配置（`-dep` 配置下 CTX 会 OOM）。不要带任何强制接受的变量。

## gen-only A/B 的测量注意事项

- gen-only 跑法（`opt-dep-short` 集群配置，CTX 1 + GEN 1，nsys 窗口 100-120 次迭代）用强制接受（2.75 个 draft / 步）。
  MoE FC2 + finalize 的 BF16 原子求和运行间不逐位确定，强制接受会把一个近似平票的 token 翻转放大成另一个退化 token 循环：
  同一配置两次运行的步长可差 2-3.5%，而且退化循环碰到的专家比真实文本少（gen-only 比 e2e 乐观）。
- 本版：只比较不依赖路由的分段（`sol_breakdown` 去掉路由专家路径），或比较同一 token 流的运行（逐迭代 cached_kv_tokens 签名 +
  FC1 逐（层, rank）指纹）。`TLLM_SPEC_DECODE_FORCE_ACCEPT_KEY=state` 去掉了抽签对接收顺序的依赖，`TLLM_SPEC_DECODE_STREAM_LOG=<dir>`
  逐请求记录 token 流，但都不能消除数值非确定性。
- 下一版：teacher-forced 回放（`TLLM_SPEC_DECODE_FORCE_REPLAY=<参考流>`，参考流来自一次不强制接受的真实运行）让同一配置的步长
  复现到 0.02%，且路由接近真实文本（b2 步长约高 5%）。
- `nsys --cuda-graph-trace=node` 会把窗口内的 PDL 边变成完整边，对不同流拓扑的配置代价不同（b1 实测 A +129、B +47 us/步）：
  PDL 结构不同的两组配置，以窗口外的步长为准。
