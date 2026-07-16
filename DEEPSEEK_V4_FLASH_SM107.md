# Running DeepSeek-V4-Flash on Rubin (SM107)

Enablement notes + workarounds to run **DeepSeek-V4-Flash** on Rubin / SM 10.7 with the
TensorRT-LLM PyTorch backend. Validated 2026-06-30 on 4× VR NVL72 (sm107), CUDA 13.4
container (`pytorch-rubin-py3-sbsa-…-trt10.15.1.29-20260512`).

> These are **bringup workarounds**, not final fixes. They unblock a functional run; the
> default/perf path (AUTO MoE backend + autotuner + MHC tcgen05 MMA) still has open sm107
> kernel bugs (see "Open issues" below).

## 1. Build (from source, in the container)

```bash
git config --global --add safe.directory '*'   # FetchContent clones are host-owned

./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks \
  --use_ccache -a "107-real" -f --nvtx --nvrtc_dynamic_linking
```
- `--nvrtc_dynamic_linking` is **required** on CUDA 13.4 (no `libnvrtc_static.a`).
- **cutlass SM107 patch** is applied **automatically** by FetchContent: `3rdparty/fetch_content.json`
  gives the cutlass dependency `"patch_file": "patches/cutlass_sm107.patch"`, and the loader
  (`3rdparty/CMakeLists.txt`) runs it (`patch -p1 --forward --batch`, idempotent) on the cloned
  cutlass v4.4.2 — same mechanism `xgrammar`/`deep_ep` use. No manual step; survives clean rebuilds.
  The patch (`3rdparty/patches/cutlass_sm107.patch`) maps `__CUDA_ARCH__==1070` / family-1070 to the
  SM100 (Blackwell) family so the `sm_107f` cubin enables SM100 MMA/TMA/tcgen05 instead of crashing
  with disabled-PTX stubs. (Public cutlass 4.4.2 strips SM107 — it's internal-release-only.)

## 2. Run

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=<repo> \
  python examples/llm-api/quickstart_advanced.py \
  --model_dir <DeepSeek-V4-Flash> \
  --tp_size 4 --trust_remote_code \
  --tokens_per_block 128 \
  --moe_backend CUTLASS \
  --disable_autotuner
```

## 3. Required flags / workarounds and why

| flag / change | reason |
|---|---|
| `--tokens_per_block 128` | `DeepseekV4CacheManager` requires 128 or 256 (default 32 → executor worker dies). |
| `--moe_backend CUTLASS` | the trtllm-gen `mxe4m3_mxe2m1` MXFP4 MoE crashes (illegal address) in attention warmup on sm107; CUTLASS MoE handles the MXFP4 W4A8_MXFP4_MXFP8 experts and works. |
| `--disable_autotuner` | the autotuner warmup hits an illegal address on sm107 (even with CUTLASS MoE). |
| MHC MMA→FMA (in-tree) | `mhc_cuda.py` forces the MHC FMA path on sm107; the tcgen05 TF32 **MMA** kernel hits `cudaErrorLaunchFailure`. (code change, automatic on sm107) |
| `3rdparty/patches/cutlass_sm107.patch` (auto-applied via `fetch_content.json`) | enables SM100-family MMA/TMA/tcgen05 for sm107 at build; see above. |
| int() casts (in-tree) | `dsa.py`/`trtllm.py` cast `max_seq_len` to `int` at DSA/MLA pybind boundaries. |

In-tree changes (this commit): `mhc_cuda.py`, `dsa.py`, `trtllm.py`, plus the cutlass SM107 patch
at `3rdparty/patches/cutlass_sm107.patch` wired in via `3rdparty/fetch_content.json` (auto-applied
during the build — nothing manual, survives clean rebuilds).

## 4. Open issues (default/perf path — to root-fix)

1. trtllm-gen `mxe4m3_mxe2m1` MXFP4 MoE → illegal address in attention warmup on sm107.
2. Autotuner warmup → illegal address on sm107.
3. MHC tcgen05 TF32 MMA kernel → `cudaErrorLaunchFailure` on sm107 (FMA path is the WAR).
4. Make the cutlass SM107 change permanent (survive clean `-c` rebuilds).
