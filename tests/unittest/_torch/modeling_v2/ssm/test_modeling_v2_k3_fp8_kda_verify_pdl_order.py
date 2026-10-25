# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: ``trtllm::k3_fp8_kda_verify`` launched with PDL reads this step's projection rows only after
``griddepcontrol.wait``, with TRTLLM_K3LL_KDA_VERIFY_EARLY_ROWS off and on.

A late writer stands in front of the verify: a PDL kernel that waits for its own predecessor, releases its dependents at
once, spins for a delay and only then writes new projection rows (both enqueued behind a busy kernel, so the verify
launches while the writer spins). The verify's CTAs are resident early and run their
prologue meanwhile; a projection load issued before the wait would read the old rows. Each output must equal the output
of a PDL-off verify of the new rows (bit for bit), alternating two sets of rows so a stale read always differs.
"""

import _kda_cells as kc
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4 import flags as F
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4.verify import (
    k3_fp8_kda_verify,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="requires an SM100-family GPU",
)

NUM_SPEC = 7
ROUNDS = 8
BUSY_CYCLES = 4_000_000  # ~2 ms of GPU time ahead of the writer
_writer = {}


def _late_writer():
    """write(src, dst, ns): one PDL launch (16 CTAs) that waits for its predecessor, releases its dependents, spins
    ``ns`` nanoseconds and then copies ``src`` into ``dst``."""
    if "run" in _writer:
        return _writer["run"]
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cutlass_dsl import dsl_user_op
    from cutlass.experimental import primitives as prims

    @dsl_user_op
    def _globaltimer(*, loc=None, ip=None):
        return cutlass.Int64(
            _llvm.inline_asm(
                _T.i64(), [], "mov.u64 $0, %globaltimer;", "=l,~{memory}", has_side_effects=True,
                is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
            )
        )  # fmt: skip

    @cute.kernel
    def _writer_kernel(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32, ns: cutlass.Int64):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        t0 = _globaltimer()
        while _globaltimer() - t0 < ns:
            pass
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        for i in cutlass.range(bidx * 256 + tidx, n, 16 * 256):
            dst[i] = src[i]

    @cute.jit
    def _writer_launch(
        src: cute.Tensor,
        dst: cute.Tensor,
        n: cutlass.Int32,
        ns: cutlass.Int64,
        stream: cuda_driver.CUstream,
    ):
        _writer_kernel(src, dst, n, ns).launch(
            grid=(16, 1, 1), block=(256, 1, 1), stream=stream, use_pdl=True
        )

    compiled = {}

    def run(src: torch.Tensor, dst: torch.Tensor, ns: int) -> None:
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        a = from_dlpack(src.view(-1).view(torch.int32), assumed_align=16)
        b = from_dlpack(dst.view(-1).view(torch.int32), assumed_align=16)
        n = src.numel() // 2
        if n not in compiled:  # the tensors' shapes are compiled in
            compiled[n] = cute.compile(
                _writer_launch, a, b, cutlass.Int32(n), cutlass.Int64(ns), stream
            )
        compiled[n](a, b, cutlass.Int32(n), cutlass.Int64(ns), stream)

    _writer["run"] = run
    return run


@pytest.mark.parametrize("early_rows", [0, 1], ids=["early0", "early1"])
@pytest.mark.parametrize("delay_ns", [0, 1500, 3000, 6000], ids=lambda d: f"late{d}ns")
@pytest.mark.parametrize("num", [2, 4], ids=["bs2", "bs4"])
def test_projection_rows_read_after_the_wait(num, delay_ns, early_rows, monkeypatch) -> None:
    monkeypatch.setattr(F, "TRTLLM_K3LL_KDA_VERIFY_HOIST", 1)
    monkeypatch.setattr(F, "TRTLLM_K3LL_KDA_VERIFY_EARLY_ROWS", early_rows)
    kc.load_ops()
    write = _late_writer()
    mgr = kc.build_manager(1, num_spec=NUM_SPEC, max_batch_size=8, state_dtype=torch.bfloat16)
    try:
        slots = kc.request_slots(mgr, num, first_id=300 * num)
        p = kc.layer_pools(mgr, 0)
        tokens = num * (NUM_SPEC + 1)
        with torch.inference_mode():
            wt = kc.make_weights(4242)
            gen = torch.Generator(device="cuda").manual_seed(77 + num)
            rows = [
                torch.randn(tokens, kc.PROJ, generator=gen, device="cuda").bfloat16()
                for _ in range(2)
            ]
            proj = torch.empty_like(rows[0])
            accepted = torch.tensor(
                [(2 - i) % (NUM_SPEC + 1) for i in range(num)], dtype=torch.int32, device="cuda"
            )

            def reset():
                kc.fill_pools(p, 900 + num)
                p["pending"][slots.long()] = accepted

            def verify():
                return k3_fp8_kda_verify(
                    proj, wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
                    p["cs_q"], p["cs_k"], p["cs_v"], p["ssm"], p["state_tok"], slots, p["pending"], NUM_SPEC,
                    kc.LOWER_BOUND, kc.SCALE, kc.EPS,
                )  # fmt: skip

            refs = []
            monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
            for r in rows:
                reset()
                proj.copy_(r)
                refs.append(verify().clone())
            assert not torch.equal(refs[0], refs[1])
            monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
            stale = []
            for k in range(ROUNDS):
                new = k % 2
                reset()
                proj.copy_(rows[1 - new])  # the old rows
                torch.cuda.synchronize()
                # Keep the GPU busy while the host enqueues the writer and the verify: otherwise the verify's host
                # side outlasts the writer and the two never overlap.
                torch.cuda._sleep(BUSY_CYCLES)
                write(rows[new], proj, delay_ns)
                out = verify()
                torch.cuda.synchronize()
                if not torch.equal(out, refs[new]):
                    stale.append(k)
        assert not stale, (
            f"rounds {stale} of {ROUNDS} read projection rows before the late writer finished"
        )
    finally:
        mgr.shutdown()
