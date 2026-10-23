# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``k3_ctm_gemv.op.run_k3_ctm_gemv_wide_ar_push`` (``k3_ctm_gemv_wide`` with the all-reduce push epilogue,
``TRTLLM_K3LL_MOE_TAIL_PUSH``) on one GPU, at the MoE tail's TP16 per-rank shape [7168, 640].

A W-rank Lamport workspace stands in on one device: W buffers of three Lamport buffers each (the ranks' unicast
mappings, armed with the sentinel), their addresses as the peer table, and rotation flags whose current buffer is
not 0 (so the buffer index is exercised). Pushing as rank ``r`` must store token ``t``'s row of the plain
``trtllm::k3_ctm_gemv_wide`` output, bit for bit with ``-0`` as ``+0``, into slot ``(t // W) * W + r`` of the scatter
stage of buffer ``t % W``'s current Lamport buffer, and nothing anywhere else; eager and from a CUDA graph replayed with
new rows. Every M of 1..64 (the 16-, 32- and 64-token tiles), every rank of W = 4 and W = 16 at a few M.
"""

import functools

import pytest
import torch


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="needs an SM100-family GPU")

N, K = 7168, 640  # the MoE tail [latent up 224 | padding 32 | shared down 384] at TP16
SENTINEL = -(2**31)
NEG_ZERO = -(2**15)
CUR = 1  # the rotation's current buffer
M_ALL = list(range(1, 65))


def _ctm():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import op

    return op


def _lamport():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_lamport_ar

    return k3_lamport_ar


@functools.lru_cache(maxsize=None)
def _weight(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device="cuda") * 0.02).bfloat16()
    w[:128] = 0  # whole zero rows: zero outputs, whatever their sign, go out as +0
    return w


def _rows(m: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(m, K, generator=g, device="cuda").bfloat16()


class _Workspace:
    """W ranks' Lamport buffers on one device and a ``LamportWorkspace`` view of them as rank ``rank``."""

    def __init__(self, world: int, rank: int, max_tokens: int = 64):
        lam = _lamport()
        self.world, self.rank = world, rank
        self.bpb = lam.required_bytes(max_tokens, N, world, False)
        self.bufs = [
            torch.full((3 * self.bpb // 4,), SENTINEL, dtype=torch.int32, device="cuda")
            for _ in range(world)
        ]
        self.flags = torch.zeros(9, dtype=torch.int32, device="cuda")
        self.flags[0], self.flags[1], self.flags[2] = CUR, (CUR + 2) % 3, self.bpb
        self.view = lam.LamportWorkspace(
            world=world,
            rank=rank,
            uc_local=self.bufs[rank].data_ptr(),
            mc_base=0,
            peer_ptrs=torch.tensor(
                [b.data_ptr() for b in self.bufs], dtype=torch.int64, device="cuda"
            ),
            flags=self.flags,
            bytes_per_buffer=self.bpb,
        )

    def rearm(self):
        for b in self.bufs:
            b.fill_(SENTINEL)

    def expected(self, y: torch.Tensor) -> list:
        """Every buffer as it must look after rank ``rank`` pushed ``y`` (int32 words, -0 halves as +0)."""
        bits = y.contiguous().view(torch.int16)
        bits = bits.masked_fill(bits == NEG_ZERO, 0).view(torch.int32)  # [M, N / 2]
        want = [torch.full_like(b, SENTINEL) for b in self.bufs]
        base = CUR * self.bpb // 4  # stage 0 (scatter) of the current buffer
        row_words = N // 2
        for t in range(y.shape[0]):
            slot = (t // self.world) * self.world + self.rank
            start = base + slot * row_words
            want[t % self.world][start : start + row_words] = bits[t]
        return want


def _check(ws: _Workspace, y: torch.Tensor, where: str):
    for owner, (got, want) in enumerate(zip(ws.bufs, ws.expected(y))):
        differ = int((got != want).sum())
        assert differ == 0, f"{where}: buffer of rank {owner}: {differ} words differ"


def _plain(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.ops.trtllm.k3_ctm_gemv_wide(x, w)


@pytest.mark.parametrize("m", M_ALL)
def test_push_rows_are_plain_rows(m):
    """Eager: the slots hold the plain kernel's bf16 rows (-0 as +0), nothing else is written, the flags unchanged."""
    op = _ctm()
    w = _weight(1)
    for rank in (0, 3):
        ws = _Workspace(4, rank)
        x = _rows(m, 2 + rank)
        assert op.wide_push_config(
            N, K, m, torch.cuda.get_device_properties(0).multi_processor_count, 4
        )
        flags = ws.flags.clone()
        op.run_k3_ctm_gemv_wide_ar_push(x, w, ws.view)
        torch.cuda.synchronize()
        _check(ws, _plain(x, w), f"M={m} rank {rank}")
        assert torch.equal(ws.flags, flags), "the push must not advance the rotation"


@pytest.mark.parametrize("m", [9, 16, 17, 32, 64])
def test_push_world16(m):
    """W = 16 (the TP16 group): tokens spread over 16 owners, slot (t // 16) * 16 + rank."""
    op = _ctm()
    w = _weight(1)
    for rank in (0, 5, 15):
        ws = _Workspace(16, rank)
        x = _rows(m, 40 + rank)
        op.run_k3_ctm_gemv_wide_ar_push(x, w, ws.view)
        torch.cuda.synchronize()
        _check(ws, _plain(x, w), f"W=16 M={m} rank {rank}")


@pytest.mark.parametrize("m", [9, 16, 24, 32])
def test_push_graph_replay(m):
    """A captured push replayed with new rows (written into the captured input) and re-armed buffers between
    replays: every replay's slots are the plain kernel's rows of that replay's input."""
    op = _ctm()
    w = _weight(1)
    ws = _Workspace(4, 2)
    x = _rows(m, 7)
    op.run_k3_ctm_gemv_wide_ar_push(x, w, ws.view)  # compiles eagerly
    torch.cuda.synchronize()
    ws.rearm()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op.run_k3_ctm_gemv_wide_ar_push(x, w, ws.view)
    for i in range(3):
        ws.rearm()
        x.copy_(_rows(m, 100 + i))
        graph.replay()
        torch.cuda.synchronize()
        _check(ws, _plain(x, w), f"graph M={m} replay {i}")


def test_compile_only_and_refusals():
    """``compile_wide_ar_push`` compiles without launching (no slot written), ``wide_ar_push_compiled`` sees it; a
    shape the push does not take (N not a multiple of 128, too many rows, an unsupported rank count) is refused."""
    op = _ctm()
    w = _weight(1)
    ws = _Workspace(4, 0)
    assert op.compile_wide_ar_push(w, 16, 4)
    assert op.wide_ar_push_compiled(N, K, 9, 4, w.device)
    torch.cuda.synchronize()
    assert all(bool((b == SENTINEL).all()) for b in ws.bufs)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert op.wide_push_config(N - 8, K, 16, sms, 4) is None
    assert op.wide_push_config(N, K, 65, sms, 4) is None
    assert op.wide_push_config(N, K, 16, sms, 3) is None
    with pytest.raises(ValueError):
        op.run_k3_ctm_gemv_wide_ar_push(_rows(65, 1), w, ws.view)
