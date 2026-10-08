# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only host routing tests. These do not execute or validate GPU math."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "tensorrt_llm/_torch/modules/kimi_kda/_kda_kernels.py"


class Tensor:
    def __init__(self, shape):
        self.shape = shape

    def detach(self):
        return self

    def __getitem__(self, key):
        return Tensor((1, key[1].stop, *self.shape[2:]))


@pytest.mark.parametrize(
    "tokens,sequences,expected_pad",
    [(65, 1, True), (64, 1, False), (65, 2, False), (65, None, False)],
)
def test_actual_dispatch_method_routing(tokens, sequences, expected_pad):
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KDAKernelDispatch"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prefill_chunk_kda"
    )
    method.returns = None
    for a in method.args.args + method.args.kwonlyargs:
        a.annotation = None
    calls = {"pad": 0, "op": 0}

    def pad(t, padding):
        calls["pad"] += 1
        return Tensor((1, t.shape[1] + padding[-1], *t.shape[2:]))

    def op(**kwargs):
        calls["op"] += 1
        assert kwargs["state_pool"] is pool
        assert kwargs["state_indices"] is slots
        assert kwargs["cu_seqlens"] is cu
        assert kwargs["single_sequence_length"] == (tokens if sequences == 1 else None)
        assert kwargs["q"].shape[1] == (128 if expected_pad else tokens)
        return kwargs["q"]

    fake_torch = SimpleNamespace(ops=SimpleNamespace(trtllm=SimpleNamespace(kda_prefill=op)))
    fn = SimpleNamespace(pad=pad)
    ns = {
        "torch": fake_torch,
        "_load_prefill_module": lambda: None,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), ns)
    inputs = [Tensor((1, tokens, 6, 128)) for _ in range(4)]
    pool, slots = object(), object()
    cu = Tensor((sequences + 1,)) if sequences is not None else None
    with patch.dict(
        sys.modules,
        {
            "torch": SimpleNamespace(nn=SimpleNamespace(functional=fn)),
            "torch.nn": SimpleNamespace(functional=fn),
            "torch.nn.functional": fn,
        },
    ):
        out, state = ns["prefill_chunk_kda"](
            SimpleNamespace(prefill_kernel_path="optimized"),
            q=inputs[0],
            k=inputs[1],
            v=inputs[2],
            g=inputs[3],
            beta=Tensor((1, tokens, 6)),
            A_log=None,
            dt_bias=None,
            scale=1.0,
            initial_state=None,
            safe_gate=True,
            lower_bound=-5.0,
            cu_seqlens=cu,
            state_pool=pool,
            state_indices=slots,
            single_sequence_length=tokens if sequences == 1 else None,
        )
    assert out.shape[1] == tokens and state is None
    assert calls == {"pad": 5 if expected_pad else 0, "op": 1}


@pytest.mark.parametrize(
    "q_tokens,k_tokens,g_tokens,beta_tokens,valid",
    [
        (128, 128, 65, 128, True),
        (128, 127, 65, 128, False),
        (128, 128, 66, 128, False),
        (128, 128, 65, 65, False),
        (96, 96, 65, 96, False),
    ],
)
def test_preproduced_padding_routing(q_tokens, k_tokens, g_tokens, beta_tokens, valid):
    """Producer-padded inputs bypass F.pad and trim only the output."""
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KDAKernelDispatch"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prefill_chunk_kda"
    )
    method.returns = None
    for arg in method.args.args + method.args.kwonlyargs:
        arg.annotation = None

    calls = {"op": 0}

    def op(**kwargs):
        calls["op"] += 1
        assert kwargs["q"].shape[1] == q_tokens
        assert kwargs["g"].shape[1] == g_tokens
        return kwargs["q"]

    namespace = {
        "torch": SimpleNamespace(ops=SimpleNamespace(trtllm=SimpleNamespace(kda_prefill=op))),
        "_load_prefill_module": lambda: None,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    inputs = {
        "q": Tensor((1, q_tokens, 6, 128)),
        "k": Tensor((1, k_tokens, 6, 128)),
        "v": Tensor((1, q_tokens, 6, 128)),
        "g": Tensor((1, g_tokens, 6, 128)),
        "beta": Tensor((1, beta_tokens, 6)),
    }
    fake_functional = SimpleNamespace(pad=lambda *_: pytest.fail("unexpected F.pad"))
    with patch.dict(
        sys.modules,
        {
            "torch": SimpleNamespace(nn=SimpleNamespace(functional=fake_functional)),
            "torch.nn": SimpleNamespace(functional=fake_functional),
            "torch.nn.functional": fake_functional,
        },
    ):

        def call():
            return namespace["prefill_chunk_kda"](
                SimpleNamespace(prefill_kernel_path="optimized"),
                **inputs,
                A_log=None,
                dt_bias=None,
                scale=1.0,
                initial_state=None,
                safe_gate=True,
                lower_bound=-5.0,
                cu_seqlens=Tensor((2,)),
                state_pool=object(),
                state_indices=object(),
                single_sequence_length=65,
            )

        if valid:
            out, state = call()
            assert out.shape[1] == 65 and state is None
            assert calls["op"] == 1
        else:
            with pytest.raises(ValueError, match="Pre-padded KDA prefill inputs"):
                call()
            assert calls["op"] == 0
