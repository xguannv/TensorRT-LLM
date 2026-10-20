# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``trtllm::k3_spec_accept``'s sampled acceptance (``samp_temp`` / ``samp_seed`` / ``samp_off``) on one GPU.

* Each row's target token is the argmax of ``logits / T + g`` with ``g`` the op's Gumbel noise
  (``op.gumbel_reference``) at the global columns (``shard_offset`` + column), up to the hardware log2's rounding on
  near-ties; rows below the greedy temperature keep the plain argmax. Given those tokens, every other output and the
  state updates are the greedy reference's (test_k3_spec_accept.reference) on the noised scores, bit for bit.
* The draws are temperature samples: over many (seed, offset) pairs the token frequencies of a few-candidate row match
  softmax(logits / T).
"""

import importlib.util
import pathlib
import sys

import pytest
import torch

_BASE = pathlib.Path(__file__).resolve().parent / "test_k3_spec_accept.py"


def _base():
    name = "_k3_spec_accept_base_for_sample"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _BASE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="needs SM100 (CTM kernel)",
)


def _sampling_inputs(gen, rows: int, temps):
    temp = torch.tensor(temps, dtype=torch.float32, device="cuda")
    seed = torch.randint(0, 2**62, (rows,), generator=gen, device="cuda", dtype=torch.int64)
    offset = torch.randint(0, 2**40, (rows,), generator=gen, device="cuda", dtype=torch.int64)
    return temp, seed, offset


@pytest.mark.parametrize("batch,drafts", [(1, 7), (2, 7), (4, 3)])
@pytest.mark.parametrize("shard_offset", [0, 81920])
def test_sampled_tokens_and_state(batch, drafts, shard_offset):
    base = _base()
    op = base._op()
    gen = torch.Generator(device="cuda").manual_seed(batch * 100 + drafts)
    rows = batch * (drafts + 1)
    block = drafts + 1
    logits, _ = base.step_inputs(gen, batch, drafts, "plain", 0)
    # Every third row greedy (T = 0), the others at 0.7 / 1.0 / 1.3.
    temps = [0.0 if r % 3 == 0 else (0.7, 1.0, 1.3)[r % 3] for r in range(rows)]
    temp, seed, offset = _sampling_inputs(gen, rows, temps)
    columns = torch.arange(logits.shape[1], device="cuda") + shard_offset
    noise = op.gumbel_reference(seed, offset, columns)
    hot = (temp >= 1e-4)[:, None]
    scores = torch.where(hot, logits / temp.clamp(min=1e-4)[:, None] + noise, logits)
    draft = base.drafts_for(gen, scores, batch, drafts, 3)
    st = base.make_state(gen, batch)
    ref_st = base.clone_state(st)

    out = torch.ops.trtllm.k3_spec_accept(
        logits, draft, st["block_off"], st["pool_idx"], st["divisor"], st["block_counts"], st["block_tables"],
        st["prev_acc"], st["state_idx"], st["dummy"], st["kv_lens"], st["batch_to_slot"], st["ctx_len"],
        base.MAX_CTX, st["embed"], st["mask_row"], st["rng_pool"], st["rng_counter"], 0.0, block,
        None, None, None, 0, 1, 1, shard_offset, temp, seed, offset,
    )  # fmt: skip
    tokens = out[0].view(-1).long()
    best = scores.argmax(-1)
    picked = scores.gather(1, tokens[:, None]).squeeze(1)
    top = scores.gather(1, best[:, None]).squeeze(1)
    # A different pick is allowed only on a near-tie of the noised scores (the kernel's log2 is the hardware's).
    assert bool(((tokens == best) | (top - picked < 1e-3)).all()), (
        "a sampled token is not the noised argmax"
    )
    assert bool((tokens[hot.squeeze(1) == 0] == logits.argmax(-1)[hot.squeeze(1) == 0]).all()), (
        "greedy rows moved"
    )
    if bool((tokens == best).all()):
        expected = base.reference(ref_st, scores, draft, 0.0, block)
        for name, got, want in zip(base.OUTPUTS, out, expected):
            assert torch.equal(got, want.to(got.dtype)), name
        for key in base.STATE_KEYS:
            assert torch.equal(st[key], ref_st[key]), key


def test_sampled_frequencies_follow_the_softmax():
    """Rows of a few candidate columns (the rest far below) over 64 rows x 48 calls of fresh (seed, offset): the
    empirical frequencies of the candidates match softmax(logits / T) within a few standard errors."""
    base = _base()
    gen = torch.Generator(device="cuda").manual_seed(5)
    batch, drafts = 8, 7
    rows = batch * (drafts + 1)
    block = drafts + 1
    cand = torch.tensor([3, 2000, 70000, 163839], device="cuda")
    cand_logits = torch.tensor([1.0, 0.5, 0.0, -0.7], device="cuda")
    temperature = 0.8
    logits = torch.full((rows, base.V), -60.0, device="cuda")
    logits[:, cand] = cand_logits
    counts = torch.zeros(len(cand), device="cuda")
    calls = 48
    for _ in range(calls):
        temp, seed, offset = _sampling_inputs(gen, rows, [temperature] * rows)
        draft = torch.zeros(batch, drafts, dtype=torch.int32, device="cuda")
        st = base.make_state(gen, batch)
        out = torch.ops.trtllm.k3_spec_accept(
            logits, draft, st["block_off"], st["pool_idx"], st["divisor"], st["block_counts"], st["block_tables"],
            st["prev_acc"], st["state_idx"], st["dummy"], st["kv_lens"], st["batch_to_slot"], st["ctx_len"],
            base.MAX_CTX, st["embed"], st["mask_row"], st["rng_pool"], st["rng_counter"], 0.0, block,
            None, None, None, 0, 1, 1, 0, temp, seed, offset,
        )  # fmt: skip
        tokens = out[0].view(-1)
        counts += (tokens[:, None] == cand[None, :]).sum(0).float()
    n = calls * rows
    assert int(counts.sum().item()) == n, "a token outside the candidates"
    p = torch.softmax(cand_logits / temperature, 0)
    freq = counts / n
    sigma = (p * (1 - p) / n).sqrt()
    assert bool(((freq - p).abs() < 4 * sigma + 1e-3).all()), (
        f"frequencies {freq.tolist()} vs softmax {p.tolist()}"
    )


def test_gumbel_reference_is_a_gumbel():
    """The reference noise over many columns has the Gumbel(0, 1) mean (Euler-Mascheroni) and variance (pi^2 / 6)."""
    op = _base()._op()
    g = op.gumbel_reference(
        torch.tensor([12345], device="cuda"),
        torch.tensor([678], device="cuda"),
        torch.arange(1 << 20, device="cuda"),
    )
    assert abs(g.mean().item() - 0.5772) < 0.01
    assert abs(g.var().item() - 1.6449) < 0.02
