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
* ``rows_per_cta`` (``TRTLLM_K3LL_SPEC_ACCEPT_ROW_SPLIT``) changes no bit of a sampled call: every output and state
  tensor equals the call with one CTA per column slice, on the same (temperature, seed, offset).
* Distribution, with and without ``rows_per_cta``: a chi-square goodness-of-fit test of the sampled tokens against
  softmax(logits / T) over the whole 163840-token vocabulary (equal-mass bins of a spread distribution; the candidate
  tokens plus the rest of a peaked one), with a negative control (the same counts against another temperature must
  be rejected), and the mean accepted length of drafts that always propose the top token against its closed form
  sum_j p^j (p the top token's probability).
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


def _sampled_call(base, st, logits, draft, block, sampling, rows_per_cta=0, shard_offset=0):
    return base.fused(st, logits, draft, 0.0, block, sampling=sampling, shard_offset=shard_offset,
                      rows_per_cta=rows_per_cta)  # fmt: skip


@pytest.mark.parametrize("rows_per_cta", [1, 2, 4])
@pytest.mark.parametrize("batch,drafts", [(1, 7), (2, 7), (4, 7), (8, 7), (3, 3)])
@pytest.mark.parametrize("shard_offset", [0, 81920])
def test_row_split_sampled_bits(batch, drafts, rows_per_cta, shard_offset):
    """``rows_per_cta``: a sampled call (a third of the rows greedy) gives the bits of the one-CTA-per-slice call on
    the same inputs, every output and state tensor, over 4 consecutive steps with the state carried."""
    base = _base()
    gen = torch.Generator(device="cuda").manual_seed(
        9000 + 100 * batch + 10 * drafts + rows_per_cta
    )
    rows = batch * (drafts + 1)
    block = drafts + 1
    st_off = base.make_state(gen, batch)
    st_on = base.clone_state(st_off)
    with torch.inference_mode():
        for step in range(4):
            logits, _ = base.step_inputs(gen, batch, drafts, "plain", step)
            temps = [0.0 if (r + step) % 3 == 0 else (0.7, 1.0, 1.3)[r % 3] for r in range(rows)]
            sampling = _sampling_inputs(gen, rows, temps)
            draft = base.drafts_for(gen, logits, batch, drafts, step)
            base.set_dummies((st_off, st_on), batch, step)
            want = _sampled_call(base, st_off, logits, draft, block, sampling, 0, shard_offset)
            got = _sampled_call(
                base, st_on, logits, draft, block, sampling, rows_per_cta, shard_offset
            )
            flags = base.same(got + base.state_of(st_on), want + base.state_of(st_off))
            names = base.OUTPUTS + base.STATE_KEYS
            assert all(flags), (step, [n for n, f in zip(names, flags) if not f])


def _chi_square(counts: torch.Tensor, probs: torch.Tensor):
    """(statistic, degrees of freedom, upper-tail p-value) of observed ``counts`` against bin probabilities."""
    counts, probs = counts.double().cpu(), probs.double().cpu()
    n = counts.sum()
    expected = n * probs
    stat = ((counts - expected) ** 2 / expected).sum()
    df = counts.numel() - 1
    p_value = torch.special.gammaincc(torch.tensor(df / 2.0, dtype=torch.float64), stat / 2.0)
    return stat.item(), df, p_value.item()


def _draw_tokens(base, logits_row, temperature, calls, gen, rows_per_cta, batch=8, drafts=7):
    """The sampled target tokens of ``calls`` calls of ``batch x (drafts + 1)`` rows, every row ``logits_row``, each
    row and call a fresh (seed, offset)."""
    rows = batch * (drafts + 1)
    logits = logits_row[None, :].expand(rows, -1).contiguous()
    draft = torch.zeros(batch, drafts, dtype=torch.int32, device="cuda")
    tokens = []
    with torch.inference_mode():
        for _ in range(calls):
            sampling = _sampling_inputs(gen, rows, [temperature] * rows)
            st = base.make_state(gen, batch)
            out = _sampled_call(base, st, logits, draft, drafts + 1, sampling, rows_per_cta)
            tokens.append(out[0].view(-1).long())
    return torch.cat(tokens)


def _spread_logits(gen):
    return torch.randn(_base().V, generator=gen, device="cuda") * 1.5


def _peaked_logits(gen, cand):
    logits = torch.randn(_base().V, generator=gen, device="cuda") - 10.0
    logits[cand] = torch.linspace(2.5, 0.0, len(cand), device="cuda")
    return logits


def _equal_mass_bins(probs: torch.Tensor, bins: int):
    """(bin index per token, bin count): tokens sorted by probability, cut into at most ``bins`` runs of about equal
    total mass; a token heavier than 1 / ``bins`` makes its run alone, and the empty cuts are dropped."""
    order = torch.argsort(probs, descending=True)
    cum = torch.cumsum(probs[order].double(), 0)
    which = torch.clamp((cum * bins).floor().long(), max=bins - 1)
    which = torch.unique(which, return_inverse=True)[1]
    index = torch.empty_like(order)
    index[order] = which
    return index, int(which.max().item()) + 1


@pytest.mark.parametrize("rows_per_cta", [0, 1], ids=["one-cta-per-slice", "row-split"])
@pytest.mark.parametrize("temperature", [0.6, 1.0])
def test_chi_square_spread_vocabulary(temperature, rows_per_cta):
    """A spread distribution over all 163840 tokens (logits ~ N(0, 1.5^2)), up to 24 equal-mass bins, 64 rows x 160
    calls: the chi-square test does not reject softmax(logits / T) (p > 1e-4); against the softmax at 1.3 T it does
    (p < 1e-6), so the test can tell temperatures apart at this sample size."""
    base = _base()
    gen = torch.Generator(device="cuda").manual_seed(4242)
    logits_row = _spread_logits(gen)
    tokens = _draw_tokens(base, logits_row, temperature, 160, gen, rows_per_cta)
    bins = 24
    probs = torch.softmax(logits_row.double() / temperature, 0)
    index, bins = _equal_mass_bins(probs, bins)
    counts = torch.bincount(index[tokens], minlength=bins)
    bin_probs = torch.zeros(bins, dtype=torch.float64, device="cuda").index_add_(0, index, probs)
    stat, df, p_value = _chi_square(counts, bin_probs)
    wrong = torch.softmax(logits_row.double() / (1.3 * temperature), 0)
    wrong_bins = torch.zeros(bins, dtype=torch.float64, device="cuda").index_add_(0, index, wrong)
    stat_w, _, p_w = _chi_square(counts, wrong_bins)
    print(f"[chi2 spread T={temperature} rows_per_cta={rows_per_cta}] n={tokens.numel()} chi2={stat:.2f} df={df} "
          f"p={p_value:.4f} | control 1.3T: chi2={stat_w:.1f} p={p_w:.2e}")  # fmt: skip
    assert p_value > 1e-4, (
        f"chi2 {stat:.1f} (df {df}) p {p_value:.2e}: draws do not follow softmax(logits / T)"
    )
    assert p_w < 1e-6, f"negative control not rejected: chi2 {stat_w:.1f} p {p_w:.2e}"


@pytest.mark.parametrize("rows_per_cta", [0, 1], ids=["one-cta-per-slice", "row-split"])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_chi_square_peaked_vocabulary(temperature, rows_per_cta):
    """12 candidate tokens across the vocabulary (CTA and shard boundaries) above a background of every other token,
    bins = the candidates + the rest, 64 rows x 120 calls: not rejected (p > 1e-4); the softmax at 1.25 T is
    (p < 1e-6)."""
    base = _base()
    gen = torch.Generator(device="cuda").manual_seed(777)
    cand = torch.tensor([0, 1, 1023, 1024, 10239, 10240, 50000, 81919, 81920, 120000, 163838, 163839],
                        device="cuda")  # fmt: skip
    logits_row = _peaked_logits(gen, cand)
    tokens = _draw_tokens(base, logits_row, temperature, 120, gen, rows_per_cta)
    index = torch.full((base.V,), len(cand), dtype=torch.long, device="cuda")
    index[cand] = torch.arange(len(cand), device="cuda")
    counts = torch.bincount(index[tokens], minlength=len(cand) + 1)

    def binned(t):
        probs = torch.softmax(logits_row.double() / t, 0)
        return torch.zeros(len(cand) + 1, dtype=torch.float64, device="cuda").index_add_(
            0, index, probs
        )

    stat, df, p_value = _chi_square(counts, binned(temperature))
    stat_w, _, p_w = _chi_square(counts, binned(1.25 * temperature))
    print(f"[chi2 peaked T={temperature} rows_per_cta={rows_per_cta}] n={tokens.numel()} chi2={stat:.2f} df={df} "
          f"p={p_value:.4f} | control 1.25T: chi2={stat_w:.1f} p={p_w:.2e}")  # fmt: skip
    assert p_value > 1e-4, f"chi2 {stat:.1f} (df {df}) p {p_value:.2e}: counts {counts.tolist()}"
    assert p_w < 1e-6, f"negative control not rejected: chi2 {stat_w:.1f} p {p_w:.2e}"


@pytest.mark.parametrize("rows_per_cta", [0, 1], ids=["one-cta-per-slice", "row-split"])
def test_acceptance_length_matches_closed_form(rows_per_cta):
    """Every row the same peaked logits (top token probability p at T = 1), every draft the top token: a request's
    accepted count n = 1 + the leading rows that sampled the top token, so E[n] = sum_{j <= K} p^j. 8 requests x 200
    calls: the mean n is within 4.5 standard errors of it, and the per-position acceptance rate within 4.5 standard
    errors of p^(j + 1)."""
    base = _base()
    gen = torch.Generator(device="cuda").manual_seed(31337)
    batch, drafts, calls, temperature = 8, 7, 200, 1.0
    rows = batch * (drafts + 1)
    logits_row = torch.randn(base.V, generator=gen, device="cuda") - 12.0
    top = 40961
    logits_row[top] = 2.5
    logits_row[[7, 90000, 163000]] = torch.tensor([1.0, 0.5, 0.2], device="cuda")
    p = torch.softmax(logits_row.double() / temperature, 0)[top].item()
    logits = logits_row[None, :].expand(rows, -1).contiguous()
    draft = torch.full((batch, drafts), top, dtype=torch.int32, device="cuda")
    n_all = []
    with torch.inference_mode():
        for _ in range(calls):
            sampling = _sampling_inputs(gen, rows, [temperature] * rows)
            st = base.make_state(gen, batch)
            out = _sampled_call(base, st, logits, draft, drafts + 1, sampling, rows_per_cta)
            n_all.append(out[1].long())
    n = torch.cat(n_all).double()
    j = torch.arange(drafts + 1, dtype=torch.float64)
    pmf = torch.cat([p ** j[:-1] * (1 - p), torch.tensor([p**drafts], dtype=torch.float64)])
    mean = float((pmf * (j + 1)).sum())
    var = float((pmf * (j + 1) ** 2).sum()) - mean**2
    se = (var / n.numel()) ** 0.5
    rates = [round((n >= k + 2).double().mean().item(), 4) for k in range(drafts)]
    print(f"[acceptance rows_per_cta={rows_per_cta}] p_top={p:.4f} requests={n.numel()} mean n={n.mean().item():.4f} "
          f"closed form {mean:.4f} (se {se:.4f}); per-position rate {rates} vs "
          f"{[round(p ** (k + 1), 4) for k in range(drafts)]}")  # fmt: skip
    assert abs(n.mean().item() - mean) < 4.5 * se, (
        f"mean accepted {n.mean().item():.4f} vs {mean:.4f} (se {se:.4f})"
    )
    for k in range(drafts):
        rate = (n >= k + 2).double().mean().item()
        want = p ** (k + 1)
        se_k = (want * (1 - want) / n.numel()) ** 0.5
        assert abs(rate - want) < 4.5 * se_k + 1e-9, f"position {k}: {rate:.4f} vs {want:.4f}"
