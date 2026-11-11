# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep16 target's attention-DP KV-balanced admission planner (``RuntimeHooks.kv_balanced_admission_planner``).

With attention DP every rank attends over the KV of its own requests and the ranks step in lockstep, so the step
time follows the rank holding the most KV tokens. Among the first ``WINDOW`` waiting requests, the planner gives
each free slot the request whose prompt brings that slot's rank closest to the mean load the ranks would have
once every slot admitted this round is filled. A request the engine marks as forced (passed over in too many
admitting rounds) is admitted first, to the least loaded rank.

The engine's ``DefaultADPRouter`` keeps the rest: it calls the planner only for a window of generation-only
requests, counts the rounds a request is passed over and forces it after ``KV_BALANCE_MAX_SKIPPED_ROUNDS`` (64),
and places the admitted requests as planned. Every rank calls the planner with the same all-gathered inputs; it is
deterministic.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

__all__ = ["K3Dep16KvBalancedAdmission", "plan_kv_balanced_admission"]


def plan_kv_balanced_admission(
    candidate_tokens: Sequence[int],
    candidate_forced: Sequence[bool],
    rank_tokens: Sequence[int],
    rank_free_slots: Sequence[int],
    max_new: int,
) -> List[Tuple[int, int]]:
    """Return ``(candidate_index, rank)`` pairs, in admission order.

    Args:
        candidate_tokens: KV tokens each waiting candidate brings (queue order).
        candidate_forced: Whether the candidate has waited too long and must be admitted before any other choice.
        rank_tokens: Current KV tokens per rank.
        rank_free_slots: Free request slots per rank.
        max_new: Upper bound on the number of admissions.
    """
    loads = list(rank_tokens)
    free = [max(0, f) for f in rank_free_slots]
    num_ranks = len(loads)
    remaining = list(range(len(candidate_tokens)))
    budget = min(max_new, sum(free), len(remaining))
    plan: List[Tuple[int, int]] = []
    if budget <= 0:
        return plan

    # Mean load once every slot this round fills holds an average-sized candidate.
    avg_candidate = sum(candidate_tokens) / len(candidate_tokens)
    target = (sum(loads) + avg_candidate * budget) / num_ranks

    for _ in range(budget):
        open_ranks = [r for r in range(num_ranks) if free[r] > 0]
        if not open_ranks or not remaining:
            break
        forced = [i for i in remaining if candidate_forced[i]]
        if forced:
            idx = forced[0]
            rank = min(open_ranks, key=lambda r: (loads[r], r))
        else:
            # Fill the rank furthest below target first; give it the candidate that lands it closest to the target
            # (ties: earlier in the queue).
            rank = min(open_ranks, key=lambda r: (loads[r] - target, r))
            want = target - loads[rank]
            idx = min(remaining, key=lambda i: (abs(candidate_tokens[i] - want), i))
        plan.append((idx, rank))
        remaining.remove(idx)
        loads[rank] += candidate_tokens[idx]
        free[rank] -= 1
    return plan


class K3Dep16KvBalancedAdmission:
    """``KvBalancedAdmissionPlanner`` of the dep16 target: a fixed head window of ``WINDOW`` requests and
    :func:`plan_kv_balanced_admission`. Admission groups are not used: every candidate is placed on its own."""

    WINDOW = 32

    def window(self, max_new_requests: int, free_slots: int) -> int:
        return self.WINDOW

    def plan(
        self,
        *,
        candidate_tokens: list[int],
        candidate_forced: list[bool],
        rank_tokens: list[int],
        rank_num_requests: list[int],
        max_requests_per_rank: int,
        max_new_requests: int,
        candidate_groups: list[int],
    ) -> list[tuple[int, int]]:
        return plan_kv_balanced_admission(
            candidate_tokens,
            candidate_forced,
            rank_tokens,
            [max_requests_per_rank - n for n in rank_num_requests],
            max_new_requests,
        )
