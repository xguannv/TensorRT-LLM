# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static routed-expert placement (``TRTLLM_K3LL_EXPERT_PLACEMENT``, flags.py), applied at load.

Expert parallelism gives EP rank ``r`` the contiguous routed-expert ids ("slots") ``[r * n, (r + 1) * n)``,
``n = num_experts / ep_size`` (896 / 4 = 224; EP rank ``r`` is the MoE-TP group of ranks ``r, r + 4, r + 8, r + 12``
under TP16 = MoE TP4 x EP4). A placement chooses, per MoE layer, which checkpoint expert each slot holds:

* every rank loads checkpoint expert ``expert_of_slot[s]`` into slot ``s`` (``KimiK3MoERuntime.local_expert_ids``
  lists them in local slot order; the inherited loader reads those checkpoint keys);
* once the checkpoint values are in place, the router's weight rows and score-correction bias are permuted into slot
  order (``permute_router_rows``), so routing returns slot ids directly and nothing runs at decode time: no kernel
  changes, CUDA graphs are unaffected.

What it changes numerically: every routed expert sees the same input and is weighted the same, so each (token,
expert) contribution is the same; the EP group that computes it changes, so the groups' bf16 partial sums and the
latent all-reduce over them add in another order. The router logits are the same values in permuted columns; top-k
breaks exact score ties by id, which is now the slot id. Layers the file does not list keep the identity placement.

File format (``ep4_place.py``'s)::

    {"format": "kimi_k3_expert_placement/v1",
     "num_experts": 896,
     "ep_size": 4,
     "expert_of_slot": {"<layer_idx>": [num_experts checkpoint expert ids], ...}}

A ``kimi_k3_expert_placement/v2`` file is v1 plus redundant copies::

     "replica_slots": S,
     "replicas": {"<layer_idx>": [[home slot, copy EP rank], ...], ...}   # priority order

Without ``TRTLLM_K3LL_EXPERT_REPLICAS`` it is read as its v1 permutation. With it (`load_replicas`,
`ExpertReplicas`), every EP rank holds ``224 + S`` local slots: its home slots, then the copies it holds (ascending
home slot), unused copy slots repeating its first home slot (never routed to). A copy slot loads its home slot's
checkpoint expert. Routing keeps returning home slot ids (896 space); the decode front's sort chooses, per step,
which copy computes each touched two-copy expert (`k3_moe_front`, ``replica_choice``), and every other path maps a
home slot ``x`` to its home copy ``x + (x // 224) * S`` (`ExpertReplicas.routed_slot_of_home`).
"""

from __future__ import annotations

import dataclasses
import functools
import json
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

FORMAT_V1 = "kimi_k3_expert_placement/v1"
FORMAT_V2 = "kimi_k3_expert_placement/v2"


@functools.lru_cache(maxsize=4)
def load_placement(path: str, num_experts: int, ep_size: int) -> Dict[int, Tuple[int, ...]]:
    """The per-layer ``expert_of_slot`` tables of placement file ``path``, identity layers dropped.

    Raises ``ValueError`` unless the file is a v1 (or v2) placement for ``num_experts`` experts balanced for
    ``ep_size`` EP ranks whose every table is a permutation of ``range(num_experts)`` keyed by a layer index.
    """
    with open(path) as f:
        data = json.load(f)
    fmt = data.get("format")
    if fmt not in (FORMAT_V1, FORMAT_V2):
        raise ValueError(f"{path}: expected format {FORMAT_V1!r} (or {FORMAT_V2!r}), got {fmt!r}")
    if data.get("num_experts") != num_experts:
        raise ValueError(
            f"{path}: placement is for {data.get('num_experts')} experts, the model has {num_experts}"
        )
    if data.get("ep_size") != ep_size:
        raise ValueError(
            f"{path}: placement is balanced for ep_size={data.get('ep_size')}, the routed MoE runs with "
            f"ep_size={ep_size}"
        )
    tables = data.get("expert_of_slot")
    if not isinstance(tables, dict):
        raise ValueError(f"{path}: expert_of_slot must map layer indices to tables")
    identity = tuple(range(num_experts))
    out: Dict[int, Tuple[int, ...]] = {}
    for layer, expert_of_slot in tables.items():
        try:
            layer_idx = int(layer)
        except ValueError as err:
            raise ValueError(f"{path}: layer key {layer!r} is not an integer") from err
        table = tuple(int(e) for e in expert_of_slot)
        if len(table) != num_experts or sorted(table) != list(identity):
            raise ValueError(
                f"{path}: layer {layer_idx} expert_of_slot is not a permutation of range({num_experts})"
            )
        if table != identity:
            out[layer_idx] = table
    return out


def check_layers(
    tables: Dict[int, Tuple[int, ...]], moe_layers: Iterable[int], num_layers: int, path: str
) -> None:
    """Raise ``ValueError`` if placement ``tables`` (from ``path``) permutes a layer of the model (``num_layers``
    decoder layers) that is not a MoE layer. Layers past the model's are ignored (a layer-truncated debug model)."""
    moe = set(moe_layers)
    bad = sorted(layer for layer in tables if layer < num_layers and layer not in moe)
    if bad:
        raise ValueError(f"{path}: permutes layers {bad}, which are not MoE layers of this model")


def local_checkpoint_experts(
    local_slots: Sequence[int], expert_of_slot: Optional[Sequence[int]]
) -> Tuple[int, ...]:
    """The checkpoint expert each of this rank's slots ``local_slots`` holds (the slots themselves without a
    placement)."""
    if expert_of_slot is None:
        return tuple(local_slots)
    return tuple(int(expert_of_slot[slot]) for slot in local_slots)


def permute_router_rows(
    weight: torch.Tensor, bias: torch.Tensor, expert_of_slot: Sequence[int]
) -> None:
    """Reorder the router in place so that row ``s`` scores checkpoint expert ``expert_of_slot[s]``.

    ``weight``: ``[num_experts, hidden]``; ``bias``: ``[num_experts]`` (the score-correction bias)."""
    index = torch.tensor(list(expert_of_slot), dtype=torch.long, device=weight.device)
    with torch.no_grad():
        weight.copy_(weight.index_select(0, index))
        bias.copy_(bias.index_select(0, index.to(bias.device)))


@dataclasses.dataclass(frozen=True)
class ExpertReplicas:
    """Second copies of some home slots of one MoE layer (the module docstring's v2 layout)."""

    num_experts: int
    ep_size: int
    #: Copy slots per EP rank (``S``); every MoE layer has them, used or not.
    slots_per_rank: int
    #: ``(home slot, EP rank of the copy)`` in priority order.
    entries: Tuple[Tuple[int, int], ...]

    @property
    def num_home(self) -> int:
        """Home slots per EP rank."""
        return self.num_experts // self.ep_size

    @property
    def num_local(self) -> int:
        """Slots per EP rank: home slots, then copy slots."""
        return self.num_home + self.slots_per_rank

    @property
    def num_slots(self) -> int:
        """Slots of the routed MoE (rank-major)."""
        return self.ep_size * self.num_local

    def copies_on(self, rank: int) -> Tuple[int, ...]:
        """Home slots whose copy EP rank ``rank`` holds, in local-slot (ascending) order."""
        return tuple(sorted(x for x, r in self.entries if r == rank))

    def home_slots_of_local(self, rank: int) -> Tuple[int, ...]:
        """The home slot whose expert each local slot of ``rank`` holds; copy slots without a copy repeat the rank's
        first home slot."""
        home = tuple(range(rank * self.num_home, (rank + 1) * self.num_home))
        copies = self.copies_on(rank)
        return home + copies + (home[0],) * (self.slots_per_rank - len(copies))

    def routed_slot_of_home(self) -> Tuple[int, ...]:
        """Routed-MoE slot of the home copy of each home slot."""
        return tuple(
            x + (x // self.num_home) * self.slots_per_rank for x in range(self.num_experts)
        )


@functools.lru_cache(maxsize=4)
def load_replicas(
    path: str, num_experts: int, ep_size: int
) -> Tuple[int, Dict[int, Tuple[Tuple[int, int], ...]]]:
    """``(replica_slots, {layer_idx: entries})`` of v2 placement file ``path``, layers without copies dropped.

    Raises ``ValueError`` unless the file is v2 for ``num_experts`` experts and ``ep_size`` EP ranks, with
    ``replica_slots >= 1``, every copy on a rank other than its home, no expert copied twice and at most
    ``replica_slots`` copies per rank and layer."""
    with open(path) as f:
        data = json.load(f)
    if data.get("format") != FORMAT_V2:
        raise ValueError(
            f"{path}: redundant copies need format {FORMAT_V2!r}, got {data.get('format')!r}"
        )
    if data.get("num_experts") != num_experts or data.get("ep_size") != ep_size:
        raise ValueError(
            f"{path}: copies are for {data.get('num_experts')} experts over ep_size={data.get('ep_size')}, the "
            f"routed MoE has {num_experts} over ep_size={ep_size}"
        )
    slots = int(data.get("replica_slots", 0))
    if slots < 1 or num_experts % ep_size:
        raise ValueError(f"{path}: needs replica_slots >= 1 and ep_size dividing {num_experts}")
    num_home = num_experts // ep_size
    layers: Dict[int, Tuple[Tuple[int, int], ...]] = {}
    for layer, raw in data.get("replicas", {}).items():
        try:
            layer_idx = int(layer)
        except ValueError as err:
            raise ValueError(f"{path}: replicas layer key {layer!r} is not an integer") from err
        entries = tuple((int(x), int(r)) for x, r in raw)
        per_rank = [0] * ep_size
        for x, r in entries:
            if not (0 <= x < num_experts and 0 <= r < ep_size and r != x // num_home):
                raise ValueError(f"{path}: layer {layer_idx}: bad copy {[x, r]}")
            per_rank[r] += 1
        if len({x for x, _ in entries}) != len(entries) or max(per_rank) > slots:
            raise ValueError(
                f"{path}: layer {layer_idx}: an expert copied twice, or more than {slots} copies on one rank"
            )
        if entries:
            layers[layer_idx] = entries
    return slots, layers


def replicas_for_layer(path: str, layer_idx: int, num_experts: int, ep_size: int) -> ExpertReplicas:
    """``layer_idx``'s copies from v2 file ``path`` (`load_replicas`); a layer without copies still gets the file's
    ``replica_slots`` (empty) copy slots, so every MoE layer has the same slot layout."""
    slots, layers = load_replicas(path, num_experts, ep_size)
    return ExpertReplicas(num_experts, ep_size, slots, layers.get(layer_idx, ()))
