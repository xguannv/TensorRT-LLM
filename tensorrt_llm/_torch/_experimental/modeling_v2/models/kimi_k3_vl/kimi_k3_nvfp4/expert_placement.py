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

A ``kimi_k3_expert_placement/v2`` file (v1 plus replica slots) is read as its v1 permutation; its replicas are not
used here.
"""

from __future__ import annotations

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
