# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Fusing the FP8 KDA projections must not leave the sources resident.

_fuse_checkpoint_projections copies q/k/v/g into one fused Linear. Until the
sources are repointed at row views of that buffer, both copies stay alive and
the projection weights are held twice. The BF16 path already repoints in
_merge_projection_weights; this covers the FP8 path doing the same. CPU-only.
"""

import torch

from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention


class _FakeLinear:
    """Minimal stand-in exposing the attributes the aliasing touches."""

    def __init__(self, rows: int, cols: int, scale_rows: int, fill: float) -> None:
        self.weight = torch.nn.Parameter(
            torch.full((rows, cols), fill, dtype=torch.bfloat16), requires_grad=False
        )
        self.weight_scale = torch.nn.Parameter(
            torch.full((scale_rows, cols // 128), fill, dtype=torch.float32),
            requires_grad=False,
        )
        self.out_features = rows


def _fused_of(modules, cols: int, scale_cols: int):
    total = sum(m.weight.shape[0] for m in modules)
    total_scale = sum(m.weight_scale.shape[0] for m in modules)
    fused = _FakeLinear(total, cols, total_scale, 0.0)
    offset = scale_offset = 0
    for module in modules:
        rows = module.weight.shape[0]
        fused.weight.data[offset : offset + rows].copy_(module.weight.data)
        offset += rows
        srows = module.weight_scale.shape[0]
        fused.weight_scale.data[scale_offset : scale_offset + srows].copy_(module.weight_scale.data)
        scale_offset += srows
    return fused


def _modules():
    return tuple(_FakeLinear(rows=4, cols=256, scale_rows=1, fill=float(i + 1)) for i in range(4))


def test_sources_share_storage_with_the_fused_buffer() -> None:
    modules = _modules()
    fused = _fused_of(modules, cols=256, scale_cols=2)
    before = [m.weight.data_ptr() for m in modules]

    KimiKDALinearAttention._alias_projections_into_fused(modules, fused)

    base = fused.weight.data_ptr()
    end = base + fused.weight.numel() * fused.weight.element_size()
    for module, old_ptr in zip(modules, before):
        ptr = module.weight.data_ptr()
        assert ptr != old_ptr, "source still points at its own storage"
        assert base <= ptr < end, "source is not a view into the fused weight"


def test_aliasing_preserves_each_projection_values() -> None:
    modules = _modules()
    expected = [m.weight.detach().clone() for m in modules]
    fused = _fused_of(modules, cols=256, scale_cols=2)

    KimiKDALinearAttention._alias_projections_into_fused(modules, fused)

    for module, want in zip(modules, expected):
        assert torch.equal(module.weight, want)


def test_scales_are_aliased_too() -> None:
    modules = _modules()
    fused = _fused_of(modules, cols=256, scale_cols=2)

    KimiKDALinearAttention._alias_projections_into_fused(modules, fused)

    base = fused.weight_scale.data_ptr()
    end = base + fused.weight_scale.numel() * fused.weight_scale.element_size()
    for module in modules:
        ptr = module.weight_scale.data_ptr()
        assert base <= ptr < end, "scale grid was left on its own storage"


def test_writing_through_the_fused_buffer_reaches_the_sources() -> None:
    # The fusion comment notes DeepGEMM resmooths codes in place; after
    # aliasing those in-place edits must be visible through the projections.
    modules = _modules()
    fused = _fused_of(modules, cols=256, scale_cols=2)
    KimiKDALinearAttention._alias_projections_into_fused(modules, fused)

    fused.weight.data.fill_(9.0)

    for module in modules:
        assert torch.all(module.weight == 9.0)
