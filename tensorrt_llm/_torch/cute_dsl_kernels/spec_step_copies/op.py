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
"""Host side of the one-model speculative decoding step's copy kernels (``spec_step_copies_kernel``).

``SlotScatter`` moves a step's per-row outputs into the sampler's slot stores (one launch for the sampler's four
``index_copy_``). ``StepInputGather`` performs the overlap scheduler's gathers from those stores into a decode step's
inputs (one launch for the engine's four). Each kernel takes every size and address as a launch argument, so it is
compiled once per process, on the first launch or by ``precompile``, outside CUDA-graph capture. The kernels are
called through TVM-FFI: a launch costs a few microseconds of host time.

Callers build them where ``build_at_load()`` holds and launch them where ``is_enabled()`` holds on that call. A call
whose arguments are outside what the kernel covers launches nothing and returns False; the caller then does that work
itself.
"""

from __future__ import annotations

import os
from typing import Any, Callable, ClassVar, Optional

import torch

from ...._utils import get_sm_version
from ...cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

SM103_ENV_VAR = "TRTLLM_K3LL_SPEC_STEP_COPIES_SM103"

# A model that flips ``TRTLLM_K3LL_SPEC_STEP_COPIES_SM103`` at run time installs its reader here
# (``set_sm103_switch``); without one the environment decides. Either is read on every call.
_sm103_reader: Optional[Callable[[], Any]] = None
_sm103_dispatched: Optional[Callable[[], None]] = None


def set_sm103_switch(
    reader: Optional[Callable[[], Any]], dispatched: Optional[Callable[[], None]] = None
) -> None:
    """Read ``TRTLLM_K3LL_SPEC_STEP_COPIES_SM103`` as ``reader()`` (true = on) on every call instead of from the
    environment, and call ``dispatched()`` after each launch on SM 103. ``reader=None`` restores the environment.

    With a reader installed the switch can change between steps, so the callers build their state and compile the
    kernels at load on SM 103 whatever its value (``precompile_at_load``).
    """
    global _sm103_reader, _sm103_dispatched
    _sm103_reader = reader
    _sm103_dispatched = dispatched if reader is not None else None


def sm103_enabled() -> bool:
    """Whether ``TRTLLM_K3LL_SPEC_STEP_COPIES_SM103`` admits SM 103 (GB300) now; default 0."""
    reader = _sm103_reader
    if reader is not None:
        return bool(reader())
    return os.environ.get(SM103_ENV_VAR, "0") == "1"


def is_available() -> bool:
    """Whether the kernels can run on the current device: SM 100 (B200 / GB200) or SM 103 (GB300) with the CuTe DSL
    installed. Callers build their state where ``build_at_load()`` holds and ask ``is_enabled()`` on every call.

    The kernels are plain SIMT code (thread and block indices, global loads and stores), with nothing specific to an
    architecture. On SM 103 a switch admits them because ``SlotScatter`` stores zeros in a row's ``new_tokens`` past
    its accepted length where the torch path stores the forward's raw columns: no reader looks there, but the stores
    are not bit-identical. SM 107 keeps the torch path.
    """
    if not (IS_CUTLASS_DSL_AVAILABLE and torch.cuda.is_available()):
        return False
    return get_sm_version() in (100, 103)


def is_enabled() -> bool:
    """Per call, on a device where ``is_available()`` holds: SM 100 always, SM 103 when the switch is on now."""
    return get_sm_version() != 103 or sm103_enabled()


def is_supported() -> bool:
    """Whether the kernels run now: ``is_available()`` and ``is_enabled()``, i.e. SM 100, and SM 103 (GB300) when
    ``TRTLLM_K3LL_SPEC_STEP_COPIES_SM103=1``."""
    return is_available() and is_enabled()


def build_at_load() -> bool:
    """Whether a caller builds its kernel state at load: the kernels run now, or a run-time reader may turn them on
    between steps. Without a reader, SM 103 with the switch off builds nothing, as before the switch could change."""
    return is_available() and (is_enabled() or _sm103_reader is not None)


def precompile_at_load() -> bool:
    """Whether callers compile the kernels at load: on SM 103 with a run-time reader installed (the switch may turn
    on between steps; the first launch must not compile in a timed step). Elsewhere they compile on first launch."""
    return _sm103_reader is not None and is_available() and get_sm_version() == 103


def _note_dispatch() -> None:
    """A launch happened: report it on SM 103 (the switched architecture)."""
    dispatched = _sm103_dispatched
    if dispatched is not None and get_sm_version() == 103:
        dispatched()


def _is_i32(t: torch.Tensor) -> bool:
    """A contiguous CUDA int32 tensor."""
    return t.is_cuda and t.dtype == torch.int32 and t.is_contiguous()


def _compile(name: str, args: tuple):
    """Compile kernel ``name`` for its scalar ``args`` with TVM-FFI; a call then takes ``args`` and the CUDA stream
    handle to launch on."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"spec_step_copies: the {name} kernel must run once outside CUDA-graph capture first "
            "(it compiles on its first launch)."
        )
    import cutlass.cute as cute

    from . import spec_step_copies_kernel as kernel

    return cute.compile(
        getattr(kernel, name), *args, cute.runtime.make_fake_stream(), options="--enable-tvm-ffi"
    )


def _gather_shapes_ok(
    store_next_new_tokens: torch.Tensor,
    store_next_draft: torch.Tensor,
    store_lens: torch.Tensor,
    rows: int,
    tokens_per_row: int,
    draft_width: int,
) -> bool:
    """The overlap gather's stores are [width, slots, 1], [slots, draft width] and [slots], wide enough for a row."""
    num_slots = store_lens.numel()
    return (
        store_lens.dim() == 1
        and store_next_new_tokens.dim() == 3
        and store_next_new_tokens.shape[1:] == (num_slots, 1)
        and store_next_draft.dim() == 2
        and store_next_draft.shape[0] == num_slots
        and 0 < tokens_per_row <= store_next_new_tokens.shape[0]
        and 0 <= draft_width < tokens_per_row
        and draft_width <= store_next_draft.shape[1]
        and rows >= 0
    )


class SlotScatter:
    """The speculative sampler's store update as one kernel.

    ``store[..., slots[r]] = outputs[row_begin + r]`` for every row ``r``, each row padded with zeros or cut to its
    store's width. A row's ``new_tokens`` at or past its ``new_tokens_lens`` are stored as zeros: the forward writes
    only column 0 of a context row, and readers of the store stop at that length.
    """

    _kernel: ClassVar[Any] = None
    # scatter_kernel's arguments: 9 addresses, 10 sizes.
    _NUM_ARGS: ClassVar[int] = 19

    def scatter(
        self,
        outputs: dict[str, torch.Tensor],
        row_begin: int,
        rows: int,
        slots: torch.Tensor,
        store_new_tokens: torch.Tensor,
        store_next_new_tokens: torch.Tensor,
        store_lens: torch.Tensor,
        store_next_draft: torch.Tensor,
    ) -> bool:
        """Launch the store update on the current stream.

        Args:
            outputs: The forward's ``new_tokens`` [rows_total, a], ``next_new_tokens`` [rows_total, b],
                ``new_tokens_lens`` [rows_total] and ``next_draft_tokens`` [rows_total, c], int32.
            row_begin: The first row of ``outputs`` to move.
            rows: The number of rows to move.
            slots: The distinct slot of each moved row, int32 [>= rows] on the device.
            store_new_tokens: int32 [width, num_slots, 1].
            store_next_new_tokens: int32 [width, num_slots, 1].
            store_lens: int32 [num_slots].
            store_next_draft: int32 [num_slots, width].

        Returns:
            False, launching nothing, when an argument is outside what the kernel covers (a non-contiguous or
            non-int32 tensor, a store of another layout, fewer output rows or slots than ``rows``); True otherwise.
        """
        if rows == 0:
            return True
        new_tokens = outputs["new_tokens"]
        next_new_tokens = outputs["next_new_tokens"]
        lens = outputs["new_tokens_lens"]
        next_draft = outputs["next_draft_tokens"]
        num_slots = store_lens.numel()
        tensors = (
            new_tokens,
            next_new_tokens,
            lens,
            next_draft,
            slots,
            store_new_tokens,
            store_next_new_tokens,
            store_lens,
            store_next_draft,
        )
        if not (
            all(_is_i32(t) for t in tensors)
            and new_tokens.dim() == next_new_tokens.dim() == next_draft.dim() == 2
            and lens.dim() == store_lens.dim() == 1
            and min(t.shape[0] for t in (new_tokens, next_new_tokens, lens, next_draft))
            >= row_begin + rows
            and row_begin >= 0
            and slots.numel() >= rows
            and store_new_tokens.dim() == store_next_new_tokens.dim() == 3
            and store_new_tokens.shape[1:] == (num_slots, 1)
            and store_next_new_tokens.shape[1:] == (num_slots, 1)
            and store_next_draft.dim() == 2
            and store_next_draft.shape[0] == num_slots
        ):
            return False
        columns = max(
            store_new_tokens.shape[0], store_next_new_tokens.shape[0], store_next_draft.shape[1], 1
        )
        args = (
            new_tokens.data_ptr(),
            next_new_tokens.data_ptr(),
            lens.data_ptr(),
            next_draft.data_ptr(),
            slots.data_ptr(),
            store_new_tokens.data_ptr(),
            store_next_new_tokens.data_ptr(),
            store_lens.data_ptr(),
            store_next_draft.data_ptr(),
            rows,
            row_begin,
            columns,
            num_slots,
            store_new_tokens.shape[0],
            new_tokens.shape[1],
            store_next_new_tokens.shape[0],
            next_new_tokens.shape[1],
            store_next_draft.shape[1],
            next_draft.shape[1],
        )
        if SlotScatter._kernel is None:
            SlotScatter._kernel = _compile("scatter", args)
        SlotScatter._kernel(*args, torch.cuda.current_stream().cuda_stream)
        _note_dispatch()
        return True

    @classmethod
    def precompile(cls) -> None:
        """Compile the kernel now, outside CUDA-graph capture, if no launch has (sizes and addresses are launch
        arguments: one build serves every call)."""
        if cls._kernel is None:
            cls._kernel = _compile("scatter", (0,) * cls._NUM_ARGS)


class StepInputGather:
    """The overlap scheduler's gathers of a decode step's inputs from the sampler's slot stores, as one kernel."""

    _kernel: ClassVar[Any] = None
    # gather_kernel's arguments: 9 addresses, 9 sizes.
    _NUM_ARGS: ClassVar[int] = 18

    def gather(
        self,
        store_next_new_tokens: torch.Tensor,
        store_next_draft: torch.Tensor,
        store_lens: torch.Tensor,
        slots: torch.Tensor,
        pos_indices: torch.Tensor,
        rows: int,
        tokens_per_row: int,
        draft_width: int,
        input_ids: torch.Tensor,
        input_begin: int,
        draft_tokens: torch.Tensor,
        draft_begin: int,
        pos_offsets: torch.Tensor,
        pos_begin: int,
        kv_offsets: torch.Tensor,
        kv_begin: int,
    ) -> bool:
        """Launch the gathers of the rows whose request ran in the previous step on the current stream.

        For ``r < rows`` with ``s = slots[r]`` and ``j < tokens_per_row``:
        ``input_ids[input_begin + r * tokens_per_row + j] = store_next_new_tokens[j, s, 0]``,
        ``pos_offsets[pos_begin + r * tokens_per_row + j] = store_lens[pos_indices[r * tokens_per_row + j]]``,
        ``draft_tokens[draft_begin + r * draft_width + j] = store_next_draft[s, j]`` for ``j < draft_width`` and
        ``kv_offsets[kv_begin + r] = store_lens[s] - tokens_per_row``. Every tensor is int32 on the device; the
        stores are [width, slots, 1], [slots, width] and [slots].

        Returns:
            False, launching nothing, when an argument is outside what the kernel covers; True otherwise.
        """
        tensors = (
            store_next_new_tokens,
            store_next_draft,
            store_lens,
            slots,
            pos_indices,
            input_ids,
            draft_tokens,
            pos_offsets,
            kv_offsets,
        )
        if not (
            all(_is_i32(t) for t in tensors)
            and _gather_shapes_ok(
                store_next_new_tokens,
                store_next_draft,
                store_lens,
                rows,
                tokens_per_row,
                draft_width,
            )
            and slots.numel() >= rows
            and pos_indices.numel() >= rows * tokens_per_row
            and input_ids.numel() >= input_begin + rows * tokens_per_row
            and draft_tokens.numel() >= draft_begin + rows * draft_width
            and pos_offsets.numel() >= pos_begin + rows * tokens_per_row
            and kv_offsets.numel() >= kv_begin + rows
            and min(input_begin, draft_begin, pos_begin, kv_begin) >= 0
        ):
            return False
        if rows == 0:
            return True
        args = (
            store_next_new_tokens.data_ptr(),
            store_next_draft.data_ptr(),
            store_lens.data_ptr(),
            slots.data_ptr(),
            pos_indices.data_ptr(),
            input_ids.data_ptr(),
            draft_tokens.data_ptr(),
            pos_offsets.data_ptr(),
            kv_offsets.data_ptr(),
            rows,
            tokens_per_row,
            draft_width,
            store_lens.numel(),
            store_next_draft.shape[1],
            input_begin,
            draft_begin,
            pos_begin,
            kv_begin,
        )
        if StepInputGather._kernel is None:
            StepInputGather._kernel = _compile("gather", args)
        StepInputGather._kernel(*args, torch.cuda.current_stream().cuda_stream)
        _note_dispatch()
        return True

    @classmethod
    def precompile(cls) -> None:
        """Compile the kernel now, outside CUDA-graph capture, if no launch has (sizes and addresses are launch
        arguments: one build serves every call)."""
        if cls._kernel is None:
            cls._kernel = _compile("gather", (0,) * cls._NUM_ARGS)
