# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K3 switches read outside this package, by the engine's generic speculative-decoding code.

``TRTLLM_K3LL_SPEC_STEP_COPIES_SM103`` (``cute_dsl_kernels/spec_step_copies/op.py``: the sampler's store update and
the overlap scheduler's input gathers) and ``TRTLLM_K3LL_SPEC_HEAD_FUSED`` (``speculative/spec_head_kernels.py``: the
decode-step head in ``DecoderRunner._preprocess_inputs``) are read there on every call. Those modules may not import
this package (it is imported only when the K3 model is selected), so ``install`` hands them readers of the flags
module's attributes instead: the value in force is ``_flags.X`` at the call, which a one-process A/B changes between
steps and CUDA-graph recaptures, and ``census`` counts each launch on the switched path
(``dispatch:spec_step_copies_sm103:<mode>``, ``dispatch:spec_head_fused:<mode>``).

Without ``install`` (any other model) those modules read the environment, as before. With it, the launch value is
the environment's too (``flags.py`` reads it at import).
"""

from tensorrt_llm._torch.cute_dsl_kernels.spec_step_copies import op as _spec_step_copies
from tensorrt_llm._torch.speculative import spec_head_kernels as _spec_head_kernels

from . import census as _census
from . import flags as _flags


def _step_copies_sm103_on() -> bool:
    return _flags.TRTLLM_K3LL_SPEC_STEP_COPIES_SM103 == 1


def _spec_head_fused_on() -> bool:
    return _flags.TRTLLM_K3LL_SPEC_HEAD_FUSED == 1


def _count_step_copies_sm103() -> None:
    _census.count(f"dispatch:spec_step_copies_sm103:{_census.mode()}")


def _count_spec_head_fused() -> None:
    _census.count(f"dispatch:spec_head_fused:{_census.mode()}")


def install() -> None:
    """Point the generic modules' switches at this package's flags (and its census, when on). Call at model
    construction, before the engine builds its sampler and runner: they build the step-copy state, and compile the
    kernels on SM103, at load whatever the switch's value."""
    counting = _census.ENABLED
    _spec_step_copies.set_sm103_switch(
        _step_copies_sm103_on, _count_step_copies_sm103 if counting else None
    )
    _spec_head_kernels.set_spec_head_fusion_switch(
        _spec_head_fused_on, _count_spec_head_fused if counting else None
    )


def uninstall() -> None:
    """Back to the environment (tests)."""
    _spec_step_copies.set_sm103_switch(None)
    _spec_head_kernels.set_spec_head_fusion_switch(None)
