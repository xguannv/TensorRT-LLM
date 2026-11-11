# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The modules every Kimi K3 SM107 target shares.

* ``layers``: the kernel-call primitives (K3-owned weights, linear, norm, MLP, embedding, the TP
  all-reduce and the Lamport pushes). The glue of each target, ``sm107_helix8_parts/`` and
  ``dspark/`` all import this one module, so its MNNVL workspace caches exist once per process.
* ``constants``: ``K3Sm107Constants``, the record of what a target runs.
"""
