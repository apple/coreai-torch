# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Graph-level optimization and kernel fusion passes for Core AI."""

from .fusion import (
    fuse_layernorm_gelu,
    fused_layernorm_gelu_kernel,
    run_graph_fusion_passes,
)

__all__ = [
    "fuse_layernorm_gelu",
    "fused_layernorm_gelu_kernel",
    "run_graph_fusion_passes",
]
