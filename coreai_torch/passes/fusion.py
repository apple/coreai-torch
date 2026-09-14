# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Execution graph optimization passes and fused Metal kernels.

Provides graph-level pattern matching and kernel fusion passes to combine
adjacent operations into single Metal GPU kernels, eliminating redundant
intermediate buffer allocations and memory bandwidth round-trips on
Apple Silicon Unified Memory Architecture (UMA).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.fx as fx
import torch.nn.functional as F
from coreai.authoring import MetalParameter

from coreai_torch._torch_metal_kernel import TorchMetalKernel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fused LayerNorm + GELU Metal Kernel (MSL)
# ---------------------------------------------------------------------------

FUSED_LAYERNORM_GELU_MSL = """
    // Fused LayerNorm + GELU kernel for Apple Silicon UMA
    // Dispatched with 1 thread per slice (row) across outer dimensions
    uint normalized_size = weight.get_extent(0);
    uint row_idx = id;
    uint total_elements = output.get_extent(0);
    uint total_rows = total_elements / normalized_size;
    if (row_idx >= total_rows) return;

    uint row_offset = row_idx * normalized_size;

    // Pass 1: Compute mean
    float sum_val = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        sum_val += static_cast<float>(x[row_offset + i]);
    }
    float mean = sum_val / static_cast<float>(normalized_size);

    // Pass 2: Compute variance
    float sq_diff_sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        float diff = static_cast<float>(x[row_offset + i]) - mean;
        sq_diff_sum += diff * diff;
    }
    float variance = sq_diff_sum / static_cast<float>(normalized_size);
    float inv_std = rsqrt(variance + 1e-5f);

    // Pass 3: Normalize, apply affine scale/bias, and compute GELU in registers
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF = 0.044715f;

    for (uint i = 0; i < normalized_size; ++i) {
        float val = static_cast<float>(x[row_offset + i]);
        float w = static_cast<float>(weight[i]);
        float b = static_cast<float>(bias[i]);

        float norm = (val - mean) * inv_std * w + b;

        // Fused GELU (tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))))
        float cube = norm * norm * norm;
        float inner = SQRT_2_OVER_PI * (norm + COEFF * cube);
        float gelu = 0.5f * norm * (1.0f + metal::tanh(inner));

        output[row_offset + i] = static_cast<TYPE>(gelu);
    }
"""


def _ref_fused_layernorm_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Reference eager PyTorch implementation of Fused LayerNorm + GELU."""
    norm_dim = weight.shape[-1]
    norm = F.layer_norm(x, (norm_dim,), weight=weight, bias=bias, eps=1e-5)
    return F.gelu(norm, approximate="tanh")


fused_layernorm_gelu_kernel = TorchMetalKernel(
    name="fused_layernorm_gelu",
    input_names=["x", "weight", "bias"],
    result_names=["output"],
    src=FUSED_LAYERNORM_GELU_MSL,
    torch_defn=_ref_fused_layernorm_gelu,
    metal_params=[
        MetalParameter("id", "uint", "thread_position_in_grid"),
    ],
)


# ---------------------------------------------------------------------------
# Graph Pattern Matching & Fusion Passes
# ---------------------------------------------------------------------------


def fuse_layernorm_gelu(graph_module: fx.GraphModule) -> fx.GraphModule:
    """Optimization pass fusing adjacent LayerNorm -> GELU nodes into a single Metal kernel.

    Detects:
        %norm = torch.ops.aten.layer_norm(%x, %normalized_shape, %weight, %bias, ...)
        %act  = torch.ops.aten.gelu(%norm, ...)

    Rewrites to:
        %fused = fused_layernorm_gelu_kernel(%x, %weight, %bias)

    Eliminating the intermediate normalized tensor buffer and kernel launch.

    Args:
        graph_module: The PyTorch FX GraphModule to optimize.

    Returns:
        The optimized FX GraphModule with fused operations.
    """
    graph = graph_module.graph
    nodes_to_fuse: list[tuple[fx.Node, fx.Node]] = []

    layernorm_targets = {
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.native_layer_norm.default,
        F.layer_norm,
    }
    gelu_targets = {
        torch.ops.aten.gelu.default,
        F.gelu,
    }

    for node in graph.nodes:
        if node.op == "call_function" and node.target in layernorm_targets:
            # Check if this LayerNorm node has exactly one consumer
            if len(node.users) == 1:
                consumer = next(iter(node.users))
                if consumer.op == "call_function" and consumer.target in gelu_targets:
                    nodes_to_fuse.append((node, consumer))

    if not nodes_to_fuse:
        return graph_module

    for ln_node, gelu_node in nodes_to_fuse:
        # Extract inputs from LayerNorm node
        # aten.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)
        args = ln_node.args
        kwargs = ln_node.kwargs

        x_input = args[0] if len(args) > 0 else kwargs.get("input")
        weight = args[2] if len(args) > 2 else kwargs.get("weight")
        bias = args[3] if len(args) > 3 else kwargs.get("bias")

        # Fallback if weight or bias is missing
        if weight is None or bias is None:
            continue

        with graph.inserting_before(ln_node):
            fused_call = graph.call_function(
                fused_layernorm_gelu_kernel.torch_custom_op,
                args=(x_input, weight, bias),
            )
            fused_call.meta = (
                gelu_node.meta.copy() if hasattr(gelu_node, "meta") else {}
            )

        gelu_node.replace_all_uses_with(fused_call)
        graph.erase_node(gelu_node)
        graph.erase_node(ln_node)
        logger.info(
            "Fused LayerNorm (%s) and GELU (%s) into %s",
            ln_node.name,
            gelu_node.name,
            fused_call.name,
        )

    graph.eliminate_dead_code()
    graph_module.recompile()
    return graph_module


def run_graph_fusion_passes(
    graph_module: fx.GraphModule,
    passes: list[Callable[[fx.GraphModule], fx.GraphModule]] | None = None,
) -> fx.GraphModule:
    """Run a pipeline of graph cleaning and kernel fusion passes on an FX GraphModule.

    Args:
        graph_module: The input graph module.
        passes: Optional sequence of graph pass callables. Defaults to standard
            fusion passes including `fuse_layernorm_gelu`.

    Returns:
        The optimized FX GraphModule.
    """
    if passes is None:
        passes = [fuse_layernorm_gelu]

    for p in passes:
        graph_module = p(graph_module)

    return graph_module
