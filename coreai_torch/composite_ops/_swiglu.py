# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Torch implementation of composite Swish Gated Linear Unit (SwiGLU) op."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing_extensions import Self

from ._utils import Version


class SwiGLUImpl(torch.nn.Module):
    """Core SwiGLU activation logic, intended to be externalized as a composite op.

    Takes both gate and value as explicit forward arguments so that both
    appear as graph inputs when externalized:
        SwiGLU(gate, value) = SiLU(gate) * value
                            = (gate * sigmoid(gate)) * value
    """

    def __init__(self: Self) -> None:
        super().__init__()
        self.version = Version.v1

    def forward(
        self: Self,
        gate: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SwiGLU activation: SiLU(gate) * value."""
        return F.silu(gate) * value


class SwiGLU(torch.nn.Module):
    """Swish Gated Linear Unit (SwiGLU) feed-forward module.

    As introduced in Shazeer (2020) "GLU Variants Improve Transformer"
    and widely used in modern LLMs (LLaMA, Mistral, Qwen, Gemma):
        FFN_SwiGLU(x) = (SiLU(x W_gate) * (x W_val)) W_out
    """

    def __init__(
        self: Self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(2 * in_features * 4 / 3)

        self.w_gate = torch.nn.Linear(in_features, hidden_features, bias=bias)
        self.w_val = torch.nn.Linear(in_features, hidden_features, bias=bias)
        self.w_out = torch.nn.Linear(hidden_features, out_features, bias=bias)
        self.swiglu_impl = SwiGLUImpl()

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass projecting x through gate and val, applying SwiGLU, and projecting out."""
        gate = self.w_gate(x)
        val = self.w_val(x)
        activated = self.swiglu_impl(gate, val)
        return self.w_out(activated)
