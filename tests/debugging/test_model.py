# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Model for hierarchical benchmarking tests."""

import math
from collections import OrderedDict

import torch

from .test_submodel import SubModel


class HierarchicalModel(torch.nn.Module):
    """A model that uses SubModel for hierarchical testing."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 8)
        self.sub_model1 = SubModel()
        self.sub_model2 = SubModel()
        self.linear2 = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multiple operations and sub-modules."""
        x = self.linear1(x)
        x = self.sub_model1(x)
        x = torch.relu(x)
        x = self.sub_model2(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x_fp32 * rms
        return y.to(x.dtype) * self.weight


class ElemwiseBlock(torch.nn.Module):
    def forward(self, x, y):
        z = x + 0.5 * y
        z = torch.nn.functional.silu(z)
        z = z * torch.sigmoid(y)
        z = z + torch.nn.functional.gelu(x)
        return torch.clamp(z, -3.0, 3.0)


class LayerNormBlock(torch.nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.ln = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x)


class NormMLPBlock(torch.nn.Module):
    def __init__(self, dim=8, hidden=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, dim)
        self.ln = torch.nn.LayerNorm(dim)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = torch.nn.functional.gelu(h)
        h = self.fc2(h)
        return x + h


class GatedResidualBlock(torch.nn.Module):
    def __init__(self, dim=8, hidden=16):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gate = torch.nn.Linear(dim, hidden, bias=False)
        self.up = torch.nn.Linear(dim, hidden, bias=False)
        self.down = torch.nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        h = self.norm(x)
        h = torch.nn.functional.silu(self.gate(h)) * self.up(h)
        h = self.down(h)
        return x + h


class BatchedMatmulBlock(torch.nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(x.shape[-1])
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)


class TinySelfAttention(torch.nn.Module):
    def __init__(self, dim=8, num_heads=2):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v)
        attn = attn.transpose(1, 2).contiguous().view(b, s, d)
        return self.proj(attn)


class SDPAAttentionBlock(torch.nn.Module):
    def __init__(self, dim=8, num_heads=2):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim, bias=False)
        self.o = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, s, d = x.shape

        def split_heads(t):
            return t.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q(x))
        k = split_heads(self.k(x))
        v = split_heads(self.v(x))

        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        attn = attn.transpose(1, 2).contiguous().view(b, s, d)
        return self.o(attn)


class ConcatSplitBlock(torch.nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.a = torch.nn.Linear(dim, dim, bias=False)
        self.b = torch.nn.Linear(dim, dim, bias=False)
        self.out = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        xa = self.a(x)
        xb = self.b(x)
        y = torch.cat([xa, xb], dim=-1)
        y1, y2 = torch.chunk(y, 2, dim=-1)
        return self.out(y1 + y2)


class EmbeddingMLPBlock(torch.nn.Module):
    def __init__(self, vocab_size=32, dim=8, hidden=16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, dim)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, tokens):
        x = self.embed(tokens)
        h = self.norm(x)
        h = torch.nn.functional.relu(self.fc1(h))
        return self.fc2(h)


class TinyTransformerBlock(torch.nn.Module):
    def __init__(self, vocab_size=32, dim=8, hidden=16, num_heads=2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = SDPAAttentionBlock(dim=dim, num_heads=num_heads)
        self.gate = torch.nn.Linear(dim, hidden, bias=False)
        self.up = torch.nn.Linear(dim, hidden, bias=False)
        self.down = torch.nn.Linear(hidden, dim, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)
        h = torch.nn.functional.silu(self.gate(h)) * self.up(h)
        x = x + self.down(h)
        return x


class LinearMulAddModel(torch.nn.Module):
    """Linear -> multiply -> add.  (from test_debug_info, test_inspector, test_location_bindings)"""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        y = self.linear(x)
        z = y * 2
        return z + 1


class LinearReluMulModel(torch.nn.Module):
    """Linear -> relu -> multiply.  (from test_graph)"""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: linear -> relu -> multiply."""
        y = self.linear(x)
        z = torch.relu(y)
        return z * 2


class TwoLinearSigmoidModel(torch.nn.Module):
    """Two linear layers with relu and sigmoid.  (from test_graph, test_location_bindings)"""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 8)
        self.linear2 = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multiple operations."""
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


class ThreeLinearModel(torch.nn.Module):
    """Three-layer network: fc1 -> relu -> fc2 -> relu -> fc3.  (from test_graph_diff)"""

    def __init__(self) -> None:
        """Initialize layers."""
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 15)
        self.fc3 = torch.nn.Linear(15, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class ModifiedActivationModel(torch.nn.Module):
    """Same as ThreeLinearModel but uses tanh instead of second relu.  (from test_graph_diff)"""

    def __init__(self) -> None:
        """Initialize layers."""
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 15)
        self.fc3 = torch.nn.Linear(15, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)  # Changed activation
        x = self.fc3(x)
        return x


class TwoLinearSkipModel(torch.nn.Module):
    """Two-layer network: missing the middle layer from ThreeLinearModel.  (from test_graph_diff)"""

    def __init__(self) -> None:
        """Initialize layers."""
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc3 = torch.nn.Linear(20, 5)  # Skip fc2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = torch.relu(x)
        # No fc2 layer
        x = self.fc3(x)
        return x


class ExtraLayerModel(torch.nn.Module):
    """Four-layer network: adds an extra layer to ThreeLinearModel.  (from test_graph_diff)"""

    def __init__(self) -> None:
        """Initialize layers."""
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 15)
        self.fc3 = torch.nn.Linear(15, 10)
        self.fc4 = torch.nn.Linear(10, 5)  # Extra layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)  # Extra activation
        x = self.fc4(x)
        return x


class ParallelBranchModel(torch.nn.Module):
    r"""
    Model with parallel branches to create multiple ops at same depth level.

    Architecture:
          input
         /  |  \
        fc1 fc2 fc3  (parallel branches - same depth)
         \\  |  /
          add       (merge branches)
           |
          fc4
           |
         output

    (from test_validator)
    """

    def __init__(self) -> None:
        """Initialize the model with parallel branches."""
        super().__init__()
        # Parallel branches
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(4, 8)
        self.fc3 = torch.nn.Linear(4, 8)
        # Final layer after merge
        self.fc4 = torch.nn.Linear(8, 2)
        self.nan_injection_branch = None  # "fc1", "fc2", "fc3", or None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through parallel branches."""
        # Process input through parallel branches (creates multiple ops at same depth)
        branch1 = self.fc1(x)
        if self.nan_injection_branch == "fc1":
            branch1 = branch1 / 0.0  # Inject NaN

        branch2 = self.fc2(x)
        if self.nan_injection_branch == "fc2":
            branch2 = branch2 / 0.0  # Inject NaN

        branch3 = self.fc3(x)
        if self.nan_injection_branch == "fc3":
            branch3 = branch3 / 0.0  # Inject NaN

        # Merge branches
        merged = branch1 + branch2 + branch3

        # Final layer
        output = self.fc4(merged)

        return output


class SimpleSequentialModel(torch.nn.Module):
    """Simple model with cascading ReLU, mul, and add operations.  (from test_comparator)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with simple cascading operations."""
        # x -> relu -> mul -> add -> relu -> mul -> add
        x1 = torch.nn.functional.relu(x)  # First ReLU
        x2 = x1 * 2.0  # First mul
        x3 = x2 + 1.0  # First add
        x4 = torch.nn.functional.relu(x3)  # Second ReLU
        x5 = x4 * 3.0  # Second mul
        x6 = x5 + 2.0  # Second add
        return x6


class SimpleLinearModel(torch.nn.Module):
    """A simple single-linear-layer model.  (from test_torch_utils)"""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.linear(x)


class TwoLayerMLPModel(torch.nn.Module):
    """A two-layer MLP model.  (from test_torch_utils)"""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 3)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


EXAMPLE_INPUTS = {
    HierarchicalModel: lambda: OrderedDict(
        x=torch.randn(1, 2, 4),
    ),
    ElemwiseBlock: lambda: OrderedDict(
        x=torch.randn(1, 2, 8),
        y=torch.randn(1, 2, 8),
    ),
    LayerNormBlock: lambda: OrderedDict(
        x=torch.randn(1, 2, 8),
    ),
    NormMLPBlock: lambda: OrderedDict(
        x=torch.randn(1, 2, 8),
    ),
    GatedResidualBlock: lambda: OrderedDict(
        x=torch.randn(1, 2, 8),
    ),
    BatchedMatmulBlock: lambda: OrderedDict(
        x=torch.randn(1, 2, 8),
    ),
    TinySelfAttention: lambda: OrderedDict(
        x=torch.randn(1, 3, 8),
    ),
    SDPAAttentionBlock: lambda: OrderedDict(
        x=torch.randn(1, 3, 8),
    ),
    ConcatSplitBlock: lambda: OrderedDict(
        x=torch.randn(1, 2, 8),
    ),
    EmbeddingMLPBlock: lambda: OrderedDict(
        tokens=torch.randint(0, 32, (1, 3), dtype=torch.int32),
    ),
    TinyTransformerBlock: lambda: OrderedDict(
        tokens=torch.randint(0, 32, (1, 3), dtype=torch.int32),
    ),
    LinearMulAddModel: lambda: OrderedDict(
        x=torch.randn(2, 4),
    ),
    LinearReluMulModel: lambda: OrderedDict(
        x=torch.randn(2, 4),
    ),
    TwoLinearSigmoidModel: lambda: OrderedDict(
        x=torch.randn(2, 4),
    ),
    ThreeLinearModel: lambda: OrderedDict(
        x=torch.randn(2, 10),
    ),
    ModifiedActivationModel: lambda: OrderedDict(
        x=torch.randn(2, 10),
    ),
    TwoLinearSkipModel: lambda: OrderedDict(
        x=torch.randn(2, 10),
    ),
    ExtraLayerModel: lambda: OrderedDict(
        x=torch.randn(2, 10),
    ),
    ParallelBranchModel: lambda: OrderedDict(
        x=torch.randn(1, 4),
    ),
    SimpleSequentialModel: lambda: OrderedDict(
        x=torch.randn(2, 4),
    ),
    SimpleLinearModel: lambda: OrderedDict(
        x=torch.randn(1, 4),
    ),
    TwoLayerMLPModel: lambda: OrderedDict(
        x=torch.randn(1, 4),
    ),
}


def get_example_inputs(
    model_cls: type[torch.nn.Module],
) -> OrderedDict[str, torch.Tensor]:
    if model_cls not in EXAMPLE_INPUTS:
        raise KeyError(
            f"No example inputs registered for model type: {model_cls.__name__}"
        )

    return EXAMPLE_INPUTS[model_cls]()
