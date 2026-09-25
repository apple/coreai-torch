# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test for rope composite op."""

import platform
import re

import numpy as np
import pytest
import torch
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3RotaryEmbedding,
    Gemma3TextConfig,
)
from transformers.models.gemma3.modeling_gemma3 import (
    apply_rotary_pos_emb as gemma3_apply_rotary_pos_emb,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextConfig,
    Llama4TextRotaryEmbedding,
)
from transformers.models.llama4.modeling_llama4 import (
    apply_rotary_emb as llama4_apply_rotary_pos_emb,
)
from transformers.models.mistral.modeling_mistral import (
    MistralConfig,
    MistralRotaryEmbedding,
)
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb as mistral_apply_rotary_pos_emb,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralConfig,
    MixtralRotaryEmbedding,
)
from transformers.models.mixtral.modeling_mixtral import (
    apply_rotary_pos_emb as mixtral_apply_rotary_pos_emb,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2RotaryEmbedding,
)
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb as qwen2_apply_rotary_pos_emb,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3RotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeRotaryEmbedding,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    apply_rotary_pos_emb as qwen3moe_apply_rotary_pos_emb,
)
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextConfig,
    Qwen3NextRotaryEmbedding,
)
from transformers.models.qwen3_next.modeling_qwen3_next import (
    apply_rotary_pos_emb as qwen3_next_apply_rotary_pos_emb,
)
from typing_extensions import Self

if platform.system() == "Darwin":
    # disable mypy checking so it would not complain on linux
    # also need to calm mypy on MacOS when it sees no error
    import mlx  # type: ignore[import-not-found, unused-ignore]
    import mlx.core  # type: ignore[import-not-found, unused-ignore]
    import mlx.core.fast  # type: ignore[import-not-found, unused-ignore]

from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table
from coreai_torch.composite_ops import RoPE
from coreai_torch.composite_ops._rope import _compute_angle, rope

from ..utils import (
    _mlx_array_to_numpy_array,
    _torch_tensor_to_numpy_array,
    filecheck_pattern,
    validate_numerical_output,
)


class TestTorchRoPE:
    """Test that torch implementation of rope composite op can be exported and numerics match mlx."""

    @pytest.mark.parametrize("scale", [1, 2])
    @pytest.mark.parametrize("base", [10000, 10])
    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rope_basic(  # noqa: PLR0913
        scale: float,
        base: float,
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test construction of cos and sin inside rope composite op."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        input = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)

        model = RoPE(scale=scale, base=base, dims=dims, interleaved=interleaved)
        model.eval()

        if dynamic:
            batch_size_dim = torch.export.Dim(name="batch_size", min=1, max=32)
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {"input": {0: batch_size_dim, 2: q_len_dim}}
        else:
            dynamic_shapes = None
        exported_program = torch.export.export(
            model,
            args=(input,),
            dynamic_shapes=dynamic_shapes,
        )

        output_torch_eager = model(input)
        output_torch_export = exported_program.module()(input)
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _torch_tensor_to_numpy_array(output_torch_export),
            atol=1e-4,
            rtol=1e-4,
        )

        input_mlx = mlx.core.array(input)
        output_mlx = mlx.core.fast.rope(
            input_mlx,
            dims=dims if dims is not None else head_dim,
            traditional=interleaved,
            base=base,
            scale=scale,
            offset=0,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _mlx_array_to_numpy_array(output_mlx),
            rtol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
        )

    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rope_with_cos_and_sin(
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test providing pre-computed cos and sin to rope composite op."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        input = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)
        is_partial_rotation = dims is not None and dims < head_dim
        rotation_dims = dims if is_partial_rotation else head_dim
        half_dim = rotation_dims // 2
        base = 1e4
        scale = 1.0
        offset = 0
        positions = (offset + torch.arange(q_len)) * scale
        freqs = 1.0 / torch.pow(base, torch.arange(half_dim) / half_dim)
        # (q_len, half_dim) = (q_len, 1) * (1, half_dim)
        theta = positions.unsqueeze(1) * freqs.unsqueeze(0)
        cos = torch.cos(theta).to(dtype)
        sin = torch.sin(theta).to(dtype)

        model = RoPE(dims=dims, interleaved=interleaved)
        model.eval()

        dynamic_shapes = None
        if dynamic:
            batch_size_dim = torch.export.Dim(name="batch_size", min=1, max=32)
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "input": {0: batch_size_dim, 2: q_len_dim},
                "cos": {0: q_len_dim},
                "sin": {0: q_len_dim},
            }
        exported_program = torch.export.export(
            model,
            args=(input,),
            kwargs={"cos": cos, "sin": sin},
            dynamic_shapes=dynamic_shapes,
        )

        output_torch_eager = model(input, cos=cos, sin=sin)
        output_torch_export = exported_program.module()(input, cos=cos, sin=sin)
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _torch_tensor_to_numpy_array(output_torch_export),
            atol=1e-4,
            rtol=1e-4,
        )

        input_mlx = mlx.core.array(input)
        output_mlx = mlx.core.fast.rope(
            input_mlx,
            dims=rotation_dims,
            traditional=interleaved,
            base=base,
            scale=scale,
            offset=offset,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _mlx_array_to_numpy_array(output_mlx),
            rtol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
        )

    @pytest.mark.parametrize("scale", [1, 2])
    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rope_with_freqs(
        scale: float,
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """
        Test providing pre-computed frequencies to rope composite op.

        This is used in Llama3 RoPE and Yarn RoPE etc.
        """
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        input = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)
        is_partial_rotation = dims is not None and dims < head_dim
        rotation_dims = dims if is_partial_rotation else head_dim
        half_dim = rotation_dims // 2
        base = 1e4
        scale = 1.0
        offset = 0
        period = torch.pow(base, torch.arange(half_dim) / half_dim)
        freqs = 1.0 / period

        model = RoPE(scale=scale, dims=dims, interleaved=interleaved)
        model.eval()

        dynamic_shapes = None
        if dynamic:
            batch_size_dim = torch.export.Dim(name="batch_size", min=1, max=32)
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "input": {0: batch_size_dim, 2: q_len_dim},
                "freqs": {},
            }
        exported_program = torch.export.export(
            model,
            args=(input,),
            kwargs={"freqs": freqs},
            dynamic_shapes=dynamic_shapes,
        )

        output_torch_eager = model(input, freqs=freqs)
        output_torch_export = exported_program.module()(input, freqs=freqs)
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _torch_tensor_to_numpy_array(output_torch_export),
            atol=1e-4,
            rtol=1e-4,
        )

        input_mlx = mlx.core.array(input)
        period_mlx = mlx.core.array(period)
        output_mlx = mlx.core.fast.rope(
            input_mlx,
            dims=rotation_dims,
            traditional=interleaved,
            base=None,
            scale=scale,
            offset=offset,
            # MLX RoPE frequency is actually period
            freqs=period_mlx,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _mlx_array_to_numpy_array(output_mlx),
            rtol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
        )

    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rope_with_offset(
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test providing offset to rope composite op, used when offset as kv-cache LLM input."""
        batch_size = 2
        context_size = 1024
        num_heads = 3
        q_len = 5
        head_dim = 14
        input = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)
        offset = torch.randint(low=0, high=context_size, size=(batch_size,))
        base = 1e4
        scale = 1.0

        model = RoPE(dims=dims, interleaved=interleaved)
        model.eval()

        input = torch.tensor(input)
        offset = torch.tensor(offset)
        dynamic_shapes = None
        if dynamic:
            batch_size_dim = torch.export.Dim(name="batch_size", min=1, max=32)
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "input": {0: batch_size_dim, 2: q_len_dim},
                "offset": {0: batch_size_dim},
            }
        exported_program = torch.export.export(
            model,
            args=(input,),
            kwargs={"offset": offset},
            dynamic_shapes=dynamic_shapes,
        )

        output_torch_eager = model(input, offset=offset)
        output_torch_export = exported_program.module()(input, offset=offset)
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _torch_tensor_to_numpy_array(output_torch_export),
            atol=1e-4,
            rtol=1e-4,
        )

        input_mlx = mlx.core.array(input)
        offset_mlx = mlx.core.array(offset)
        output_mlx = mlx.core.fast.rope(
            input_mlx,
            dims=dims if dims is not None else head_dim,
            traditional=interleaved,
            base=base,
            scale=scale,
            offset=offset_mlx,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _mlx_array_to_numpy_array(output_mlx),
            rtol={torch.float32: 5e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 5e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
        )

    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rope_with_position_ids(
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test providing position_ids to rope composite op, used when position_ids as kv-cache LLM input."""
        batch_size = 2
        context_size = 1024
        num_heads = 3
        q_len = 5
        head_dim = 14
        input = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)
        offset = torch.randint(low=0, high=context_size, size=(batch_size,))
        int_scale = 1
        position_ids = (
            offset.unsqueeze(-1) + torch.arange(q_len).unsqueeze(0)
        ) * int_scale
        scale = float(int_scale)
        base = 1e4

        model = RoPE(dims=dims, interleaved=interleaved)
        model.eval()

        dynamic_shapes = None
        if dynamic:
            batch_size_dim = torch.export.Dim(name="batch_size", min=1, max=32)
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "input": {0: batch_size_dim, 2: q_len_dim},
                "position_ids": {0: batch_size_dim, 1: q_len_dim},
            }
        exported_program = torch.export.export(
            model,
            args=(input,),
            kwargs={"position_ids": position_ids},
            dynamic_shapes=dynamic_shapes,
        )

        output_torch_eager = model(input, position_ids=position_ids)
        output_torch_export = exported_program.module()(
            input,
            position_ids=position_ids,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _torch_tensor_to_numpy_array(output_torch_export),
            atol=1e-4,
            rtol=1e-4,
        )

        input_mlx = mlx.core.array(input)
        offset_mlx = mlx.core.array(offset)
        output_mlx = mlx.core.fast.rope(
            input_mlx,
            dims=dims if dims is not None else head_dim,
            traditional=interleaved,
            base=base,
            scale=scale,
            offset=offset_mlx,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _mlx_array_to_numpy_array(output_mlx),
            rtol={torch.float32: 5e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 5e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
        )


class TestTorchRoPEDetails:
    """Test torch implementation details that are vital for numerics."""

    @staticmethod
    def test_compute_angle_numerical() -> None:
        """
        Test the numerical precision of the _compute_angle function.

        This test verifies that the _compute_angle function produces exactly the same
        values as the current approach, ensuring numerical consistency.
        """
        # Test parameters
        batch_size = 2
        num_heads = 3
        q_len = 5234
        head_dim = 14
        half_dim = head_dim // 2
        scale = 2.0
        base = 1e4

        # Create test inputs
        rng = np.random.default_rng(seed=42)  # Use fixed seed for reproducibility
        input_data = rng.random((batch_size, num_heads, q_len, head_dim))
        input_torch = torch.tensor(input_data)

        # Create position_ids
        position_ids = torch.arange(q_len, dtype=torch.float32).expand(batch_size, -1)

        # Create freqs
        exponent = torch.arange(half_dim, dtype=torch.float32) / half_dim
        inv_freq = torch.pow(base, -exponent)

        # Call the _compute_angle function
        angle = _compute_angle(
            inv_freq,
            scale,
            input_torch,
            position_ids,
            use_hf_impl=True,
        )

        # Compute the expected result manually using the same logic
        # First, scale the frequencies and cast to input dtype
        scaled_freqs = (inv_freq * scale).to(input_torch.dtype)

        # Expand position_ids and multiply with freqs
        position_ids_expanded = position_ids.unsqueeze(-1)
        expected_angle = position_ids_expanded * scaled_freqs.float()

        # Verify the results match exactly
        assert torch.allclose(angle, expected_angle, rtol=0, atol=0), (
            "The _compute_angle function should produce exactly the same values as the current approach"
        )

        # Test with different dtypes to ensure precision handling is correct
        input_fp16 = input_torch.to(torch.float16)
        angle_fp16 = _compute_angle(
            inv_freq,
            scale,
            input_fp16,
            position_ids,
            use_hf_impl=True,
        )

        # Compute expected result for fp16
        scaled_freqs_fp16 = (inv_freq * scale).to(torch.float16)
        expected_angle_fp16 = position_ids_expanded * scaled_freqs_fp16.float()

        # Verify the results match exactly for fp16
        assert torch.allclose(angle_fp16, expected_angle_fp16, rtol=0, atol=0), (
            "The _compute_angle function should handle fp16 precision correctly"
        )

    @staticmethod
    def test_rope_scale_application() -> None:
        """
        Test that scale is correctly applied whether position_ids is provided or not.

        This test verifies the fix for a bug where scale was only applied when position_ids
        was None. The fix ensures scale is applied in both cases by moving the scale
        multiplication logic to line 98 in _construct_cos_and_sin.
        """
        # Test parameters
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        scale = 3.14159  # Use non-default scale to test the fix
        base = 1e4
        interleaved = False
        offset = 0

        # Create random input
        rng = np.random.default_rng(seed=42)  # Use fixed seed for reproducibility
        input_data = rng.random((batch_size, num_heads, q_len, head_dim))
        input_torch = torch.tensor(input_data)

        # Case 1: No position_ids provided (internally computed with scale)
        class ModelWithoutPositionIds(torch.nn.Module):
            def forward(self: Self, input: torch.Tensor) -> torch.Tensor:
                return rope(
                    input,
                    offset=offset,
                    scale=scale,
                    base=base,
                    interleaved=interleaved,
                )

        # Case 2: Explicitly provide position_ids with scale pre-applied
        class ModelWithPositionIds(torch.nn.Module):
            def forward(self: Self, input: torch.Tensor) -> torch.Tensor:
                # Create position_ids with scale already applied
                # This mimics what would happen internally if scale is applied correctly
                position_ids = torch.arange(q_len).expand(batch_size, -1)
                return rope(
                    input,
                    position_ids=position_ids,
                    scale=scale,  # Scale should be applied to position_ids internally
                    base=base,
                    interleaved=interleaved,
                )

        # Initialize models
        model1 = ModelWithoutPositionIds()
        model2 = ModelWithPositionIds()
        model1.eval()
        model2.eval()

        # Get outputs
        output1 = model1(input_torch)
        output2 = model2(input_torch)

        # If the fix is working correctly, both approaches should produce identical outputs
        # because scale is applied in both cases
        np.testing.assert_allclose(
            output1.detach().numpy(),
            output2.detach().numpy(),
            atol=1e-4,
            rtol=1e-4,
            err_msg="Scale is not applied consistently between implicit and explicit position_ids",
        )


class TestTorchRoPEHuggingFace:
    """Test that torch implementation of rope composite op numerics match transformers.

    Uses transformers' rotary embedding implementations as independent reference,
    complementing the MLX-based checks in TestTorchRoPE.
    """

    @staticmethod
    def _hf_reference_standard(
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        config_class: type,
        rotary_embedding_class: type,
        apply_rotary_fn: object,
        head_dim: int,
        hidden_size: int,
        n_heads: int,
        theta: float,
        head_is_different: bool,
        precision: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run standard HF RoPE (Mistral/Mixtral/Qwen style) and return expected outputs."""
        resolved_head_dim = head_dim if head_is_different else hidden_size // n_heads
        config = config_class(
            hidden_size=hidden_size,
            num_attention_heads=n_heads,
            head_dim=resolved_head_dim,
            rope_theta=theta,
            max_position_embeddings=123456,
        )
        x = torch.rand(query.shape[0], query.shape[2], 123, dtype=precision)
        hf_rotary = rotary_embedding_class(config).to(precision)
        hf_cos, hf_sin = hf_rotary(x, position_ids)
        return apply_rotary_fn(query, key, hf_cos, hf_sin)

    @staticmethod
    def _hf_reference_gemma3(
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        head_dim: int,
        hidden_size: int,
        num_attention_heads: int,
        theta: float,
        factor: float | None,
        head_is_different: bool,
        precision: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Gemma3 HF RoPE and return expected outputs."""
        resolved_head_dim = (
            head_dim if head_is_different else hidden_size // num_attention_heads
        )
        config = Gemma3TextConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=resolved_head_dim,
            rope_theta=theta,
            max_position_embeddings=123456,
        )
        if factor is not None:
            config.rope_scaling = {"rope_type": "linear", "factor": factor}
        x = torch.rand(query.shape[0], query.shape[2], 123, dtype=precision)
        hf_rotary = Gemma3RotaryEmbedding(config).to(precision)
        hf_cos, hf_sin = hf_rotary(x, position_ids)
        return gemma3_apply_rotary_pos_emb(query, key, hf_cos, hf_sin)

    @staticmethod
    def _hf_reference_llama4(
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        theta: float,
        precision: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Llama4 HF RoPE and return expected outputs."""
        config = Llama4TextConfig(
            hidden_size=128,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
            rope_theta=theta,
        )
        x = torch.rand(query.shape[0], query.shape[2], 123, dtype=precision)
        hf_rotary = Llama4TextRotaryEmbedding(config).to(precision)
        pos_embedding = hf_rotary(x, position_ids)
        # HF Llama4 expects [batch, seq, n_heads, head_dim]
        query_hf, key_hf = llama4_apply_rotary_pos_emb(
            query.transpose(2, 1), key.transpose(2, 1), pos_embedding
        )
        return query_hf.transpose(2, 1), key_hf.transpose(2, 1)

    @staticmethod
    def _hf_reference_qwen3_next(
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        head_dim: int,
        hidden_size: int,
        n_heads: int,
        theta: float,
        partial_rotary_factor: float,
        head_is_different: bool,
        precision: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen3Next HF RoPE and return expected outputs."""
        resolved_head_dim = head_dim if head_is_different else hidden_size // n_heads
        config = Qwen3NextConfig(
            hidden_size=hidden_size,
            num_attention_heads=n_heads,
            head_dim=resolved_head_dim,
            rope_theta=theta,
            partial_rotary_factor=partial_rotary_factor,
            max_position_embeddings=123456,
        )
        x = torch.rand(query.shape[0], query.shape[2], 123, dtype=precision)
        hf_rotary = Qwen3NextRotaryEmbedding(config).to(precision)
        hf_cos, hf_sin = hf_rotary(x, position_ids)
        return qwen3_next_apply_rotary_pos_emb(query, key, hf_cos, hf_sin)

    @pytest.mark.parametrize("head_is_different", [True, False])
    @pytest.mark.parametrize(
        "precision", [torch.float32, torch.float16, torch.bfloat16]
    )
    @staticmethod
    def test_standard_mistral(
        head_is_different: bool,
        precision: torch.dtype,
    ) -> None:
        """Validate standard RoPE numerics match HF Mistral implementation."""
        # Important to set batch to 2 not 1 — this caught a bug in the ANE rope.
        batch = 2
        seq_len = 10
        head_dim = 8
        hidden_size = 16
        n_heads = 2
        n_kv_heads = 4
        theta = 314159

        resolved_head_dim = head_dim if head_is_different else hidden_size // n_heads
        offset = 3
        position_ids = offset + torch.arange(seq_len, dtype=torch.int32).unsqueeze(
            0
        ).expand(batch, -1)
        query = torch.rand(batch, n_heads, seq_len, resolved_head_dim, dtype=precision)
        key = torch.rand(batch, n_kv_heads, seq_len, resolved_head_dim, dtype=precision)

        rope_model = RoPE(base=theta, _use_hf_impl=True).to(precision)
        query_hf, key_hf = TestTorchRoPEHuggingFace._hf_reference_standard(
            query,
            key,
            position_ids,
            config_class=MistralConfig,
            rotary_embedding_class=MistralRotaryEmbedding,
            apply_rotary_fn=mistral_apply_rotary_pos_emb,
            head_dim=head_dim,
            hidden_size=hidden_size,
            n_heads=n_heads,
            theta=theta,
            head_is_different=head_is_different,
            precision=precision,
        )

        for x, x_hf in zip((query, key), (query_hf, key_hf)):
            x_out = rope_model(x, position_ids=position_ids)
            np.testing.assert_allclose(
                _torch_tensor_to_numpy_array(x_out),
                _torch_tensor_to_numpy_array(x_hf),
                rtol=0,
                atol={torch.float32: 1e-3, torch.float16: 1e-3, torch.bfloat16: 1e-3}[
                    precision
                ],
            )

    @pytest.mark.slow  # mark "slow" to skip in CI
    @pytest.mark.parametrize(
        "config_class,rotary_embedding_class,apply_rotary_fn",
        [
            (MixtralConfig, MixtralRotaryEmbedding, mixtral_apply_rotary_pos_emb),
            (Qwen2Config, Qwen2RotaryEmbedding, qwen2_apply_rotary_pos_emb),
            (Qwen3Config, Qwen3RotaryEmbedding, qwen3_apply_rotary_pos_emb),
            (Qwen3MoeConfig, Qwen3MoeRotaryEmbedding, qwen3moe_apply_rotary_pos_emb),
        ],
    )
    @pytest.mark.parametrize("head_is_different", [True, False])
    @pytest.mark.parametrize(
        "precision", [torch.float32, torch.float16, torch.bfloat16]
    )
    @staticmethod
    def test_standard_slow(  # noqa: PLR0913
        config_class: type,
        rotary_embedding_class: type,
        apply_rotary_fn: object,
        head_is_different: bool,
        precision: torch.dtype,
    ) -> None:
        """Validate standard RoPE numerics match HF Mixtral/Qwen2/Qwen3/Qwen3Moe.

        Since they are the same to Mistral, no need to run them in CI.
        """
        batch = 2
        seq_len = 10
        head_dim = 8
        hidden_size = 16
        n_heads = 2
        n_kv_heads = 4
        theta = 314159

        resolved_head_dim = head_dim if head_is_different else hidden_size // n_heads
        offset = 3
        position_ids = offset + torch.arange(seq_len, dtype=torch.int32).unsqueeze(
            0
        ).expand(batch, -1)
        query = torch.rand(batch, n_heads, seq_len, resolved_head_dim, dtype=precision)
        key = torch.rand(batch, n_kv_heads, seq_len, resolved_head_dim, dtype=precision)

        rope_model = RoPE(base=theta, _use_hf_impl=True).to(precision)
        query_hf, key_hf = TestTorchRoPEHuggingFace._hf_reference_standard(
            query,
            key,
            position_ids,
            config_class=config_class,
            rotary_embedding_class=rotary_embedding_class,
            apply_rotary_fn=apply_rotary_fn,
            head_dim=head_dim,
            hidden_size=hidden_size,
            n_heads=n_heads,
            theta=theta,
            head_is_different=head_is_different,
            precision=precision,
        )

        for x, x_hf in zip((query, key), (query_hf, key_hf)):
            x_out = rope_model(x, position_ids=position_ids)
            np.testing.assert_allclose(
                _torch_tensor_to_numpy_array(x_out),
                _torch_tensor_to_numpy_array(x_hf),
                rtol=0,
                atol={torch.float32: 1e-3, torch.float16: 1e-3, torch.bfloat16: 1e-3}[
                    precision
                ],
            )

    @pytest.mark.parametrize(
        "head_is_different,factor",
        [(False, None), (True, 22.3)],
    )
    @pytest.mark.parametrize(
        "precision", [torch.float32, torch.float16, torch.bfloat16]
    )
    @staticmethod
    def test_gemma3(
        head_is_different: bool,
        factor: float | None,
        precision: torch.dtype,
    ) -> None:
        """Validate Gemma3 RoPE with optional linear scaling numerics match HF."""
        batch = 1
        seq_len = 10
        head_dim = 8
        hidden_size = 16
        num_attention_heads = 2
        n_kv_heads = 4
        theta = 31415

        resolved_head_dim = (
            head_dim if head_is_different else hidden_size // num_attention_heads
        )
        offset = 3
        position_ids = offset + torch.arange(seq_len, dtype=torch.int32).unsqueeze(
            0
        ).expand(batch, -1)
        query = torch.rand(
            batch, num_attention_heads, seq_len, resolved_head_dim, dtype=precision
        )
        key = torch.rand(batch, n_kv_heads, seq_len, resolved_head_dim, dtype=precision)

        scale = 1.0 if factor is None else float(1 / factor)
        rope_model = RoPE(base=theta, scale=scale, _use_hf_impl=True).to(precision)
        query_hf, key_hf = TestTorchRoPEHuggingFace._hf_reference_gemma3(
            query,
            key,
            position_ids,
            head_dim=head_dim,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            theta=theta,
            factor=factor,
            head_is_different=head_is_different,
            precision=precision,
        )

        for x, x_hf in zip((query, key), (query_hf, key_hf)):
            x_out = rope_model(x, position_ids=position_ids)
            np.testing.assert_allclose(
                _torch_tensor_to_numpy_array(x_out),
                _torch_tensor_to_numpy_array(x_hf),
                rtol=0,
                atol={torch.float32: 1e-3, torch.float16: 1e-3, torch.bfloat16: 1e-3}[
                    precision
                ],
            )

    @pytest.mark.parametrize(
        "precision", [torch.float32, torch.float16, torch.bfloat16]
    )
    @staticmethod
    def test_llama4(precision: torch.dtype) -> None:
        """Validate Llama4 interleaved RoPE numerics match HF."""
        batch = 1
        seq_len = 10
        n_heads = 2
        n_kv_heads = 4
        head_dim = 8
        theta = 31415

        offset = 3
        position_ids = offset + torch.arange(seq_len, dtype=torch.int32).unsqueeze(
            0
        ).expand(batch, -1)
        query = torch.rand(batch, n_heads, seq_len, head_dim, dtype=precision)
        key = torch.rand(batch, n_kv_heads, seq_len, head_dim, dtype=precision)

        rope_model = RoPE(base=theta, interleaved=True, _use_hf_impl=True).to(precision)
        query_hf, key_hf = TestTorchRoPEHuggingFace._hf_reference_llama4(
            query,
            key,
            position_ids,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            theta=theta,
            precision=precision,
        )

        for x, x_hf in zip((query, key), (query_hf, key_hf)):
            x_out = rope_model(x, position_ids=position_ids)
            np.testing.assert_allclose(
                _torch_tensor_to_numpy_array(x_out),
                _torch_tensor_to_numpy_array(x_hf),
                rtol=0,
                atol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 1e-2}[
                    precision
                ],
            )

    @pytest.mark.parametrize(
        "head_is_different,partial_rotary_factor",
        [(False, 0.5), (True, 1.0)],
    )
    @pytest.mark.parametrize(
        "precision", [torch.float32, torch.float16, torch.bfloat16]
    )
    @staticmethod
    def test_qwen3_next(
        head_is_different: bool,
        partial_rotary_factor: float,
        precision: torch.dtype,
    ) -> None:
        """Validate Qwen3Next RoPE with partial rotary factor numerics match HF."""
        batch = 1
        seq_len = 10
        head_dim = 8
        hidden_size = 16
        n_heads = 2
        n_kv_heads = 4
        theta = 3.14

        resolved_head_dim = head_dim if head_is_different else hidden_size // n_heads
        rope_dims = int(resolved_head_dim * partial_rotary_factor)
        offset = 3
        position_ids = offset + torch.arange(seq_len, dtype=torch.int32).unsqueeze(
            0
        ).expand(batch, -1)
        query = torch.rand(batch, n_heads, seq_len, resolved_head_dim, dtype=precision)
        key = torch.rand(batch, n_kv_heads, seq_len, resolved_head_dim, dtype=precision)

        rope_model = RoPE(dims=rope_dims, base=theta, _use_hf_impl=True).to(precision)
        query_hf, key_hf = TestTorchRoPEHuggingFace._hf_reference_qwen3_next(
            query,
            key,
            position_ids,
            head_dim=head_dim,
            hidden_size=hidden_size,
            n_heads=n_heads,
            theta=theta,
            partial_rotary_factor=partial_rotary_factor,
            head_is_different=head_is_different,
            precision=precision,
        )

        for x, x_hf in zip((query, key), (query_hf, key_hf)):
            x_out = rope_model(x, position_ids=position_ids)
            np.testing.assert_allclose(
                _torch_tensor_to_numpy_array(x_out),
                _torch_tensor_to_numpy_array(x_hf),
                rtol=0,
                atol={torch.float32: 1e-3, torch.float16: 1e-3, torch.bfloat16: 1e-3}[
                    precision
                ],
            )


class TestTorchRoPEConversion:
    """Tests for RoPE composite op conversion using ExternalizeSpec.

    Each test corresponds to a variant of the forward call.  All tests are
    parametrised over dynamic shapes, dtypes, dims and interleaved
    to provide comprehensive coverage.
    """

    @staticmethod
    def _make_externalize_spec() -> ExternalizeSpec:
        return ExternalizeSpec(
            target_class=RoPE,
            composite_op_name="rope",
            composite_attrs=["scale", "base", "dims", "interleaved"],
        )

    @staticmethod
    def _op_attrs(scale: object, base: object, dims: object, interleaved: bool) -> str:
        """Build the expected op_attributes string for a RoPE composite_decl."""
        return (
            f"{{'scale': {scale!r}, 'base': {base!r}, 'dims': {dims!r}, "
            f"'interleaved': {str(interleaved).lower()}, 'version': 1}}"
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("scale", [1.0, 2.0])
    @pytest.mark.parametrize("base", [10000.0, 10.0])
    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rope_basic_ir(  # noqa: PLR0913
        scale: float,
        base: float,
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test rope externalization IR with input only (no optional tensors)."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        x = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE(
                    scale=scale, base=base, dims=dims, interleaved=interleaved
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.rope(x)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            export_dynamic_shapes = {"x": {0: batch_dim, 2: q_dim}}

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x,), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        b = "?" if dynamic else batch_size
        q = "?" if dynamic else q_len
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @rope_[[S:.*]](%arg0: tensor<{b}x{num_heads}x{q}x{head_dim}x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rope" =
        // CHECK-SAME: input_names = ["input"]
        // CHECK-SAME: base = {base:e} : f32
        {f"// CHECK-SAME: dims = {dims} : si64" if dims is not None else ""}
        // CHECK-SAME: interleaved = {str(interleaved).lower()}
        // CHECK-SAME: scale = {scale:e} : f32
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @rope_[[S]](
        """
        filecheck_pattern(str(converted_program._module._mlir_module), check_file=truth)

    @pytest.mark.parametrize("scale", [1.0, 2.0])
    @pytest.mark.parametrize("base", [10000.0, 10.0])
    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_rope_basic(  # noqa: PLR0913
        scale: float,
        base: float,
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test rope externalization with input only (no optional tensors)."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        x = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE(
                    scale=scale, base=base, dims=dims, interleaved=interleaved
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.rope(x)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            export_dynamic_shapes = {"x": {0: batch_dim, 2: q_dim}}

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x,), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        output_torch_eager = model(x)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            x=x,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rope_with_cos_and_sin_ir(
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test rope externalization IR with pre-computed cos and sin inputs."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        rotation_dims = dims if (dims is not None and dims < head_dim) else head_dim
        half_dim = rotation_dims // 2
        x = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)
        cos = torch.rand(q_len, half_dim, dtype=dtype)
        sin = torch.rand(q_len, half_dim, dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE(dims=dims, interleaved=interleaved)

            def forward(
                self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
            ) -> torch.Tensor:
                return self.rope(x, cos=cos, sin=sin)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            export_dynamic_shapes = {
                "x": {0: batch_dim, 2: q_dim},
                "cos": {0: q_dim},
                "sin": {0: q_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x, cos, sin), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        b = "?" if dynamic else batch_size
        q = "?" if dynamic else q_len
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @rope_[[S:.*]](%arg0: tensor<{b}x{num_heads}x{q}x{head_dim}x{dt}>
        // CHECK-SAME: %arg1: tensor<{q}x{half_dim}x{dt}>
        // CHECK-SAME: %arg2: tensor<{q}x{half_dim}x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rope" =
        // CHECK-SAME: input_names = ["input", "cos", "sin"]
        // CHECK-SAME: base = 1.000000e+04 : f32
        {f"// CHECK-SAME: dims = {dims} : si64" if dims is not None else ""}
        // CHECK-SAME: interleaved = {str(interleaved).lower()}
        // CHECK-SAME: scale = 1.000000e+00 : f32
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @rope_[[S]](
        """
        filecheck_pattern(str(converted_program._module._mlir_module), check_file=truth)

    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_rope_with_cos_and_sin(
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test rope externalization with pre-computed cos and sin inputs."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        rotation_dims = dims if (dims is not None and dims < head_dim) else head_dim
        half_dim = rotation_dims // 2
        x = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)
        cos = torch.rand(q_len, half_dim, dtype=dtype)
        sin = torch.rand(q_len, half_dim, dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE(dims=dims, interleaved=interleaved)

            def forward(
                self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
            ) -> torch.Tensor:
                return self.rope(x, cos=cos, sin=sin)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            export_dynamic_shapes = {
                "x": {0: batch_dim, 2: q_dim},
                "cos": {0: q_dim},
                "sin": {0: q_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x, cos, sin), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        output_torch_eager = model(x, cos, sin)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            x=x,
            cos=cos,
            sin=sin,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rope_with_freqs_ir(
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test rope externalization IR with pre-computed frequencies (Llama3/YaRN style)."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        rotation_dims = dims if (dims is not None and dims < head_dim) else head_dim
        half_dim = rotation_dims // 2
        base = 1e4
        scale = 1.0
        period = torch.pow(base, torch.arange(half_dim) / half_dim)
        freqs = 1.0 / period
        x = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE(scale=scale, dims=dims, interleaved=interleaved)

            def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
                return self.rope(x, freqs=freqs)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            export_dynamic_shapes = {
                "x": {0: batch_dim, 2: q_dim},
                "freqs": {},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x, freqs), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        b = "?" if dynamic else batch_size
        q = "?" if dynamic else q_len
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @rope_[[S:.*]](%arg0: tensor<{b}x{num_heads}x{q}x{head_dim}x{dt}>
        // CHECK-SAME: %arg1: tensor<{half_dim}xf32>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rope" =
        // CHECK: coreai.invoke @rope_[[S]](
        """
        filecheck_pattern(str(converted_program._module._mlir_module), check_file=truth)

    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_rope_with_freqs(
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test rope externalization with pre-computed frequencies (Llama3/YaRN style)."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        rotation_dims = dims if (dims is not None and dims < head_dim) else head_dim
        half_dim = rotation_dims // 2
        base = 1e4
        scale = 1.0
        period = torch.pow(base, torch.arange(half_dim) / half_dim)
        freqs = 1.0 / period
        x = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE(scale=scale, dims=dims, interleaved=interleaved)

            def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
                return self.rope(x, freqs=freqs)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            export_dynamic_shapes = {
                "x": {0: batch_dim, 2: q_dim},
                "freqs": {},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x, freqs), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        output_torch_eager = model(x, freqs)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            x=x,
            freqs=freqs,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rope_with_offset_ir(
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test rope externalization IR with tensor offset (KV-cache decoding)."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        x = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)
        offset = torch.randint(low=0, high=1024, size=(batch_size,))

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE(dims=dims, interleaved=interleaved)

            def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
                return self.rope(x, offset=offset)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            export_dynamic_shapes = {
                "x": {0: batch_dim, 2: q_dim},
                "offset": {0: batch_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x, offset), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        b = "?" if dynamic else batch_size
        q = "?" if dynamic else q_len
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @rope_[[S:.*]](%arg0: tensor<{b}x{num_heads}x{q}x{head_dim}x{dt}>
        // CHECK-SAME: %arg1: tensor<{b}x
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rope" =
        // CHECK-SAME: input_names = ["input", "offset"]
        // CHECK-SAME: base = 1.000000e+04 : f32
        {f"// CHECK-SAME: dims = {dims} : si64" if dims is not None else ""}
        // CHECK-SAME: interleaved = {str(interleaved).lower()}
        // CHECK-SAME: scale = 1.000000e+00 : f32
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @rope_[[S]](
        """
        filecheck_pattern(str(converted_program._module._mlir_module), check_file=truth)

    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_rope_with_offset(
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test rope externalization with tensor offset (KV-cache decoding)."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        x = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)
        offset = torch.randint(low=0, high=1024, size=(batch_size,))

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE(dims=dims, interleaved=interleaved)

            def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
                return self.rope(x, offset=offset)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            export_dynamic_shapes = {
                "x": {0: batch_dim, 2: q_dim},
                "offset": {0: batch_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x, offset), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        output_torch_eager = model(x, offset)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            x=x,
            offset=offset,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rope_with_position_ids_ir(
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test rope externalization IR with explicit position indices."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        x = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)
        position_ids = torch.arange(q_len).unsqueeze(0).expand(batch_size, -1)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE(dims=dims, interleaved=interleaved)

            def forward(
                self, x: torch.Tensor, position_ids: torch.Tensor
            ) -> torch.Tensor:
                return self.rope(x, position_ids=position_ids)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            export_dynamic_shapes = {
                "x": {0: batch_dim, 2: q_dim},
                "position_ids": {0: batch_dim, 1: q_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x, position_ids), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        b = "?" if dynamic else batch_size
        q = "?" if dynamic else q_len
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @rope_[[S:.*]](%arg0: tensor<{b}x{num_heads}x{q}x{head_dim}x{dt}>
        // CHECK-SAME: %arg1: tensor<{b}x{q}x
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rope" =
        // CHECK-SAME: input_names = ["input", "position_ids"]
        // CHECK-SAME: base = 1.000000e+04 : f32
        {f"// CHECK-SAME: dims = {dims} : si64" if dims is not None else ""}
        // CHECK-SAME: interleaved = {str(interleaved).lower()}
        // CHECK-SAME: scale = 1.000000e+00 : f32
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @rope_[[S]](
        """
        filecheck_pattern(str(converted_program._module._mlir_module), check_file=truth)

    @pytest.mark.parametrize("dims", [None, 2, 8])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_rope_with_position_ids(
        dims: int | None,
        interleaved: bool,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test rope externalization with explicit position indices."""
        batch_size = 2
        num_heads = 3
        q_len = 5
        head_dim = 14
        x = torch.rand((batch_size, num_heads, q_len, head_dim), dtype=dtype)
        position_ids = torch.arange(q_len).unsqueeze(0).expand(batch_size, -1)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE(dims=dims, interleaved=interleaved)

            def forward(
                self, x: torch.Tensor, position_ids: torch.Tensor
            ) -> torch.Tensor:
                return self.rope(x, position_ids=position_ids)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            export_dynamic_shapes = {
                "x": {0: batch_dim, 2: q_dim},
                "position_ids": {0: batch_dim, 1: q_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x, position_ids), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        output_torch_eager = model(x, position_ids)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            x=x,
            position_ids=position_ids,
        )


class TestTorchRoPEConversionDetails:
    """Test Core AI graph details that are vital for numerics."""

    @staticmethod
    def test_rope_compute_precision() -> None:
        """Test the rope is computed in fp32 precision."""
        batch_size = 2
        context_size = 1024
        num_heads = 3
        q_len = 5
        head_dim = 14
        input_tensor = torch.rand(
            (batch_size, num_heads, q_len, head_dim),
            dtype=torch.float16,
        )
        offset = torch.randint(low=0, high=context_size, size=(batch_size,))

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE()

            def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
                return self.rope(x, offset=offset)

        model = WrapperModel().eval()
        args = (input_tensor, offset)

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=args
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        # Verify that cos/sin operations are performed in fp32 and then cast to fp16
        # This confirms the fix for precision handling
        coreai_module_str = str(converted_program._module._mlir_module)

        # Use more flexible patterns that match various IR formats
        cos_pattern = r"coreai\.cos.*?f32"
        sin_pattern = r"coreai\.sin.*?f32"
        cast_pattern = r"coreai\.cast.*?f32.*?f16"

        # Check for fp32 cos/sin operations in the entire IR
        assert re.search(cos_pattern, coreai_module_str, re.DOTALL), (
            "No cos operations on fp32 tensors found in the IR"
        )

        assert re.search(sin_pattern, coreai_module_str, re.DOTALL), (
            "No sin operations on fp32 tensors found in the IR"
        )

        # Check for casting from fp32 to fp16
        assert re.search(cast_pattern, coreai_module_str, re.DOTALL), (
            "No cast operations from fp32 to fp16 found in the IR"
        )

        # Check for the sequence pattern: cos/sin on fp32 followed by cast to fp16
        # This is the critical check that verifies the fix
        sequence_pattern = r"coreai\.(cos|sin).*?f32.*?coreai\.cast.*?f32.*?f16"
        assert re.search(sequence_pattern, coreai_module_str, re.DOTALL), (
            "Could not find the sequence of cos/sin followed by cast in the IR"
        )

    @pytest.mark.ir
    @staticmethod
    def test_rope_derived_symbol_dynamic_shapes() -> None:
        """Test that externalized RoPE handles derived symbolic shapes at the op boundary.

        When two tensors with independent dynamic sequence lengths are
        concatenated before being passed to RoPE, the custom op node receives
        an input whose seq_len dim is a derived expression (``q_len1 + q_len2``).
        This exercises:

        1. Dim name sanitisation — derived expressions like ``"s0 + s1"`` are
           not valid Python identifiers, so ``torch.export.Dim`` would crash
           without sanitisation.
        2. Dynamic shape propagation — the inner exported program must have
           a dynamic seq_len dim, not a static one.
        """
        batch_size = 2
        num_heads = 3
        q_len1 = 3
        q_len2 = 4
        head_dim = 14
        x1 = torch.rand((batch_size, num_heads, q_len1, head_dim), dtype=torch.float32)
        x2 = torch.rand((batch_size, num_heads, q_len2, head_dim), dtype=torch.float32)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rope = RoPE()

            def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
                x = torch.cat([x1, x2], dim=2)
                return self.rope(x)

        model = WrapperModel().eval()

        batch_dim = torch.export.Dim("batch", min=1, max=32)
        q1_dim = torch.export.Dim("q_len1", min=1, max=64)
        q2_dim = torch.export.Dim("q_len2", min=1, max=64)
        export_dynamic_shapes = {
            "x1": {0: batch_dim, 2: q1_dim},
            "x2": {0: batch_dim, 2: q2_dim},
        }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x1, x2), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchRoPEConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        # The inner graph must have dynamic batch and seq_len dims.
        # Without the Dim-name sanitisation fix, this crashes with:
        #   AssertionError: Dim name must be a valid identifier, got s0 + s1
        truth = f"""
        // CHECK: coreai.graph private noinline @rope_[[S:.*]](%arg0: tensor<?x{num_heads}x?x{head_dim}xf32>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rope"
        // CHECK: coreai.invoke @rope_[[S]](
        """
        filecheck_pattern(str(converted_program._module._mlir_module), check_file=truth)
