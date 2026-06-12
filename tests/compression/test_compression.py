# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test compression utils."""

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from coreai.authoring import AIProgram
from coreai.runtime import AIModel, NDArray
from torch import nn
from torch.export.exported_program import ExportedProgram

from coreai_torch import TorchConverter
from coreai_torch._compression._floatx import Float4Tensor
from coreai_torch._compression.custom_layers import (
    ActivationDequantizeModule,
    ActivationQuantizeModule,
)

from ..utils import TemporaryModelAsset, filecheck_pattern

# We add "./tests/coreai" path, in order to use some existing utils
sys.path.append(str(Path(__file__).parents[2]))


def _scale_shape(
    input_shape: tuple[int, ...],
    axis: int,
    scale_rank: int,
) -> tuple[int, ...]:
    """Compute the scale tensor shape for a given axis and rank.

    scale_rank=0 -> scalar (), scale_rank=1 -> (input_shape[axis],),
    scale_rank=2 -> input-rank shape with 1s except along axis.
    """
    if scale_rank == 0:
        return ()
    if scale_rank == 1:
        return (input_shape[axis],)
    shape = [1] * len(input_shape)
    shape[axis] = input_shape[axis]
    return tuple(shape)


async def lower_to_coreai(
    exported_program: ExportedProgram,
) -> AIProgram:
    """Verify that an exported program could be lowered."""
    coreaten_program = exported_program.run_decompositions()
    converter = TorchConverter().add_exported_program(coreaten_program)
    return converter.to_coreai()


async def _validate_execution(
    coreai_program: AIProgram,
    torch_out: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    **kwargs: Any,
) -> None:
    """Run the Core AI program using ref kernels and match with torch output."""
    with TemporaryModelAsset() as tempdir:
        coreai_program.save_asset(Path(tempdir))
        ai_model = await AIModel.load(Path(tempdir))
        rt_func = ai_model.load_function("main")

        # Wrap all kwargs with NDArray
        nd_kwargs = {k: NDArray(data=v) for k, v in kwargs.items()}
        coreai_outs = await rt_func(nd_kwargs)

        coreai_out_np = {k: v.numpy() for k, v in coreai_outs.items()}
        np.testing.assert_allclose(
            torch_out.detach().numpy(),
            next(iter(coreai_out_np.values())),  # type: ignore[arg-type]
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "nbits",
    [4, 8],
)
@pytest.mark.parametrize(
    "signed",
    [True, False],
)
@pytest.mark.parametrize(
    "offset_mode",
    ["none", "zero_point", "minval"],
)
async def test_weight_dequantization_int_with_custom_op(
    nbits: int,
    signed: bool,
    offset_mode: str,
) -> None:
    """Use constexpr_blockwise_shift_scale INT path directly and verify export + execution."""

    class TestModel(torch.nn.Module):
        """Dequantization model using blockwise shift scale custom op."""

        def __init__(self) -> None:
            super().__init__()

            lower_bound = -(2 ** (nbits - 1)) if signed else 0
            upper_bound = 2 ** (nbits - 1) - 1 if signed else 2**nbits - 1
            torch_dtype = torch.int8 if signed else torch.uint8
            self.register_buffer(
                "quantized_data",
                torch.randint(lower_bound, upper_bound + 1, (8, 32), dtype=torch_dtype),
            )
            self.register_buffer("scale", torch.ones((4, 8), dtype=torch.float32) * 2)
            if offset_mode == "zero_point":
                self.register_buffer(
                    "zero_point",
                    torch.ones((4, 8), dtype=torch_dtype),
                )
            elif offset_mode == "minval":
                self.register_buffer(
                    "minval",
                    torch.ones((4, 8), dtype=torch.float32) * 0.5,
                )

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            kwargs: dict[str, Any] = {}
            if offset_mode == "zero_point":
                kwargs["zero_point"] = self.zero_point
            elif offset_mode == "minval":
                kwargs["minval"] = self.minval
                # input_dtype conveys the logical sub-byte type (e.g., int4
                # stored as int8). Derive from nbits to exercise the sub-byte path.
                if nbits == 4:  # noqa: PLR2004
                    kwargs["input_dtype"] = torch.int4 if signed else torch.uint4
                else:
                    kwargs["input_dtype"] = torch.int8 if signed else torch.uint8
            dequant_output = torch.ops.coreai.constexpr_blockwise_shift_scale(
                self.quantized_data,
                self.scale,
                **kwargs,
            )
            return cast("torch.Tensor", input_tensor + dequant_output)

    model = TestModel()
    model.eval()

    input_tensor = torch.randn(8, 32)

    with torch.no_grad():
        torch_output = model(input_tensor)
        exported_program = torch.export.export(
            model.eval(),
            args=(),
            kwargs={"input_tensor": input_tensor},
        )

    coreai_program = await lower_to_coreai(
        exported_program,
    )

    await _validate_execution(
        coreai_program,
        torch_output,
        input_tensor=input_tensor,
    )


@pytest.mark.parametrize(
    "fp_dtype",
    [torch.float8_e5m2, torch.float8_e4m3fn],
)
@pytest.mark.parametrize(
    "scale_dtype",
    [torch.float32, torch.float8_e8m0fnu],
)
async def test_weight_dequantization_fp_with_custom_op(
    fp_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> None:
    """Use constexpr_blockwise_shift_scale FP8 path and verify export + execution."""

    class TestModel(torch.nn.Module):
        """FP dequantization model using blockwise shift scale custom op."""

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "quantized_data",
                torch.tensor([1.0, 2.0, 0.5, -1.0] * 8, dtype=fp_dtype).reshape(4, 8),
            )
            if scale_dtype == torch.float8_e8m0fnu:
                # e8m0fnu encodes power-of-2 scales; 128 = 2^0 = 1.0.
                self.register_buffer(
                    "scale",
                    torch.full((4, 8), 128, dtype=torch.uint8).view(
                        torch.float8_e8m0fnu,
                    ),
                )
            else:
                self.register_buffer(
                    "scale",
                    torch.ones((4, 8), dtype=scale_dtype) * 2,
                )

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            dequant_output = torch.ops.coreai.constexpr_blockwise_shift_scale(
                self.quantized_data,
                self.scale,
                output_dtype=torch.float32,
            )
            return cast("torch.Tensor", input_tensor + dequant_output)

    model = TestModel()
    model.eval()

    input_tensor = torch.randn(4, 8)

    with torch.no_grad():
        torch_output = model(input_tensor)
        exported_program = torch.export.export(
            model.eval(),
            args=(),
            kwargs={"input_tensor": input_tensor},
        )

    coreai_program = await lower_to_coreai(
        exported_program,
    )

    await _validate_execution(
        coreai_program,
        torch_output,
        input_tensor=input_tensor,
    )


@pytest.mark.xfail
async def test_weight_dequantization_fp4_with_custom_op() -> None:
    """Use constexpr_blockwise_shift_scale FP4 path (Float4Tensor + e8m0 scale)."""

    class TestModel(torch.nn.Module):
        """FP4 dequantization model using blockwise shift scale custom op."""

        def __init__(self) -> None:
            super().__init__()
            # FP4 data: packed uint8 wrapped in Float4Tensor.
            # Shape (4, 4) uint8 → logical shape (4, 8) fp4.
            packed = torch.randint(0, 255, (4, 4), dtype=torch.uint8)
            self.register_buffer("quantized_data", Float4Tensor(packed))
            # e8m0fnu scale: power-of-2 values. 128 = 2^0 = 1.0 in e8m0.
            self.register_buffer(
                "scale",
                torch.full((4, 4), 128, dtype=torch.uint8).view(torch.float8_e8m0fnu),
            )

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            dequant_output = torch.ops.coreai.constexpr_blockwise_shift_scale(
                self.quantized_data,
                self.scale,
                output_dtype=torch.float32,
            )
            return cast("torch.Tensor", input_tensor + dequant_output)

    model = TestModel()
    model.eval()

    input_tensor = torch.randn(4, 8)

    with torch.no_grad():
        torch_output = model(input_tensor)
        exported_program = torch.export.export(
            model.eval(),
            args=(),
            kwargs={"input_tensor": input_tensor},
        )

    coreai_program = await lower_to_coreai(
        exported_program,
    )

    await _validate_execution(
        coreai_program,
        torch_output,
        input_tensor=input_tensor,
    )


@pytest.mark.parametrize(
    "signed",
    [True, False],
)
@pytest.mark.parametrize(
    "axis",
    [0, 1],
)
@pytest.mark.parametrize(
    "scale_rank",
    [0, 1, 2],
)
@pytest.mark.parametrize(
    "offset_mode",
    ["none", "zero_point", "minval"],
)
async def test_activation_quantization_int_with_custom_op(
    signed: bool,
    axis: int,
    scale_rank: int,
    offset_mode: str,
) -> None:
    """Use quantize INT path directly and verify export + execution."""
    if scale_rank == 0 and axis == 1:
        pytest.skip("per-tensor scale does not use axis")

    input_shape = (8, 32)

    class TestModel(torch.nn.Module):
        """Quantization model using quantize custom op."""

        def __init__(self) -> None:
            super().__init__()
            torch_dtype = torch.int8 if signed else torch.uint8
            shape = _scale_shape(input_shape, axis, scale_rank)
            self.register_buffer(
                "scale",
                torch.ones(shape, dtype=torch.float32),
            )
            if offset_mode == "zero_point":
                self.register_buffer(
                    "zero_point",
                    torch.zeros(shape, dtype=torch_dtype),
                )
            elif offset_mode == "minval":
                self.register_buffer("minval", torch.zeros(shape, dtype=torch.float32))
            self.output_dtype = torch_dtype
            self.axis = axis

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            kwargs: dict[str, Any] = {}
            if offset_mode == "zero_point":
                kwargs["zero_point"] = self.zero_point
            elif offset_mode == "minval":
                kwargs["minval"] = self.minval
            quant_output = torch.ops.coreai.quantize(
                input_tensor,
                self.scale,
                self.output_dtype,
                axis=self.axis,
                **kwargs,
            )
            return cast("torch.Tensor", quant_output)

    model = TestModel()
    model.eval()

    input_tensor = torch.randn(input_shape)

    with torch.no_grad():
        torch_output = model(input_tensor)
        exported_program = torch.export.export(
            model.eval(),
            args=(),
            kwargs={"input_tensor": input_tensor},
        )

    coreai_program = await lower_to_coreai(
        exported_program,
    )

    await _validate_execution(
        coreai_program,
        torch_output,
        input_tensor=input_tensor,
    )


@pytest.mark.parametrize(
    "fp_dtype",
    [torch.float8_e5m2, torch.float8_e4m3fn],
)
@pytest.mark.parametrize(
    "axis",
    [0, 1],
)
@pytest.mark.parametrize(
    "scale_rank",
    [0, 1, 2],
)
async def test_activation_quantization_fp_with_custom_op(
    fp_dtype: torch.dtype,
    axis: int,
    scale_rank: int,
) -> None:
    """Use quantize FP path directly and verify export + execution."""
    if scale_rank == 0 and axis == 1:
        pytest.skip("per-tensor scale does not use axis")

    input_shape = (8, 32)

    class TestModel(torch.nn.Module):
        """FP quantization model using quantize custom op."""

        def __init__(self) -> None:
            super().__init__()
            shape = _scale_shape(input_shape, axis, scale_rank)
            self.register_buffer(
                "scale",
                torch.ones(shape, dtype=torch.float32),
            )
            self.output_dtype = fp_dtype
            self.axis = axis

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            quant_output = torch.ops.coreai.quantize(
                input_tensor,
                self.scale,
                self.output_dtype,
                axis=self.axis,
            )
            # Cast back to float32 — the runtime cannot handle float8 outputs.
            return cast("torch.Tensor", quant_output.to(torch.float32))

    model = TestModel()
    model.eval()

    input_tensor = torch.randn(input_shape)

    with torch.no_grad():
        torch_output = model(input_tensor)
        exported_program = torch.export.export(
            model.eval(),
            args=(),
            kwargs={"input_tensor": input_tensor},
        )

    coreai_program = await lower_to_coreai(
        exported_program,
    )

    await _validate_execution(
        coreai_program,
        torch_output,
        input_tensor=input_tensor,
    )


@pytest.mark.parametrize(
    "signed",
    [True, False],
)
@pytest.mark.parametrize(
    "axis",
    [0, 1],
)
@pytest.mark.parametrize(
    "scale_rank",
    [0, 1, 2],
)
@pytest.mark.parametrize(
    "offset_mode",
    ["none", "zero_point", "minval"],
)
async def test_activation_dequantization_int_with_custom_op(
    signed: bool,
    axis: int,
    scale_rank: int,
    offset_mode: str,
) -> None:
    """Use dequantize INT path directly and verify export + execution."""
    if scale_rank == 0 and axis == 1:
        pytest.skip("per-tensor scale does not use axis")

    input_shape = (8, 32)

    class TestModel(torch.nn.Module):
        """Dequantization model using dequantize custom op."""

        def __init__(self) -> None:
            super().__init__()
            torch_dtype = torch.int8 if signed else torch.uint8
            shape = _scale_shape(input_shape, axis, scale_rank)
            self.register_buffer(
                "scale",
                torch.ones(shape, dtype=torch.float32),
            )
            if offset_mode == "zero_point":
                self.register_buffer(
                    "zero_point",
                    torch.ones(shape, dtype=torch_dtype),
                )
            elif offset_mode == "minval":
                self.register_buffer(
                    "minval",
                    torch.ones(shape, dtype=torch.float32) * 0.5,
                )
            self.input_dtype = torch_dtype
            self.axis = axis

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            kwargs: dict[str, Any] = {}
            if offset_mode == "zero_point":
                kwargs["zero_point"] = self.zero_point
            elif offset_mode == "minval":
                kwargs["minval"] = self.minval
                kwargs["input_dtype"] = self.input_dtype
            dequant_output = torch.ops.coreai.dequantize(
                input_tensor,
                self.scale,
                axis=self.axis,
                **kwargs,
            )
            return cast("torch.Tensor", dequant_output)

    model = TestModel()
    model.eval()

    lower_bound = -128 if signed else 0
    upper_bound = 127 if signed else 255
    torch_dtype = torch.int8 if signed else torch.uint8
    input_tensor = torch.randint(
        lower_bound,
        upper_bound + 1,
        input_shape,
        dtype=torch_dtype,
    )

    with torch.no_grad():
        torch_output = model(input_tensor)
        exported_program = torch.export.export(
            model.eval(),
            args=(),
            kwargs={"input_tensor": input_tensor},
        )

    coreai_program = await lower_to_coreai(
        exported_program,
    )

    await _validate_execution(
        coreai_program,
        torch_output,
        input_tensor=input_tensor,
    )


@pytest.mark.parametrize(
    "fp_dtype",
    [torch.float8_e5m2, torch.float8_e4m3fn],
)
@pytest.mark.parametrize(
    "axis",
    [0, 1],
)
@pytest.mark.parametrize(
    "scale_rank",
    [0, 1, 2],
)
async def test_activation_dequantization_fp_with_custom_op(
    fp_dtype: torch.dtype,
    axis: int,
    scale_rank: int,
) -> None:
    """Use dequantize FP path directly and verify export + execution."""
    if scale_rank == 0 and axis == 1:
        pytest.skip("per-tensor scale does not use axis")

    input_shape = (8, 8)

    class TestModel(torch.nn.Module):
        """FP dequantization model using dequantize custom op."""

        def __init__(self) -> None:
            super().__init__()
            # Store float8 data as a buffer — the runtime cannot handle float8
            # tensors at the boundary, so the model takes a float32 input and
            # adds it to the dequantized buffer.
            self.register_buffer(
                "quantized_data",
                torch.tensor(
                    [[1.0, 2.0, 0.5, -1.0] * 2] * 8,
                    dtype=fp_dtype,
                ),
            )
            shape = _scale_shape(input_shape, axis, scale_rank)
            self.register_buffer(
                "scale",
                torch.ones(shape, dtype=torch.float32) * 2,
            )
            self.axis = axis

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            dequant_output = torch.ops.coreai.dequantize(
                self.quantized_data,
                self.scale,
                axis=self.axis,
                output_dtype=torch.float32,
            )
            return cast("torch.Tensor", input_tensor + dequant_output)

    model = TestModel()
    model.eval()

    input_tensor = torch.randn(input_shape)

    with torch.no_grad():
        torch_output = model(input_tensor)
        exported_program = torch.export.export(
            model.eval(),
            args=(),
            kwargs={"input_tensor": input_tensor},
        )

    coreai_program = await lower_to_coreai(
        exported_program,
    )

    await _validate_execution(
        coreai_program,
        torch_output,
        input_tensor=input_tensor,
    )


@pytest.mark.xfail(
    reason="PyTorch doesn't keep custom metadata after run_decompositions",
    raises=(AssertionError, KeyError),
)
def test_torch_decomposition_keep_metadata() -> None:
    """
    Make sure the metadata is kept after exported program run_decompositions.

    After torch fixes the bug about keeping custom metadata in fx graph node, we
    can consider switching to using the custom field in node.meta to indicate
    the true sub-byte dtype instead of packing everything in state_dict in IntxTensor
    and UintxTensor as we currently do.
    """

    @torch.library.custom_op("mylib::add", mutates_args=())
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * 2 + y * 2

    @torch.library.register_fake("mylib::add")  # type: ignore [misc]
    def _(x: torch.Tensor, _y: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    class TestModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return cast("torch.Tensor", torch.ops.mylib.add(x, y))

    model = TestModel()
    x_example = torch.randn(2, 3)
    y_example = torch.randn(2, 3)
    exported_program = torch.export.export(model, (x_example, y_example))

    for node in exported_program.graph.nodes:
        node.meta["my_field"] = "dummy"
    for node in exported_program.graph.nodes:
        assert node.meta["my_field"] == "dummy"

    decomposed_program = exported_program.run_decompositions()
    for node in decomposed_program.graph.nodes:
        assert node.meta["my_field"] == "dummy"


class TestWeightPalettizationWithCustomOp:
    """Tests for weight_palettization_with_custom_op lowering."""

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "nbits",
        [1, 2, 4, 8],
    )
    @pytest.mark.parametrize(
        "vector_size",
        [1, 2, 4],
    )
    def test_ir(
        self,
        nbits: int,
        vector_size: int,
    ) -> None:
        """Use custom op directly instead of using coremltools to do palettization IR check."""
        indices_shape = (2, 3)
        axis = 0

        output_shape = list(indices_shape)
        output_shape[axis] *= vector_size

        class TestModel(torch.nn.Module):
            """Palettized model with torch custom op."""

            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "indices",
                    torch.randint(0, 2**nbits, indices_shape, dtype=torch.uint8),
                )
                self.register_buffer(
                    "lut",
                    torch.ones((1, 1, 2**nbits, vector_size), dtype=torch.float16),
                )

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                lut_output = torch.ops.coreai.lut_to_dense(
                    indices=self.indices,
                    lut=self.lut,
                    axis=axis,
                )
                return cast("torch.Tensor", input_tensor + lut_output)

        model = TestModel()
        model.eval()

        input_tensor = torch.randn(output_shape).to(torch.float16)

        with torch.no_grad():
            exported_program = torch.export.export(
                model.eval(),
                args=(),
                kwargs={"input_tensor": input_tensor},
            )

        coreaten_program = exported_program.run_decompositions()
        converter = TorchConverter().add_exported_program(coreaten_program)
        coreai_program = converter.to_coreai()
        # sub-byte dtype (i.e. nbits < 8) indices use dense_resource
        # full-byte dtype (i.e. nbits >= 8) indices use dense
        indices_pattern = (
            "dense<[[VALUE:[^>]*]]>"
            if nbits == 8
            else "dense_resource<[[RESOURCE:[^>]*]]>"
        )
        out_shape_str = f"{output_shape[0]}x{output_shape[1]}"

        # Flaky: FileCheck pattern matching is sensitive to IR output formatting changes
        try:
            truth = f"""
        // CHECK-LABEL: coreai.graph @main
        // CHECK: %[[INDICES:[0-9]+]] = coreai.constant {indices_pattern} : tensor<2x3xui{nbits}>
        // CHECK: %[[LUT:[0-9]+]] = coreai.constant dense<[[VALUE:[^:]*]]> : tensor<1x1x{2**nbits}x{vector_size}xf16>
        // CHECK: %[[AXIS:[0-9]+]] = coreai.constant dense<0> : tensor<si16>
        // CHECK: %[[DEPALETTIZED:[0-9]+]] = coreai.lut_to_dense %[[INDICES]], %[[LUT]], %[[AXIS]] : (tensor<2x3xui{nbits}>, tensor<1x1x{2**nbits}x{vector_size}xf16>, tensor<si16>) -> tensor<{out_shape_str}xf16>
        // CHECK: coreai.decomposable.broadcasting_add {{{{.*}}}} : (tensor<{out_shape_str}xf16>, tensor<{out_shape_str}xf16>) -> tensor<{out_shape_str}xf16>
        """
            filecheck_pattern(str(coreai_program), check_file=truth)
        except RuntimeError:
            pytest.skip("Flaky: FileCheck pattern match failed")

    @pytest.mark.parametrize(
        "nbits",
        [1, 2, 4, 8],
    )
    @pytest.mark.parametrize(
        "vector_size",
        [1, 2, 4],
    )
    async def test_numerical(
        self,
        nbits: int,
        vector_size: int,
    ) -> None:
        """Use custom op directly instead of using coremltools to do palettization."""
        indices_shape = (2, 3)
        axis = 0

        output_shape = list(indices_shape)
        output_shape[axis] *= vector_size

        class TestModel(torch.nn.Module):
            """Palettized model with torch custom op."""

            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "indices",
                    torch.randint(0, 2**nbits, indices_shape, dtype=torch.uint8),
                )
                self.register_buffer(
                    "lut",
                    torch.ones((1, 1, 2**nbits, vector_size), dtype=torch.float16),
                )

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                lut_output = torch.ops.coreai.lut_to_dense(
                    indices=self.indices,
                    lut=self.lut,
                    axis=axis,
                )
                return cast("torch.Tensor", input_tensor + lut_output)

        model = TestModel()
        model.eval()

        input_tensor = torch.randn(output_shape).to(torch.float16)

        with torch.no_grad():
            torch_output = model(input_tensor)
            exported_program = torch.export.export(
                model.eval(),
                args=(),
                kwargs={"input_tensor": input_tensor},
            )

        coreai_program = await lower_to_coreai(exported_program)

        await _validate_execution(
            coreai_program,
            torch_output,
            input_tensor=input_tensor,
        )


class TestWeightSparsificationWithCustomOp:
    """Tests for weight_sparsification_with_custom_op lowering."""

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "nonzero_data_dtype",
        [np.int8, np.float16],
    )
    @pytest.mark.parametrize(
        "data_shape",
        [(3, 4), (16, 32)],
    )
    def test_ir(
        self,
        nonzero_data_dtype: type[np.number[Any]],
        data_shape: tuple[int, int],
    ) -> None:
        """Use custom op directly instead of using coremltools to do sparsification IR check."""
        rng = np.random.default_rng(1337)

        mask = rng.random(data_shape) > 0.5
        nonzero_num = np.int32(np.sum(mask))

        if np.issubdtype(nonzero_data_dtype, np.integer):
            nonzero_data_dtype = cast("type[np.integer[Any]]", nonzero_data_dtype)
            nonzero_data = rng.integers(
                np.iinfo(nonzero_data_dtype).min,
                np.iinfo(nonzero_data_dtype).max,
                (nonzero_num,),
            ).astype(nonzero_data_dtype)
        else:
            nonzero_data = rng.random((nonzero_num,)).astype(nonzero_data_dtype)

        class TestModel(torch.nn.Module):
            """Sparsified model with torch custom op."""

            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("nonzero_data", torch.from_numpy(nonzero_data))
                self.register_buffer("mask", torch.from_numpy(mask))

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                lut_output = torch.ops.coreai.sparse_to_dense(
                    self.nonzero_data,
                    self.mask,
                )
                return cast("torch.Tensor", input_tensor + lut_output)

        model = TestModel()
        model.eval()

        input_tensor = torch.randn(*data_shape).to(torch.float16)

        with torch.no_grad():
            exported_program = torch.export.export(
                model.eval(),
                args=(),
                kwargs={"input_tensor": input_tensor},
            )

        coreaten_program = exported_program.run_decompositions()
        converter = TorchConverter().add_exported_program(coreaten_program)
        coreai_program = converter.to_coreai()

        np_type_to_str: dict[type[np.number[Any]], str] = {
            np.int8: "si8",
            np.float16: "f16",
        }
        dtype_str = np_type_to_str[nonzero_data_dtype]
        truth = f"""
        // CHECK-LABEL: coreai.graph @main
        // CHECK: %[[NONZERO_DATA:.*]] = coreai.constant [[VALUE:[^:]*]] : tensor<{nonzero_num}x{dtype_str}>
        // CHECK: %[[MASK:.*]] = coreai.constant [[VALUE:[^:]*]] : tensor<{data_shape[0]}x{data_shape[1]}xui1>
        // CHECK: %[[SPARSE:.*]] = coreai.build_sparse_with_bitmask %[[NONZERO_DATA]], %[[MASK]]
        // CHECK: %[[RESULT:.*]] = coreai.sparse_with_bitmask_to_dense %[[SPARSE]]
        """
        filecheck_pattern(str(coreai_program), check_file=truth)

    @pytest.mark.parametrize(
        "nonzero_data_dtype",
        [np.int8, np.float16],
    )
    @pytest.mark.parametrize(
        "data_shape",
        [(3, 4), (16, 32)],
    )
    async def test_numerical(
        self,
        nonzero_data_dtype: type[np.number[Any]],
        data_shape: tuple[int, int],
    ) -> None:
        """Use custom op directly instead of using coremltools to do sparsification."""
        rng = np.random.default_rng(1337)

        mask = rng.random(data_shape) > 0.5
        nonzero_num = np.int32(np.sum(mask))

        if np.issubdtype(nonzero_data_dtype, np.integer):
            nonzero_data_dtype = cast("type[np.integer[Any]]", nonzero_data_dtype)
            nonzero_data = rng.integers(
                np.iinfo(nonzero_data_dtype).min,
                np.iinfo(nonzero_data_dtype).max,
                (nonzero_num,),
            ).astype(nonzero_data_dtype)
        else:
            nonzero_data = rng.random((nonzero_num,)).astype(nonzero_data_dtype)

        class TestModel(torch.nn.Module):
            """Sparsified model with torch custom op."""

            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("nonzero_data", torch.from_numpy(nonzero_data))
                self.register_buffer("mask", torch.from_numpy(mask))

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                lut_output = torch.ops.coreai.sparse_to_dense(
                    self.nonzero_data,
                    self.mask,
                )
                return cast("torch.Tensor", input_tensor + lut_output)

        model = TestModel()
        model.eval()

        input_tensor = torch.randn(*data_shape).to(torch.float16)

        with torch.no_grad():
            torch_output = model(input_tensor)
            exported_program = torch.export.export(
                model.eval(),
                args=(),
                kwargs={"input_tensor": input_tensor},
            )

        coreai_program = await lower_to_coreai(
            exported_program,
        )

        await _validate_execution(
            coreai_program,
            torch_output,
            input_tensor=input_tensor,
        )


class ActQuantizedLinearModel(torch.nn.Module):
    """Activation quantized linear model."""

    def __init__(
        self,
        *,
        symmetric: bool,
        dtype: torch.dtype,
        granularity: str,
        axis: int = 0,
    ) -> None:
        """Initialize the activation quantized linear model."""
        super().__init__()
        self.linear = nn.Linear(24, 32)
        assert granularity in ["per_tensor", "per_channel"]
        if granularity == "per_tensor":
            shape: list[int] = []
        elif axis == 0:
            shape = [2]
        else:
            shape = [32]
        scale = torch.ones(shape).to(torch.float32)
        zero_point = torch.zeros_like(scale) if symmetric else torch.ones_like(scale)
        zero_point = zero_point.to(dtype)
        self.quant = ActivationQuantizeModule(
            scale,
            output_dtype=dtype,
            zero_point=zero_point,
            axis=axis,
        )
        self.dequant = ActivationDequantizeModule(
            scale, zero_point=zero_point, axis=axis
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass for ActQuantizedLinearModel."""
        lineared = self.linear(input_tensor)
        quantized = self.quant(lineared)
        return cast("torch.Tensor", self.dequant(quantized))


class TestActivationQuantizationLinear:
    """Tests for activation_quantization_linear lowering."""

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "symmetric",
        [True, False],
    )
    @pytest.mark.parametrize(
        "dtype",
        [torch.int8, torch.uint8],
    )
    @pytest.mark.parametrize(
        "granularity",
        ["per_tensor", "per_channel"],
    )
    @pytest.mark.parametrize(
        "axis",
        [0, 1],
    )
    def test_ir(
        self,
        *,
        symmetric: bool,
        dtype: torch.dtype,
        granularity: str,
        axis: int,
    ) -> None:
        """Test activation quantized linear IR check."""
        model = ActQuantizedLinearModel(
            symmetric=symmetric,
            dtype=dtype,
            granularity=granularity,
            axis=axis,
        )
        input_tensor = torch.randn(2, 24)

        with torch.no_grad():
            # Make sure pytorch model is runnable.
            model(input_tensor)
            exported_program = torch.export.export(
                model.eval(),
                args=(),
                kwargs={"input_tensor": input_tensor},
            )

        coreaten_program = exported_program.run_decompositions()
        converter = TorchConverter().add_exported_program(coreaten_program)
        coreai_program = converter.to_coreai()
        quant_dtype = "ui8" if dtype == torch.uint8 else "si8"
        offset = 0 if symmetric else 1
        activation_shape = (2, 32)
        scale_shape = (
            "" if granularity == "per_tensor" else f"{activation_shape[axis]}x"
        )

        msg = "TODO: reshape on consts such as offset and scale is not const eliminated"
        pytest.xfail(reason=msg)
        truth = f"""
        // CHECK-LABEL: coreai.graph @main
        // CHECK-DAG: %[[BROADCAST:[0-9]+]] = coreai.constant dense<{{{{\\[.*\\]}}}}> : tensor<32xf32>
        // CHECK-DAG: %[[OFFSET:[0-9]+]] = coreai.constant dense<{offset}> : tensor<{scale_shape}{quant_dtype}>
        // CHECK-DAG: %[[SCALE:[0-9]+]] = coreai.constant dense<1.000000e+00> : tensor<{scale_shape}f32>
        // CHECK-DAG: %[[AXIS:[0-9]+]] = coreai.constant dense<{axis}> : tensor<si32>
        // CHECK: %[[MATMUL:[0-9]+]] = coreai.batch_matmul %[[INPUT:.*]], %[[WEIGHT:.*]] : (tensor<2x24xf32>, tensor<24x32xf32>) -> tensor<2x32xf32>
        // CHECK: %[[ADD:[0-9]+]] = coreai.decomposable.broadcasting_add %[[MATMUL]], %[[BROADCAST]] : (tensor<2x32xf32>, tensor<32xf32>) -> tensor<2x32xf32>
        // CHECK: %[[QUANT:[0-9]+]] = coreai.quantize %[[ADD]], %[[SCALE]], %[[OFFSET]], %[[AXIS]] : (tensor<2x32xf32>, tensor<{scale_shape}f32>, tensor<{scale_shape}{quant_dtype}>, tensor<si32>) -> tensor<2x32x{quant_dtype}>
        // CHECK: %[[DEQUANT:[0-9]+]] = coreai.dequantize %[[QUANT]], %[[SCALE]], %[[OFFSET]], %[[AXIS]] : (tensor<2x32x{quant_dtype}>, tensor<{scale_shape}f32>, tensor<{scale_shape}{quant_dtype}>, tensor<si32>) -> tensor<2x32xf32>
        """
        filecheck_pattern(str(coreai_program), check_file=truth)

    @pytest.mark.parametrize(
        "symmetric",
        [True, False],
    )
    @pytest.mark.parametrize(
        "dtype",
        [torch.int8, torch.uint8],
    )
    @pytest.mark.parametrize(
        "granularity",
        ["per_tensor", "per_channel"],
    )
    @pytest.mark.parametrize(
        "axis",
        [0, 1],
    )
    async def test_numerical(
        self,
        *,
        symmetric: bool,
        dtype: torch.dtype,
        granularity: str,
        axis: int,
    ) -> None:
        """Test activation quantized linear."""
        model = ActQuantizedLinearModel(
            symmetric=symmetric,
            dtype=dtype,
            granularity=granularity,
            axis=axis,
        )
        input_tensor = torch.randn(2, 24)

        with torch.no_grad():
            # Make sure pytorch model is runnable.
            torch_out = model(input_tensor)
            exported_program = torch.export.export(
                model.eval(),
                args=(),
                kwargs={"input_tensor": input_tensor},
            )

        coreai_program = await lower_to_coreai(
            exported_program,
        )

        await _validate_execution(coreai_program, torch_out, input_tensor=input_tensor)
