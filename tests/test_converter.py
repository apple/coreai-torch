# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from coreai._compiler._mlir_libs._coreaiIR._bindings.mlir.dialects.coreai import (
    register_should_const_folding_hook,
)
from coreai._compiler.dialects import coreai
from coreai.runtime import NDArray
from torch import Tensor
from torch.export import ExportedProgram

import coreai_torch
from coreai_torch import TorchConverter, get_decomp_table
from coreai_torch._aten_to_core import (
    _aten_to_core_resolver,
    _higher_order_resolver,
)
from coreai_torch._custom_to_core import _custom_to_core_resolver
from coreai_torch._utils import (
    get_operand,
    get_operands,
    get_promoted_type,
)

from .utils import filecheck_pattern

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_converter() -> TorchConverter:
    return TorchConverter()


def _noop(values_map, node, loc): ...


# ---------------------------------------------------------------------------
# Custom torch op for integration tests — registered once at module level.
# coreai_test::neg negates its input; our lowering uses broadcasting_mul(x, -1)
# so filecheck can assert the exact IR pattern.
# ---------------------------------------------------------------------------


@torch.library.custom_op("coreai_test::neg", mutates_args=())
def _coreai_test_neg(x: torch.Tensor) -> torch.Tensor:
    return -x


@_coreai_test_neg.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _make_add_program() -> object:
    class _Add(nn.Module):
        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return x + y

    x = torch.rand(2, 3, dtype=torch.float32)
    return torch.export.export(_Add(), args=(x, x)).run_decompositions()


def _sub_lowering(values_map, node, loc):
    """Stand-in for aten::add.Tensor that emits subtraction instead."""
    a, b = get_operands(values_map, node, [0, 1])
    promoted = get_promoted_type(a.type, b.type)
    return coreai.broadcasting_sub(
        coreai.cast(a, promoted),
        coreai.cast(b, promoted),
    )


# ---------------------------------------------------------------------------
# TestRegisterTorchLowering
# ---------------------------------------------------------------------------


class TestRegisterTorchLowering:
    @pytest.fixture
    def converter(self) -> TorchConverter:
        return _make_converter()

    # --- qualified_name format ---

    @pytest.mark.parametrize(
        "bad_name",
        [
            "my_op",  # no separator
            "",  # empty
            "::my_op",  # empty namespace
            "my_lib::",  # empty op name
            "::",  # both empty
            "a::b::c",  # too many separators
        ],
    )
    def test_invalid_format_raises(
        self, converter: TorchConverter, bad_name: str
    ) -> None:
        with pytest.raises(ValueError, match="namespace::op_name"):
            converter.register_torch_lowering(bad_name)(_noop)

    def test_format_error_echoes_bad_name(self, converter: TorchConverter) -> None:
        with pytest.raises(ValueError, match="totally_wrong"):
            converter.register_torch_lowering("totally_wrong")(_noop)

    # --- successful registration ---

    def test_register_stores_function(self, converter: TorchConverter) -> None:
        converter.register_torch_lowering("my_lib::my_op")(_noop)
        assert converter._user_defined_torch_lowering["my_lib::my_op"] is _noop

    def test_decorator_returns_original_function(
        self, converter: TorchConverter
    ) -> None:
        result = converter.register_torch_lowering("my_lib::my_op")(_noop)
        assert result is _noop

    def test_register_multiple_ops(self, converter: TorchConverter) -> None:
        def fn_a(values_map, node, loc): ...
        def fn_b(values_map, node, loc): ...

        converter.register_torch_lowering("my_lib::op_a")(fn_a)
        converter.register_torch_lowering("my_lib::op_b")(fn_b)
        assert converter._user_defined_torch_lowering["my_lib::op_a"] is fn_a
        assert converter._user_defined_torch_lowering["my_lib::op_b"] is fn_b

    # --- decorator syntax ---

    def test_decorator_default(self, converter: TorchConverter) -> None:
        @converter.register_torch_lowering("my_lib::op")
        def fn(values_map, node, loc): ...

        assert converter._user_defined_torch_lowering["my_lib::op"] is fn

    def test_decorator_explicit_false(self, converter: TorchConverter) -> None:
        @converter.register_torch_lowering("my_lib::op", allow_override=False)
        def fn(values_map, node, loc): ...

        assert converter._user_defined_torch_lowering["my_lib::op"] is fn

    def test_decorator_override_true(self, converter: TorchConverter) -> None:
        @converter.register_torch_lowering("aten::abs.default", allow_override=True)
        def fn(values_map, node, loc): ...

        assert converter._user_defined_torch_lowering["aten::abs.default"] is fn

    def test_decorator_raises_for_existing_op(self, converter: TorchConverter) -> None:
        with pytest.raises(ValueError, match="already registered"):

            @converter.register_torch_lowering(
                "aten::abs.default", allow_override=False
            )
            def fn(values_map, node, loc): ...

    # --- duplicate / override ---

    def test_duplicate_raises(self, converter: TorchConverter) -> None:
        converter.register_torch_lowering("my_lib::my_op")(_noop)
        with pytest.raises(ValueError, match="my_lib::my_op") as exc_info:
            converter.register_torch_lowering("my_lib::my_op")(_noop)
        assert "allow_override" in str(exc_info.value)

    def test_duplicate_with_override_replaces(self, converter: TorchConverter) -> None:
        def fn_v1(values_map, node, loc): ...
        def fn_v2(values_map, node, loc): ...

        converter.register_torch_lowering("my_lib::op")(fn_v1)
        result = converter.register_torch_lowering("my_lib::op", allow_override=True)(
            fn_v2
        )
        assert result is fn_v2
        assert converter._user_defined_torch_lowering["my_lib::op"] is fn_v2

    # --- override built-in ops ---

    @pytest.mark.parametrize(
        "qualified_name",
        [
            "aten::abs.default",
            "higher_order::cond",
            "coreai::lut_to_dense",
        ],
    )
    def test_built_in_op_raises_without_override(
        self, converter: TorchConverter, qualified_name: str
    ) -> None:
        with pytest.raises(ValueError, match="already registered"):
            converter.register_torch_lowering(qualified_name)(_noop)

    @pytest.mark.parametrize(
        "qualified_name,resolver",
        [
            ("aten::abs.default", _aten_to_core_resolver),
            ("higher_order::cond", _higher_order_resolver),
            ("coreai::lut_to_dense", _custom_to_core_resolver),
        ],
    )
    def test_built_in_op_override_does_not_mutate_resolver(
        self, converter: TorchConverter, qualified_name: str, resolver: dict
    ) -> None:
        original = dict(resolver)
        converter.register_torch_lowering(qualified_name, allow_override=True)(_noop)
        assert resolver == original

    # --- instance isolation ---

    def test_registration_is_per_instance(self) -> None:
        converter_a, converter_b = _make_converter(), _make_converter()
        converter_a.register_torch_lowering("my_lib::op")(_noop)
        assert "my_lib::op" not in converter_b._user_defined_torch_lowering

    def test_override_is_per_instance(self) -> None:
        converter_a, converter_b = _make_converter(), _make_converter()
        converter_a.register_torch_lowering("aten::abs.default", allow_override=True)(
            _noop
        )
        assert "aten::abs.default" not in converter_b._user_defined_torch_lowering


# ---------------------------------------------------------------------------
# TestRegisterTorchLoweringIntegration
# ---------------------------------------------------------------------------


class TestRegisterTorchLoweringIntegration:
    """End-to-end: export → register lowering → convert → verify IR."""

    @pytest.mark.ir
    def test_custom_op_lowering_is_invoked(self) -> None:
        """Custom lowering for coreai_test::neg produces the correct op and output shape."""

        class NegModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.coreai_test.neg(x)  # type: ignore[attr-defined]

        x = torch.rand(4, 5, dtype=torch.float32)
        program = torch.export.export(NegModel(), args=(x,)).run_decompositions()
        converter = TorchConverter()

        @converter.register_torch_lowering("coreai_test::neg.default")
        def _neg_lowering(values_map, node, loc):
            inp = get_operand(values_map, node, 0, loc)
            return coreai.broadcasting_mul(
                inp, coreai.constant(-1.0, dtype=inp.type.element_type)
            )

        filecheck_pattern(
            str(converter.add_exported_program(program).to_coreai()),
            check_file="""
                // CHECK: coreai.graph @main(%{{.*}}: tensor<4x5xf32>
                // CHECK-SAME: -> (tensor<4x5xf32>
                // CHECK: coreai.decomposable.broadcasting_mul
                // CHECK-NOT: coreai.neg
            """,
        )

    @pytest.mark.ir
    def test_override_aten_add_uses_custom_lowering(self) -> None:
        """Overriding aten::add.Tensor — the converter uses the custom lowering, not the built-in."""
        converter = TorchConverter()
        converter.register_torch_lowering("aten::add.Tensor", allow_override=True)(
            _sub_lowering
        )

        filecheck_pattern(
            str(converter.add_exported_program(_make_add_program()).to_coreai()),
            check_file="""
                // CHECK: coreai.decomposable.broadcasting_sub
                // CHECK-NOT: coreai.decomposable.broadcasting_add
            """,
        )

    def test_override_aten_add_does_not_mutate_resolver(self) -> None:
        """Overriding aten::add.Tensor must not modify _aten_to_core_resolver."""
        original_add = _aten_to_core_resolver["add.Tensor"]

        converter = TorchConverter()
        converter.register_torch_lowering("aten::add.Tensor", allow_override=True)(
            _sub_lowering
        )
        converter.add_exported_program(_make_add_program()).to_coreai()

        assert _aten_to_core_resolver["add.Tensor"] is original_add


# ---------------------------------------------------------------------------
# TestConvertToCoreaiNoOptimization
# ---------------------------------------------------------------------------


class TestConvertToCoreaiIR:
    """to_coreai emits Core AI IR with the pre-compilation rewrites already applied."""

    @pytest.mark.ir
    def test_cast_chain_fused(self) -> None:
        """An si32→f32→f16 chain is fused into a single si32→f16 cast by to_coreai."""

        class CastChainModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.to(torch.float32).to(torch.float16)

        x = torch.randint(0, 10, (3, 4), dtype=torch.int32)
        program = torch.export.export(CastChainModel(), args=(x,)).run_decompositions()
        coreai_program = TorchConverter().add_exported_program(program).to_coreai()

        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK-NOT: coreai.cast %{{.*}} : tensor<3x4xsi32> to tensor<3x4xf32>
                // CHECK: coreai.cast %{{.*}} : tensor<3x4xsi32> to tensor<3x4xf16>
            """,
        )

    @pytest.mark.ir
    def test_bfloat16_constant_model(self) -> None:
        """Models with bfloat16 parameters can be imported without error."""

        class BFloat16Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 4, dtype=torch.bfloat16))

            def forward(self, x: Tensor) -> Tensor:
                return x + self.weight

        model = BFloat16Model()
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        program = torch.export.export(model, args=(x,)).run_decompositions()
        coreai_program = TorchConverter().add_exported_program(program).to_coreai()

        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: coreai.constant dense<{{.*}}> : tensor<3x4xbf16>
            """,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "fp8_dtype,coreai_dtype",
        [
            (torch.float8_e4m3fn, "f8E4M3FN"),
            (torch.float8_e5m2, "f8E5M2"),
            (torch.float8_e8m0fnu, "f8E8M0FNU"),
        ],
    )
    @pytest.mark.ir
    def test_reduced_precision_float_constant_model(
        self, fp8_dtype: torch.dtype, coreai_dtype: str
    ) -> None:
        """Models with reduced-precision float (fp8) buffers are imported correctly."""

        class FP8ConstantModule(nn.Module):
            def __init__(self, dtype: torch.dtype) -> None:
                super().__init__()
                self.register_buffer("fp8_weight", torch.ones((6,), dtype=dtype))
                self.register_buffer("fp8_bias", torch.full((3,), 0.5, dtype=dtype))

            def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
                return x, self.fp8_weight, self.fp8_bias

        model = FP8ConstantModule(fp8_dtype).eval()
        input_tensor = torch.rand(2, 3, dtype=torch.float32)
        program = torch.export.export(model, args=(input_tensor,)).run_decompositions()
        coreai_program = TorchConverter().add_exported_program(program).to_coreai()

        filecheck_pattern(
            str(coreai_program),
            check_file=f"""
                // CHECK: coreai.graph @main(%arg0: tensor<2x3xf32>
                // CHECK-DAG: coreai.constant dense<{{{{.*}}}}> : tensor<6x{coreai_dtype}>
                // CHECK-DAG: coreai.constant dense<{{{{.*}}}}> : tensor<3x{coreai_dtype}>
            """,
        )

    @pytest.mark.ir
    def test_const_folding_hook_prevents_cast_folding(self) -> None:
        """register_should_const_folding_hook can suppress constant folding for specific ops.

        Without the hook, the pre-compilation rewrite run by ``to_coreai`` folds
        cast(constant(7, si32), f32) into a single float constant. A hook
        returning False for coreai.cast keeps both ops intact.
        """

        class ConstCastModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("bias", torch.tensor([7], dtype=torch.int32))

            def forward(self, _: Tensor) -> Tensor:
                return self.bias.to(torch.float32)

        program = torch.export.export(
            ConstCastModel(), args=(torch.rand(1),)
        ).run_decompositions()

        # The hook has to be installed on the converter's context before
        # to_coreai runs the rewrite.
        converter = TorchConverter()
        register_should_const_folding_hook(
            callable=lambda op: op.name != "coreai.cast",
            context=converter.context._mlir_context,
        )
        coreai_program = converter.add_exported_program(program).to_coreai()

        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: coreai.constant dense<7> : tensor<1xsi32>
                // CHECK: coreai.cast %{{.*}} : tensor<1xsi32> to tensor<1xf32>
            """,
        )


# ---------------------------------------------------------------------------
# TestConvertToCoreaiInputNames
# ---------------------------------------------------------------------------


def _export(model: nn.Module, *args: torch.Tensor):
    return torch.export.export(model, args=args).run_decompositions()


class TestConvertToCoreaiInputNames:
    @staticmethod
    def _identity_program():
        class _Id(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x

        return _export(_Id(), torch.rand(2, 3))

    @staticmethod
    def _add_program():
        class _Add(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x + y

        # Distinct tensors prevent torch.export from deduplicating the inputs.
        return _export(_Add(), torch.rand(2, 3), torch.rand(2, 3) + 1)

    @staticmethod
    def _add3_program():
        class _Add3(nn.Module):
            def forward(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
                return x + y + z

        t = torch.rand(1, 2, dtype=torch.float32)
        return _export(_Add3(), t, t + 1, t + 2)

    @pytest.mark.ir
    def test_single_custom_name_appears_in_ir(self):
        result = (
            TorchConverter()
            .add_exported_program(self._identity_program(), input_names=["my_input"])
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK: coreai.graph @main(%arg0: tensor<2x3xf32> {coreai.name = "my_input"}
            """,
        )

    @pytest.mark.ir
    def test_default_names_use_original_placeholder_targets(self):
        result = (
            TorchConverter().add_exported_program(self._identity_program()).to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK: coreai.graph @main(%arg0: tensor<2x3xf32> {coreai.name = "x"}
            """,
        )

    @pytest.mark.ir
    def test_empty_list_keeps_original_names(self):
        result = (
            TorchConverter()
            .add_exported_program(self._identity_program(), input_names=[])
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK: coreai.graph @main(%arg0: tensor<2x3xf32> {coreai.name = "x"}
            """,
        )

    @pytest.mark.ir
    def test_multi_input_custom_names_appear_in_ir(self):
        result = (
            TorchConverter()
            .add_exported_program(self._add_program(), input_names=["alpha", "beta"])
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK: coreai.graph @main(%arg0: tensor<2x3xf32> {coreai.name = "alpha"}, %arg1: tensor<2x3xf32> {coreai.name = "beta"}
            """,
        )

    @pytest.mark.ir
    def test_renamed_inputs_downstream_add_still_emitted(self):
        result = (
            TorchConverter()
            .add_exported_program(self._add_program(), input_names=["a", "b"])
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK: coreai.graph @main(%arg0: tensor<2x3xf32> {coreai.name = "a"}, %arg1: tensor<2x3xf32> {coreai.name = "b"}
                // CHECK: coreai.decomposable.broadcasting_add
            """,
        )

    @pytest.mark.ir
    def test_permuted_names_matching_other_placeholder_targets(self):
        """Regression: user names that are a permutation of the original placeholder names."""
        result = (
            TorchConverter()
            .add_exported_program(self._add3_program(), input_names=["z", "x", "y"])
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK: coreai.graph @main(%arg0: tensor<1x2xf32> {coreai.name = "z"}, %arg1: tensor<1x2xf32> {coreai.name = "x"}, %arg2: tensor<1x2xf32> {coreai.name = "y"}
                // CHECK: coreai.decomposable.broadcasting_add
            """,
        )

    def test_too_few_names_raises(self):
        with pytest.raises(ValueError, match="live inputs"):
            TorchConverter().add_exported_program(
                self._add_program(), input_names=["only_one"]
            ).to_coreai()

    def test_too_many_names_raises(self):
        with pytest.raises(ValueError, match="live inputs"):
            TorchConverter().add_exported_program(
                self._identity_program(), input_names=["a", "b"]
            ).to_coreai()

    async def test_user_provided_names_used_at_runtime(self):
        from tempfile import TemporaryDirectory

        x_val = torch.rand(2, 3)

        class _Id(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x

        coreai_program = (
            TorchConverter()
            .add_exported_program(_export(_Id(), x_val), input_names=["my_input"])
            .to_coreai()
        )

        with TemporaryDirectory(suffix=".aimodel") as tmp:
            asset = coreai_program.save_asset(Path(tmp))
            async with asset.executable() as ai_model:
                rt_func = ai_model.load_function("main")
                outputs = await rt_func(inputs={"my_input": NDArray(x_val)})
                assert np.allclose(list(outputs.values())[0].numpy(), x_val.numpy())


class TestMultiGraphChaining:
    """Tests for multi-graph chaining via repeated add_exported_program calls."""

    @staticmethod
    def _add_model() -> ExportedProgram:
        class AddModel(torch.nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x + y

        return torch.export.export(
            AddModel().eval(),
            args=(
                torch.randn(2, 2, 3, dtype=torch.float32),
                torch.randn(2, 2, 3, dtype=torch.float32),
            ),
        )

    @staticmethod
    def _mul_model() -> ExportedProgram:
        class MulModel(torch.nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x * y

        return torch.export.export(
            MulModel().eval(),
            args=(
                torch.randn(4, 5, dtype=torch.float64),
                torch.randn(4, 5, dtype=torch.float64),
            ),
        )

    @pytest.mark.ir
    def test_chaining_multiple_exported_programs(self) -> None:
        """Chain multiple add_exported_program calls into a single coreai_program."""
        add_model = self._add_model()
        mul_model = self._mul_model()

        coreai_program = (
            TorchConverter()
            .add_exported_program(
                add_model,
                input_names=["x", "y"],
                output_names=["added"],
                entrypoint_name="add",
            )
            .add_exported_program(
                mul_model,
                input_names=["a", "b"],
                output_names=["muled"],
                entrypoint_name="mul",
            )
            .to_coreai()
        )

        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: coreai.graph @add
                // CHECK-SAME: {coreai.name = "x"}
                // CHECK-SAME: {coreai.name = "y"}
                // CHECK-SAME: {coreai.name = "added"}
                // CHECK: coreai.graph @mul
                // CHECK-SAME: {coreai.name = "a"}
                // CHECK-SAME: {coreai.name = "b"}
                // CHECK-SAME: {coreai.name = "muled"}
            """,
        )

    @pytest.mark.ir
    def test_chaining_exported_program_and_pytorch_module(self) -> None:
        """Chain add_exported_program and add_pytorch_module on the same converter."""

        class NegModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return -x

        add_ep = self._add_model()
        neg_module = NegModel().eval()
        sample = (torch.randn(3, 4, dtype=torch.float32),)

        coreai_program = (
            TorchConverter()
            .add_exported_program(
                add_ep,
                input_names=["x", "y"],
                output_names=["added"],
                entrypoint_name="add",
            )
            .add_pytorch_module(
                neg_module,
                export_fn=lambda m: torch.export.export(
                    m, args=sample
                ).run_decompositions(get_decomp_table()),
                input_names=["inp"],
                output_names=["negated"],
                entrypoint_name="negate",
            )
            .to_coreai()
        )

        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: coreai.graph @add
                // CHECK-SAME: {coreai.name = "x"}
                // CHECK-SAME: {coreai.name = "y"}
                // CHECK-SAME: {coreai.name = "added"}
                // CHECK: coreai.graph @negate
                // CHECK-SAME: {coreai.name = "inp"}
                // CHECK-SAME: {coreai.name = "negated"}
            """,
        )


# ---------------------------------------------------------------------------
# TestConverterReusability
# ---------------------------------------------------------------------------


class TestConverterReusability:
    """Tests for converter reuse, state management, and repr."""

    @staticmethod
    def _make_ep():
        model = nn.Linear(4, 4).eval()
        ep = torch.export.export(model, args=(torch.randn(1, 4),))
        return ep.run_decompositions()

    @staticmethod
    def _make_abs_ep():
        class AbsModel(nn.Module):
            def forward(self, x):
                return torch.abs(x)

        model = AbsModel().eval()
        ep = torch.export.export(model, args=(torch.randn(2, 3),))
        return ep.run_decompositions()

    def test_to_coreai_no_staged_raises(self):
        converter = TorchConverter()
        with pytest.raises(RuntimeError, match="No programs to convert"):
            converter.to_coreai()

    def test_programs_persist_after_to_coreai(self):
        converter = TorchConverter()
        converter.add_exported_program(self._make_ep())
        converter.to_coreai()
        # Programs still staged — second call works without re-adding
        d2 = converter.to_coreai()
        assert d2 is not None

    def test_clear_removes_staged_programs(self):
        converter = TorchConverter()
        converter.add_exported_program(self._make_ep())
        converter.clear()
        with pytest.raises(RuntimeError, match="No programs to convert"):
            converter.to_coreai()

    def test_selective_conversion_by_entrypoint(self):
        converter = TorchConverter()
        converter.add_exported_program(self._make_ep(), entrypoint_name="encoder")
        converter.add_exported_program(self._make_ep(), entrypoint_name="decoder")
        coreai_program = converter.to_coreai(entrypoints=["encoder"])
        ir = str(coreai_program)
        assert "encoder" in ir
        assert "decoder" not in ir

    def test_selective_conversion_unknown_entrypoint_raises(self):
        converter = TorchConverter()
        converter.add_exported_program(self._make_ep(), entrypoint_name="add")
        with pytest.raises(RuntimeError, match="No programs to convert"):
            converter.to_coreai(entrypoints=["nonexistent"])

    def test_custom_ops_persist_across_conversions(self):
        converter = TorchConverter()

        @converter.register_torch_lowering("aten::abs.default", allow_override=True)
        def lower_abs(values_map, node, loc):
            from coreai_torch._utils import get_operand

            x = get_operand(values_map, node, 0, loc)
            # Custom lowering: abs(x) = x * x then sqrt, just to prove
            # the lowering runs. Use broadcasting_mul which we know exists.
            return coreai.broadcasting_mul(x, x, loc=loc)

        converter.add_exported_program(self._make_abs_ep())
        d1 = converter.to_coreai()
        assert d1 is not None
        converter.clear()
        converter.add_exported_program(self._make_abs_ep())
        d2 = converter.to_coreai()
        assert d2 is not None

    def test_repr_shows_staged_programs(self):
        converter = TorchConverter()
        converter.add_exported_program(
            self._make_ep(),
            entrypoint_name="enc",
            input_names=["img"],
            output_names=["feat"],
        )
        r = repr(converter)
        assert "enc: ExportedProgram" in r
        assert "['img'] -> ['feat']" in r

    def test_clear_by_entrypoint(self):
        converter = TorchConverter()
        converter.add_exported_program(self._make_ep(), entrypoint_name="keep")
        converter.add_exported_program(self._make_ep(), entrypoint_name="drop")
        converter.clear(entrypoints=["drop"])
        assert len(converter._staged) == 1
        assert converter._staged[0].entrypoint_name == "keep"

    def test_duplicate_entrypoint_raises(self):
        converter = TorchConverter()
        converter.add_exported_program(self._make_ep(), entrypoint_name="main")
        with pytest.raises(ValueError, match="entrypoint_name='main'.*already staged"):
            converter.add_exported_program(self._make_ep(), entrypoint_name="main")

    def test_duplicate_entrypoint_raises_add_pytorch_module(self):
        converter = TorchConverter()
        converter.add_exported_program(self._make_ep(), entrypoint_name="shared")

        class _Id(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x

        with pytest.raises(
            ValueError, match="entrypoint_name='shared'.*already staged"
        ):
            converter.add_pytorch_module(
                _Id(),
                export_fn=lambda m: torch.export.export(
                    m, args=(torch.randn(2, 3),)
                ).run_decompositions(get_decomp_table()),
                entrypoint_name="shared",
            )

    def test_duplicate_after_clear_allowed(self):
        converter = TorchConverter()
        converter.add_exported_program(self._make_ep(), entrypoint_name="main")
        converter.clear(entrypoints=["main"])
        # Re-staging with the same name should succeed after clear
        converter.add_exported_program(self._make_ep(), entrypoint_name="main")
        assert len(converter._staged) == 1

    def test_repr_empty(self):
        converter = TorchConverter()
        r = repr(converter)
        assert "TorchConverter" in r
        assert "(none)" in r


# ---------------------------------------------------------------------------
# TestGetDefaultDecompositionTable
# ---------------------------------------------------------------------------


class TestGetDefaultDecompositionTable:
    """Unit tests for coreai_torch.get_decomp_table()."""

    def test_importable_from_top_level_package(self):
        assert hasattr(coreai_torch, "get_decomp_table")
        assert callable(coreai_torch.get_decomp_table)

    def test_returns_dict(self):
        table = get_decomp_table()
        assert isinstance(table, dict)

    def test_non_empty(self):
        """The table is a non-empty subset of the default decompositions."""
        table = get_decomp_table()
        assert len(table) > 0

    def test_instance_norm_excluded(self):
        table = get_decomp_table()
        assert torch.ops.aten.instance_norm.default not in table

    def test_pixel_shuffle_excluded(self):
        table = get_decomp_table()
        assert torch.ops.aten.pixel_shuffle.default not in table

    def test_scaled_dot_product_attention_excluded(self):
        table = get_decomp_table()
        assert torch.ops.aten.scaled_dot_product_attention.default not in table

    def test_table_is_strictly_smaller_than_default(self):
        """The table is a proper subset of default_decompositions (3 composite ops removed)."""
        default = torch.export.default_decompositions()
        table = get_decomp_table()
        assert len(table) < len(default)

    def test_returns_independent_copy(self):
        """Each call returns a fresh dict; mutating one does not affect another."""
        t1 = get_decomp_table()
        t2 = get_decomp_table()
        t1.clear()
        assert len(t2) > 0

    def test_is_subset_of_default_decompositions(self):
        """Every key in the table is present in torch.export.default_decompositions()."""
        default = torch.export.default_decompositions()
        table = get_decomp_table()
        for key in table:
            assert key in default


# ---------------------------------------------------------------------------
# TestDefaultDecompTableIntegration
# ---------------------------------------------------------------------------


def _make_instance_norm_model():
    class _InstanceNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.InstanceNorm2d(4, affine=True)

        def forward(self, x: Tensor) -> Tensor:
            return self.norm(x)

    return _InstanceNormModel().eval()


def _make_pixel_shuffle_model():
    class _PixelShuffleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ps = nn.PixelShuffle(2)

        def forward(self, x: Tensor) -> Tensor:
            return self.ps(x)

    return _PixelShuffleModel().eval()


def _make_sdpa_model():
    class _SDPAModel(nn.Module):
        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return nn.functional.scaled_dot_product_attention(q, k, v)

    return _SDPAModel().eval()


class TestDefaultDecompTableIntegration:
    """End-to-end: composite ops appear in the IR when the default decomp table is used."""

    # --- add_exported_program path (caller applies the table) ---

    @pytest.mark.ir
    def test_add_exported_program_instance_norm_composite(self):
        """Passing get_decomp_table() to run_decompositions preserves instance_norm."""
        model = _make_instance_norm_model()
        ep = torch.export.export(model, args=(torch.randn(1, 4, 8, 8),))
        ep = ep.run_decompositions(get_decomp_table())
        coreai_program = TorchConverter().add_exported_program(ep).to_coreai()
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: composite_decl = #coreai.composite_declaration<"instance_norm"
            """,
        )

    @pytest.mark.ir
    def test_add_exported_program_pixel_shuffle_composite(self):
        """Passing get_decomp_table() to run_decompositions preserves pixel_shuffle."""
        model = _make_pixel_shuffle_model()
        ep = torch.export.export(model, args=(torch.randn(1, 16, 4, 4),))
        ep = ep.run_decompositions(get_decomp_table())
        coreai_program = TorchConverter().add_exported_program(ep).to_coreai()
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: composite_decl = #coreai.composite_declaration<"pixel_shuffle"
            """,
        )

    @pytest.mark.ir
    def test_add_exported_program_sdpa_composite(self):
        """Passing get_decomp_table() to run_decompositions preserves sdpa."""
        model = _make_sdpa_model()
        q = torch.randn(1, 2, 4, 8)
        ep = torch.export.export(model, args=(q, q, q))
        ep = ep.run_decompositions(get_decomp_table())
        coreai_program = TorchConverter().add_exported_program(ep).to_coreai()
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention"
            """,
        )

    # --- add_pytorch_module path (default table in export_fn) ---

    @pytest.mark.ir
    def test_add_pytorch_module_instance_norm_composite_by_default(self):
        """export_fn with default table — instance_norm becomes a composite op."""
        model = _make_instance_norm_model()
        coreai_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(torch.randn(1, 4, 8, 8),)
                ).run_decompositions(get_decomp_table()),
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: composite_decl = #coreai.composite_declaration<"instance_norm"
            """,
        )

    @pytest.mark.ir
    def test_add_pytorch_module_pixel_shuffle_composite_by_default(self):
        """export_fn with default table — pixel_shuffle becomes a composite op."""
        model = _make_pixel_shuffle_model()
        coreai_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(torch.randn(1, 16, 4, 4),)
                ).run_decompositions(get_decomp_table()),
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: composite_decl = #coreai.composite_declaration<"pixel_shuffle"
            """,
        )

    @pytest.mark.ir
    def test_add_pytorch_module_sdpa_composite_by_default(self):
        """add_pytorch_module applies the default table — sdpa becomes a composite op."""
        model = _make_sdpa_model()
        q = torch.randn(1, 2, 4, 8)
        coreai_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(q, q, q)
                ).run_decompositions(get_decomp_table()),
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention"
            """,
        )

    # --- explicit full decomp table overrides the default ---

    def test_add_pytorch_module_full_table_decomposes_instance_norm(self):
        """Using torch.export.default_decompositions() in export_fn decomposes instance_norm.

        The full table decomposes instance_norm into lower-level primitives that are not
        composite ops.  Some of those primitives (e.g. _native_batch_norm_legit) are not
        yet supported by the converter, so validation correctly rejects them.
        """
        model = _make_instance_norm_model()
        converter = TorchConverter()
        with pytest.raises(ValueError, match="unsupported ATen ops"):
            converter.add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(torch.randn(1, 4, 8, 8),)
                ).run_decompositions(torch.export.default_decompositions()),
            )

    def test_add_pytorch_module_full_table_decomposes_pixel_shuffle(self):
        """Using torch.export.default_decompositions() in export_fn decomposes pixel_shuffle.

        Same rationale as instance_norm: we verify at the FX graph level.
        """
        model = _make_pixel_shuffle_model()
        converter = TorchConverter()
        converter.add_pytorch_module(
            model,
            export_fn=lambda m: torch.export.export(
                m, args=(torch.randn(1, 16, 4, 4),)
            ).run_decompositions(torch.export.default_decompositions()),
        )
        ep = converter._staged[0].exported_program
        op_targets = {
            node.target for node in ep.graph.nodes if node.op == "call_function"
        }
        assert torch.ops.aten.pixel_shuffle.default not in op_targets
