# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Stateful IO tests — IR-level (filecheck) and runtime functional.

This file consolidates all tests related to stateful inputs/outputs:
- Mutable buffer annotations (IR-level)
- Mutable user input annotations (IR-level)
- Combined buffer + user input annotations (IR-level)
- Input/output name validation with stateful models
- state_names parameter behavior
- Runtime functional tests for buffer mutations, user input mutations,
  combined mutations, dtype coverage, default names, dynamic shapes,
  and corner cases.
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from coreai_torch import TorchConverter, get_decomp_table

from .utils import (
    filecheck_pattern,
    make_dynamic_shapes,
    validate_numerical_output,
)

# ===========================================================================
# IR-level stateful tests
# ===========================================================================


# ---------------------------------------------------------------------------
# TestMutableBufferAnnotation
# ---------------------------------------------------------------------------


class TestMutableBufferAnnotation:
    """MutableBuffers.buffer_mutation is annotated on mutable-buffer inputs.

    Regression coverage: the annotation must survive ``to_coreai(input_names=...)``.
    Before the fix the annotation loop used the user-visible ``graph_input_names``
    for the ``inputs_to_buffers`` lookup (keyed by original FX placeholder names),
    silently dropping the annotation whenever the caller supplied custom
    ``input_names``.
    """

    @staticmethod
    def _mutable_buffer_program():
        """One user input + one mutable buffer; populates both
        ``inputs_to_buffers`` and ``buffers_to_mutate`` in the graph signature.
        """

        class _BufMutate(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(1, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.state.copy_(x)
                return self.state

        x = torch.rand(1, 4, dtype=torch.float32)
        return torch.export.export(_BufMutate(), args=(x,)).run_decompositions()

    @staticmethod
    def _input_count(program) -> int:
        """Number of graph-level inputs seen by TorchConverter
        (user inputs + mutable buffer inputs).
        """
        sig = program.graph_signature
        mutable_bufs = {
            k
            for k, v in sig.inputs_to_buffers.items()
            if v in sig.buffers_to_mutate.values()
        }
        return len(sig.user_inputs) + len(mutable_bufs)

    @pytest.mark.ir
    def test_annotation_added_with_default_names(self) -> None:
        """MutableBuffers.buffer_mutation appears in the IR when no custom names."""
        result = (
            TorchConverter()
            .add_exported_program(self._mutable_buffer_program())
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<1x4xf32> {MutableBuffers.buffer_mutation = "b_state", coreai.name = "b_state"}, %{{.*}}: tensor<1x4xf32> {coreai.name = "x"}) -> (tensor<1x4xf32> {coreai.name = "b_state"}) {
                // CHECK:     coreai.output %{{.*}} : tensor<1x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    def test_annotation_added_with_custom_names(self) -> None:
        """Regression: annotation survives when the caller renames all inputs.

        Before the fix, the loop used the renamed ``graph_input_names`` for the
        ``inputs_to_buffers`` lookup.  Because that dict is keyed by the original
        FX placeholder names, the lookup always missed when names were renamed,
        so ``MutableBuffers.buffer_mutation`` was never written to the IR.
        """
        program = self._mutable_buffer_program()
        result = (
            TorchConverter()
            .add_exported_program(
                program, state_names=["custom_buf"], input_names=["custom_x"]
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<1x4xf32> {MutableBuffers.buffer_mutation = "custom_buf", coreai.name = "custom_buf"}, %{{.*}}: tensor<1x4xf32> {coreai.name = "custom_x"}) -> (tensor<1x4xf32> {coreai.name = "custom_buf"}) {
                // CHECK:     coreai.output %{{.*}} : tensor<1x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    def test_non_buffer_inputs_have_no_annotation(self) -> None:
        """Regular user inputs on a buffer-free model must not carry the annotation."""

        class _Add(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x + y

        t = torch.rand(2, 3, dtype=torch.float32)
        program = torch.export.export(_Add(), args=(t, t + 1)).run_decompositions()
        result = TorchConverter().add_exported_program(program).to_coreai()
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x3xf32> {coreai.name = "x"}, %{{.*}}: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "add"}) {
                // CHECK-NOT: MutableBuffers.buffer_mutation
                // CHECK:     coreai.decomposable.broadcasting_add
                // CHECK:     coreai.output %{{.*}} : tensor<2x3xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    def test_buffer_mutation_input_output_names(self) -> None:
        """Buffer mutation IR carries correct input/output names and mutation attr."""

        class _BufAdd(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.state.add_(x)
                return self.state + 1

        ep = torch.export.export(
            _BufAdd().eval(), args=(torch.rand(2, 4),)
        ).run_decompositions()
        result = TorchConverter().add_exported_program(ep).to_coreai()
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "b_state", coreai.name = "b_state"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "x"}) -> (tensor<2x4xf32> {coreai.name = "b_state"}, tensor<2x4xf32> {coreai.name = "add_1"}) {
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     coreai.output %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    def test_buffer_mutation_custom_input_output_names(self) -> None:
        """Buffer mutation attr survives custom input/output names."""

        class _BufAdd(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.state.add_(x)
                return self.state + 1

        ep = torch.export.export(
            _BufAdd().eval(), args=(torch.rand(2, 4),)
        ).run_decompositions()
        result = (
            TorchConverter()
            .add_exported_program(
                ep,
                state_names=["my_buf"],
                input_names=["my_x"],
                output_names=["result"],
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "my_buf", coreai.name = "my_buf"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "my_x"}) -> (tensor<2x4xf32> {coreai.name = "my_buf"}, tensor<2x4xf32> {coreai.name = "result"}) {
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     coreai.output %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    async def test_buffer_mutation_after_make_stateful(self) -> None:
        """After make_stateful, buffer input becomes handle type and output becomes token."""
        result = (
            TorchConverter()
            .add_exported_program(self._mutable_buffer_program())
            .to_coreai()
        )
        result.optimize()
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: !coreai.handle<tensor<1x4xf32>> {MutableBuffers.buffer_mutation = "b_state", coreai.name = "b_state"}, %{{.*}}: tensor<1x4xf32> {coreai.name = "x"}) -> (!coreai.token {coreai.name = "b_state"}) attributes {__coreai_pure__} {
                // CHECK:     %{{.*}} = coreai.create_token : !coreai.token
                // CHECK:     %{{.*}}, %{{.*}} = coreai.read_handle %{{.*}}, %{{.*}} : (<tensor<1x4xf32>>, !coreai.token) -> (tensor<1x4xf32>, !coreai.token)
                // CHECK:     %{{.*}} = coreai.write_handle %{{.*}}, %{{.*}}, %{{.*}} : (<tensor<1x4xf32>>, tensor<1x4xf32>, !coreai.token) -> !coreai.token
                // CHECK:     coreai.output %{{.*}} : !coreai.token
                // CHECK:   }
                // CHECK: }
            """,
        )


# ---------------------------------------------------------------------------
# TestMutableUserInputAnnotation
# ---------------------------------------------------------------------------


class TestMutableUserInputAnnotation:
    """MutableBuffers.buffer_mutation is annotated on mutated user inputs.

    When a user input is mutated in-place (e.g. ``x.mul_(5)``), the graph
    signature populates ``user_inputs_to_mutate``.  The converter must emit
    ``MutableBuffers.buffer_mutation`` on the corresponding input.
    """

    @staticmethod
    def _mutating_program():
        """Model that mutates user input ``x`` in-place."""

        class _MulInplace(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                x.mul_(5)
                return y + 1

        t = torch.rand(2, 4, dtype=torch.float32)
        return torch.export.export(
            _MulInplace(), args=(t, t.clone())
        ).run_decompositions()

    @pytest.mark.ir
    def test_annotation_on_mutated_user_input(self) -> None:
        """MutableBuffers.buffer_mutation appears for a mutated user input."""
        result = (
            TorchConverter().add_exported_program(self._mutating_program()).to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "x", coreai.name = "x"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "y"}) -> (tensor<2x4xf32> {coreai.name = "x"}, tensor<2x4xf32> {coreai.name = "add"}) {
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_mul %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     coreai.output %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    def test_annotation_survives_custom_input_names(self) -> None:
        """Annotation persists when the caller supplies custom input_names."""
        program = self._mutating_program()
        result = (
            TorchConverter()
            .add_exported_program(
                program, state_names=["x_state"], input_names=["my_y"]
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "x_state", coreai.name = "x_state"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "my_y"}) -> (tensor<2x4xf32> {coreai.name = "x_state"}, tensor<2x4xf32> {coreai.name = "add"}) {
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_mul %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     coreai.output %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    def test_non_mutated_input_has_no_annotation(self) -> None:
        """A model with no in-place mutation must not carry the annotation."""

        class _NoMutation(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x * 5

        t = torch.rand(2, 4, dtype=torch.float32)
        program = torch.export.export(_NoMutation(), args=(t,)).run_decompositions()
        result = TorchConverter().add_exported_program(program).to_coreai()
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x4xf32> {coreai.name = "x"}) -> (tensor<2x4xf32> {coreai.name = "mul"}) {
                // CHECK-NOT: MutableBuffers.buffer_mutation
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_mul %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     coreai.output %{{.*}} : tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    def test_user_input_mutation_input_output_names(self) -> None:
        """User input mutation IR carries correct input/output names and attr."""
        result = (
            TorchConverter().add_exported_program(self._mutating_program()).to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "x", coreai.name = "x"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "y"}) -> (tensor<2x4xf32> {coreai.name = "x"}, tensor<2x4xf32> {coreai.name = "add"}) {
                // CHECK:     %{{.*}} = coreai.constant dense<5> : tensor<si32>
                // CHECK:     %{{.*}} = coreai.cast %{{.*}} : tensor<si32> to tensor<f32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_mul %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.constant dense<1> : tensor<si32>
                // CHECK:     %{{.*}} = coreai.cast %{{.*}} : tensor<si32> to tensor<f32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     coreai.output %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    def test_user_input_mutation_custom_input_output_names(self) -> None:
        """User input mutation attr survives custom input/output names."""
        program = self._mutating_program()
        result = (
            TorchConverter()
            .add_exported_program(
                program,
                state_names=["x_state"],
                input_names=["my_y"],
                output_names=["result"],
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "x_state", coreai.name = "x_state"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "my_y"}) -> (tensor<2x4xf32> {coreai.name = "x_state"}, tensor<2x4xf32> {coreai.name = "result"}) {
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_mul %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     coreai.output %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    async def test_user_input_mutation_after_make_stateful(self) -> None:
        """After make_stateful, mutated user input becomes handle type and output becomes token."""
        result = (
            TorchConverter().add_exported_program(self._mutating_program()).to_coreai()
        )
        result.optimize()
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: !coreai.handle<tensor<2x4xf32>> {MutableBuffers.buffer_mutation = "x", coreai.name = "x"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "y"}) -> (!coreai.token {coreai.name = "x"}, tensor<2x4xf32> {coreai.name = "add"}) attributes {__coreai_pure__} {
                // CHECK:     %{{.*}} = coreai.create_token : !coreai.token
                // CHECK:     %{{.*}}, %{{.*}} = coreai.read_handle %{{.*}}, %{{.*}} : (<tensor<2x4xf32>>, !coreai.token) -> (tensor<2x4xf32>, !coreai.token)
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_mul %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.write_handle %{{.*}}, %{{.*}}, %{{.*}} : (<tensor<2x4xf32>>, tensor<2x4xf32>, !coreai.token) -> !coreai.token
                // CHECK:     coreai.output %{{.*}}, %{{.*}} : !coreai.token, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )


# ---------------------------------------------------------------------------
# TestMutableBufferAndUserInputAnnotation
# ---------------------------------------------------------------------------


class TestMutableBufferAndUserInputAnnotation:
    """Tests for models with both buffer mutations and user input mutations."""

    @staticmethod
    def _both_mutations_program():
        class _BothMutate(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                self.state.add_(x)
                y.mul_(2)
                return self.state + y

        return torch.export.export(
            _BothMutate().eval(),
            args=(torch.rand(2, 4), torch.rand(2, 4)),
        ).run_decompositions()

    def test_both_annotations_present(self) -> None:
        """Both buffer and user input mutations produce annotations."""
        result = (
            TorchConverter()
            .add_exported_program(self._both_mutations_program())
            .to_coreai()
        )
        ir = str(result)
        assert ir.count("MutableBuffers.buffer_mutation") == 2

    @pytest.mark.ir
    def test_both_mutations_input_output_names(self) -> None:
        """Both mutation annotations carry correct input/output names."""
        result = (
            TorchConverter()
            .add_exported_program(self._both_mutations_program())
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "b_state", coreai.name = "b_state"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "x"}, %{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "y", coreai.name = "y"}) -> (tensor<2x4xf32> {coreai.name = "b_state"}, tensor<2x4xf32> {coreai.name = "y"}, tensor<2x4xf32> {coreai.name = "add_1"}) {
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_mul %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
                // CHECK:     coreai.output %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    def test_both_mutations_custom_names(self) -> None:
        """Both mutation annotations survive custom input/output names."""
        result = (
            TorchConverter()
            .add_exported_program(
                self._both_mutations_program(),
                state_names=["my_buf", "my_y"],
                input_names=["my_x"],
                output_names=["result"],
            )
            .to_coreai()
        )
        ir = str(result)
        assert ir.count("MutableBuffers.buffer_mutation") == 2
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "my_buf", coreai.name = "my_buf"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "my_x"}, %{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "my_y", coreai.name = "my_y"}) -> (tensor<2x4xf32> {coreai.name = "my_buf"}, tensor<2x4xf32> {coreai.name = "my_y"}, tensor<2x4xf32> {coreai.name = "result"}) {
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_mul %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
                // CHECK:     coreai.output %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    async def test_both_mutations_after_make_stateful(self) -> None:
        """After make_stateful, both mutated inputs become handles and outputs become tokens."""
        result = (
            TorchConverter()
            .add_exported_program(self._both_mutations_program())
            .to_coreai()
        )
        result.optimize()
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: !coreai.handle<tensor<2x4xf32>> {MutableBuffers.buffer_mutation = "b_state", coreai.name = "b_state"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "x"}, %{{.*}}: !coreai.handle<tensor<2x4xf32>> {MutableBuffers.buffer_mutation = "y", coreai.name = "y"}) -> (!coreai.token {coreai.name = "b_state"}, !coreai.token {coreai.name = "y"}, tensor<2x4xf32> {coreai.name = "add_1"}) attributes {__coreai_pure__} {
                // CHECK:     %{{.*}} = coreai.create_token : !coreai.token
                // CHECK:     %{{.*}}, %{{.*}} = coreai.read_handle %{{.*}}, %{{.*}} : (<tensor<2x4xf32>>, !coreai.token) -> (tensor<2x4xf32>, !coreai.token)
                // CHECK:     %{{.*}}, %{{.*}} = coreai.read_handle %{{.*}}, %{{.*}} : (<tensor<2x4xf32>>, !coreai.token) -> (tensor<2x4xf32>, !coreai.token)
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.write_handle %{{.*}}, %{{.*}}, %{{.*}} : (<tensor<2x4xf32>>, tensor<2x4xf32>, !coreai.token) -> !coreai.token
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_mul %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                // CHECK:     %{{.*}} = coreai.write_handle %{{.*}}, %{{.*}}, %{{.*}} : (<tensor<2x4xf32>>, tensor<2x4xf32>, !coreai.token) -> !coreai.token
                // CHECK:     %{{.*}} = coreai.decomposable.broadcasting_add %{{.*}}, %{{.*}} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
                // CHECK:     coreai.output %{{.*}}, %{{.*}}, %{{.*}} : !coreai.token, !coreai.token, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )


# ---------------------------------------------------------------------------
# TestInputOutputNameValidation
# ---------------------------------------------------------------------------


class TestInputOutputNameValidation:
    """Validation of user-provided input_names / output_names counts.

    The converter must raise ``ValueError`` when the number of user-provided
    names does not match the number of live graph inputs or outputs.
    Mutations (buffer and user input) affect both counts.
    """

    def test_input_names_too_few(self) -> None:
        class _Add(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x + y

        ep = torch.export.export(
            _Add(), args=(torch.rand(2, 4), torch.rand(2, 4))
        ).run_decompositions()
        with pytest.raises(ValueError, match="live inputs"):
            TorchConverter().add_exported_program(ep, input_names=["a"]).to_coreai()

    def test_input_names_too_many(self) -> None:
        class _Add(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x + y

        ep = torch.export.export(
            _Add(), args=(torch.rand(2, 4), torch.rand(2, 4))
        ).run_decompositions()
        with pytest.raises(ValueError, match="live inputs"):
            TorchConverter().add_exported_program(
                ep, input_names=["a", "b", "c"]
            ).to_coreai()

    def test_output_names_too_few(self) -> None:
        class _Add(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x + y

        ep = torch.export.export(
            _Add(), args=(torch.rand(2, 4), torch.rand(2, 4))
        ).run_decompositions()
        # Graph has 1 output, providing 2 should fail
        with pytest.raises(ValueError, match="live outputs"):
            TorchConverter().add_exported_program(
                ep, output_names=["a", "b"]
            ).to_coreai()

    def test_output_names_too_many(self) -> None:
        class _Add(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x + y

        ep = torch.export.export(
            _Add(), args=(torch.rand(2, 4), torch.rand(2, 4))
        ).run_decompositions()
        with pytest.raises(ValueError, match="live outputs"):
            TorchConverter().add_exported_program(
                ep, output_names=["a", "b", "c"]
            ).to_coreai()

    def test_input_names_mismatch_with_buffer_mutation(self) -> None:
        """Buffer mutation makes the buffer a stateful input; input_names covers only non-state."""

        class _BufMut(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.state.add_(x)
                return self.state + 1

        ep = torch.export.export(
            _BufMut().eval(), args=(torch.rand(2, 4),)
        ).run_decompositions()
        # 1 non-state live input (x); providing 2 should fail
        with pytest.raises(ValueError, match="live inputs"):
            TorchConverter().add_exported_program(
                ep, input_names=["a", "b"]
            ).to_coreai()
        # Correct count (1 non-state input) succeeds
        dp = TorchConverter().add_exported_program(ep, input_names=["my_x"]).to_coreai()
        ir = str(dp)
        assert 'coreai.name = "my_x"' in ir
        # Use state_names to also rename the buffer
        dp2 = (
            TorchConverter()
            .add_exported_program(ep, state_names=["my_buf"], input_names=["my_x"])
            .to_coreai()
        )
        ir2 = str(dp2)
        assert 'coreai.name = "my_buf"' in ir2
        assert 'coreai.name = "my_x"' in ir2

    def test_output_names_mismatch_with_buffer_mutation(self) -> None:
        """Buffer mutation output is handled by state_names; output_names covers only non-state."""

        class _BufMut(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.state.add_(x)
                return self.state + 1

        ep = torch.export.export(
            _BufMut().eval(), args=(torch.rand(2, 4),)
        ).run_decompositions()
        # 1 non-state output; providing 2 should fail
        with pytest.raises(ValueError, match="live outputs"):
            TorchConverter().add_exported_program(
                ep, output_names=["a", "b"]
            ).to_coreai()
        # Correct count (1 non-state output) succeeds
        dp = (
            TorchConverter()
            .add_exported_program(ep, output_names=["result"])
            .to_coreai()
        )
        ir = str(dp)
        assert 'coreai.name = "result"' in ir

    def test_input_names_mismatch_with_user_input_mutation(self) -> None:
        """User input mutation makes the mutated input stateful; input_names covers only non-state."""

        class _MulInplace(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                x.mul_(5)
                return y + 1

        ep = torch.export.export(
            _MulInplace(), args=(torch.rand(2, 4), torch.rand(2, 4))
        ).run_decompositions()
        # 1 non-state live input (y); providing 2 should fail
        with pytest.raises(ValueError, match="live inputs"):
            TorchConverter().add_exported_program(
                ep, input_names=["a", "b"]
            ).to_coreai()
        # Correct count (1 non-state input) succeeds
        dp = TorchConverter().add_exported_program(ep, input_names=["my_y"]).to_coreai()
        ir = str(dp)
        assert 'coreai.name = "my_y"' in ir

    def test_output_names_mismatch_with_user_input_mutation(self) -> None:
        """User input mutation output is handled by state_names; output_names covers only non-state."""

        class _MulInplace(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                x.mul_(5)
                return y + 1

        ep = torch.export.export(
            _MulInplace(), args=(torch.rand(2, 4), torch.rand(2, 4))
        ).run_decompositions()
        # 1 non-state output (add=user output); providing 2 should fail
        with pytest.raises(ValueError, match="live outputs"):
            TorchConverter().add_exported_program(
                ep, output_names=["a", "b"]
            ).to_coreai()
        # Correct count (1 non-state output) succeeds
        dp = (
            TorchConverter()
            .add_exported_program(ep, output_names=["result"])
            .to_coreai()
        )
        ir = str(dp)
        assert 'coreai.name = "result"' in ir

    def test_names_mismatch_with_both_mutations(self) -> None:
        """Combined buffer + user input mutation: 1 non-state input, 1 non-state output."""

        class _BothMut(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                self.state.add_(x)
                y.mul_(2)
                return self.state + y

        ep = torch.export.export(
            _BothMut().eval(), args=(torch.rand(2, 4), torch.rand(2, 4))
        ).run_decompositions()
        # input_names covers only non-state inputs (x); providing 2 should fail
        with pytest.raises(ValueError, match="live inputs"):
            TorchConverter().add_exported_program(
                ep, input_names=["a", "b"]
            ).to_coreai()
        # output_names covers only non-state outputs (add_1); providing 2 should fail
        with pytest.raises(ValueError, match="live outputs"):
            TorchConverter().add_exported_program(
                ep, output_names=["a", "b"]
            ).to_coreai()
        # Correct: state_names for 2 states, input_names for 1 non-state, output_names for 1 non-state
        dp = (
            TorchConverter()
            .add_exported_program(
                ep,
                state_names=["my_buf", "my_y"],
                input_names=["my_x"],
                output_names=["result"],
            )
            .to_coreai()
        )
        ir = str(dp)
        for name in ["my_buf", "my_x", "my_y", "result"]:
            assert f'coreai.name = "{name}"' in ir


# ---------------------------------------------------------------------------
# TestStateNames
# ---------------------------------------------------------------------------


class TestStateNames:
    """Tests for the state_names parameter in add_exported_program.

    When state_names is provided:
    - input_names covers only non-stateful user inputs
    - state_names provides a single name per state, applied to both the
      state input AND its corresponding mutation output
    - output_names covers only non-state outputs (actual return values)
    This matches the Core AI runtime API where states have one name.
    """

    @pytest.mark.ir
    def test_buffer_mutation_state_names(self) -> None:
        """state_names applies to both buffer input and mutation output."""

        class _BufMut(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.state.add_(x)
                return self.state + 1

        ep = torch.export.export(
            _BufMut().eval(), args=(torch.rand(2, 4),)
        ).run_decompositions()
        result = (
            TorchConverter()
            .add_exported_program(
                ep,
                input_names=["my_x"],
                state_names=["kv_cache"],
                output_names=["result"],
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "kv_cache", coreai.name = "kv_cache"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "my_x"}) -> (tensor<2x4xf32> {coreai.name = "kv_cache"}, tensor<2x4xf32> {coreai.name = "result"}) {
                // CHECK:     coreai.output %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    def test_user_input_mutation_state_names(self) -> None:
        """state_names applies to both mutated user input and mutation output."""

        class _MulInplace(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                x.mul_(5)
                return y + 1

        t = torch.rand(2, 4, dtype=torch.float32)
        ep = torch.export.export(
            _MulInplace(), args=(t, t.clone())
        ).run_decompositions()
        result = (
            TorchConverter()
            .add_exported_program(
                ep,
                input_names=["my_y"],
                state_names=["x_state"],
                output_names=["result"],
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%{{.*}}: tensor<2x4xf32> {MutableBuffers.buffer_mutation = "x_state", coreai.name = "x_state"}, %{{.*}}: tensor<2x4xf32> {coreai.name = "my_y"}) -> (tensor<2x4xf32> {coreai.name = "x_state"}, tensor<2x4xf32> {coreai.name = "result"}) {
                // CHECK:     coreai.output %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xf32>
                // CHECK:   }
                // CHECK: }
            """,
        )

    @pytest.mark.ir
    def test_both_mutations_state_names(self) -> None:
        """state_names covers both mutable buffer and mutated user input."""

        class _BothMut(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                self.state.add_(x)
                y.mul_(2)
                return self.state + y

        ep = torch.export.export(
            _BothMut().eval(), args=(torch.rand(2, 4), torch.rand(2, 4))
        ).run_decompositions()
        result = (
            TorchConverter()
            .add_exported_program(
                ep,
                input_names=["my_x"],
                state_names=["kv_cache", "y_state"],
                output_names=["result"],
            )
            .to_coreai()
        )
        ir = str(result)
        # State inputs use state_names
        assert 'coreai.name = "kv_cache"' in ir
        assert 'coreai.name = "y_state"' in ir
        # Non-state input uses input_names
        assert 'coreai.name = "my_x"' in ir
        # State outputs also use state_names (same name for input and output)
        assert 'MutableBuffers.buffer_mutation = "kv_cache"' in ir
        assert 'MutableBuffers.buffer_mutation = "y_state"' in ir
        # Non-state output uses output_names
        assert 'coreai.name = "result"' in ir

    @pytest.mark.ir
    def test_state_names_defaults_from_module(self) -> None:
        """When state_names=[] (empty), defaults use FX placeholder names."""

        class _BufMut(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("kv_cache", torch.zeros(2, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.kv_cache.add_(x)
                return self.kv_cache + 1

        ep = torch.export.export(
            _BufMut().eval(), args=(torch.rand(2, 4),)
        ).run_decompositions()
        # state_names=[] triggers stateful mode with auto-derived defaults
        result = TorchConverter().add_exported_program(ep, state_names=[]).to_coreai()
        ir = str(result)
        # Default state name keeps FX placeholder name "b_kv_cache"
        assert 'coreai.name = "b_kv_cache"' in ir
        assert 'MutableBuffers.buffer_mutation = "b_kv_cache"' in ir
        # Non-state input keeps FX name "x"
        assert 'coreai.name = "x"' in ir

    @pytest.mark.ir
    def test_state_names_defaults_user_input_mutation(self) -> None:
        """Default state name for mutated user input is the forward arg name."""

        class _MulInplace(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                x.mul_(5)
                return y + 1

        t = torch.rand(2, 4, dtype=torch.float32)
        ep = torch.export.export(
            _MulInplace(), args=(t, t.clone())
        ).run_decompositions()
        result = TorchConverter().add_exported_program(ep, state_names=[]).to_coreai()
        ir = str(result)
        # Default state name is forward arg name "x"
        assert 'MutableBuffers.buffer_mutation = "x"' in ir
        # Non-state input "y" keeps its name
        assert 'coreai.name = "y"' in ir

    @pytest.mark.ir
    def test_state_names_without_input_or_output_names(self) -> None:
        """state_names alone renames states; other IO keeps FX defaults."""

        class _BufMut(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.state.add_(x)
                return self.state + 1

        ep = torch.export.export(
            _BufMut().eval(), args=(torch.rand(2, 4),)
        ).run_decompositions()
        result = (
            TorchConverter()
            .add_exported_program(ep, state_names=["my_state"])
            .to_coreai()
        )
        ir = str(result)
        # State input and output both use "my_state"
        assert 'coreai.name = "my_state"' in ir
        assert 'MutableBuffers.buffer_mutation = "my_state"' in ir
        # Non-state input keeps FX name
        assert 'coreai.name = "x"' in ir

    def test_state_names_count_mismatch(self) -> None:
        """Wrong number of state_names raises ValueError."""

        class _BufMut(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.state.add_(x)
                return self.state + 1

        ep = torch.export.export(
            _BufMut().eval(), args=(torch.rand(2, 4),)
        ).run_decompositions()
        with pytest.raises(ValueError, match="stateful inputs"):
            TorchConverter().add_exported_program(
                ep, state_names=["s1", "s2"]
            ).to_coreai()

    def test_input_names_count_mismatch_with_state_names(self) -> None:
        """Wrong number of input_names (when state_names is set) raises ValueError."""

        class _BufMut(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.state.add_(x)
                return self.state + 1

        ep = torch.export.export(
            _BufMut().eval(), args=(torch.rand(2, 4),)
        ).run_decompositions()
        with pytest.raises(ValueError, match="live inputs"):
            TorchConverter().add_exported_program(
                ep,
                input_names=["a", "b"],
                state_names=["my_state"],
            ).to_coreai()

    def test_output_names_count_mismatch_with_state_names(self) -> None:
        """output_names must match non-state output count in stateful mode."""

        class _BufMut(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.state.add_(x)
                return self.state + 1

        ep = torch.export.export(
            _BufMut().eval(), args=(torch.rand(2, 4),)
        ).run_decompositions()
        # There's 1 non-state output; providing 2 should fail
        with pytest.raises(ValueError, match="live outputs"):
            TorchConverter().add_exported_program(
                ep,
                state_names=["my_state"],
                output_names=["a", "b"],
            ).to_coreai()


# ===========================================================================
# Runtime functional stateful tests
# ===========================================================================


# ---------------------------------------------------------------------------
# Buffer mutation tests
# ---------------------------------------------------------------------------


class TestBufferMutationRuntime:
    """Runtime validation for register_buffer + in-place mutation."""

    async def test_buffer_add_accumulates(self) -> None:
        """Buffer accumulates values across calls."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("acc", torch.zeros(1, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.acc.add_(x)
                return self.acc + 1

        await validate_numerical_output(
            model=Model(),
            x=torch.ones(1, 4),
            state_names=["acc"],
            input_names=["x"],
            output_names=["result"],
            num_calls=2,
            atol=1e-5,
        )

    async def test_buffer_copy(self) -> None:
        """Buffer is overwritten (copy_) each call."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 3))

            def forward(self, x: Tensor) -> Tensor:
                self.cache.copy_(x)
                return self.cache * 2

        await validate_numerical_output(
            model=Model(),
            x=torch.randn(2, 3),
            state_names=["cache"],
            input_names=["x"],
            output_names=["result"],
            num_calls=2,
            atol=1e-5,
        )

    async def test_multiple_buffers(self) -> None:
        """Multiple buffers mutated independently."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("sum_buf", torch.zeros(4))
                self.register_buffer("count", torch.zeros(1, dtype=torch.int32))

            def forward(self, x: Tensor) -> Tensor:
                self.sum_buf.add_(x)
                self.count.add_(1)
                return self.sum_buf

        await validate_numerical_output(
            model=Model(),
            x=torch.ones(4),
            state_names=["sum_buf", "count"],
            input_names=["x"],
            num_calls=2,
            atol=1e-5,
        )

    async def test_buffer_mul_accumulate(self) -> None:
        """Buffer with multiplicative update."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.ones(1, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.scale.mul_(x)
                return self.scale

        await validate_numerical_output(
            model=Model(),
            x=torch.full((1, 4), 2.0),
            state_names=["scale"],
            input_names=["x"],
            num_calls=2,
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# User input mutation tests
# ---------------------------------------------------------------------------


class TestUserInputMutationRuntime:
    """Runtime validation for user input mutations (forward arg mutated in-place)."""

    async def test_user_input_mul_inplace(self) -> None:
        """Mutated user input is reflected in state output."""

        class Model(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                x.mul_(2)
                return x + y

        await validate_numerical_output(
            model=Model(),
            x=torch.ones(2, 4),
            y=torch.ones(2, 4),
            state_names=["x_state"],
            input_names=["y"],
            output_names=["result"],
            num_calls=2,
            atol=1e-5,
        )

    async def test_user_input_add_inplace(self) -> None:
        """User input add_ mutation."""

        class Model(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                y.add_(x)
                return y * 3

        await validate_numerical_output(
            model=Model(),
            x=torch.randn(3, 3),
            y=torch.randn(3, 3),
            state_names=["y_state"],
            input_names=["x"],
            output_names=["result"],
            num_calls=2,
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Combined buffer + user input mutation tests
# ---------------------------------------------------------------------------


class TestCombinedMutationsRuntime:
    """Runtime validation for models with both buffer and user input mutations."""

    async def test_both_buffer_and_user_mutation(self) -> None:
        """Both mutations tracked as separate states."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4))

            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                self.cache.add_(x)
                y.mul_(2)
                return self.cache + y

        await validate_numerical_output(
            model=Model(),
            x=torch.ones(2, 4),
            y=torch.ones(2, 4),
            state_names=["cache", "y_state"],
            input_names=["x"],
            output_names=["result"],
            num_calls=2,
            atol=1e-5,
        )

    async def test_kv_cache_pattern(self) -> None:
        """Realistic KV-cache pattern: accumulate keys/values."""

        class KVCacheModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("k_cache", torch.zeros(1, 4, 8))
                self.register_buffer("v_cache", torch.zeros(1, 4, 8))

            def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
                self.k_cache.copy_(k)
                self.v_cache.copy_(v)
                attn = torch.matmul(q, self.k_cache.transpose(-2, -1))
                return torch.matmul(attn, self.v_cache)

        await validate_numerical_output(
            model=KVCacheModel(),
            q=torch.randn(1, 4, 8),
            k=torch.randn(1, 4, 8),
            v=torch.randn(1, 4, 8),
            state_names=["k_cache", "v_cache"],
            input_names=["q", "k", "v"],
            output_names=["result"],
            num_calls=2,
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Dtype tests
# ---------------------------------------------------------------------------


class TestStatefulDtypes:
    """Verify state works across different dtypes."""

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.int32,
        ],
    )
    async def test_buffer_dtype(self, dtype: torch.dtype) -> None:
        """Buffer mutation works for various dtypes."""

        class Model(nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 4, dtype=dtype))

            def forward(self, x: Tensor) -> Tensor:
                self.state.add_(x)
                return self.state

        x = torch.ones(2, 4, dtype=dtype)
        atol = 1e-3 if dtype == torch.float16 else 1e-5

        await validate_numerical_output(
            model=Model(dtype),
            x=x,
            state_names=["state"],
            input_names=["x"],
            atol=atol,
            num_calls=2,
        )


# ---------------------------------------------------------------------------
# Default names (no explicit state_names)
# ---------------------------------------------------------------------------


class TestStatefulDefaultNames:
    """Verify the runtime works with default FX placeholder names (no overrides)."""

    async def test_buffer_default_names(self) -> None:
        """No state_names provided — uses FX defaults (e.g. "b_pos")."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("pos", torch.zeros(4))

            def forward(self, x: Tensor) -> Tensor:
                self.pos.add_(1)
                return x + self.pos

        # No state_names passed — runtime should use FX default "b_pos".
        # If the default name resolution were broken, _init_runtime_state
        # would fail to allocate state for "b_pos" and the run would error.
        await validate_numerical_output(
            model=Model(),
            x=torch.randn(4),
            num_calls=2,
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Dynamic shapes
# ---------------------------------------------------------------------------


class TestStatefulDynamicShapes:
    """Verify stateful models with dynamic shapes."""

    async def test_buffer_with_dynamic_input(self) -> None:
        """Buffer is static, input has dynamic batch dim.

        Verifies that a stateful model with a buffer mutation works when
        the user input has a dynamic dimension — buffer accumulates across
        calls and outputs match PyTorch reference.
        """

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.zeros(4))

            def forward(self, x: Tensor) -> Tensor:
                self.bias.add_(x.mean(dim=0))
                return x + self.bias

        model = Model().eval()
        x = torch.randn(2, 4)

        dynamic_shapes = make_dynamic_shapes(x=["batch", None])

        await validate_numerical_output(
            model=model,
            x=x,
            state_names=["bias"],
            input_names=["x"],
            output_names=["result"],
            dynamic_shapes=dynamic_shapes,
            num_calls=2,
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Corner cases
# ---------------------------------------------------------------------------


class TestStatefulCornerCases:
    """Edge cases for stateful models."""

    async def test_no_non_state_outputs(self) -> None:
        """Model where the only output IS the mutated state (no return value)."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x: Tensor) -> Tensor:
                self.state.add_(x)
                return self.state

        # Here "result" is the same value as the state — no non-state outputs
        await validate_numerical_output(
            model=Model(),
            x=torch.ones(4),
            state_names=["state"],
            input_names=["x"],
            num_calls=2,
            atol=1e-5,
        )

    async def test_state_only_no_user_inputs(self) -> None:
        """Model with buffer mutation but no user inputs (counter)."""

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("count", torch.zeros(1, dtype=torch.int32))

            def forward(self) -> Tensor:
                self.count.add_(1)
                return self.count

        # The model returns the state itself, so output comparison validates
        # that state accumulates correctly across calls.
        await validate_numerical_output(
            model=Counter(),
            state_names=["count"],
            num_calls=3,
            atol=0,
        )

    async def test_multiple_return_values(self) -> None:
        """Model returning multiple values alongside state."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                self.state.add_(x)
                return self.state * 2, self.state + 1

        await validate_numerical_output(
            model=Model(),
            x=torch.ones(4),
            state_names=["state"],
            input_names=["x"],
            output_names=["doubled", "plus_one"],
            num_calls=2,
            atol=1e-5,
        )

    async def test_scalar_buffer(self) -> None:
        """Scalar (0-dim) buffer state."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("step", torch.tensor(0.0))

            def forward(self, x: Tensor) -> Tensor:
                self.step.add_(0.1)
                return x * self.step

        await validate_numerical_output(
            model=Model(),
            x=torch.randn(3, 3),
            state_names=["step"],
            input_names=["x"],
            output_names=["result"],
            num_calls=5,
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# add_pytorch_module path
# ---------------------------------------------------------------------------


class TestAddPytorchModuleStateful:
    """Verify state_names works through the add_pytorch_module path."""

    @pytest.mark.ir
    def test_add_pytorch_module_with_state_names(self) -> None:
        """state_names wires correctly through add_pytorch_module."""

        class _BufMut(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4))

            def forward(self, x: Tensor) -> Tensor:
                self.cache.add_(x)
                return self.cache + 1

        model = _BufMut().eval()
        x = torch.rand(2, 4)

        result = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x,)
                ).run_decompositions(get_decomp_table()),
                state_names=["my_cache"],
                input_names=["my_x"],
                output_names=["result"],
            )
            .to_coreai()
        )
        ir = str(result)
        assert 'coreai.name = "my_cache"' in ir
        assert 'MutableBuffers.buffer_mutation = "my_cache"' in ir
        assert 'coreai.name = "my_x"' in ir
        assert 'coreai.name = "result"' in ir
