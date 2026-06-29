# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import operator

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from ..utils import _all_dims_dynamic, filecheck_pattern, get_ir


class TestUnaryOpsIR:
    @pytest.mark.parametrize(
        "op_fn, coreai_op",
        [
            (torch.abs, "coreai.abs"),
            (torch.acos, "coreai.acos"),
            (torch.acosh, "coreai.acosh"),
            (torch.asin, "coreai.asin"),
            (torch.asinh, "coreai.asinh"),
            (torch.atan, "coreai.atan"),
            (torch.atanh, "coreai.atanh"),
            (torch.cos, "coreai.cos"),
            (torch.cosh, "coreai.cosh"),
            (torch.erf, "coreai.erf"),
            (torch.exp, "coreai.exp"),
            (torch.log, "coreai.log"),
            (torch.round, "coreai.round"),
            (torch.rsqrt, "coreai.rsqrt"),
            (torch.sin, "coreai.sin"),
            (torch.sinh, "coreai.sinh"),
            (torch.sqrt, "coreai.sqrt"),
            (torch.tan, "coreai.tan"),
            (torch.tanh, "coreai.tanh"),
        ],
    )
    def test_static(self, op_fn, coreai_op) -> None:
        class UnaryModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return op_fn(x)

        ir = get_ir(UnaryModel().eval(), x=torch.rand(2, 3) + 1.0)
        filecheck_pattern(
            ir,
            check_file=f"""
                // CHECK-LABEL: module {{
                // CHECK-NEXT:    coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {{coreai.name = "x"}}) -> (tensor<2x3xf32> {{coreai.name = "{{{{.*}}}}"}}) attributes {{__coreai_pure__}} {{
                // CHECK-NEXT:      %[[R:.*]] = {coreai_op} %[[X]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:      coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:    }}
                // CHECK-NEXT:  }}
            """,
        )

    @pytest.mark.parametrize(
        "op_fn, coreai_op",
        [
            (torch.abs, "coreai.abs"),
            (torch.acos, "coreai.acos"),
            (torch.acosh, "coreai.acosh"),
            (torch.asin, "coreai.asin"),
            (torch.asinh, "coreai.asinh"),
            (torch.atan, "coreai.atan"),
            (torch.atanh, "coreai.atanh"),
            (torch.cos, "coreai.cos"),
            (torch.cosh, "coreai.cosh"),
            (torch.erf, "coreai.erf"),
            (torch.exp, "coreai.exp"),
            (torch.log, "coreai.log"),
            (torch.round, "coreai.round"),
            (torch.rsqrt, "coreai.rsqrt"),
            (torch.sin, "coreai.sin"),
            (torch.sinh, "coreai.sinh"),
            (torch.sqrt, "coreai.sqrt"),
            (torch.tan, "coreai.tan"),
            (torch.tanh, "coreai.tanh"),
        ],
    )
    def test_dynamic(self, op_fn, coreai_op) -> None:
        class UnaryModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return op_fn(x)

        x = torch.rand(2, 3) + 1.0
        ir = get_ir(
            UnaryModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file=f"""
                // CHECK-LABEL: module {{
                // CHECK-NEXT:    coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {{coreai.name = "x"}}) -> (tensor<?x?xf32> {{coreai.name = "{{{{.*}}}}"}}) attributes {{__coreai_pure__}} {{
                // CHECK-NEXT:      %[[R:.*]] = {coreai_op} %[[X]] : tensor<?x?xf32> -> tensor<?x?xf32>
                // CHECK-NEXT:      coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:    }}
                // CHECK-NEXT:  }}
            """,
        )

    def test_int32(self) -> None:
        class AbsModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.abs(x)

        ir = get_ir(AbsModel().eval(), x=torch.tensor([-5, 0, 3], dtype=torch.int32))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<3xsi32> {coreai.name = "x"}) -> (tensor<3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.abs %[[ARG0]] : tensor<3xsi32> -> tensor<3xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestBinaryOpsIR:
    @pytest.mark.parametrize(
        "op_fn, coreai_op",
        [
            (torch.add, "coreai.decomposable.broadcasting_add"),
        ],
    )
    def test_static_tensor(self, op_fn, coreai_op) -> None:
        class BinaryModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return op_fn(x, y)

        ir = get_ir(BinaryModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file=f"""
                // CHECK-LABEL: module {{
                // CHECK-NEXT:    coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {{coreai.name = "x"}}, %[[Y:.*]]: tensor<2x3xf32> {{coreai.name = "y"}}) -> (tensor<2x3xf32> {{coreai.name = "{{{{.*}}}}"}}) attributes {{__coreai_pure__}} {{
                // CHECK-NEXT:      %[[R:.*]] = {coreai_op} %[[X]], %[[Y]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:      coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:    }}
                // CHECK-NEXT:  }}
            """,
        )

    @pytest.mark.parametrize(
        "op_fn, coreai_op",
        [
            (torch.add, "coreai.decomposable.broadcasting_add"),
        ],
    )
    def test_dynamic_tensor(self, op_fn, coreai_op) -> None:
        class BinaryModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return op_fn(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        ir = get_ir(
            BinaryModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": _all_dims_dynamic(x), "y": _all_dims_dynamic(y)},
        )
        filecheck_pattern(
            ir,
            check_file=f"""
                // CHECK-LABEL: module {{
                // CHECK-NEXT:    coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {{coreai.name = "x"}}, %[[Y:.*]]: tensor<?x?xf32> {{coreai.name = "y"}}) -> (tensor<?x?xf32> {{coreai.name = "{{{{.*}}}}"}}) attributes {{__coreai_pure__}} {{
                // CHECK-NEXT:      %[[R:.*]] = {coreai_op} %[[X]], %[[Y]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:      coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:    }}
                // CHECK-NEXT:  }}
            """,
        )

    def test_broadcast(self) -> None:
        class AddBroadcastModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.add(x, y)

        ir = get_ir(AddBroadcastModel().eval(), x=torch.rand(2, 3), y=torch.rand(3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_add %[[ARG0]], %[[ARG1]] : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    @pytest.mark.parametrize(
        "op_fn, coreai_op",
        [
            (operator.add, "coreai.decomposable.broadcasting_add"),
        ],
    )
    def test_static_scalar(self, op_fn, coreai_op) -> None:
        class ScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return op_fn(x, 2.0)

        ir = get_ir(ScalarModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file=f"""
                // CHECK-LABEL: module {{
                // CHECK-NEXT:    coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {{coreai.name = "x"}}) -> (tensor<2x3xf32> {{coreai.name = "{{{{.*}}}}"}}) attributes {{__coreai_pure__}} {{
                // CHECK-NEXT:      %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:      %[[R:.*]] = {coreai_op} %[[X]], %[[C]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:      coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:    }}
                // CHECK-NEXT:  }}
            """,
        )

    @pytest.mark.parametrize(
        "op_fn, coreai_op",
        [
            (operator.add, "coreai.decomposable.broadcasting_add"),
        ],
    )
    def test_dynamic_scalar(self, op_fn, coreai_op) -> None:
        class ScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return op_fn(x, 2.0)

        x = torch.rand(2, 3)
        ir = get_ir(
            ScalarModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file=f"""
                // CHECK-LABEL: module {{
                // CHECK-NEXT:    coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {{coreai.name = "x"}}) -> (tensor<?x?xf32> {{coreai.name = "{{{{.*}}}}"}}) attributes {{__coreai_pure__}} {{
                // CHECK-NEXT:      %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:      %[[R:.*]] = {coreai_op} %[[X]], %[[C]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:      coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:    }}
                // CHECK-NEXT:  }}
            """,
        )


class TestAddmmIR:
    def test_static(self) -> None:
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)

            def forward(self, x: Tensor) -> Tensor:
                return self.linear(x)

        ir = get_ir(LinearModel().eval(), x=torch.rand(2, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x4xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<3x4xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<{{.*}}> : tensor<3xf32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<[1, 0]> : tensor<2xui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.transpose %[[V0]], %[[V2]] : (tensor<3x4xf32>, tensor<2xui32>) -> tensor<4x3xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.decomposable.broadcasting_batch_matmul %[[ARG0]], %[[V3]] : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.decomposable.broadcasting_add %[[V4]], %[[V1]] : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V5]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)

            def forward(self, x: Tensor) -> Tensor:
                return self.linear(x)

        x = torch.rand(2, 4)
        ir = get_ir(
            LinearModel().eval(),
            x=x,
            dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x4xf32> {coreai.name = "x"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<3x4xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<{{.*}}> : tensor<3xf32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<[1, 0]> : tensor<2xui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.transpose %[[V0]], %[[V2]] : (tensor<3x4xf32>, tensor<2xui32>) -> tensor<4x3xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.decomposable.broadcasting_batch_matmul %[[ARG0]], %[[V3]] : (tensor<?x4xf32>, tensor<4x3xf32>) -> tensor<?x3xf32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.decomposable.broadcasting_add %[[V4]], %[[V1]] : (tensor<?x3xf32>, tensor<3xf32>) -> tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[V5]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestAliasIR:
    def test_static(self) -> None:
        class AliasModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.detach()

        ir = get_ir(AliasModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     coreai.output %[[ARG0]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class AliasModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.detach()

        x = torch.rand(2, 3)
        ir = get_ir(
            AliasModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     coreai.output %[[ARG0]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestAmaxAminIR:
    @pytest.mark.parametrize(
        "op_fn, coreai_op",
        [
            (torch.amax, "coreai.reduce_max"),
            (torch.amin, "coreai.reduce_min"),
        ],
    )
    def test_static(self, op_fn, coreai_op) -> None:
        class ReduceModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return op_fn(x, dim=1)

        ir = get_ir(ReduceModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file=f"""
                // CHECK-LABEL: module {{
                // CHECK-NEXT:    coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {{coreai.name = "x"}}) -> (tensor<2xf32> {{coreai.name = "{{{{.*}}}}"}}) attributes {{__coreai_pure__}} {{
                // CHECK-NEXT:      %[[SHAPE:.*]] = coreai.constant dense<2> : tensor<1xui32>
                // CHECK-NEXT:      %[[DIMS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:      %[[RED:.*]] = {coreai_op} %[[X]], %[[DIMS]] : (tensor<2x3xf32>, tensor<1xsi32>) -> tensor<2x1xf32>
                // CHECK-NEXT:      %[[R:.*]] = coreai.reshape %[[RED]], %[[SHAPE]] : (tensor<2x1xf32>, tensor<1xui32>) -> tensor<2xf32>
                // CHECK-NEXT:      coreai.output %[[R]] : tensor<2xf32>
                // CHECK-NEXT:    }}
                // CHECK-NEXT:  }}
            """,
        )

    @pytest.mark.parametrize(
        "op_fn, coreai_op",
        [
            (torch.amax, "coreai.reduce_max"),
            (torch.amin, "coreai.reduce_min"),
        ],
    )
    def test_dynamic(self, op_fn, coreai_op) -> None:
        class ReduceModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return op_fn(x, dim=1)

        x = torch.rand(2, 3)
        ir = get_ir(
            ReduceModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file=f"""
                // CHECK-LABEL: module {{
                // CHECK-NEXT:    coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {{coreai.name = "x"}}) -> (tensor<?xf32> {{coreai.name = "{{{{.*}}}}"}}) attributes {{__coreai_pure__}} {{
                // CHECK-NEXT:      %[[DIMS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:      %[[RED:.*]] = {coreai_op} %[[X]], %[[DIMS]] : (tensor<?x?xf32>, tensor<1xsi32>) -> tensor<?x1xf32>
                // CHECK-NEXT:      %[[R:.*]] = coreai.shrink_dims %[[RED]], %[[DIMS]] : (tensor<?x1xf32>, tensor<1xsi32>) to tensor<?xf32>
                // CHECK-NEXT:      coreai.output %[[R]] : tensor<?xf32>
                // CHECK-NEXT:    }}
                // CHECK-NEXT:  }}
            """,
        )

    @pytest.mark.parametrize(
        "op_fn, coreai_op",
        [
            (torch.amax, "coreai.reduce_max"),
            (torch.amin, "coreai.reduce_min"),
        ],
    )
    def test_keepdim(self, op_fn, coreai_op) -> None:
        class ReduceKeepDimModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return op_fn(x, dim=1, keepdim=True)

        ir = get_ir(ReduceKeepDimModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file=f"""
                // CHECK-LABEL: module {{
                // CHECK-NEXT:    coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {{coreai.name = "x"}}) -> (tensor<2x1xf32> {{coreai.name = "{{{{.*}}}}"}}) attributes {{__coreai_pure__}} {{
                // CHECK-NEXT:      %[[DIMS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:      %[[R:.*]] = {coreai_op} %[[X]], %[[DIMS]] : (tensor<2x3xf32>, tensor<1xsi32>) -> tensor<2x1xf32>
                // CHECK-NEXT:      coreai.output %[[R]] : tensor<2x1xf32>
                // CHECK-NEXT:    }}
                // CHECK-NEXT:  }}
            """,
        )


class TestAdaptiveAvgPool2dIR:
    def test_static_divisible(self) -> None:
        class AdaptiveAvgPool2dModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))

        ir = get_ir(AdaptiveAvgPool2dModel().eval(), x=torch.rand(1, 3, 8, 8))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x3x8x8xf32> {coreai.name = "x"}) -> (tensor<1x3x2x2xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1.600000e+01> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<2xui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<4> : tensor<2xui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.sum_pool_2d %[[ARG0]], %[[V2]], %[[V2]], %[[V1]] : (tensor<1x3x8x8xf32>, tensor<2xui32>, tensor<2xui32>, tensor<2xui32>) -> tensor<1x3x2x2xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.decomposable.broadcasting_divide %[[V3]], %[[V0]] : (tensor<1x3x2x2xf32>, tensor<f32>) -> tensor<1x3x2x2xf32>
                // CHECK-NEXT:     coreai.output %[[V4]] : tensor<1x3x2x2xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSoftmaxIR:
    def test_static(self) -> None:
        class SoftmaxModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.softmax(x, dim=-1)

        ir = get_ir(SoftmaxModel().eval(), x=torch.rand(2, 5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x5xf32> {coreai.name = "x"}) -> (tensor<2x5xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.softmax %[[ARG0]], %[[V0]] : (tensor<2x5xf32>, tensor<si32>) -> tensor<2x5xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x5xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class SoftmaxModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.softmax(x, dim=-1)

        x = torch.rand(2, 5)
        ir = get_ir(
            SoftmaxModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.softmax %[[ARG0]], %[[V0]] : (tensor<?x?xf32>, tensor<si32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLogSoftmaxIR:
    def test_static_composite(self) -> None:
        class LogSoftmaxModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.log_softmax(x, dim=-1)

        ir = get_ir(LogSoftmaxModel().eval(), x=torch.rand(2, 5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @log_softmax_{{.*}}(%[[ARG0:.*]]: tensor<2x5xf32> {coreai.name = "input"}) -> tensor<2x5xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"log_softmax" = {input_names = ["input"], op_attrs = {axis = 1 : si64, version = 1 : si64}, output_names = ["output"]}>, template_op = "log_softmax"} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.reduce_max %[[ARG0]], %[[V0]] : (tensor<2x5xf32>, tensor<1xsi32>) -> tensor<2x1xf32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.decomposable.broadcasting_sub %[[ARG0]], %[[V1]] : (tensor<2x5xf32>, tensor<2x1xf32>) -> tensor<2x5xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.exp %[[V2]] : tensor<2x5xf32> -> tensor<2x5xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.reduce_sum %[[V3]], %[[V0]] : (tensor<2x5xf32>, tensor<1xsi32>) -> tensor<2x1xf32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.log %[[V4]] : tensor<2x1xf32> -> tensor<2x1xf32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.decomposable.broadcasting_sub %[[V2]], %[[V5]] : (tensor<2x5xf32>, tensor<2x1xf32>) -> tensor<2x5xf32>
                // CHECK-NEXT:     coreai.output %[[V6]] : tensor<2x5xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0]]: tensor<2x5xf32> {coreai.name = "x"}) -> (tensor<2x5xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0]] = coreai.invoke @log_softmax_{{.*}}(%[[ARG0]])  : (tensor<2x5xf32>) -> tensor<2x5xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x5xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic_composite(self) -> None:
        class LogSoftmaxModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.log_softmax(x, dim=-1)

        x = torch.rand(2, 5)
        ir = get_ir(
            LogSoftmaxModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @log_softmax_{{.*}}(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "input"}) -> tensor<?x?xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"log_softmax" = {input_names = ["input"], op_attrs = {axis = 1 : si64, version = 1 : si64}, output_names = ["output"]}>, template_op = "log_softmax"} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.reduce_max %[[ARG0]], %[[V0]] : (tensor<?x?xf32>, tensor<1xsi32>) -> tensor<?x1xf32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.decomposable.broadcasting_sub %[[ARG0]], %[[V1]] : (tensor<?x?xf32>, tensor<?x1xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.exp %[[V2]] : tensor<?x?xf32> -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.reduce_sum %[[V3]], %[[V0]] : (tensor<?x?xf32>, tensor<1xsi32>) -> tensor<?x1xf32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.log %[[V4]] : tensor<?x1xf32> -> tensor<?x1xf32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.decomposable.broadcasting_sub %[[V2]], %[[V5]] : (tensor<?x?xf32>, tensor<?x1xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V6]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0]] = coreai.invoke @log_softmax_{{.*}}(%[[ARG0]])  : (tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestBatchNormIR:
    def test_static_composite(self) -> None:
        class BatchNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(3)

            def forward(self, x: Tensor) -> Tensor:
                return self.bn(x)

        ir = get_ir(BatchNormModel().eval(), x=torch.rand(1, 3, 4, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @batch_norm_{{.*}}(%[[ARG0:.*]]: tensor<1x3x4x4xf32> {coreai.name = "input"}, %[[ARG1:.*]]: tensor<3xf32> {coreai.name = "gamma"}, %[[ARG2:.*]]: tensor<3xf32> {coreai.name = "beta"}, %[[ARG3:.*]]: tensor<3xf32> {coreai.name = "mean"}, %[[ARG4:.*]]: tensor<3xf32> {coreai.name = "variance"}) -> tensor<1x3x4x4xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"batch_norm" = {input_names = ["input", "gamma", "beta", "mean", "variance"], op_attrs = {eps = 9.99999974E-6 : f32, version = 1 : si64}, output_names = ["output"]}>, template_op = "batch_norm"} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<9.99999974E-6> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<[1, 3, 1, 1]> : tensor<4xui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.reshape %[[ARG1]], %[[V1]] : (tensor<3xf32>, tensor<4xui32>) -> tensor<1x3x1x1xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.reshape %[[ARG2]], %[[V1]] : (tensor<3xf32>, tensor<4xui32>) -> tensor<1x3x1x1xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.reshape %[[ARG3]], %[[V1]] : (tensor<3xf32>, tensor<4xui32>) -> tensor<1x3x1x1xf32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.decomposable.broadcasting_add %[[ARG4]], %[[V0]] : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.reshape %[[V5]], %[[V1]] : (tensor<3xf32>, tensor<4xui32>) -> tensor<1x3x1x1xf32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.sqrt %[[V6]] : tensor<1x3x1x1xf32> -> tensor<1x3x1x1xf32>
                // CHECK-NEXT:     %[[V8:.*]] = coreai.decomposable.broadcasting_sub %[[ARG0]], %[[V4]] : (tensor<1x3x4x4xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x4x4xf32>
                // CHECK-NEXT:     %[[V9:.*]] = coreai.decomposable.broadcasting_divide %[[V8]], %[[V7]] : (tensor<1x3x4x4xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x4x4xf32>
                // CHECK-NEXT:     %[[V10:.*]] = coreai.decomposable.broadcasting_mul %[[V9]], %[[V2]] : (tensor<1x3x4x4xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x4x4xf32>
                // CHECK-NEXT:     %[[V11:.*]] = coreai.decomposable.broadcasting_add %[[V10]], %[[V3]] : (tensor<1x3x4x4xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[V11]] : tensor<1x3x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0]]: tensor<1x3x4x4xf32> {coreai.name = "x"}) -> (tensor<1x3x4x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0]] = coreai.constant dense<1.000000e+00> : tensor<3xf32>
                // CHECK-NEXT:     %[[V1]] = coreai.constant dense<0.000000e+00> : tensor<3xf32>
                // CHECK-NEXT:     %[[V2]] = coreai.invoke @batch_norm_{{.*}}(%[[ARG0]], %[[V0]], %[[V1]], %[[V1]], %[[V0]])  : (tensor<1x3x4x4xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x3x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<1x3x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestToCopyIR:
    def test_cast_f32_to_f16(self) -> None:
        class ToCopyModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.to(torch.float16)

        ir = get_ir(ToCopyModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf16> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.cast %[[ARG0]] : tensor<2x3xf32> to tensor<2x3xf16>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xf16>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_cast_f32_to_si32(self) -> None:
        class ToCopyModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.to(torch.int32)

        ir = get_ir(ToCopyModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.cast %[[ARG0]] : tensor<2x3xf32> to tensor<2x3xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestUnsafeViewIR:
    def test_static(self) -> None:
        class ViewFlattenModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.view(6)

        ir = get_ir(ViewFlattenModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<6xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<6> : tensor<1xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.reshape %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<1xui32>) -> tensor<6xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<6xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestPolarIR:
    def test_static(self) -> None:
        class PolarModel(nn.Module):
            def forward(self, abs_val: Tensor, angle: Tensor) -> Tensor:
                return torch.polar(abs_val, angle)

        ir = get_ir(
            PolarModel().eval(), abs_val=torch.rand(2, 3), angle=torch.rand(2, 3)
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "abs_val"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "angle"}) -> (tensor<2x3xcomplex<f32>> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.cos %[[ARG1]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_mul %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.sin %[[ARG1]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.decomposable.broadcasting_mul %[[ARG0]], %[[V2]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.create_complex %[[V1]], %[[V3]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xcomplex<f32>>
                // CHECK-NEXT:     coreai.output %[[V4]] : tensor<2x3xcomplex<f32>>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class PolarModel(nn.Module):
            def forward(self, abs_val: Tensor, angle: Tensor) -> Tensor:
                return torch.polar(abs_val, angle)

        abs_val = torch.rand(2, 3)
        angle = torch.rand(2, 3)
        ir = get_ir(
            PolarModel().eval(),
            abs_val=abs_val,
            angle=angle,
            dynamic_shapes={
                "abs_val": _all_dims_dynamic(abs_val),
                "angle": _all_dims_dynamic(angle),
            },
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "abs_val"}, %[[ARG1:.*]]: tensor<?x?xf32> {coreai.name = "angle"}) -> (tensor<?x?xcomplex<f32>> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.cos %[[ARG1]] : tensor<?x?xf32> -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_mul %[[ARG0]], %[[V0]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.sin %[[ARG1]] : tensor<?x?xf32> -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.decomposable.broadcasting_mul %[[ARG0]], %[[V2]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.create_complex %[[V1]], %[[V3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
                // CHECK-NEXT:     coreai.output %[[V4]] : tensor<?x?xcomplex<f32>>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_1d(self) -> None:
        class PolarModel(nn.Module):
            def forward(self, abs_val: Tensor, angle: Tensor) -> Tensor:
                return torch.polar(abs_val, angle)

        ir = get_ir(PolarModel().eval(), abs_val=torch.rand(5), angle=torch.rand(5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<5xf32> {coreai.name = "abs_val"}, %[[ARG1:.*]]: tensor<5xf32> {coreai.name = "angle"}) -> (tensor<5xcomplex<f32>> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.cos %[[ARG1]] : tensor<5xf32> -> tensor<5xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_mul %[[ARG0]], %[[V0]] : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.sin %[[ARG1]] : tensor<5xf32> -> tensor<5xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.decomposable.broadcasting_mul %[[ARG0]], %[[V2]] : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.create_complex %[[V1]], %[[V3]] : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xcomplex<f32>>
                // CHECK-NEXT:     coreai.output %[[V4]] : tensor<5xcomplex<f32>>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestAnyDefaultIR:
    def test_static(self) -> None:
        class AnyDefaultModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.any(x)

        ir = get_ir(AnyDefaultModel().eval(), x=(torch.rand(2, 3) > 0.5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xi1> {coreai.name = "x"}) -> (tensor<i1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<> : tensor<0xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.any %[[ARG0]], %[[V1]] : (tensor<2x3xi1>, tensor<2xsi32>) -> tensor<1x1xi1>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.reshape %[[V2]], %[[V0]] : (tensor<1x1xi1>, tensor<0xui32>) -> tensor<i1>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<i1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class AnyDefaultModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.any(x)

        x = torch.rand(2, 3) > 0.5
        ir = get_ir(
            AnyDefaultModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xi1> {coreai.name = "x"}) -> (tensor<i1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<> : tensor<0xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.any %[[ARG0]], %[[V1]] : (tensor<?x?xi1>, tensor<2xsi32>) -> tensor<1x1xi1>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.reshape %[[V2]], %[[V0]] : (tensor<1x1xi1>, tensor<0xui32>) -> tensor<i1>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<i1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestAnyDimIR:
    def test_static(self) -> None:
        class AnyDimModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.any(x, dim=1)

        ir = get_ir(AnyDimModel().eval(), x=(torch.rand(2, 3) > 0.5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xi1> {coreai.name = "x"}) -> (tensor<2xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2> : tensor<1xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.any %[[ARG0]], %[[V1]] : (tensor<2x3xi1>, tensor<1xsi32>) -> tensor<2x1xi1>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.reshape %[[V2]], %[[V0]] : (tensor<2x1xi1>, tensor<1xui32>) -> tensor<2xi1>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<2xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class AnyDimModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.any(x, dim=1)

        x = torch.rand(2, 3) > 0.5
        ir = get_ir(
            AnyDimModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xi1> {coreai.name = "x"}) -> (tensor<?xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.any %[[ARG0]], %[[V0]] : (tensor<?x?xi1>, tensor<1xsi32>) -> tensor<?x1xi1>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.shrink_dims %[[V1]], %[[V0]] : (tensor<?x1xi1>, tensor<1xsi32>) to tensor<?xi1>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<?xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_keepdim(self) -> None:
        class AnyDimKeepModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.any(x, dim=1, keepdim=True)

        ir = get_ir(AnyDimKeepModel().eval(), x=(torch.rand(2, 3) > 0.5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xi1> {coreai.name = "x"}) -> (tensor<2x1xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.any %[[ARG0]], %[[V0]] : (tensor<2x3xi1>, tensor<1xsi32>) -> tensor<2x1xi1>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x1xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_float_input(self) -> None:
        class AnyDimFloatModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.any(x, dim=1)

        ir = get_ir(AnyDimFloatModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2> : tensor<1xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.decomposable.broadcasting_not_equal %[[ARG0]], %[[V2]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.any %[[V3]], %[[V1]] : (tensor<2x3xi1>, tensor<1xsi32>) -> tensor<2x1xi1>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.reshape %[[V4]], %[[V0]] : (tensor<2x1xi1>, tensor<1xui32>) -> tensor<2xi1>
                // CHECK-NEXT:     coreai.output %[[V5]] : tensor<2xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestAnyDimsIR:
    def test_static(self) -> None:
        class AnyDimsModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.any(x, dim=[0, 1])

        ir = get_ir(AnyDimsModel().eval(), x=(torch.rand(2, 3, 4) > 0.5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3x4xi1> {coreai.name = "x"}) -> (tensor<4xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<4> : tensor<1xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.any %[[ARG0]], %[[V1]] : (tensor<2x3x4xi1>, tensor<2xsi32>) -> tensor<1x1x4xi1>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.reshape %[[V2]], %[[V0]] : (tensor<1x1x4xi1>, tensor<1xui32>) -> tensor<4xi1>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<4xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class AnyDimsModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.any(x, dim=[0, 1])

        x = torch.rand(2, 3, 4) > 0.5
        ir = get_ir(
            AnyDimsModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?x?xi1> {coreai.name = "x"}) -> (tensor<?xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.any %[[ARG0]], %[[V0]] : (tensor<?x?x?xi1>, tensor<2xsi32>) -> tensor<1x1x?xi1>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.shrink_dims %[[V1]], %[[V0]] : (tensor<1x1x?xi1>, tensor<2xsi32>) to tensor<?xi1>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<?xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_keepdim(self) -> None:
        class AnyDimsKeepModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.any(x, dim=[0, 1], keepdim=True)

        ir = get_ir(AnyDimsKeepModel().eval(), x=(torch.rand(2, 3, 4) > 0.5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3x4xi1> {coreai.name = "x"}) -> (tensor<1x1x4xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.any %[[ARG0]], %[[V0]] : (tensor<2x3x4xi1>, tensor<2xsi32>) -> tensor<1x1x4xi1>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<1x1x4xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestArangeIR:
    def test_static(self) -> None:
        class ArangeModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.arange(
                    0, x.shape[0], 1, dtype=torch.float32, device=x.device
                )

        ir = get_ir(ArangeModel().eval(), x=torch.rand(5, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<5x3xf32> {coreai.name = "x"}) -> (tensor<5xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<5xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<5xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class ArangeModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.arange(
                    0, x.shape[0], 1, dtype=torch.float32, device=x.device
                )

        x = torch.rand(5, 3)
        ir = get_ir(
            ArangeModel().eval(),
            x=x,
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}},
        )
        # ``end`` is a SymInt (rank-1 si32) sliced from x.shape[0];
        # ``start``/``step`` are scalar (rank-0) si32 constants.
        # The lowering keeps all operands as si32 so that coreai.range_
        # can infer a static shape from constant int operands, then casts
        # the result to the requested output dtype.  (Casting operands to
        # float *before* range_ causes it to return tensor<?> even when the
        # bounds are compile-time constants.)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: coreai.graph @main
                // CHECK-SAME:    %[[ARG0:.*]]: tensor<?x3xf32>
                // CHECK-SAME:    -> (tensor<?xf32>
                //
                // start and step remain as si32 constants; no pre-range cast:
                // CHECK-DAG:     %[[STEP:.+]] = coreai.constant dense<{{.*}}> : tensor<si32>
                // CHECK-DAG:     %[[START:.+]] = coreai.constant dense<{{.*}}> : tensor<si32>
                //
                // end: get_shape -> slice -> cast(ui32->si32) -> reshape(rank-1 to rank-0);
                // stays si32, no cast to f32 before range_:
                // CHECK:         %[[END_RANK1:.+]] = coreai.cast {{.*}} : tensor<1xui32> to tensor<1xsi32>
                // CHECK:         %[[END_RANK0:.+]] = coreai.reshape %[[END_RANK1]], {{.*}} : (tensor<1xsi32>, tensor<0xui32>) -> tensor<si32>
                //
                // range_ called with all-si32 scalars; result is si32 with dynamic shape:
                // CHECK:         %[[RANGE_OUT:.+]] = coreai.range %[[START]], %[[END_RANK0]], %[[STEP]] : (tensor<si32>, tensor<si32>, tensor<si32>) -> tensor<?xsi32>
                //
                // post-range cast to the requested f32 output dtype:
                // CHECK:         %[[OUT:.+]] = coreai.cast %[[RANGE_OUT]] : tensor<?xsi32> to tensor<?xf32>
                // CHECK:         coreai.output %[[OUT]] : tensor<?xf32>
            """,
        )

    def test_static_float_dtype_preserves_shape(self) -> None:
        """Regression: arange with a float dtype must keep a static output shape.

        Casting operands to float *before* coreai.range_ causes the op to
        return tensor<?xf32> even when all bounds are compile-time constants.
        The fix is to run range_ on int operands and cast the result.
        """

        class ArangeFloat(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                # Constant int bounds, float output dtype — the problematic case.
                return torch.arange(0, 8, 2, dtype=torch.float32, device=x.device)

        ir = get_ir(ArangeFloat().eval(), x=torch.rand(4))
        # The output shape must be static (4 elements: 0,2,4,6).
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<4xf32>
                // CHECK-SAME:     -> (tensor<4xf32>
                // The optimizer constant-folds the whole thing to a dense literal:
                // CHECK:           coreai.constant dense<{{.*}}> : tensor<4xf32>
            """,
        )


class TestArgmaxIR:
    def test_static(self) -> None:
        class ArgmaxModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.argmax(x, dim=1)

        ir = get_ir(ArgmaxModel().eval(), x=torch.rand(2, 5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x5xf32> {coreai.name = "x"}) -> (tensor<2xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2> : tensor<1xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.argmax %[[ARG0]], %[[V1]] : (tensor<2x5xf32>, tensor<si32>) -> tensor<2x1xui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.cast %[[V2]] : tensor<2x1xui32> to tensor<2x1xsi32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.reshape %[[V3]], %[[V0]] : (tensor<2x1xsi32>, tensor<1xui32>) -> tensor<2xsi32>
                // CHECK-NEXT:     coreai.output %[[V4]] : tensor<2xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class ArgmaxModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.argmax(x, dim=1)

        x = torch.rand(2, 5)
        ir = get_ir(
            ArgmaxModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.argmax %[[ARG0]], %[[V1]] : (tensor<?x?xf32>, tensor<si32>) -> tensor<?x1xui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.cast %[[V2]] : tensor<?x1xui32> to tensor<?x1xsi32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.shrink_dims %[[V3]], %[[V0]] : (tensor<?x1xsi32>, tensor<1xsi32>) to tensor<?xsi32>
                // CHECK-NEXT:     coreai.output %[[V4]] : tensor<?xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_no_dim(self) -> None:
        class ArgmaxNoDimModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.argmax(x)

        ir = get_ir(ArgmaxNoDimModel().eval(), x=torch.rand(2, 5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x5xf32> {coreai.name = "x"}) -> (tensor<si32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<> : tensor<0xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<10> : tensor<1xui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.reshape %[[ARG0]], %[[V2]] : (tensor<2x5xf32>, tensor<1xui32>) -> tensor<10xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.argmax %[[V3]], %[[V1]] : (tensor<10xf32>, tensor<si32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.reshape %[[V4]], %[[V0]] : (tensor<1xui32>, tensor<0xui32>) -> tensor<ui32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.cast %[[V5]] : tensor<ui32> to tensor<si32>
                // CHECK-NEXT:     coreai.output %[[V6]] : tensor<si32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_no_dim_dynamic(self) -> None:
        class ArgmaxNoDimModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.argmax(x)

        x = torch.rand(2, 5)
        ir = get_ir(
            ArgmaxNoDimModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<si32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<> : tensor<0xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<0> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.get_shape %[[ARG0]] : tensor<?x?xf32> -> tensor<2xui32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.reduce_product %[[V3]], %[[V2]] : (tensor<2xui32>, tensor<1xsi32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.reshape %[[ARG0]], %[[V4]] : (tensor<?x?xf32>, tensor<1xui32>) -> tensor<?xf32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.argmax %[[V5]], %[[V1]] : (tensor<?xf32>, tensor<si32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.reshape %[[V6]], %[[V0]] : (tensor<1xui32>, tensor<0xui32>) -> tensor<ui32>
                // CHECK-NEXT:     %[[V8:.*]] = coreai.cast %[[V7]] : tensor<ui32> to tensor<si32>
                // CHECK-NEXT:     coreai.output %[[V8]] : tensor<si32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestAvgPool2dIR:
    def test_static(self) -> None:
        class AvgPool2dModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

            def forward(self, x: Tensor) -> Tensor:
                return self.pool(x)

        ir = get_ir(AvgPool2dModel().eval(), x=torch.rand(1, 3, 8, 8))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @avg_pool2d_composite_{{.*}}(%[[ARG0:.*]]: tensor<1x3x8x8xf32> {coreai.name = "input_tensor"}, %[[ARG1:.*]]: tensor<2xsi32> {coreai.name = "kernel_size"}, %[[ARG2:.*]]: tensor<2xsi32> {coreai.name = "stride"}, %[[ARG3:.*]]: tensor<2xsi32> {coreai.name = "padding"}, %[[ARG4:.*]]: tensor<i1> {coreai.name = "ceil_mode"}, %[[ARG5:.*]]: tensor<i1> {coreai.name = "count_include_pad"}, %[[ARG6:.*]]: tensor<si32> {coreai.name = "divisor_override"}) -> tensor<1x3x4x4xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"avg_pool_2d" = {input_names = ["input", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"], op_attrs = {version = 1 : si64}, output_names = ["output"]}>, template_op = "avg_pool2d_composite"} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<2xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<4.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.constant dense<0> : tensor<4xui32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.constant dense<2> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.constant dense<0> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V8:.*]] = coreai.cast %[[ARG3]] : tensor<2xsi32> to tensor<2xui32>
                // CHECK-NEXT:     %[[V9:.*]] = coreai.slice %[[V8]], %[[V7]], %[[V6]], %[[V6]] : (tensor<2xui32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[V10:.*]] = coreai.slice %[[V8]], %[[V6]], %[[V5]], %[[V6]] : (tensor<2xui32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[V11:.*]] = coreai.concat %[[V3]], %[[V4]], %[[V9]], %[[V9]], %[[V10]], %[[V10]] : (tensor<si32>, tensor<4xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<8xui32>
                // CHECK-NEXT:     %[[V12:.*]] = coreai.pad %[[ARG0]], %[[V11]], %[[V2]] mode = <constant> : (tensor<1x3x8x8xf32>, tensor<8xui32>, tensor<f32>) -> tensor<1x3x?x?xf32>
                // CHECK-NEXT:     %[[V13:.*]] = coreai.cast %[[ARG1]] : tensor<2xsi32> to tensor<2xui32>
                // CHECK-NEXT:     %[[V14:.*]] = coreai.cast %[[ARG2]] : tensor<2xsi32> to tensor<2xui32>
                // CHECK-NEXT:     %[[V15:.*]] = coreai.sum_pool_2d %[[V12]], %[[V13]], %[[V14]], %[[V0]] : (tensor<1x3x?x?xf32>, tensor<2xui32>, tensor<2xui32>, tensor<2xui32>) -> tensor<1x3x4x4xf32>
                // CHECK-NEXT:     %[[V16:.*]] = coreai.decomposable.broadcasting_divide %[[V15]], %[[V1]] : (tensor<1x3x4x4xf32>, tensor<f32>) -> tensor<1x3x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[V16]] : tensor<1x3x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0]]: tensor<1x3x8x8xf32> {coreai.name = "x"}) -> (tensor<1x3x4x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0]] = coreai.constant dense<2> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V1]] = coreai.constant dense<0> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V2]] = coreai.constant dense<false> : tensor<i1>
                // CHECK-NEXT:     %[[V3]] = coreai.constant dense<true> : tensor<i1>
                // CHECK-NEXT:     %[[V4]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[V5]] = coreai.invoke @avg_pool2d_composite_{{.*}}(%[[ARG0]], %[[V0]], %[[V0]], %[[V1]], %[[V2]], %[[V3]], %[[V4]])  : (tensor<1x3x8x8xf32>, tensor<2xsi32>, tensor<2xsi32>, tensor<2xsi32>, tensor<i1>, tensor<i1>, tensor<si32>) -> tensor<1x3x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[V5]] : tensor<1x3x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_with_padding(self) -> None:
        class AvgPool2dPaddedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

            def forward(self, x: Tensor) -> Tensor:
                return self.pool(x)

        ir = get_ir(AvgPool2dPaddedModel().eval(), x=torch.rand(1, 3, 8, 8))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @avg_pool2d_composite_{{.*}}(%[[ARG0:.*]]: tensor<1x3x8x8xf32> {coreai.name = "input_tensor"}, %[[ARG1:.*]]: tensor<2xsi32> {coreai.name = "kernel_size"}, %[[ARG2:.*]]: tensor<2xsi32> {coreai.name = "stride"}, %[[ARG3:.*]]: tensor<2xsi32> {coreai.name = "padding"}, %[[ARG4:.*]]: tensor<i1> {coreai.name = "ceil_mode"}, %[[ARG5:.*]]: tensor<i1> {coreai.name = "count_include_pad"}, %[[ARG6:.*]]: tensor<si32> {coreai.name = "divisor_override"}) -> tensor<1x3x4x4xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"avg_pool_2d" = {input_names = ["input", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"], op_attrs = {version = 1 : si64}, output_names = ["output"]}>, template_op = "avg_pool2d_composite"} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<2xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<9.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.constant dense<0> : tensor<4xui32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.constant dense<2> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.constant dense<0> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V8:.*]] = coreai.cast %[[ARG3]] : tensor<2xsi32> to tensor<2xui32>
                // CHECK-NEXT:     %[[V9:.*]] = coreai.slice %[[V8]], %[[V7]], %[[V6]], %[[V6]] : (tensor<2xui32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[V10:.*]] = coreai.slice %[[V8]], %[[V6]], %[[V5]], %[[V6]] : (tensor<2xui32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[V11:.*]] = coreai.concat %[[V3]], %[[V4]], %[[V9]], %[[V9]], %[[V10]], %[[V10]] : (tensor<si32>, tensor<4xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<8xui32>
                // CHECK-NEXT:     %[[V12:.*]] = coreai.pad %[[ARG0]], %[[V11]], %[[V2]] mode = <constant> : (tensor<1x3x8x8xf32>, tensor<8xui32>, tensor<f32>) -> tensor<1x3x?x?xf32>
                // CHECK-NEXT:     %[[V13:.*]] = coreai.cast %[[ARG1]] : tensor<2xsi32> to tensor<2xui32>
                // CHECK-NEXT:     %[[V14:.*]] = coreai.cast %[[ARG2]] : tensor<2xsi32> to tensor<2xui32>
                // CHECK-NEXT:     %[[V15:.*]] = coreai.sum_pool_2d %[[V12]], %[[V13]], %[[V14]], %[[V0]] : (tensor<1x3x?x?xf32>, tensor<2xui32>, tensor<2xui32>, tensor<2xui32>) -> tensor<1x3x4x4xf32>
                // CHECK-NEXT:     %[[V16:.*]] = coreai.decomposable.broadcasting_divide %[[V15]], %[[V1]] : (tensor<1x3x4x4xf32>, tensor<f32>) -> tensor<1x3x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[V16]] : tensor<1x3x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0]]: tensor<1x3x8x8xf32> {coreai.name = "x"}) -> (tensor<1x3x4x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0]] = coreai.constant dense<3> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V1]] = coreai.constant dense<2> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V2]] = coreai.constant dense<1> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V3]] = coreai.constant dense<false> : tensor<i1>
                // CHECK-NEXT:     %[[V4]] = coreai.constant dense<true> : tensor<i1>
                // CHECK-NEXT:     %[[V5]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[V6]] = coreai.invoke @avg_pool2d_composite_{{.*}}(%[[ARG0]], %[[V0]], %[[V1]], %[[V2]], %[[V3]], %[[V4]], %[[V5]])  : (tensor<1x3x8x8xf32>, tensor<2xsi32>, tensor<2xsi32>, tensor<2xsi32>, tensor<i1>, tensor<i1>, tensor<si32>) -> tensor<1x3x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[V6]] : tensor<1x3x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestAvgPool3dIR:
    def test_static(self) -> None:
        class AvgPool3dModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

            def forward(self, x: Tensor) -> Tensor:
                return self.pool(x)

        ir = get_ir(AvgPool3dModel().eval(), x=torch.rand(1, 3, 4, 4, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @avg_pool3d_composite_{{.*}}(%[[ARG0:.*]]: tensor<1x3x4x4x4xf32> {coreai.name = "input_tensor"}, %[[ARG1:.*]]: tensor<3xsi32> {coreai.name = "kernel_size"}, %[[ARG2:.*]]: tensor<3xsi32> {coreai.name = "stride"}, %[[ARG3:.*]]: tensor<3xsi32> {coreai.name = "padding"}, %[[ARG4:.*]]: tensor<i1> {coreai.name = "ceil_mode"}, %[[ARG5:.*]]: tensor<i1> {coreai.name = "count_include_pad"}, %[[ARG6:.*]]: tensor<si32> {coreai.name = "divisor_override"}) -> tensor<1x3x2x2x2xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"avg_pool_3d" = {input_names = ["input", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"], op_attrs = {version = 1 : si64}, output_names = ["output"]}>, template_op = "avg_pool3d_composite"} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<3xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<8.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.constant dense<0> : tensor<4xui32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.constant dense<3> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.constant dense<2> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V8:.*]] = coreai.constant dense<0> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V9:.*]] = coreai.cast %[[ARG3]] : tensor<3xsi32> to tensor<3xui32>
                // CHECK-NEXT:     %[[V10:.*]] = coreai.slice %[[V9]], %[[V8]], %[[V7]], %[[V7]] : (tensor<3xui32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[V11:.*]] = coreai.slice %[[V9]], %[[V7]], %[[V6]], %[[V7]] : (tensor<3xui32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[V12:.*]] = coreai.slice %[[V9]], %[[V6]], %[[V5]], %[[V7]] : (tensor<3xui32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[V13:.*]] = coreai.concat %[[V3]], %[[V4]], %[[V10]], %[[V10]], %[[V11]], %[[V11]], %[[V12]], %[[V12]] : (tensor<si32>, tensor<4xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<10xui32>
                // CHECK-NEXT:     %[[V14:.*]] = coreai.pad %[[ARG0]], %[[V13]], %[[V2]] mode = <constant> : (tensor<1x3x4x4x4xf32>, tensor<10xui32>, tensor<f32>) -> tensor<1x3x?x?x?xf32>
                // CHECK-NEXT:     %[[V15:.*]] = coreai.cast %[[ARG1]] : tensor<3xsi32> to tensor<3xui32>
                // CHECK-NEXT:     %[[V16:.*]] = coreai.cast %[[ARG2]] : tensor<3xsi32> to tensor<3xui32>
                // CHECK-NEXT:     %[[V17:.*]] = coreai.sum_pool_3d %[[V14]], %[[V15]], %[[V16]], %[[V0]] : (tensor<1x3x?x?x?xf32>, tensor<3xui32>, tensor<3xui32>, tensor<3xui32>) -> tensor<1x3x2x2x2xf32>
                // CHECK-NEXT:     %[[V18:.*]] = coreai.decomposable.broadcasting_divide %[[V17]], %[[V1]] : (tensor<1x3x2x2x2xf32>, tensor<f32>) -> tensor<1x3x2x2x2xf32>
                // CHECK-NEXT:     coreai.output %[[V18]] : tensor<1x3x2x2x2xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0]]: tensor<1x3x4x4x4xf32> {coreai.name = "x"}) -> (tensor<1x3x2x2x2xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0]] = coreai.constant dense<2> : tensor<3xsi32>
                // CHECK-NEXT:     %[[V1]] = coreai.constant dense<0> : tensor<3xsi32>
                // CHECK-NEXT:     %[[V2]] = coreai.constant dense<false> : tensor<i1>
                // CHECK-NEXT:     %[[V3]] = coreai.constant dense<true> : tensor<i1>
                // CHECK-NEXT:     %[[V4]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[V5]] = coreai.invoke @avg_pool3d_composite_{{.*}}(%[[ARG0]], %[[V0]], %[[V0]], %[[V1]], %[[V2]], %[[V3]], %[[V4]])  : (tensor<1x3x4x4x4xf32>, tensor<3xsi32>, tensor<3xsi32>, tensor<3xsi32>, tensor<i1>, tensor<i1>, tensor<si32>) -> tensor<1x3x2x2x2xf32>
                // CHECK-NEXT:     coreai.output %[[V5]] : tensor<1x3x2x2x2xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestBitwiseAndIR:
    def test_static(self) -> None:
        class BitwiseAndModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.bitwise_and(x, y)

        ir = get_ir(
            BitwiseAndModel().eval(),
            x=torch.randint(0, 255, (2, 3), dtype=torch.int32),
            y=torch.randint(0, 255, (2, 3), dtype=torch.int32),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xsi32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xsi32> {coreai.name = "y"}) -> (tensor<2x3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_bitwise_and %[[ARG0]], %[[ARG1]] : (tensor<2x3xsi32>, tensor<2x3xsi32>) -> tensor<2x3xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class BitwiseAndModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.bitwise_and(x, y)

        x = torch.randint(0, 255, (2, 3), dtype=torch.int32)
        y = torch.randint(0, 255, (2, 3), dtype=torch.int32)
        ir = get_ir(
            BitwiseAndModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": _all_dims_dynamic(x), "y": _all_dims_dynamic(y)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xsi32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x?xsi32> {coreai.name = "y"}) -> (tensor<?x?xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_bitwise_and %[[ARG0]], %[[ARG1]] : (tensor<?x?xsi32>, tensor<?x?xsi32>) -> tensor<?x?xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x?xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestBitwiseNotIR:
    def test_bool_static(self) -> None:
        class BitwiseNotBoolModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.bitwise_not(x)

        ir = get_ir(BitwiseNotBoolModel().eval(), x=(torch.rand(2, 3) > 0.5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xi1> {coreai.name = "x"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.not %[[ARG0]] : tensor<2x3xi1> -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_bool_dynamic(self) -> None:
        class BitwiseNotBoolModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.bitwise_not(x)

        x = torch.rand(2, 3) > 0.5
        ir = get_ir(
            BitwiseNotBoolModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xi1> {coreai.name = "x"}) -> (tensor<?x?xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.not %[[ARG0]] : tensor<?x?xi1> -> tensor<?x?xi1>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x?xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_int_static(self) -> None:
        class BitwiseNotIntModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.bitwise_not(x)

        ir = get_ir(
            BitwiseNotIntModel().eval(),
            x=torch.randint(0, 255, (2, 3), dtype=torch.int32),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xsi32> {coreai.name = "x"}) -> (tensor<2x3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<-1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.cast %[[ARG0]] : tensor<2x3xsi32> to tensor<2x3xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.decomposable.broadcasting_add %[[V2]], %[[V1]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.decomposable.broadcasting_mul %[[V3]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.cast %[[V4]] : tensor<2x3xf32> to tensor<2x3xsi32>
                // CHECK-NEXT:     coreai.output %[[V5]] : tensor<2x3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_int_dynamic(self) -> None:
        class BitwiseNotIntModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.bitwise_not(x)

        x = torch.randint(0, 255, (2, 3), dtype=torch.int32)
        ir = get_ir(
            BitwiseNotIntModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xsi32> {coreai.name = "x"}) -> (tensor<?x?xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<-1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.cast %[[ARG0]] : tensor<?x?xsi32> to tensor<?x?xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.decomposable.broadcasting_add %[[V2]], %[[V1]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.decomposable.broadcasting_mul %[[V3]], %[[V0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.cast %[[V4]] : tensor<?x?xf32> to tensor<?x?xsi32>
                // CHECK-NEXT:     coreai.output %[[V5]] : tensor<?x?xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestBitwiseOrIR:
    def test_static(self) -> None:
        class BitwiseOrModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.bitwise_or(x, y)

        ir = get_ir(
            BitwiseOrModel().eval(),
            x=torch.randint(0, 255, (2, 3), dtype=torch.int32),
            y=torch.randint(0, 255, (2, 3), dtype=torch.int32),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xsi32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xsi32> {coreai.name = "y"}) -> (tensor<2x3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_bitwise_or %[[ARG0]], %[[ARG1]] : (tensor<2x3xsi32>, tensor<2x3xsi32>) -> tensor<2x3xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class BitwiseOrModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.bitwise_or(x, y)

        x = torch.randint(0, 255, (2, 3), dtype=torch.int32)
        y = torch.randint(0, 255, (2, 3), dtype=torch.int32)
        ir = get_ir(
            BitwiseOrModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": _all_dims_dynamic(x), "y": _all_dims_dynamic(y)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xsi32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x?xsi32> {coreai.name = "y"}) -> (tensor<?x?xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_bitwise_or %[[ARG0]], %[[ARG1]] : (tensor<?x?xsi32>, tensor<?x?xsi32>) -> tensor<?x?xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x?xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestBitwiseXorIR:
    def test_static(self) -> None:
        class BitwiseXorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.bitwise_xor(x, y)

        ir = get_ir(
            BitwiseXorModel().eval(),
            x=torch.randint(0, 255, (2, 3), dtype=torch.int32),
            y=torch.randint(0, 255, (2, 3), dtype=torch.int32),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xsi32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xsi32> {coreai.name = "y"}) -> (tensor<2x3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_bitwise_xor %[[ARG0]], %[[ARG1]] : (tensor<2x3xsi32>, tensor<2x3xsi32>) -> tensor<2x3xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class BitwiseXorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.bitwise_xor(x, y)

        x = torch.randint(0, 255, (2, 3), dtype=torch.int32)
        y = torch.randint(0, 255, (2, 3), dtype=torch.int32)
        ir = get_ir(
            BitwiseXorModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": _all_dims_dynamic(x), "y": _all_dims_dynamic(y)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xsi32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x?xsi32> {coreai.name = "y"}) -> (tensor<?x?xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_bitwise_xor %[[ARG0]], %[[ARG1]] : (tensor<?x?xsi32>, tensor<?x?xsi32>) -> tensor<?x?xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x?xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestBmmIR:
    def test_static(self) -> None:
        class BmmModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.bmm(x, y)

        ir = get_ir(BmmModel().eval(), x=torch.rand(2, 3, 4), y=torch.rand(2, 4, 5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3x4xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x4x5xf32> {coreai.name = "y"}) -> (tensor<2x3x5xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_batch_matmul %[[ARG0]], %[[ARG1]] : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3x5xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class BmmModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.bmm(x, y)

        x = torch.rand(2, 3, 4)
        y = torch.rand(2, 4, 5)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            BmmModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={
                "x": {0: batch},
                "y": {0: batch},
            },
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3x4xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x4x5xf32> {coreai.name = "y"}) -> (tensor<?x3x5xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_batch_matmul %[[ARG0]], %[[ARG1]] : (tensor<?x3x4xf32>, tensor<?x4x5xf32>) -> tensor<?x3x5xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x3x5xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestCatIR:
    def test_static(self) -> None:
        class CatModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.cat([x, y], dim=1)

        ir = get_ir(CatModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x4xf32> {coreai.name = "y"}) -> (tensor<2x7xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.concat %[[V0]], %[[ARG0]], %[[ARG1]] : (tensor<si32>, tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x7xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class CatModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.cat([x, y], dim=1)

        x = torch.rand(2, 3)
        y = torch.rand(2, 4)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            CatModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x4xf32> {coreai.name = "y"}) -> (tensor<?x7xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.concat %[[V0]], %[[ARG0]], %[[ARG1]] : (tensor<si32>, tensor<?x3xf32>, tensor<?x4xf32>) -> tensor<?x7xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x7xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dim0(self) -> None:
        class CatDim0Model(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.cat([x, y], dim=0)

        ir = get_ir(CatDim0Model().eval(), x=torch.rand(2, 3), y=torch.rand(4, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<4x3xf32> {coreai.name = "y"}) -> (tensor<6x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.concat %[[V0]], %[[ARG0]], %[[ARG1]] : (tensor<si32>, tensor<2x3xf32>, tensor<4x3xf32>) -> tensor<6x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<6x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dim0_dynamic(self) -> None:
        class CatDim0Model(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.cat([x, y], dim=0)

        x = torch.rand(2, 3)
        y = torch.rand(4, 3)
        ir = get_ir(
            CatDim0Model().eval(),
            x=x,
            y=y,
            dynamic_shapes={
                "x": {0: torch.export.Dim("xb")},
                "y": {0: torch.export.Dim("yb")},
            },
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.concat %[[V0]], %[[ARG0]], %[[ARG1]] : (tensor<si32>, tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestCeilIR:
    def test_static(self) -> None:
        class CeilModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.ceil(x)

        ir = get_ir(CeilModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<-1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.decomposable.broadcasting_mul %[[ARG0]], %[[V1]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.decomposable.broadcasting_floor_divide %[[V2]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.decomposable.broadcasting_mul %[[V3]], %[[V1]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V4]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class CeilModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.ceil(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            CeilModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<-1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.decomposable.broadcasting_mul %[[ARG0]], %[[V1]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.decomposable.broadcasting_floor_divide %[[V2]], %[[V0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.decomposable.broadcasting_mul %[[V3]], %[[V1]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V4]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestClampIR:
    def test_static(self) -> None:
        class ClampModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.clamp(x, min=0.0, max=1.0)

        ir = get_ir(ClampModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.decomposable.broadcasting_maximum %[[ARG0]], %[[V1]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.decomposable.broadcasting_minimum %[[V2]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class ClampModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.clamp(x, min=0.0, max=1.0)

        x = torch.rand(2, 3)
        ir = get_ir(
            ClampModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.decomposable.broadcasting_maximum %[[ARG0]], %[[V1]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.decomposable.broadcasting_minimum %[[V2]], %[[V0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_min_only(self) -> None:
        class ClampMinModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.clamp(x, min=0.0)

        ir = get_ir(ClampMinModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_maximum %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_max_only(self) -> None:
        class ClampMaxModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.clamp(x, max=1.0)

        ir = get_ir(ClampMaxModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_minimum %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestClampTensorIR:
    def test_static(self) -> None:
        class ClampTensorModel(nn.Module):
            def forward(self, x: Tensor, low: Tensor, high: Tensor) -> Tensor:
                return torch.clamp(x, min=low, max=high)

        ir = get_ir(
            ClampTensorModel().eval(),
            x=torch.rand(2, 3),
            low=torch.zeros(2, 3),
            high=torch.ones(2, 3),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "low"}, %[[ARG2:.*]]: tensor<2x3xf32> {coreai.name = "high"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_maximum %[[ARG0]], %[[ARG1]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_minimum %[[V0]], %[[ARG2]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class ClampTensorModel(nn.Module):
            def forward(self, x: Tensor, low: Tensor, high: Tensor) -> Tensor:
                return torch.clamp(x, min=low, max=high)

        x = torch.rand(2, 3)
        low = torch.zeros(2, 3)
        high = torch.ones(2, 3)
        ir = get_ir(
            ClampTensorModel().eval(),
            x=x,
            low=low,
            high=high,
            dynamic_shapes={
                "x": _all_dims_dynamic(x),
                "low": _all_dims_dynamic(low),
                "high": _all_dims_dynamic(high),
            },
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x?xf32> {coreai.name = "low"}, %[[ARG2:.*]]: tensor<?x?xf32> {coreai.name = "high"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_maximum %[[ARG0]], %[[ARG1]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_minimum %[[V0]], %[[ARG2]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestCloneIR:
    def test_static(self) -> None:
        class CloneModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.clone()

        ir = get_ir(CloneModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     coreai.output %[[ARG0]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class CloneModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.clone()

        x = torch.rand(2, 3)
        ir = get_ir(
            CloneModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     coreai.output %[[ARG0]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestComplexIR:
    def test_static(self) -> None:
        class ComplexModel(nn.Module):
            def forward(self, real: Tensor, imag: Tensor) -> Tensor:
                return torch.complex(real, imag)

        ir = get_ir(ComplexModel().eval(), real=torch.rand(2, 3), imag=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "real"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "imag"}) -> (tensor<2x3xcomplex<f32>> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.create_complex %[[ARG0]], %[[ARG1]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xcomplex<f32>>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xcomplex<f32>>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class ComplexModel(nn.Module):
            def forward(self, real: Tensor, imag: Tensor) -> Tensor:
                return torch.complex(real, imag)

        real = torch.rand(2, 3)
        imag = torch.rand(2, 3)
        ir = get_ir(
            ComplexModel().eval(),
            real=real,
            imag=imag,
            dynamic_shapes={
                "real": _all_dims_dynamic(real),
                "imag": _all_dims_dynamic(imag),
            },
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "real"}, %[[ARG1:.*]]: tensor<?x?xf32> {coreai.name = "imag"}) -> (tensor<?x?xcomplex<f32>> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.create_complex %[[ARG0]], %[[ARG1]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x?xcomplex<f32>>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestCondIR:
    def test_static(self) -> None:
        def true_fn(x):
            return x + 1.0

        def false_fn(x):
            return x - 1.0

        class CondModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

        ir = get_ir(CondModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<> : tensor<0xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.reduce_sum %[[ARG0]], %[[V3]] : (tensor<2x3xf32>, tensor<2xsi32>) -> tensor<1x1xf32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.decomposable.broadcasting_greater %[[V4]], %[[V1]] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xi1>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.reshape %[[V5]], %[[V0]] : (tensor<1x1xi1>, tensor<0xui32>) -> tensor<i1>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.if %[[V6]] {
                // CHECK-NEXT:       %[[V8:.*]] = coreai.decomposable.broadcasting_add %[[ARG0]], %[[V2]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:       coreai.yield %[[V8]] : tensor<2x3xf32>
                // CHECK-NEXT:     } else {
                // CHECK-NEXT:       %[[V8]] = coreai.decomposable.broadcasting_sub %[[ARG0]], %[[V2]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:       coreai.yield %[[V8]] : tensor<2x3xf32>
                // CHECK-NEXT:     } : tensor<i1> -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V7]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        def true_fn(x):
            return x + 1.0

        def false_fn(x):
            return x - 1.0

        class CondModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

        x = torch.rand(2, 3)
        ir = get_ir(
            CondModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<> : tensor<0xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.reduce_sum %[[ARG0]], %[[V3]] : (tensor<?x?xf32>, tensor<2xsi32>) -> tensor<1x1xf32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.decomposable.broadcasting_greater %[[V4]], %[[V1]] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xi1>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.reshape %[[V5]], %[[V0]] : (tensor<1x1xi1>, tensor<0xui32>) -> tensor<i1>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.if %[[V6]] {
                // CHECK-NEXT:       %[[V8:.*]] = coreai.decomposable.broadcasting_add %[[ARG0]], %[[V2]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:       coreai.yield %[[V8]] : tensor<?x?xf32>
                // CHECK-NEXT:     } else {
                // CHECK-NEXT:       %[[V8]] = coreai.decomposable.broadcasting_sub %[[ARG0]], %[[V2]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:       coreai.yield %[[V8]] : tensor<?x?xf32>
                // CHECK-NEXT:     } : tensor<i1> -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V7]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestConstantPadNdIR:
    def test_static(self) -> None:
        class ConstantPadModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.pad(x, (1, 2), mode="constant", value=0.0)

        ir = get_ir(ConstantPadModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x6xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<[0, 0, 1, 2]> : tensor<4xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.pad %[[ARG0]], %[[V0]], %[[V1]] mode = <constant> : (tensor<2x3xf32>, tensor<4xui32>, tensor<f32>) -> tensor<2x6xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<2x6xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class ConstantPadModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.pad(x, (1, 2), mode="constant", value=0.0)

        x = torch.rand(2, 3)
        ir = get_ir(
            ConstantPadModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<[0, 0, 1, 2]> : tensor<4xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.pad %[[ARG0]], %[[V0]], %[[V1]] mode = <constant> : (tensor<?x?xf32>, tensor<4xui32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_nonzero_value(self) -> None:
        class ConstantPadValueModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.pad(
                    x, (1, 2, 3, 4), mode="constant", value=1.5
                )

        ir = get_ir(ConstantPadValueModel().eval(), x=torch.rand(2, 5, 5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x5x5xf32> {coreai.name = "x"}) -> (tensor<2x12x8xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<[0, 0, 3, 4, 1, 2]> : tensor<6xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1.500000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.pad %[[ARG0]], %[[V0]], %[[V1]] mode = <constant> : (tensor<2x5x5xf32>, tensor<6xui32>, tensor<f32>) -> tensor<2x12x8xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<2x12x8xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_negative_pad_static(self) -> None:
        class NegPadModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.pad(x, (-1, -2), mode="constant", value=0.0)

        ir = get_ir(NegPadModel().eval(), x=torch.rand(2, 8))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x8xf32> {coreai.name = "x"}) -> (tensor<2x5xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<[2147483647, 6]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<1> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.slice %[[ARG0]], %[[V0]], %[[V1]], %[[V2]] : (tensor<2x8xf32>, tensor<2xsi32>, tensor<2xsi32>, tensor<2xsi32>) -> tensor<2x5xf32>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<2x5xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_negative_pad_dynamic(self) -> None:
        class NegPadModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.pad(x, (-1, -2), mode="constant", value=0.0)

        x = torch.rand(2, 8)
        ir = get_ir(
            NegPadModel().eval(),
            x=x,
            dynamic_shapes={"x": {1: torch.export.Dim("seq", min=5)}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<2147483647> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.constant dense<-2> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.constant dense<2> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.get_shape %[[ARG0]] : tensor<2x?xf32> -> tensor<2xui32>
                // CHECK-NEXT:     %[[V8:.*]] = coreai.cast %[[V7]] : tensor<2xui32> to tensor<2xsi32>
                // CHECK-NEXT:     %[[V9:.*]] = coreai.slice %[[V8]], %[[V6]], %[[V5]], %[[V6]] : (tensor<2xsi32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xsi32>
                // CHECK-NEXT:     %[[V10:.*]] = coreai.decomposable.broadcasting_add %[[V9]], %[[V4]] : (tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xsi32>
                // CHECK-NEXT:     %[[V11:.*]] = coreai.concat %[[V3]], %[[V2]], %[[V10]] : (tensor<si32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<2xsi32>
                // CHECK-NEXT:     %[[V12:.*]] = coreai.slice %[[ARG0]], %[[V1]], %[[V11]], %[[V0]] : (tensor<2x?xf32>, tensor<2xsi32>, tensor<2xsi32>, tensor<2xsi32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V12]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_mixed_positive_negative_pad(self) -> None:
        class MixedPadModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.pad(
                    x, (1, -2, -1, 2), mode="constant", value=0.0
                )

        ir = get_ir(MixedPadModel().eval(), x=torch.rand(2, 6, 8))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x6x8xf32> {coreai.name = "x"}) -> (tensor<2x7x7xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<[0, 0, 0, 2, 1, 0]> : tensor<6xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<[0, 1, 0]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense<[2147483647, 2147483647, 6]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.constant dense<1> : tensor<3xsi32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.slice %[[ARG0]], %[[V2]], %[[V3]], %[[V4]] : (tensor<2x6x8xf32>, tensor<3xsi32>, tensor<3xsi32>, tensor<3xsi32>) -> tensor<2x5x6xf32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.pad %[[V5]], %[[V0]], %[[V1]] mode = <constant> : (tensor<2x5x6xf32>, tensor<6xui32>, tensor<f32>) -> tensor<2x7x7xf32>
                // CHECK-NEXT:     coreai.output %[[V6]] : tensor<2x7x7xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestConvolutionIR:
    def test_conv2d_static(self) -> None:
        class Conv2dModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=0, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                return self.conv(x)

        ir = get_ir(Conv2dModel().eval(), x=torch.rand(1, 3, 8, 8))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x3x8x8xf32> {coreai.name = "x"}) -> (tensor<1x8x6x6xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<8x3x3x3xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<2xui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<1> : tensor<ui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.conv2d %[[ARG0]], %[[V0]], %[[V1]], %[[V1]], %[[V2]] : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>) -> tensor<1x8x6x6xf32>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<1x8x6x6xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_conv2d_with_padding_and_bias(self) -> None:
        class Conv2dBiasModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

            def forward(self, x: Tensor) -> Tensor:
                return self.conv(x)

        ir = get_ir(Conv2dBiasModel().eval(), x=torch.rand(1, 3, 8, 8))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x3x8x8xf32> {coreai.name = "x"}) -> (tensor<1x16x8x8xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<1x16x1x1xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<ui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<1> : tensor<2xui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense_resource<{{.*}}> : tensor<16x3x3x3xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.constant dense<[0, 0, 0, 0, 1, 1, 1, 1]> : tensor<8xui32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.pad %[[ARG0]], %[[V5]], %[[V4]] mode = <constant> : (tensor<1x3x8x8xf32>, tensor<8xui32>, tensor<f32>) -> tensor<1x3x10x10xf32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.conv2d %[[V6]], %[[V3]], %[[V2]], %[[V2]], %[[V1]] : (tensor<1x3x10x10xf32>, tensor<16x3x3x3xf32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>) -> tensor<1x16x8x8xf32>
                // CHECK-NEXT:     %[[V8:.*]] = coreai.decomposable.broadcasting_add %[[V7]], %[[V0]] : (tensor<1x16x8x8xf32>, tensor<1x16x1x1xf32>) -> tensor<1x16x8x8xf32>
                // CHECK-NEXT:     coreai.output %[[V8]] : tensor<1x16x8x8xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_conv1d_static(self) -> None:
        class Conv1dModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(3, 8, kernel_size=3, padding=0, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                return self.conv(x)

        ir = get_ir(Conv1dModel().eval(), x=torch.rand(1, 3, 10))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x3x10xf32> {coreai.name = "x"}) -> (tensor<1x8x8xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<[1, 8, 8]> : tensor<3xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<{{.*}}> : tensor<8x3x1x3xf32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<1> : tensor<ui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense<1> : tensor<2xui32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.constant dense<[1, 3, 1, 10]> : tensor<4xui32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.reshape %[[ARG0]], %[[V4]] : (tensor<1x3x10xf32>, tensor<4xui32>) -> tensor<1x3x1x10xf32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.conv2d %[[V5]], %[[V1]], %[[V3]], %[[V3]], %[[V2]] : (tensor<1x3x1x10xf32>, tensor<8x3x1x3xf32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>) -> tensor<1x8x1x8xf32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.reshape %[[V6]], %[[V0]] : (tensor<1x8x1x8xf32>, tensor<3xui32>) -> tensor<1x8x8xf32>
                // CHECK-NEXT:     coreai.output %[[V7]] : tensor<1x8x8xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_conv1d_with_padding_and_bias(self) -> None:
        class Conv1dPadBiasModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(3, 8, kernel_size=3, padding=1, bias=True)

            def forward(self, x: Tensor) -> Tensor:
                return self.conv(x)

        ir = get_ir(Conv1dPadBiasModel().eval(), x=torch.rand(1, 3, 10))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x3x10xf32> {coreai.name = "x"}) -> (tensor<1x8x10xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<1x8x1xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<[1, 8, 10]> : tensor<3xui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<{{.*}}> : tensor<8x3x1x3xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense<1> : tensor<ui32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.constant dense<1> : tensor<2xui32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.constant dense<[0, 0, 0, 0, 0, 0, 1, 1]> : tensor<8xui32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.constant dense<[1, 3, 1, 10]> : tensor<4xui32>
                // CHECK-NEXT:     %[[V8:.*]] = coreai.reshape %[[ARG0]], %[[V7]] : (tensor<1x3x10xf32>, tensor<4xui32>) -> tensor<1x3x1x10xf32>
                // CHECK-NEXT:     %[[V9:.*]] = coreai.pad %[[V8]], %[[V5]], %[[V6]] mode = <constant> : (tensor<1x3x1x10xf32>, tensor<8xui32>, tensor<f32>) -> tensor<1x3x1x12xf32>
                // CHECK-NEXT:     %[[V10:.*]] = coreai.conv2d %[[V9]], %[[V2]], %[[V4]], %[[V4]], %[[V3]] : (tensor<1x3x1x12xf32>, tensor<8x3x1x3xf32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>) -> tensor<1x8x1x10xf32>
                // CHECK-NEXT:     %[[V11:.*]] = coreai.reshape %[[V10]], %[[V1]] : (tensor<1x8x1x10xf32>, tensor<3xui32>) -> tensor<1x8x10xf32>
                // CHECK-NEXT:     %[[V12:.*]] = coreai.decomposable.broadcasting_add %[[V11]], %[[V0]] : (tensor<1x8x10xf32>, tensor<1x8x1xf32>) -> tensor<1x8x10xf32>
                // CHECK-NEXT:     coreai.output %[[V12]] : tensor<1x8x10xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_conv2d_dynamic_batch(self) -> None:
        class Conv2dModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=0, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                return self.conv(x)

        x = torch.rand(2, 3, 8, 8)
        ir = get_ir(
            Conv2dModel().eval(),
            x=x,
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3x8x8xf32> {coreai.name = "x"}) -> (tensor<?x8x6x6xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<8x3x3x3xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<2xui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<1> : tensor<ui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.conv2d %[[ARG0]], %[[V0]], %[[V1]], %[[V1]], %[[V2]] : (tensor<?x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>) -> tensor<?x8x6x6xf32>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<?x8x6x6xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_conv2d_grouped(self) -> None:
        class GroupedConv2dModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(
                    4, 8, kernel_size=3, padding=0, groups=2, bias=False
                )

            def forward(self, x: Tensor) -> Tensor:
                return self.conv(x)

        ir = get_ir(GroupedConv2dModel().eval(), x=torch.rand(1, 4, 6, 6))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x4x6x6xf32> {coreai.name = "x"}) -> (tensor<1x8x4x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<8x2x3x3xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<2xui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<2> : tensor<ui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.conv2d %[[ARG0]], %[[V0]], %[[V1]], %[[V1]], %[[V2]] : (tensor<1x4x6x6xf32>, tensor<8x2x3x3xf32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>) -> tensor<1x8x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<1x8x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_conv3d_static(self) -> None:
        class Conv3dModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(3, 4, kernel_size=3, padding=0, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                return self.conv(x)

        ir = get_ir(Conv3dModel().eval(), x=torch.rand(1, 3, 6, 6, 6))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x3x6x6x6xf32> {coreai.name = "x"}) -> (tensor<1x4x4x4x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense_resource<{{.*}}> : tensor<4x3x3x3x3xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<3xui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<1> : tensor<ui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.conv3d %[[ARG0]], %[[V0]], %[[V1]], %[[V1]], %[[V2]] : (tensor<1x3x6x6x6xf32>, tensor<4x3x3x3x3xf32>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>) -> tensor<1x4x4x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<1x4x4x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_conv3d_with_padding_and_bias(self) -> None:
        class Conv3dBiasModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(3, 4, kernel_size=3, padding=1, bias=True)

            def forward(self, x: Tensor) -> Tensor:
                return self.conv(x)

        ir = get_ir(Conv3dBiasModel().eval(), x=torch.rand(1, 3, 4, 4, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x3x4x4x4xf32> {coreai.name = "x"}) -> (tensor<1x4x4x4x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<1x4x1x1x1xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<ui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<1> : tensor<3xui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense_resource<{{.*}}> : tensor<4x3x3x3x3xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.constant dense<[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]> : tensor<10xui32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.pad %[[ARG0]], %[[V5]], %[[V4]] mode = <constant> : (tensor<1x3x4x4x4xf32>, tensor<10xui32>, tensor<f32>) -> tensor<1x3x6x6x6xf32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.conv3d %[[V6]], %[[V3]], %[[V2]], %[[V2]], %[[V1]] : (tensor<1x3x6x6x6xf32>, tensor<4x3x3x3x3xf32>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>) -> tensor<1x4x4x4x4xf32>
                // CHECK-NEXT:     %[[V8:.*]] = coreai.decomposable.broadcasting_add %[[V7]], %[[V0]] : (tensor<1x4x4x4x4xf32>, tensor<1x4x1x1x1xf32>) -> tensor<1x4x4x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[V8]] : tensor<1x4x4x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_conv_transpose2d(self) -> None:
        class ConvTranspose2dModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.ConvTranspose2d(
                    3,
                    8,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                )

            def forward(self, x: Tensor) -> Tensor:
                return self.conv(x)

        ir = get_ir(ConvTranspose2dModel().eval(), x=torch.rand(1, 3, 4, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x3x4x4xf32> {coreai.name = "x"}) -> (tensor<1x8x8x8xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<[1, 8, 9, 9]> : tensor<4xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<4xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<[0, 0, 1, 1]> : tensor<4xsi32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.constant dense<{{.*}}> : tensor<3x8x3x3xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.constant dense<2> : tensor<2xui32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.constant dense<0> : tensor<2xui32>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.constant dense<1> : tensor<2xui32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.constant dense<1> : tensor<ui32>
                // CHECK-NEXT:     %[[V8:.*]] = coreai.conv_transpose2d %[[ARG0]], %[[V3]], %[[V4]], %[[V5]], %[[V6]], %[[V5]], %[[V7]] : (tensor<1x3x4x4xf32>, tensor<3x8x3x3xf32>, tensor<2xui32>, tensor<2xui32>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>) -> tensor<1x8x9x9xf32>
                // CHECK-NEXT:     %[[V9:.*]] = coreai.slice %[[V8]], %[[V2]], %[[V0]], %[[V1]] : (tensor<1x8x9x9xf32>, tensor<4xsi32>, tensor<4xsi32>, tensor<4xsi32>) -> tensor<1x8x8x8xf32>
                // CHECK-NEXT:     coreai.output %[[V9]] : tensor<1x8x8x8xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestCopyIR:
    def test_broadcast_static(self) -> None:
        class CopyModel(nn.Module):
            def forward(self, dest: Tensor, src: Tensor) -> Tensor:
                return torch.ops.aten.copy.default(dest, src)

        ir = get_ir(
            CopyModel().eval(),
            dest=torch.zeros(2, 3),
            src=torch.rand(1, 3),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "dest"}, %[[ARG1:.*]]: tensor<1x3xf32> {coreai.name = "src"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2> : tensor<1xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.broadcast_in_dims %[[ARG1]], %[[V0]], %[[V1]] : (tensor<1x3xf32>, tensor<1xui32>, tensor<1xsi32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dtype_cast(self) -> None:
        class CopyModel(nn.Module):
            def forward(self, dest: Tensor, src: Tensor) -> Tensor:
                return torch.ops.aten.copy.default(dest, src)

        ir = get_ir(
            CopyModel().eval(),
            dest=torch.zeros(2, 3, dtype=torch.float16),
            src=torch.rand(2, 3),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf16> {coreai.name = "dest"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "src"}) -> (tensor<2x3xf16> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.cast %[[ARG1]] : tensor<2x3xf32> to tensor<2x3xf16>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xf16>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class CopyModel(nn.Module):
            def forward(self, dest: Tensor, src: Tensor) -> Tensor:
                return torch.ops.aten.copy.default(dest, src)

        dest = torch.zeros(2, 3)
        src = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            CopyModel().eval(),
            dest=dest,
            src=src,
            dynamic_shapes={"dest": {0: batch}, "src": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "dest"}, %[[ARG1:.*]]: tensor<?x3xf32> {coreai.name = "src"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.get_shape %[[ARG0]] : tensor<?x3xf32> -> tensor<2xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.broadcast_to %[[ARG1]], %[[V0]] : (tensor<?x3xf32>, tensor<2xui32>) -> tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestCumsumIR:
    def test_static(self) -> None:
        class CumsumModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.cumsum(x, dim=1)

        ir = get_ir(CumsumModel().eval(), x=torch.rand(2, 5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x5xf32> {coreai.name = "x"}) -> (tensor<2x5xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<ui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<false> : tensor<i1>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.scan %[[ARG0]], %[[V0]], %[[V1]] combiner = <sum> : (tensor<2x5xf32>, tensor<ui32>, tensor<i1>) -> tensor<2x5xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<2x5xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    @pytest.mark.parametrize("dim", [-1, -2])
    def test_negative_dim(self, dim: int) -> None:
        class CumsumModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.cumsum(x, dim=dim)

        normalized = dim + 2  # rank=2
        ir = get_ir(CumsumModel().eval(), x=torch.rand(2, 5))
        filecheck_pattern(
            ir,
            check_file=f"""
                // CHECK-LABEL: module {{
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x5xf32> {{coreai.name = "x"}}) -> (tensor<2x5xf32> {{coreai.name = "{{{{.*}}}}"}}) attributes {{__coreai_pure__}} {{
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{normalized}> : tensor<ui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<false> : tensor<i1>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.scan %[[ARG0]], %[[V0]], %[[V1]] combiner = <sum> : (tensor<2x5xf32>, tensor<ui32>, tensor<i1>) -> tensor<2x5xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<2x5xf32>
                // CHECK-NEXT:   }}
                // CHECK-NEXT: }}
            """,
        )

    def test_dynamic(self) -> None:
        class CumsumModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.cumsum(x, dim=1)

        x = torch.rand(2, 5)
        ir = get_ir(
            CumsumModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<ui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<false> : tensor<i1>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.scan %[[ARG0]], %[[V0]], %[[V1]] combiner = <sum> : (tensor<?x?xf32>, tensor<ui32>, tensor<i1>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestDivScalarIR:
    def test_static(self) -> None:
        class DivScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.div(x, 2.0)

        ir = get_ir(DivScalarModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_divide %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class DivScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.div(x, 2.0)

        x = torch.rand(2, 3)
        ir = get_ir(
            DivScalarModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_divide %[[ARG0]], %[[V0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestDivTensorIR:
    def test_static(self) -> None:
        class DivTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.div(x, y)

        ir = get_ir(DivTensorModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_divide %[[ARG0]], %[[ARG1]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class DivTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.div(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            DivTensorModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_divide %[[ARG0]], %[[ARG1]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestDivTensorModeIR:
    def test_floor(self) -> None:
        class DivFloorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.div(x, y, rounding_mode="floor")

        ir = get_ir(DivFloorModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_floor_divide %[[ARG0]], %[[ARG1]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_floor_dynamic(self) -> None:
        class DivFloorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.div(x, y, rounding_mode="floor")

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            DivFloorModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_floor_divide %[[ARG0]], %[[ARG1]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_trunc_float(self) -> None:
        class DivTruncModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.div(x, y, rounding_mode="trunc")

        ir = get_ir(DivTruncModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.decomposable.broadcasting_divide %[[ARG0]], %[[ARG1]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.decomposable.broadcasting_greater %[[V2]], %[[V1]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.cast %[[V3]] : tensor<2x3xi1> to tensor<2x3xf32>
                // CHECK-NEXT:     %[[V5:.*]] = coreai.decomposable.broadcasting_greater %[[V1]], %[[V2]] : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[V6:.*]] = coreai.cast %[[V5]] : tensor<2x3xi1> to tensor<2x3xf32>
                // CHECK-NEXT:     %[[V7:.*]] = coreai.decomposable.broadcasting_sub %[[V4]], %[[V6]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V8:.*]] = coreai.abs %[[V2]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V9:.*]] = coreai.decomposable.broadcasting_floor_divide %[[V8]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V10:.*]] = coreai.decomposable.broadcasting_mul %[[V7]], %[[V9]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V10]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_trunc_int(self) -> None:
        class DivTruncModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.div(x, y, rounding_mode="trunc")

        ir = get_ir(
            DivTruncModel().eval(),
            x=torch.randint(1, 10, (2, 3), dtype=torch.int32),
            y=torch.randint(1, 10, (2, 3), dtype=torch.int32),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xsi32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xsi32> {coreai.name = "y"}) -> (tensor<2x3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_divide %[[ARG0]], %[[ARG1]] : (tensor<2x3xsi32>, tensor<2x3xsi32>) -> tensor<2x3xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_trunc_int_dynamic(self) -> None:
        class DivTruncModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.div(x, y, rounding_mode="trunc")

        x = torch.randint(1, 10, (2, 3), dtype=torch.int32)
        y = torch.randint(1, 10, (2, 3), dtype=torch.int32)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            DivTruncModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xsi32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x3xsi32> {coreai.name = "y"}) -> (tensor<?x3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_divide %[[ARG0]], %[[ARG1]] : (tensor<?x3xsi32>, tensor<?x3xsi32>) -> tensor<?x3xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestEmbeddingIR:
    def test_static(self) -> None:
        class EmbeddingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 4)

            def forward(self, x: Tensor) -> Tensor:
                return self.emb(x)

        ir = get_ir(
            EmbeddingModel().eval(),
            x=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xsi32> {coreai.name = "x"}) -> (tensor<2x3x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<10x4xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<[2, 3, 1]> : tensor<3xui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.reshape %[[ARG0]], %[[V1]] : (tensor<2x3xsi32>, tensor<3xui32>) -> tensor<2x3x1xsi32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.gather_nd %[[V0]] at %[[V2]] : (tensor<10x4xf32>, tensor<2x3x1xsi32>) to tensor<2x3x4xf32>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<2x3x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class EmbeddingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 4)

            def forward(self, x: Tensor) -> Tensor:
                return self.emb(x)

        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        ir = get_ir(
            EmbeddingModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xsi32> {coreai.name = "x"}) -> (tensor<?x?x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<10x4xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<2> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.expand_dims %[[ARG0]], %[[V1]] : (tensor<?x?xsi32>, tensor<1xsi32>) to tensor<?x?x1xsi32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.gather_nd %[[V0]] at %[[V2]] : (tensor<10x4xf32>, tensor<?x?x1xsi32>) to tensor<?x?x4xf32>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<?x?x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_rank1_static(self) -> None:
        class EmbeddingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 4)

            def forward(self, x: Tensor) -> Tensor:
                return self.emb(x)

        ir = get_ir(
            EmbeddingModel().eval(),
            x=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<4xsi32> {coreai.name = "x"}) -> (tensor<4x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<10x4xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<[4, 1]> : tensor<2xui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.reshape %[[ARG0]], %[[V1]] : (tensor<4xsi32>, tensor<2xui32>) -> tensor<4x1xsi32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.gather_nd %[[V0]] at %[[V2]] : (tensor<10x4xf32>, tensor<4x1xsi32>) to tensor<4x4xf32>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_rank1_dynamic(self) -> None:
        class EmbeddingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 4)

            def forward(self, x: Tensor) -> Tensor:
                return self.emb(x)

        x = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        ir = get_ir(
            EmbeddingModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?xsi32> {coreai.name = "x"}) -> (tensor<?x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<{{.*}}> : tensor<10x4xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.expand_dims %[[ARG0]], %[[V1]] : (tensor<?xsi32>, tensor<1xsi32>) to tensor<?x1xsi32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.gather_nd %[[V0]] at %[[V2]] : (tensor<10x4xf32>, tensor<?x1xsi32>) to tensor<?x4xf32>
                // CHECK-NEXT:     coreai.output %[[V3]] : tensor<?x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestEmptyIR:
    # NOTE: torch.empty and Tensor.new_empty both decompose to
    # aten.empty.memory_format; aten.empty.default is not directly reachable
    # from torch APIs but shares the same `replace_empty` lowering.
    def test_static(self) -> None:
        class EmptyModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x + torch.empty(2, 3)

        ir = get_ir(EmptyModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<0.000000e+00> : tensor<2x3xf32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_add %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class EmptyModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x + x.new_empty(x.shape[0], 3)

        x = torch.rand(2, 3)
        ir = get_ir(
            EmptyModel().eval(),
            x=x,
            dynamic_shapes={"x": {0: torch.export.Dim("b")}},
        )
        # The dynamic path builds a shape tensor and broadcasts a 1x1 zeros
        # constant; verify the empty-specific ops without pinning every
        # intermediate constant.
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK:           coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK:             %[[ZERO:.*]] = coreai.constant dense<0.000000e+00> : tensor<1x1xf32>
                // CHECK:             %[[BCAST:.*]] = coreai.broadcast_to %[[ZERO]], %{{.*}} : (tensor<1x1xf32>, tensor<2xui32>) -> tensor<?x3xf32>
                // CHECK:             %[[OUT:.*]] = coreai.decomposable.broadcasting_add %[[ARG0]], %[[BCAST]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
                // CHECK:             coreai.output %[[OUT]] : tensor<?x3xf32>
                // CHECK:           }
                // CHECK:         }
            """,
        )


class TestEqScalarIR:
    def test_static(self) -> None:
        class EqScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.eq(x, 2.0)

        ir = get_ir(EqScalarModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_equal %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class EqScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.eq(x, 2.0)

        x = torch.rand(2, 3)
        ir = get_ir(
            EqScalarModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_equal %[[ARG0]], %[[V0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x?xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestEqTensorIR:
    def test_static(self) -> None:
        class EqTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.eq(x, y)

        ir = get_ir(EqTensorModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_equal %[[ARG0]], %[[ARG1]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class EqTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.eq(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            EqTensorModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_equal %[[ARG0]], %[[ARG1]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xi1>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestExp2IR:
    def test_static(self) -> None:
        class Exp2Model(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.exp2(x)

        ir = get_ir(Exp2Model().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_pow %[[V0]], %[[ARG0]] : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class Exp2Model(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.exp2(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            Exp2Model().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_pow %[[V0]], %[[ARG0]] : (tensor<f32>, tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestExpandIR:
    def test_static(self) -> None:
        class ExpandModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.expand(4, 3)

        ir = get_ir(ExpandModel().eval(), x=torch.rand(1, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x3xf32> {coreai.name = "x"}) -> (tensor<4x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<4> : tensor<1xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.broadcast_in_dims %[[ARG0]], %[[V0]], %[[V1]] : (tensor<1x3xf32>, tensor<1xui32>, tensor<1xsi32>) -> tensor<4x3xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<4x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_with_leading_dim(self) -> None:
        class ExpandModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.expand(2, 4, 3)

        ir = get_ir(ExpandModel().eval(), x=torch.rand(1, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x3xf32> {coreai.name = "x"}) -> (tensor<2x4x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<[2, 4]> : tensor<2xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.constant dense<[1, 1, 3]> : tensor<3xui32>
                // CHECK-NEXT:     %[[V3:.*]] = coreai.reshape %[[ARG0]], %[[V2]] : (tensor<1x3xf32>, tensor<3xui32>) -> tensor<1x1x3xf32>
                // CHECK-NEXT:     %[[V4:.*]] = coreai.broadcast_in_dims %[[V3]], %[[V0]], %[[V1]] : (tensor<1x1x3xf32>, tensor<2xui32>, tensor<2xsi32>) -> tensor<2x4x3xf32>
                // CHECK-NEXT:     coreai.output %[[V4]] : tensor<2x4x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class ExpandModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x.expand(y.shape[0], 3)

        x = torch.rand(1, 3)
        y = torch.rand(4)
        ir = get_ir(
            ExpandModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {}, "y": {0: torch.export.Dim("b")}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<1x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?xf32> {coreai.name = "y"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<0> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.get_shape %[[ARG1]] : tensor<?xf32> -> tensor<1xui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.broadcast_in_dims %[[ARG0]], %[[V1]], %[[V0]] : (tensor<1x3xf32>, tensor<1xui32>, tensor<1xsi32>) -> tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestExpm1IR:
    def test_static(self) -> None:
        class Expm1Model(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.expm1(x)

        ir = get_ir(Expm1Model().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.exp %[[ARG0]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.decomposable.broadcasting_sub %[[V1]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class Expm1Model(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.expm1(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            Expm1Model().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.exp %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.decomposable.broadcasting_sub %[[V1]], %[[V0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestFlipIR:
    def test_static(self) -> None:
        class FlipModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.flip(x, [0])

        ir = get_ir(FlipModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<0> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.reverse %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<1xsi32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class FlipModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.flip(x, [0])

        x = torch.rand(2, 3)
        ir = get_ir(
            FlipModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<0> : tensor<1xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.reverse %[[ARG0]], %[[V0]] : (tensor<?x?xf32>, tensor<1xsi32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_multi_dim(self) -> None:
        class FlipModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.flip(x, [0, 1])

        ir = get_ir(FlipModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.reverse %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<2xsi32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestFloorIR:
    def test_static(self) -> None:
        class FloorModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.floor(x)

        ir = get_ir(FloorModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_floor_divide %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class FloorModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.floor(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            FloorModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_floor_divide %[[ARG0]], %[[V0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestFloorDivideIR:
    def test_static_tensor(self) -> None:
        class FloorDivideModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.floor_divide(x, y)

        ir = get_ir(
            FloorDivideModel().eval(),
            x=torch.rand(2, 3) + 1.0,
            y=torch.rand(2, 3) + 1.0,
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_floor_divide %[[ARG0]], %[[ARG1]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_static_scalar(self) -> None:
        class FloorDivideScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.floor_divide(x, 2.0)

        ir = get_ir(FloorDivideScalarModel().eval(), x=torch.rand(2, 3) + 1.0)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_floor_divide %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class FloorDivideModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.floor_divide(x, y)

        x = torch.rand(2, 3) + 1.0
        y = torch.rand(2, 3) + 1.0
        batch = torch.export.Dim("batch")
        ir = get_ir(
            FloorDivideModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_floor_divide %[[ARG0]], %[[ARG1]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestFloordivIR:
    def test_static_tensor(self) -> None:
        class FloordivModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x // y

        ir = get_ir(
            FloordivModel().eval(),
            x=torch.randint(1, 10, (2, 3), dtype=torch.int32),
            y=torch.randint(1, 10, (2, 3), dtype=torch.int32),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xsi32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xsi32> {coreai.name = "y"}) -> (tensor<2x3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_floor_divide %[[ARG0]], %[[ARG1]] : (tensor<2x3xsi32>, tensor<2x3xsi32>) -> tensor<2x3xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_static_scalar(self) -> None:
        class FloordivScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x // 3

        ir = get_ir(
            FloordivScalarModel().eval(),
            x=torch.randint(1, 100, (2, 3), dtype=torch.int32),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xsi32> {coreai.name = "x"}) -> (tensor<2x3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<3> : tensor<si32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_floor_divide %[[ARG0]], %[[V0]] : (tensor<2x3xsi32>, tensor<si32>) -> tensor<2x3xsi32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class FloordivModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x // y

        x = torch.randint(1, 10, (2, 3), dtype=torch.int32)
        y = torch.randint(1, 10, (2, 3), dtype=torch.int32)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            FloordivModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xsi32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x3xsi32> {coreai.name = "y"}) -> (tensor<?x3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_floor_divide %[[ARG0]], %[[ARG1]] : (tensor<?x3xsi32>, tensor<?x3xsi32>) -> tensor<?x3xsi32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestFmodIR:
    def test_static_tensor(self) -> None:
        class FmodTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.fmod(x, y)

        ir = get_ir(
            FmodTensorModel().eval(),
            x=torch.rand(2, 3) + 1.0,
            y=torch.rand(2, 3) + 1.0,
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_modulo %[[ARG0]], %[[ARG1]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic_tensor(self) -> None:
        class FmodTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.fmod(x, y)

        x = torch.rand(2, 3) + 1.0
        y = torch.rand(2, 3) + 1.0
        batch = torch.export.Dim("batch")
        ir = get_ir(
            FmodTensorModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_modulo %[[ARG0]], %[[ARG1]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_static_scalar(self) -> None:
        class FmodScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.fmod(x, 2.0)

        ir = get_ir(FmodScalarModel().eval(), x=torch.rand(2, 3) + 1.0)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_modulo %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic_scalar(self) -> None:
        class FmodScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.fmod(x, 2.0)

        x = torch.rand(2, 3) + 1.0
        ir = get_ir(
            FmodScalarModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_modulo %[[ARG0]], %[[V0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestFullIR:
    def test_static(self) -> None:
        class FullModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.full((2, 3), 5.0)

        ir = get_ir(FullModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<5.000000e+00> : tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class FullDynModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.full((x.shape[0], 3), 5.0)

        x = torch.rand(2, 3)
        ir = get_ir(
            FullDynModel().eval(),
            x=x,
            dynamic_shapes={"x": {0: torch.export.Dim("b")}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK:          %[[FILL:.*]] = coreai.constant dense<5.000000e+00> : tensor<1x1xf32>
                // CHECK:          %[[SHAPE:.*]] = coreai.get_shape %[[ARG0]] : tensor<?x3xf32> -> tensor<2xui32>
                // CHECK:          coreai.broadcast_to %[[FILL]], %{{.*}} : (tensor<1x1xf32>, tensor<2xui32>) -> tensor<?x3xf32>
                // CHECK:          coreai.output
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestFullLikeIR:
    def test_static(self) -> None:
        class FullLikeModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.full_like(x, 7.0)

        ir = get_ir(FullLikeModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<7.000000e+00> : tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class FullLikeModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.full_like(x, 7.0)

        x = torch.rand(2, 3)
        ir = get_ir(
            FullLikeModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<7.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.get_shape %[[ARG0]] : tensor<?x?xf32> -> tensor<2xui32>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.broadcast_to %[[V0]], %[[V1]] : (tensor<f32>, tensor<2xui32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestGatherIR:
    def test_static(self) -> None:
        class GatherModel(nn.Module):
            def forward(self, x: Tensor, idx: Tensor) -> Tensor:
                return torch.gather(x, 1, idx)

        ir = get_ir(
            GatherModel().eval(),
            x=torch.rand(2, 3),
            idx=torch.zeros(2, 3, dtype=torch.int64),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xsi32> {coreai.name = "idx"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.gather_along_axis %[[ARG0]] at %[[ARG1]] along %[[V0]] : (tensor<2x3xf32>, tensor<2x3xsi32>, tensor<si32>) to tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class GatherModel(nn.Module):
            def forward(self, x: Tensor, idx: Tensor) -> Tensor:
                return torch.gather(x, 1, idx)

        x = torch.rand(2, 3)
        idx = torch.zeros(2, 3, dtype=torch.int64)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            GatherModel().eval(),
            x=x,
            idx=idx,
            dynamic_shapes={"x": {0: batch}, "idx": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x3xsi32> {coreai.name = "idx"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.gather_along_axis %[[ARG0]] at %[[ARG1]] along %[[V0]] : (tensor<?x3xf32>, tensor<?x3xsi32>, tensor<si32>) to tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestGeScalarIR:
    def test_static(self) -> None:
        class GeScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x >= 2.0

        ir = get_ir(GeScalarModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_greater %[[V0]], %[[ARG0]] : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.not %[[V1]] : tensor<2x3xi1> -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class GeScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x >= 2.0

        x = torch.rand(2, 3)
        ir = get_ir(
            GeScalarModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_greater %[[V0]], %[[ARG0]] : (tensor<f32>, tensor<?x?xf32>) -> tensor<?x?xi1>
                // CHECK-NEXT:     %[[V2:.*]] = coreai.not %[[V1]] : tensor<?x?xi1> -> tensor<?x?xi1>
                // CHECK-NEXT:     coreai.output %[[V2]] : tensor<?x?xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestGeTensorIR:
    def test_static(self) -> None:
        class GeTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.ge(x, y)

        ir = get_ir(GeTensorModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_greater %[[ARG1]], %[[ARG0]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.not %[[V0]] : tensor<2x3xi1> -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class GeTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.ge(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            GeTensorModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_greater %[[ARG1]], %[[ARG0]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xi1>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.not %[[V0]] : tensor<?x3xi1> -> tensor<?x3xi1>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestGeluIR:
    def test_static(self) -> None:
        class GeluModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.gelu(x)

        ir = get_ir(GeluModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.gelu %[[ARG0]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class GeluModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.gelu(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            GeluModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.gelu %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_approximate_tanh(self) -> None:
        class GeluTanhModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.gelu(x, approximate="tanh")

        ir = get_ir(GeluTanhModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.gelu %[[ARG0]] approximate = <tanh> : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_approximate_tanh_dynamic(self) -> None:
        class GeluTanhModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.gelu(x, approximate="tanh")

        x = torch.rand(2, 3)
        ir = get_ir(
            GeluTanhModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.gelu %[[ARG0]] approximate = <tanh> : tensor<?x?xf32> -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestGetitemIR:
    def test_static(self) -> None:
        class GetitemModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                a, b = torch.split(x, 2, dim=0)
                return a + b

        ir = get_ir(GetitemModel().eval(), x=torch.rand(4, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<4x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2> : tensor<2xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[SPLIT:.*]]:2 = coreai.split %[[ARG0]], %[[V0]], %[[V1]] : (tensor<4x3xf32>, tensor<2xui32>, tensor<si32>) -> (tensor<2x3xf32>, tensor<2x3xf32>)
                // CHECK-NEXT:     %[[ADD:.*]] = coreai.decomposable.broadcasting_add %[[SPLIT]]#0, %[[SPLIT]]#1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[ADD]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class GetitemModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                a, b = torch.split(x, 2, dim=0)
                return a + b

        x = torch.rand(4, 3)
        ir = get_ir(
            GetitemModel().eval(),
            x=x,
            dynamic_shapes={"x": {1: torch.export.Dim("d1")}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<4x?xf32> {coreai.name = "x"}) -> (tensor<2x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2> : tensor<2xui32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[SPLIT:.*]]:2 = coreai.split %[[ARG0]], %[[V0]], %[[V1]] : (tensor<4x?xf32>, tensor<2xui32>, tensor<si32>) -> (tensor<2x?xf32>, tensor<2x?xf32>)
                // CHECK-NEXT:     %[[ADD:.*]] = coreai.decomposable.broadcasting_add %[[SPLIT]]#0, %[[SPLIT]]#1 : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
                // CHECK-NEXT:     coreai.output %[[ADD]] : tensor<2x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestGtScalarIR:
    def test_static(self) -> None:
        class GtScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x > 2.0

        ir = get_ir(GtScalarModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_greater %[[ARG0]], %[[V0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class GtScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x > 2.0

        x = torch.rand(2, 3)
        ir = get_ir(
            GtScalarModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[V1:.*]] = coreai.decomposable.broadcasting_greater %[[ARG0]], %[[V0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
                // CHECK-NEXT:     coreai.output %[[V1]] : tensor<?x?xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestGtTensorIR:
    def test_static(self) -> None:
        class GtTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.gt(x, y)

        ir = get_ir(GtTensorModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_greater %[[ARG0]], %[[ARG1]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class GtTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.gt(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            GtTensorModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[ARG0:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[ARG1:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[V0:.*]] = coreai.decomposable.broadcasting_greater %[[ARG0]], %[[ARG1]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xi1>
                // CHECK-NEXT:     coreai.output %[[V0]] : tensor<?x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestHardsigmoidIR:
    def test_static(self) -> None:
        class HardsigmoidModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.hardsigmoid(x)

        ir = get_ir(
            HardsigmoidModel().eval(),
            x=torch.rand(2, 3),
            remove_decomps=[torch.ops.aten.hardsigmoid.default],
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @hard_sigmoid_{{.*}}(%[[INPUT:.*]]: tensor<2x3xf32> {coreai.name = "input"}) -> tensor<2x3xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"hard_sigmoid" = {input_names = ["input"], op_attrs = {version = 1 : si64}, output_names = ["output"]}>, template_op = "hard_sigmoid"} {
                // CHECK-NEXT:     %[[C3:.*]] = coreai.constant dense<3.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[C0:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[C6:.*]] = coreai.constant dense<6.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[ADD:.*]] = coreai.decomposable.broadcasting_add %[[INPUT]], %[[C3]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[MAX:.*]] = coreai.decomposable.broadcasting_maximum %[[ADD]], %[[C0]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[MIN:.*]] = coreai.decomposable.broadcasting_minimum %[[MAX]], %[[C6]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[DIV:.*]] = coreai.decomposable.broadcasting_divide %[[MIN]], %[[C6]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[DIV]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.invoke @hard_sigmoid_{{.*}}(%[[X]])  : (tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class HardsigmoidModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.hardsigmoid(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            HardsigmoidModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
            remove_decomps=[torch.ops.aten.hardsigmoid.default],
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @hard_sigmoid_{{.*}}(%[[INPUT:.*]]: tensor<?x?xf32> {coreai.name = "input"}) -> tensor<?x?xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"hard_sigmoid" = {input_names = ["input"], op_attrs = {version = 1 : si64}, output_names = ["output"]}>, template_op = "hard_sigmoid"} {
                // CHECK-NEXT:     %[[C3:.*]] = coreai.constant dense<3.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[C0:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[C6:.*]] = coreai.constant dense<6.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[ADD:.*]] = coreai.decomposable.broadcasting_add %[[INPUT]], %[[C3]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[MAX:.*]] = coreai.decomposable.broadcasting_maximum %[[ADD]], %[[C0]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[MIN:.*]] = coreai.decomposable.broadcasting_minimum %[[MAX]], %[[C6]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[DIV:.*]] = coreai.decomposable.broadcasting_divide %[[MIN]], %[[C6]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[DIV]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.invoke @hard_sigmoid_{{.*}}(%[[X]])  : (tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestHardswishIR:
    def test_static(self) -> None:
        class HardswishModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.hardswish(x)

        ir = get_ir(
            HardswishModel().eval(),
            x=torch.rand(2, 3),
            remove_decomps=[torch.ops.aten.hardswish.default],
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @hard_sigmoid_{{.*}}(%[[INPUT:.*]]: tensor<2x3xf32> {coreai.name = "input"}) -> tensor<2x3xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"hard_sigmoid" = {input_names = ["input"], op_attrs = {version = 1 : si64}, output_names = ["output"]}>, template_op = "hard_sigmoid"} {
                // CHECK:           coreai.output %{{.*}} : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[H:.*]] = coreai.invoke @hard_sigmoid_{{.*}}(%[[X]])  : (tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_mul %[[X]], %[[H]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class HardswishModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.hardswish(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            HardswishModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
            remove_decomps=[torch.ops.aten.hardswish.default],
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @hard_sigmoid_{{.*}}(%[[INPUT:.*]]: tensor<?x?xf32> {coreai.name = "input"}) -> tensor<?x?xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"hard_sigmoid" = {input_names = ["input"], op_attrs = {version = 1 : si64}, output_names = ["output"]}>, template_op = "hard_sigmoid"} {
                // CHECK:           coreai.output %{{.*}} : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[H:.*]] = coreai.invoke @hard_sigmoid_{{.*}}(%[[X]])  : (tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_mul %[[X]], %[[H]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestHardtanhIR:
    def test_static(self) -> None:
        class HardtanhModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.hardtanh(x)

        ir = get_ir(HardtanhModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[CMIN:.*]] = coreai.constant dense<-1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[CMAX:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[MAX:.*]] = coreai.decomposable.broadcasting_maximum %[[X]], %[[CMIN]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[MIN:.*]] = coreai.decomposable.broadcasting_minimum %[[MAX]], %[[CMAX]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[MIN]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class HardtanhModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.hardtanh(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            HardtanhModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[CMIN:.*]] = coreai.constant dense<-1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[CMAX:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[MAX:.*]] = coreai.decomposable.broadcasting_maximum %[[X]], %[[CMIN]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[MIN:.*]] = coreai.decomposable.broadcasting_minimum %[[MAX]], %[[CMAX]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[MIN]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestIndexPutIR:
    def test_static(self) -> None:
        class IndexPutModel(nn.Module):
            def forward(self, x: Tensor, idx: Tensor, vals: Tensor) -> Tensor:
                x = x.clone()
                x[idx] = vals
                return x

        ir = get_ir(
            IndexPutModel().eval(),
            x=torch.rand(5, 3),
            idx=torch.tensor([0, 2], dtype=torch.int64),
            vals=torch.rand(2, 3),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<5x3xf32> {coreai.name = "x"}, %[[IDX:.*]]: tensor<2xsi32> {coreai.name = "idx"}, %[[VALS:.*]]: tensor<2x3xf32> {coreai.name = "vals"}) -> (tensor<5x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[2, 1]> : tensor<2xui32>
                // CHECK-NEXT:     %[[RESHAPED:.*]] = coreai.reshape %[[IDX]], %[[SHAPE]] : (tensor<2xsi32>, tensor<2xui32>) -> tensor<2x1xsi32>
                // CHECK-NEXT:     %[[CAST:.*]] = coreai.cast %[[RESHAPED]] : tensor<2x1xsi32> to tensor<?x1xsi32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.scatter_nd %[[X]] with %[[VALS]] at %[[CAST]] {
                // CHECK-NEXT:     ^bb0(%[[A:.*]]: tensor<f32>, %[[B:.*]]: tensor<f32>):
                // CHECK-NEXT:       coreai.yield %[[B]] : tensor<f32>
                // CHECK-NEXT:     } : (tensor<5x3xf32>, tensor<2x3xf32>, tensor<?x1xsi32>) to tensor<5x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<5x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class IndexPutModel(nn.Module):
            def forward(self, x: Tensor, idx: Tensor, vals: Tensor) -> Tensor:
                x = x.clone()
                x[idx] = vals
                return x

        x = torch.rand(5, 3)
        idx = torch.tensor([0, 2], dtype=torch.int64)
        vals = torch.rand(2, 3)
        ir = get_ir(
            IndexPutModel().eval(),
            x=x,
            idx=idx,
            vals=vals,
            dynamic_shapes={"x": {0: torch.export.Dim("n")}, "idx": None, "vals": None},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[IDX:.*]]: tensor<2xsi32> {coreai.name = "idx"}, %[[VALS:.*]]: tensor<2x3xf32> {coreai.name = "vals"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[2, 1]> : tensor<2xui32>
                // CHECK-NEXT:     %[[RESHAPED:.*]] = coreai.reshape %[[IDX]], %[[SHAPE]] : (tensor<2xsi32>, tensor<2xui32>) -> tensor<2x1xsi32>
                // CHECK-NEXT:     %[[CAST:.*]] = coreai.cast %[[RESHAPED]] : tensor<2x1xsi32> to tensor<?x1xsi32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.scatter_nd %[[X]] with %[[VALS]] at %[[CAST]] {
                // CHECK-NEXT:     ^bb0(%[[A:.*]]: tensor<f32>, %[[B:.*]]: tensor<f32>):
                // CHECK-NEXT:       coreai.yield %[[B]] : tensor<f32>
                // CHECK-NEXT:     } : (tensor<?x3xf32>, tensor<2x3xf32>, tensor<?x1xsi32>) to tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestIndexSelectIR:
    def test_static(self) -> None:
        class IndexSelectModel(nn.Module):
            def forward(self, x: Tensor, idx: Tensor) -> Tensor:
                return torch.index_select(x, 0, idx)

        ir = get_ir(
            IndexSelectModel().eval(),
            x=torch.rand(5, 3),
            idx=torch.tensor([0, 2, 4], dtype=torch.int64),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<5x3xf32> {coreai.name = "x"}, %[[IDX:.*]]: tensor<3xsi32> {coreai.name = "idx"}) -> (tensor<3x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[AXES:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[SIZE:.*]] = coreai.constant dense<3> : tensor<1xui32>
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[3, 1]> : tensor<2xui32>
                // CHECK-NEXT:     %[[RESHAPED:.*]] = coreai.reshape %[[IDX]], %[[SHAPE]] : (tensor<3xsi32>, tensor<2xui32>) -> tensor<3x1xsi32>
                // CHECK-NEXT:     %[[BCAST:.*]] = coreai.broadcast_in_dims %[[RESHAPED]], %[[SIZE]], %[[AXES]] : (tensor<3x1xsi32>, tensor<1xui32>, tensor<1xsi32>) -> tensor<3x3xsi32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.gather_along_axis %[[X]] at %[[BCAST]] along %[[DIM]] : (tensor<5x3xf32>, tensor<3x3xsi32>, tensor<si32>) to tensor<3x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<3x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class IndexSelectModel(nn.Module):
            def forward(self, x: Tensor, idx: Tensor) -> Tensor:
                return torch.index_select(x, 0, idx)

        x = torch.rand(5, 3)
        idx = torch.tensor([0, 2], dtype=torch.int64)
        ir = get_ir(
            IndexSelectModel().eval(),
            x=x,
            idx=idx,
            dynamic_shapes={"x": {1: torch.export.Dim("n")}, "idx": None},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<5x?xf32> {coreai.name = "x"}, %[[IDX:.*]]: tensor<2xsi32> {coreai.name = "idx"}) -> (tensor<2x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[END:.*]] = coreai.constant dense<2> : tensor<1xsi32>
                // CHECK-NEXT:     %[[START:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[2, 1]> : tensor<2xui32>
                // CHECK-NEXT:     %[[RESHAPED:.*]] = coreai.reshape %[[IDX]], %[[SHAPE]] : (tensor<2xsi32>, tensor<2xui32>) -> tensor<2x1xsi32>
                // CHECK-NEXT:     %[[XSHAPE:.*]] = coreai.get_shape %[[X]] : tensor<5x?xf32> -> tensor<2xui32>
                // CHECK-NEXT:     %[[SLICE:.*]] = coreai.slice %[[XSHAPE]], %[[START]], %[[END]], %[[START]] : (tensor<2xui32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[BCAST:.*]] = coreai.broadcast_in_dims %[[RESHAPED]], %[[SLICE]], %[[START]] : (tensor<2x1xsi32>, tensor<1xui32>, tensor<1xsi32>) -> tensor<2x?xsi32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.gather_along_axis %[[X]] at %[[BCAST]] along %[[DIM]] : (tensor<5x?xf32>, tensor<2x?xsi32>, tensor<si32>) to tensor<2x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestIndexTensorIR:
    def test_static(self) -> None:
        class IndexTensorModel(nn.Module):
            def forward(self, x: Tensor, idx: Tensor) -> Tensor:
                return x[idx]

        ir = get_ir(
            IndexTensorModel().eval(),
            x=torch.rand(5, 3),
            idx=torch.tensor([0, 2, 4], dtype=torch.int64),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<5x3xf32> {coreai.name = "x"}, %[[IDX:.*]]: tensor<3xsi32> {coreai.name = "idx"}) -> (tensor<3x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[3, 1]> : tensor<2xui32>
                // CHECK-NEXT:     %[[RESHAPED:.*]] = coreai.reshape %[[IDX]], %[[SHAPE]] : (tensor<3xsi32>, tensor<2xui32>) -> tensor<3x1xsi32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.gather_nd %[[X]] at %[[RESHAPED]] : (tensor<5x3xf32>, tensor<3x1xsi32>) to tensor<3x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<3x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class IndexTensorModel(nn.Module):
            def forward(self, x: Tensor, idx: Tensor) -> Tensor:
                return x[idx]

        x = torch.rand(5, 3)
        idx = torch.tensor([0, 2, 4], dtype=torch.int64)
        ir = get_ir(
            IndexTensorModel().eval(),
            x=x,
            idx=idx,
            dynamic_shapes={"x": {0: torch.export.Dim("n")}, "idx": None},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[IDX:.*]]: tensor<3xsi32> {coreai.name = "idx"}) -> (tensor<3x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[3, 1]> : tensor<2xui32>
                // CHECK-NEXT:     %[[RESHAPED:.*]] = coreai.reshape %[[IDX]], %[[SHAPE]] : (tensor<3xsi32>, tensor<2xui32>) -> tensor<3x1xsi32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.gather_nd %[[X]] at %[[RESHAPED]] : (tensor<?x3xf32>, tensor<3x1xsi32>) to tensor<3x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<3x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestInstanceNormIR:
    def test_static(self) -> None:
        class InstanceNormModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.InstanceNorm2d(3, affine=True)

            def forward(self, x: Tensor) -> Tensor:
                return self.norm(x)

        ir = get_ir(
            InstanceNormModel().eval(),
            x=torch.rand(2, 3, 4, 4),
            remove_decomps=[torch.ops.aten.instance_norm.default],
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @instance_norm_{{.*}}(%[[INPUT:.*]]: tensor<2x3x4x4xf32> {coreai.name = "input"}, %[[GAMMA:.*]]: tensor<3x1x1xf32> {coreai.name = "gamma"}, %[[BETA:.*]]: tensor<3x1x1xf32> {coreai.name = "beta"}) -> tensor<2x3x4x4xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"instance_norm" = {input_names = ["input", "gamma", "beta"], op_attrs = {eps = 9.99999974E-6 : f32, version = 1 : si64}, output_names = ["output"]}>, template_op = "instance_norm"} {
                // CHECK:           coreai.output %{{.*}} : tensor<2x3x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3x4x4xf32> {coreai.name = "x"}) -> (tensor<2x3x4x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[W:.*]] = coreai.constant dense<1.000000e+00> : tensor<3x1x1xf32>
                // CHECK-NEXT:     %[[B:.*]] = coreai.constant dense<0.000000e+00> : tensor<3x1x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.invoke @instance_norm_{{.*}}(%[[X]], %[[W]], %[[B]])  : (tensor<2x3x4x4xf32>, tensor<3x1x1xf32>, tensor<3x1x1xf32>) -> tensor<2x3x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class InstanceNormModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.InstanceNorm2d(3, affine=True)

            def forward(self, x: Tensor) -> Tensor:
                return self.norm(x)

        x = torch.rand(2, 3, 4, 4)
        ir = get_ir(
            InstanceNormModel().eval(),
            x=x,
            dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
            remove_decomps=[torch.ops.aten.instance_norm.default],
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @instance_norm_{{.*}}(%[[INPUT:.*]]: tensor<?x3x4x4xf32> {coreai.name = "input"}, %[[GAMMA:.*]]: tensor<3x1x1xf32> {coreai.name = "gamma"}, %[[BETA:.*]]: tensor<3x1x1xf32> {coreai.name = "beta"}) -> tensor<?x3x4x4xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"instance_norm" = {input_names = ["input", "gamma", "beta"], op_attrs = {eps = 9.99999974E-6 : f32, version = 1 : si64}, output_names = ["output"]}>, template_op = "instance_norm"} {
                // CHECK:           coreai.output %{{.*}} : tensor<?x3x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x3x4x4xf32> {coreai.name = "x"}) -> (tensor<?x3x4x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[W:.*]] = coreai.constant dense<1.000000e+00> : tensor<3x1x1xf32>
                // CHECK-NEXT:     %[[B:.*]] = coreai.constant dense<0.000000e+00> : tensor<3x1x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.invoke @instance_norm_{{.*}}(%[[X]], %[[W]], %[[B]])  : (tensor<?x3x4x4xf32>, tensor<3x1x1xf32>, tensor<3x1x1xf32>) -> tensor<?x3x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x3x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestIsinfIR:
    def test_static(self) -> None:
        class IsinfModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.isinf(x)

        ir = get_ir(IsinfModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[NINF:.*]] = coreai.constant dense<0xFF800000> : tensor<f32>
                // CHECK-NEXT:     %[[PINF:.*]] = coreai.constant dense<0x7F800000> : tensor<f32>
                // CHECK-NEXT:     %[[EQP:.*]] = coreai.decomposable.broadcasting_equal %[[X]], %[[PINF]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[EQN:.*]] = coreai.decomposable.broadcasting_equal %[[X]], %[[NINF]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[OR:.*]] = coreai.decomposable.broadcasting_or %[[EQP]], %[[EQN]] : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[OR]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class IsinfModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.isinf(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            IsinfModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[NINF:.*]] = coreai.constant dense<0xFF800000> : tensor<f32>
                // CHECK-NEXT:     %[[PINF:.*]] = coreai.constant dense<0x7F800000> : tensor<f32>
                // CHECK-NEXT:     %[[EQP:.*]] = coreai.decomposable.broadcasting_equal %[[X]], %[[PINF]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
                // CHECK-NEXT:     %[[EQN:.*]] = coreai.decomposable.broadcasting_equal %[[X]], %[[NINF]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
                // CHECK-NEXT:     %[[OR:.*]] = coreai.decomposable.broadcasting_or %[[EQP]], %[[EQN]] : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
                // CHECK-NEXT:     coreai.output %[[OR]] : tensor<?x?xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLeScalarIR:
    def test_static(self) -> None:
        class LeScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x <= 2.0

        ir = get_ir(LeScalarModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[GT:.*]] = coreai.decomposable.broadcasting_greater %[[X]], %[[C]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.not %[[GT]] : tensor<2x3xi1> -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LeScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x <= 2.0

        x = torch.rand(2, 3)
        ir = get_ir(
            LeScalarModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[GT:.*]] = coreai.decomposable.broadcasting_greater %[[X]], %[[C]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.not %[[GT]] : tensor<?x?xi1> -> tensor<?x?xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLeTensorIR:
    def test_static(self) -> None:
        class LeTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.le(x, y)

        ir = get_ir(LeTensorModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[GT:.*]] = coreai.decomposable.broadcasting_greater %[[X]], %[[Y]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.not %[[GT]] : tensor<2x3xi1> -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LeTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.le(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            LeTensorModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[GT:.*]] = coreai.decomposable.broadcasting_greater %[[X]], %[[Y]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.not %[[GT]] : tensor<?x3xi1> -> tensor<?x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLeakyReluIR:
    def test_static(self) -> None:
        class LeakyReluModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.leaky_relu(x, 0.1)

        ir = get_ir(LeakyReluModel().eval(), x=torch.rand(2, 3) - 0.5)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[ZERO:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[SLOPE:.*]] = coreai.constant dense<1.000000e-01> : tensor<f32>
                // CHECK-NEXT:     %[[POS:.*]] = coreai.decomposable.broadcasting_maximum %[[X]], %[[ZERO]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[NEG:.*]] = coreai.decomposable.broadcasting_minimum %[[X]], %[[ZERO]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[SCALED:.*]] = coreai.decomposable.broadcasting_mul %[[NEG]], %[[SLOPE]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_add %[[POS]], %[[SCALED]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LeakyReluModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.leaky_relu(x, 0.1)

        x = torch.rand(2, 3) - 0.5
        ir = get_ir(
            LeakyReluModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[ZERO:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[SLOPE:.*]] = coreai.constant dense<1.000000e-01> : tensor<f32>
                // CHECK-NEXT:     %[[POS:.*]] = coreai.decomposable.broadcasting_maximum %[[X]], %[[ZERO]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[NEG:.*]] = coreai.decomposable.broadcasting_minimum %[[X]], %[[ZERO]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[SCALED:.*]] = coreai.decomposable.broadcasting_mul %[[NEG]], %[[SLOPE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_add %[[POS]], %[[SCALED]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLiftFreshCopyIR:
    def test_static(self) -> None:
        class LiftFreshCopyModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.aten.lift_fresh_copy.default(x)

        ir = get_ir(LiftFreshCopyModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     coreai.output %[[X]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LiftFreshCopyModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.aten.lift_fresh_copy.default(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            LiftFreshCopyModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     coreai.output %[[X]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLinalgVectorNormIR:
    def test_static(self) -> None:
        class LinalgNormModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.linalg.vector_norm(x)

        ir = get_ir(LinalgNormModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @linalg_vector_norm_{{.*}}(%[[INPUT:.*]]: tensor<2x3xf32> {coreai.name = "input"}) -> tensor<f32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"linalg_vector_norm" = {input_names = ["input"], op_attrs = {axes = [0 : si64, 1 : si64], keep_dim = false, ord = 2.000000e+00 : f32, version = 1 : si64}, output_names = ["output"]}>, template_op = "linalg_vector_norm"} {
                // CHECK:           coreai.output %{{.*}} : tensor<f32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<f32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.invoke @linalg_vector_norm_{{.*}}(%[[X]])  : (tensor<2x3xf32>) -> tensor<f32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<f32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LinalgNormModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.linalg.vector_norm(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            LinalgNormModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @linalg_vector_norm_{{.*}}(%[[INPUT:.*]]: tensor<?x?xf32> {coreai.name = "input"}) -> tensor<f32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"linalg_vector_norm" = {input_names = ["input"], op_attrs = {axes = [0 : si64, 1 : si64], keep_dim = false, ord = 2.000000e+00 : f32, version = 1 : si64}, output_names = ["output"]}>, template_op = "linalg_vector_norm"} {
                // CHECK:           coreai.output %{{.*}} : tensor<f32>
                // CHECK-NEXT:   }
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<f32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.invoke @linalg_vector_norm_{{.*}}(%[[X]])  : (tensor<?x?xf32>) -> tensor<f32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<f32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLog10IR:
    def test_static(self) -> None:
        class Log10Model(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.log10(x)

        ir = get_ir(Log10Model().eval(), x=torch.rand(2, 3) + 1.0)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.30258512> : tensor<f32>
                // CHECK-NEXT:     %[[L:.*]] = coreai.log %[[X]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_divide %[[L]], %[[C]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class Log10Model(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.log10(x)

        x = torch.rand(2, 3) + 1.0
        ir = get_ir(
            Log10Model().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.30258512> : tensor<f32>
                // CHECK-NEXT:     %[[L:.*]] = coreai.log %[[X]] : tensor<?x?xf32> -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_divide %[[L]], %[[C]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLog1pIR:
    def test_static(self) -> None:
        class Log1pModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.log1p(x)

        ir = get_ir(Log1pModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[ADD:.*]] = coreai.decomposable.broadcasting_add %[[X]], %[[C]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.log %[[ADD]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class Log1pModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.log1p(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            Log1pModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[ADD:.*]] = coreai.decomposable.broadcasting_add %[[X]], %[[C]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.log %[[ADD]] : tensor<?x?xf32> -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLog2IR:
    def test_static(self) -> None:
        class Log2Model(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.log2(x)

        ir = get_ir(Log2Model().eval(), x=torch.rand(2, 3) + 1.0)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<0.693147182> : tensor<f32>
                // CHECK-NEXT:     %[[L:.*]] = coreai.log %[[X]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_divide %[[L]], %[[C]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class Log2Model(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.log2(x)

        x = torch.rand(2, 3) + 1.0
        ir = get_ir(
            Log2Model().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<0.693147182> : tensor<f32>
                // CHECK-NEXT:     %[[L:.*]] = coreai.log %[[X]] : tensor<?x?xf32> -> tensor<?x?xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_divide %[[L]], %[[C]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLogicalAndIR:
    def test_bool_static(self) -> None:
        class LogicalAndModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.logical_and(x, y)

        ir = get_ir(
            LogicalAndModel().eval(),
            x=torch.tensor([True, False]),
            y=torch.tensor([True, True]),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2xi1> {coreai.name = "x"}, %[[Y:.*]]: tensor<2xi1> {coreai.name = "y"}) -> (tensor<2xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_and %[[X]], %[[Y]] : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_float_static(self) -> None:
        class LogicalAndModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.logical_and(x, y)

        ir = get_ir(LogicalAndModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[XB:.*]] = coreai.cast %[[X]] : tensor<2x3xf32> to tensor<2x3xi1>
                // CHECK-NEXT:     %[[YB:.*]] = coreai.cast %[[Y]] : tensor<2x3xf32> to tensor<2x3xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_and %[[XB]], %[[YB]] : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LogicalAndModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.logical_and(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            LogicalAndModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[XB:.*]] = coreai.cast %[[X]] : tensor<?x3xf32> to tensor<?x3xi1>
                // CHECK-NEXT:     %[[YB:.*]] = coreai.cast %[[Y]] : tensor<?x3xf32> to tensor<?x3xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_and %[[XB]], %[[YB]] : (tensor<?x3xi1>, tensor<?x3xi1>) -> tensor<?x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLogicalNotIR:
    def test_bool_static(self) -> None:
        class LogicalNotModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.logical_not(x)

        ir = get_ir(LogicalNotModel().eval(), x=torch.tensor([True, False, True]))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<3xi1> {coreai.name = "x"}) -> (tensor<3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.not %[[X]] : tensor<3xi1> -> tensor<3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_float_static(self) -> None:
        class LogicalNotModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.logical_not(x)

        ir = get_ir(LogicalNotModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[ZERO:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[NEQ:.*]] = coreai.decomposable.broadcasting_not_equal %[[X]], %[[ZERO]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.not %[[NEQ]] : tensor<2x3xi1> -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LogicalNotModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.logical_not(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            LogicalNotModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[ZERO:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[NEQ:.*]] = coreai.decomposable.broadcasting_not_equal %[[X]], %[[ZERO]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.not %[[NEQ]] : tensor<?x?xi1> -> tensor<?x?xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLogicalOrIR:
    def test_bool_static(self) -> None:
        class LogicalOrModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.logical_or(x, y)

        ir = get_ir(
            LogicalOrModel().eval(),
            x=torch.tensor([True, False]),
            y=torch.tensor([True, True]),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2xi1> {coreai.name = "x"}, %[[Y:.*]]: tensor<2xi1> {coreai.name = "y"}) -> (tensor<2xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_or %[[X]], %[[Y]] : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_float_static(self) -> None:
        class LogicalOrModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.logical_or(x, y)

        ir = get_ir(LogicalOrModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[XB:.*]] = coreai.cast %[[X]] : tensor<2x3xf32> to tensor<2x3xi1>
                // CHECK-NEXT:     %[[YB:.*]] = coreai.cast %[[Y]] : tensor<2x3xf32> to tensor<2x3xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_or %[[XB]], %[[YB]] : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LogicalOrModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.logical_or(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            LogicalOrModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[XB:.*]] = coreai.cast %[[X]] : tensor<?x3xf32> to tensor<?x3xi1>
                // CHECK-NEXT:     %[[YB:.*]] = coreai.cast %[[Y]] : tensor<?x3xf32> to tensor<?x3xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_or %[[XB]], %[[YB]] : (tensor<?x3xi1>, tensor<?x3xi1>) -> tensor<?x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLogicalXorIR:
    def test_bool_static(self) -> None:
        class LogicalXorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.logical_xor(x, y)

        ir = get_ir(
            LogicalXorModel().eval(),
            x=torch.tensor([True, False]),
            y=torch.tensor([True, True]),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2xi1> {coreai.name = "x"}, %[[Y:.*]]: tensor<2xi1> {coreai.name = "y"}) -> (tensor<2xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_xor %[[X]], %[[Y]] : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_float_static(self) -> None:
        class LogicalXorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.logical_xor(x, y)

        ir = get_ir(LogicalXorModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[XB:.*]] = coreai.cast %[[X]] : tensor<2x3xf32> to tensor<2x3xi1>
                // CHECK-NEXT:     %[[YB:.*]] = coreai.cast %[[Y]] : tensor<2x3xf32> to tensor<2x3xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_xor %[[XB]], %[[YB]] : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LogicalXorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.logical_xor(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            LogicalXorModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[XB:.*]] = coreai.cast %[[X]] : tensor<?x3xf32> to tensor<?x3xi1>
                // CHECK-NEXT:     %[[YB:.*]] = coreai.cast %[[Y]] : tensor<?x3xf32> to tensor<?x3xi1>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_xor %[[XB]], %[[YB]] : (tensor<?x3xi1>, tensor<?x3xi1>) -> tensor<?x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLtScalarIR:
    def test_static(self) -> None:
        class LtScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x < 2.0

        ir = get_ir(LtScalarModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_greater %[[C]], %[[X]] : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LtScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x < 2.0

        x = torch.rand(2, 3)
        ir = get_ir(
            LtScalarModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_greater %[[C]], %[[X]] : (tensor<f32>, tensor<?x?xf32>) -> tensor<?x?xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestLtTensorIR:
    def test_static(self) -> None:
        class LtTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.lt(x, y)

        ir = get_ir(LtTensorModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_greater %[[Y]], %[[X]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class LtTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.lt(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            LtTensorModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_greater %[[Y]], %[[X]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestMaxDefaultIR:
    def test_static(self) -> None:
        class MaxDefaultModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.max(x)

        ir = get_ir(MaxDefaultModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<f32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<> : tensor<0xui32>
                // CHECK-NEXT:     %[[AXES:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_max %[[X]], %[[AXES]] : (tensor<2x3xf32>, tensor<2xsi32>) -> tensor<1x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[REDUCE]], %[[SHAPE]] : (tensor<1x1xf32>, tensor<0xui32>) -> tensor<f32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<f32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class MaxDefaultModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.max(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            MaxDefaultModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<f32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<> : tensor<0xui32>
                // CHECK-NEXT:     %[[AXES:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_max %[[X]], %[[AXES]] : (tensor<?x?xf32>, tensor<2xsi32>) -> tensor<1x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[REDUCE]], %[[SHAPE]] : (tensor<1x1xf32>, tensor<0xui32>) -> tensor<f32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<f32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestMaxDimIR:
    def test_static(self) -> None:
        class MaxDimModel(nn.Module):
            def forward(self, x: Tensor):
                return torch.max(x, dim=1)

        ir = get_ir(MaxDimModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2xf32> {coreai.name = "{{.*}}"}, tensor<2xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<2> : tensor<1xui32>
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[AXIS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_max %[[X]], %[[AXIS]] : (tensor<2x3xf32>, tensor<1xsi32>) -> tensor<2x1xf32>
                // CHECK-NEXT:     %[[ARGMAX:.*]] = coreai.argmax %[[X]], %[[DIM]] : (tensor<2x3xf32>, tensor<si32>) -> tensor<2x1xui32>
                // CHECK-NEXT:     %[[CAST:.*]] = coreai.cast %[[ARGMAX]] : tensor<2x1xui32> to tensor<2x1xsi32>
                // CHECK-NEXT:     %[[VAL:.*]] = coreai.reshape %[[REDUCE]], %[[SHAPE]] : (tensor<2x1xf32>, tensor<1xui32>) -> tensor<2xf32>
                // CHECK-NEXT:     %[[IDX:.*]] = coreai.reshape %[[CAST]], %[[SHAPE]] : (tensor<2x1xsi32>, tensor<1xui32>) -> tensor<2xsi32>
                // CHECK-NEXT:     coreai.output %[[VAL]], %[[IDX]] : tensor<2xf32>, tensor<2xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class MaxDimModel(nn.Module):
            def forward(self, x: Tensor):
                return torch.max(x, dim=1)

        x = torch.rand(2, 3)
        ir = get_ir(
            MaxDimModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?xf32> {coreai.name = "{{.*}}"}, tensor<?xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[AXIS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_max %[[X]], %[[AXIS]] : (tensor<?x?xf32>, tensor<1xsi32>) -> tensor<?x1xf32>
                // CHECK-NEXT:     %[[ARGMAX:.*]] = coreai.argmax %[[X]], %[[DIM]] : (tensor<?x?xf32>, tensor<si32>) -> tensor<?x1xui32>
                // CHECK-NEXT:     %[[CAST:.*]] = coreai.cast %[[ARGMAX]] : tensor<?x1xui32> to tensor<?x1xsi32>
                // CHECK-NEXT:     %[[VAL:.*]] = coreai.shrink_dims %[[REDUCE]], %[[AXIS]] : (tensor<?x1xf32>, tensor<1xsi32>) to tensor<?xf32>
                // CHECK-NEXT:     %[[IDX:.*]] = coreai.shrink_dims %[[CAST]], %[[AXIS]] : (tensor<?x1xsi32>, tensor<1xsi32>) to tensor<?xsi32>
                // CHECK-NEXT:     coreai.output %[[VAL]], %[[IDX]] : tensor<?xf32>, tensor<?xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_keepdim(self) -> None:
        class MaxDimKeepdimModel(nn.Module):
            def forward(self, x: Tensor):
                return torch.max(x, dim=1, keepdim=True)

        ir = get_ir(MaxDimKeepdimModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x1xf32> {coreai.name = "{{.*}}"}, tensor<2x1xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[AXIS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_max %[[X]], %[[AXIS]] : (tensor<2x3xf32>, tensor<1xsi32>) -> tensor<2x1xf32>
                // CHECK-NEXT:     %[[ARGMAX:.*]] = coreai.argmax %[[X]], %[[DIM]] : (tensor<2x3xf32>, tensor<si32>) -> tensor<2x1xui32>
                // CHECK-NEXT:     %[[CAST:.*]] = coreai.cast %[[ARGMAX]] : tensor<2x1xui32> to tensor<2x1xsi32>
                // CHECK-NEXT:     coreai.output %[[REDUCE]], %[[CAST]] : tensor<2x1xf32>, tensor<2x1xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestMaxPool2dWithIndicesIR:
    # NOTE: coreai.max_pool_2d does not return real indices, so the converter
    # emits a placeholder dense<0> constant for the indices output. If the
    # underlying op gains real index support, update this test accordingly.
    def test_static(self) -> None:
        class MaxPoolModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)

            def forward(self, x: Tensor):
                return self.pool(x)

        ir = get_ir(MaxPoolModel().eval(), x=torch.rand(1, 3, 4, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<1x3x4x4xf32> {coreai.name = "x"}) -> (tensor<1x3x2x2xf32> {coreai.name = "{{.*}}"}, tensor<1x3x2x2xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[ZERO:.*]] = coreai.constant dense<0> : tensor<1x3x2x2xsi32>
                // CHECK-NEXT:     %[[CEIL:.*]] = coreai.constant dense<false> : tensor<i1>
                // CHECK-NEXT:     %[[DIL:.*]] = coreai.constant dense<1> : tensor<2xui32>
                // CHECK-NEXT:     %[[KS:.*]] = coreai.constant dense<2> : tensor<2xui32>
                // CHECK-NEXT:     %[[POOL:.*]] = coreai.max_pool_2d %[[X]], %[[KS]], %[[KS]], %[[DIL]], %[[CEIL]] : (tensor<1x3x4x4xf32>, tensor<2xui32>, tensor<2xui32>, tensor<2xui32>, tensor<i1>) -> tensor<1x3x2x2xf32>
                // CHECK-NEXT:     coreai.output %[[POOL]], %[[ZERO]] : tensor<1x3x2x2xf32>, tensor<1x3x2x2xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestMaximumIR:
    def test_static(self) -> None:
        class MaximumModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.maximum(x, y)

        ir = get_ir(MaximumModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_maximum %[[X]], %[[Y]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class MaximumModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.maximum(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            MaximumModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_maximum %[[X]], %[[Y]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestMeanDefaultIR:
    def test_static(self) -> None:
        class MeanDefaultModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.mean(x)

        ir = get_ir(MeanDefaultModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<f32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<> : tensor<0xui32>
                // CHECK-NEXT:     %[[AXES:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_mean %[[X]], %[[AXES]] : (tensor<2x3xf32>, tensor<2xsi32>) -> tensor<1x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[REDUCE]], %[[SHAPE]] : (tensor<1x1xf32>, tensor<0xui32>) -> tensor<f32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<f32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestMeanDimIR:
    def test_static(self) -> None:
        class MeanDimModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.mean(x, dim=1)

        ir = get_ir(MeanDimModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<2> : tensor<1xui32>
                // CHECK-NEXT:     %[[AXIS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_mean %[[X]], %[[AXIS]] : (tensor<2x3xf32>, tensor<1xsi32>) -> tensor<2x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[REDUCE]], %[[SHAPE]] : (tensor<2x1xf32>, tensor<1xui32>) -> tensor<2xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class MeanDimModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.mean(x, dim=1)

        x = torch.rand(2, 3)
        ir = get_ir(
            MeanDimModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[AXIS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_mean %[[X]], %[[AXIS]] : (tensor<?x?xf32>, tensor<1xsi32>) -> tensor<?x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.shrink_dims %[[REDUCE]], %[[AXIS]] : (tensor<?x1xf32>, tensor<1xsi32>) to tensor<?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_keepdim(self) -> None:
        class MeanDimKeepdimModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.mean(x, dim=1, keepdim=True)

        ir = get_ir(MeanDimKeepdimModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x1xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[AXIS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reduce_mean %[[X]], %[[AXIS]] : (tensor<2x3xf32>, tensor<1xsi32>) -> tensor<2x1xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x1xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestMinDefaultIR:
    def test_static(self) -> None:
        class MinDefaultModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.min(x)

        ir = get_ir(MinDefaultModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<f32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<> : tensor<0xui32>
                // CHECK-NEXT:     %[[AXES:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_min %[[X]], %[[AXES]] : (tensor<2x3xf32>, tensor<2xsi32>) -> tensor<1x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[REDUCE]], %[[SHAPE]] : (tensor<1x1xf32>, tensor<0xui32>) -> tensor<f32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<f32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestMinDimIR:
    def test_static(self) -> None:
        class MinDimModel(nn.Module):
            def forward(self, x: Tensor):
                return torch.min(x, dim=1)

        ir = get_ir(MinDimModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2xf32> {coreai.name = "{{.*}}"}, tensor<2xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<2> : tensor<1xui32>
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[NEG1:.*]] = coreai.constant dense<-1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[AXIS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_min %[[X]], %[[AXIS]] : (tensor<2x3xf32>, tensor<1xsi32>) -> tensor<2x1xf32>
                // CHECK-NEXT:     %[[NEGX:.*]] = coreai.decomposable.broadcasting_mul %[[X]], %[[NEG1]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[ARGMAX:.*]] = coreai.argmax %[[NEGX]], %[[DIM]] : (tensor<2x3xf32>, tensor<si32>) -> tensor<2x1xui32>
                // CHECK-NEXT:     %[[CAST:.*]] = coreai.cast %[[ARGMAX]] : tensor<2x1xui32> to tensor<2x1xsi32>
                // CHECK-NEXT:     %[[VAL:.*]] = coreai.reshape %[[REDUCE]], %[[SHAPE]] : (tensor<2x1xf32>, tensor<1xui32>) -> tensor<2xf32>
                // CHECK-NEXT:     %[[IDX:.*]] = coreai.reshape %[[CAST]], %[[SHAPE]] : (tensor<2x1xsi32>, tensor<1xui32>) -> tensor<2xsi32>
                // CHECK-NEXT:     coreai.output %[[VAL]], %[[IDX]] : tensor<2xf32>, tensor<2xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestMinimumIR:
    def test_static(self) -> None:
        class MinimumModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.minimum(x, y)

        ir = get_ir(MinimumModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_minimum %[[X]], %[[Y]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class MinimumModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.minimum(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        batch = torch.export.Dim("batch")
        ir = get_ir(
            MinimumModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<?x3xf32> {coreai.name = "y"}) -> (tensor<?x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_minimum %[[X]], %[[Y]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestMmIR:
    def test_static(self) -> None:
        class MmModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.mm(x, y)

        ir = get_ir(MmModel().eval(), x=torch.rand(2, 3), y=torch.rand(3, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<3x4xf32> {coreai.name = "y"}) -> (tensor<2x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_batch_matmul %[[X]], %[[Y]] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class MmModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.mm(x, y)

        n = torch.export.Dim("n")
        m = torch.export.Dim("m")
        k = torch.export.Dim("k")
        ir = get_ir(
            MmModel().eval(),
            x=torch.rand(2, 3),
            y=torch.rand(3, 4),
            dynamic_shapes={"x": {0: n, 1: k}, "y": {0: k, 1: m}},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<?x?xf32> {coreai.name = "y"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_batch_matmul %[[X]], %[[Y]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestMulIR:
    def test_tensor(self) -> None:
        class MulTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.mul(x, y)

        ir = get_ir(MulTensorModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_mul %[[X]], %[[Y]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_scalar(self) -> None:
        class MulScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.mul(x, 2.0)

        ir = get_ir(MulScalarModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_mul %[[X]], %[[C]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class MulTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.mul(x, y)

        x = torch.rand(2, 3)
        y = torch.rand(2, 3)
        ir = get_ir(
            MulTensorModel().eval(),
            x=x,
            y=y,
            dynamic_shapes={"x": _all_dims_dynamic(x), "y": _all_dims_dynamic(y)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<?x?xf32> {coreai.name = "y"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_mul %[[X]], %[[Y]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestNativeGroupNormIR:
    def test_static(self) -> None:
        class GroupNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gn = nn.GroupNorm(2, 4)

            def forward(self, x: Tensor) -> Tensor:
                return self.gn(x)

        ir = get_ir(GroupNormModel().eval(), x=torch.rand(1, 4, 8, 8))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @group_norm_{{.*}}(%[[INPUT:.*]]: tensor<1x4x8x8xf32> {coreai.name = "input"}, %[[WEIGHT:.*]]: tensor<4xf32> {coreai.name = "weight"}, %[[BIAS:.*]]: tensor<4xf32> {coreai.name = "bias"}) -> tensor<1x4x8x8xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"group_norm" = {input_names = ["input", "weight", "bias"], op_attrs = {eps = 9.99999974E-6 : f32, num_channels = 4 : si64, num_groups = 2 : si64, version = 1 : si64}, output_names = ["output"]}>, template_op = "group_norm"} {
                // CHECK:           coreai.reduce_mean
                // CHECK:           coreai.decomposable.broadcasting_sub
                // CHECK:           coreai.decomposable.broadcasting_mul
                // CHECK:           coreai.reduce_mean
                // CHECK:           coreai.decomposable.broadcasting_add
                // CHECK:           coreai.rsqrt
                // CHECK:           coreai.decomposable.broadcasting_mul
                // CHECK:           coreai.decomposable.broadcasting_mul
                // CHECK:           coreai.decomposable.broadcasting_add
                // CHECK:           coreai.output
                // CHECK-NEXT:    }
                // CHECK-NEXT:    coreai.graph @main(%[[X:.*]]: tensor<1x4x8x8xf32> {coreai.name = "x"}) -> (tensor<1x4x8x8xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:      %[[ONE:.*]] = coreai.constant dense<1.000000e+00> : tensor<4xf32>
                // CHECK-NEXT:      %[[ZERO:.*]] = coreai.constant dense<0.000000e+00> : tensor<4xf32>
                // CHECK-NEXT:      %[[R:.*]] = coreai.invoke @group_norm_{{.*}}(%[[X]], %[[ONE]], %[[ZERO]])  : (tensor<1x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
                // CHECK-NEXT:      coreai.output %[[R]] : tensor<1x4x8x8xf32>
                // CHECK-NEXT:    }
                // CHECK-NEXT:  }
            """,
        )


class TestNativeLayerNormIR:
    def test_static(self) -> None:
        class LayerNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(3)

            def forward(self, x: Tensor) -> Tensor:
                return self.ln(x)

        ir = get_ir(LayerNormModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph private noinline @layer_norm_{{.*}}(%[[INPUT:.*]]: tensor<2x3xf32> {coreai.name = "input"}, %[[GAMMA:.*]]: tensor<3xf32> {coreai.name = "gamma"}, %[[BETA:.*]]: tensor<3xf32> {coreai.name = "beta"}) -> tensor<2x3xf32> attributes {__coreai_pure__, composite_decl = #coreai.composite_declaration<"layer_norm" = {input_names = ["input", "gamma", "beta"], op_attrs = {axes = [1 : si64], eps = 9.99999974E-6 : f32, version = 1 : si64}, output_names = ["output"]}>, template_op = "layer_norm"} {
                // CHECK:           coreai.reduce_mean
                // CHECK:           coreai.decomposable.broadcasting_sub
                // CHECK:           coreai.decomposable.broadcasting_mul
                // CHECK:           coreai.reduce_mean
                // CHECK:           coreai.decomposable.broadcasting_add
                // CHECK:           coreai.rsqrt
                // CHECK:           coreai.decomposable.broadcasting_mul
                // CHECK:           coreai.decomposable.broadcasting_mul
                // CHECK:           coreai.decomposable.broadcasting_add
                // CHECK:           coreai.output
                // CHECK-NEXT:    }
                // CHECK-NEXT:    coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:      %[[ONE:.*]] = coreai.constant dense<1.000000e+00> : tensor<3xf32>
                // CHECK-NEXT:      %[[ZERO:.*]] = coreai.constant dense<0.000000e+00> : tensor<3xf32>
                // CHECK-NEXT:      %[[R:.*]] = coreai.invoke @layer_norm_{{.*}}(%[[X]], %[[ONE]], %[[ZERO]])  : (tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:      coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:    }
                // CHECK-NEXT:  }
            """,
        )


class TestNeScalarIR:
    def test_static(self) -> None:
        class NeScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.ne(x, 2.0)

        ir = get_ir(NeScalarModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_not_equal %[[X]], %[[C]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestNeTensorIR:
    def test_static(self) -> None:
        class NeTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.ne(x, y)

        ir = get_ir(NeTensorModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xi1> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_not_equal %[[X]], %[[Y]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xi1>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestNegIR:
    def test_static(self) -> None:
        class NegModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.neg(x)

        ir = get_ir(NegModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[NEG1:.*]] = coreai.constant dense<-1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_mul %[[X]], %[[NEG1]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class NegModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.neg(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            NegModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[NEG1:.*]] = coreai.constant dense<-1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_mul %[[X]], %[[NEG1]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestNonzeroIR:
    def test_static(self) -> None:
        class NonzeroModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nonzero(x)

        ir = get_ir(NonzeroModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<?x2xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.non_zero %[[X]] : tensor<2x3xf32> -> tensor<?x2xsi32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x2xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestNonzeroNumpyIR:
    def test_static(self) -> None:
        class NonzeroNumpyModel(nn.Module):
            def forward(self, x: Tensor):
                return torch.nonzero(x, as_tuple=True)

        ir = get_ir(NonzeroNumpyModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<?xsi32> {coreai.name = "{{.*}}"}, tensor<?xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[ONES:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[END1:.*]] = coreai.constant dense<[2147483647, 2]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[START1:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[STRIDE:.*]] = coreai.constant dense<1> : tensor<2xsi32>
                // CHECK-NEXT:     %[[END0:.*]] = coreai.constant dense<[2147483647, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[START0:.*]] = coreai.constant dense<0> : tensor<2xsi32>
                // CHECK-NEXT:     %[[NZ:.*]] = coreai.non_zero %[[X]] : tensor<2x3xf32> -> tensor<?x2xsi32>
                // CHECK-NEXT:     %[[S0:.*]] = coreai.slice %[[NZ]], %[[START0]], %[[END0]], %[[STRIDE]] : (tensor<?x2xsi32>, tensor<2xsi32>, tensor<2xsi32>, tensor<2xsi32>) -> tensor<?x1xsi32>
                // CHECK-NEXT:     %[[S1:.*]] = coreai.slice %[[NZ]], %[[START1]], %[[END1]], %[[STRIDE]] : (tensor<?x2xsi32>, tensor<2xsi32>, tensor<2xsi32>, tensor<2xsi32>) -> tensor<?x1xsi32>
                // CHECK-NEXT:     %[[R0:.*]] = coreai.shrink_dims %[[S0]], %[[ONES]] : (tensor<?x1xsi32>, tensor<1xsi32>) to tensor<?xsi32>
                // CHECK-NEXT:     %[[R1:.*]] = coreai.shrink_dims %[[S1]], %[[ONES]] : (tensor<?x1xsi32>, tensor<1xsi32>) to tensor<?xsi32>
                // CHECK-NEXT:     coreai.output %[[R0]], %[[R1]] : tensor<?xsi32>, tensor<?xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestPermuteIR:
    def test_static(self) -> None:
        class PermuteModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.permute(x, (1, 0, 2))

        ir = get_ir(PermuteModel().eval(), x=torch.rand(2, 3, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3x4xf32> {coreai.name = "x"}) -> (tensor<3x2x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[PERM:.*]] = coreai.constant dense<[1, 0, 2]> : tensor<3xui32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.transpose %[[X]], %[[PERM]] : (tensor<2x3x4xf32>, tensor<3xui32>) -> tensor<3x2x4xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<3x2x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class PermuteModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.permute(x, (1, 0, 2))

        x = torch.rand(2, 3, 4)
        ir = get_ir(
            PermuteModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?x?xf32> {coreai.name = "x"}) -> (tensor<?x?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[PERM:.*]] = coreai.constant dense<[1, 0, 2]> : tensor<3xui32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.transpose %[[X]], %[[PERM]] : (tensor<?x?x?xf32>, tensor<3xui32>) -> tensor<?x?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestPixelShuffleIR:
    def test_static(self) -> None:
        class PixelShuffleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.PixelShuffle(2)

            def forward(self, x: Tensor) -> Tensor:
                return self.p(x)

        ir = get_ir(PixelShuffleModel().eval(), x=torch.rand(1, 4, 2, 2))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<1x4x2x2xf32> {coreai.name = "x"}) -> (tensor<1x1x4x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[OUT:.*]] = coreai.constant dense<[1, 1, 4, 4]> : tensor<4xui32>
                // CHECK-NEXT:     %[[PERM:.*]] = coreai.constant dense<[2, 0, 3, 1]> : tensor<4xui32>
                // CHECK-NEXT:     %[[INNER:.*]] = coreai.constant dense<2> : tensor<4xui32>
                // CHECK-NEXT:     %[[R1:.*]] = coreai.reshape %[[X]], %[[INNER]] : (tensor<1x4x2x2xf32>, tensor<4xui32>) -> tensor<2x2x2x2xf32>
                // CHECK-NEXT:     %[[T:.*]] = coreai.transpose %[[R1]], %[[PERM]] : (tensor<2x2x2x2xf32>, tensor<4xui32>) -> tensor<2x2x2x2xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[T]], %[[OUT]] : (tensor<2x2x2x2xf32>, tensor<4xui32>) -> tensor<1x1x4x4xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<1x1x4x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestPowIR:
    def test_tensor_tensor(self) -> None:
        class PowTTModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.pow(x, y)

        ir = get_ir(PowTTModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_pow %[[X]], %[[Y]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_tensor_scalar(self) -> None:
        class PowTSModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.pow(x, 2.0)

        ir = get_ir(PowTSModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_pow %[[X]], %[[C]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_scalar(self) -> None:
        class PowSModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.pow(2.0, x)

        ir = get_ir(PowSModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_pow %[[C]], %[[X]] : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestProdDefaultIR:
    def test_static(self) -> None:
        class ProdDefaultModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.prod(x)

        ir = get_ir(ProdDefaultModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<f32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<> : tensor<0xui32>
                // CHECK-NEXT:     %[[AXES:.*]] = coreai.constant dense<[0, 1]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_product %[[X]], %[[AXES]] : (tensor<2x3xf32>, tensor<2xsi32>) -> tensor<1x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[REDUCE]], %[[SHAPE]] : (tensor<1x1xf32>, tensor<0xui32>) -> tensor<f32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<f32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestProdDimIntIR:
    def test_static(self) -> None:
        class ProdDimIntModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.prod(x, dim=1)

        ir = get_ir(ProdDimIntModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<2> : tensor<1xui32>
                // CHECK-NEXT:     %[[AXIS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_product %[[X]], %[[AXIS]] : (tensor<2x3xf32>, tensor<1xsi32>) -> tensor<2x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[REDUCE]], %[[SHAPE]] : (tensor<2x1xf32>, tensor<1xui32>) -> tensor<2xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestReciprocalIR:
    def test_static(self) -> None:
        class ReciprocalModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.reciprocal(x)

        ir = get_ir(ReciprocalModel().eval(), x=torch.rand(2, 3) + 1.0)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[ONE:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_divide %[[ONE]], %[[X]] : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class ReciprocalModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.reciprocal(x)

        x = torch.rand(2, 3) + 1.0
        ir = get_ir(
            ReciprocalModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[ONE:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_divide %[[ONE]], %[[X]] : (tensor<f32>, tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestReluIR:
    def test_static(self) -> None:
        class ReluModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.relu(x)

        ir = get_ir(ReluModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.relu %[[X]] : (tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class ReluModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.relu(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            ReluModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.relu %[[X]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestRemainderIR:
    def test_static(self) -> None:
        class RemainderModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.remainder(x, y)

        ir = get_ir(
            RemainderModel().eval(),
            x=torch.rand(2, 3),
            y=torch.rand(2, 3) + 1.0,
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[FD:.*]] = coreai.decomposable.broadcasting_floor_divide %[[X]], %[[Y]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[MUL:.*]] = coreai.decomposable.broadcasting_mul %[[FD]], %[[Y]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_sub %[[X]], %[[MUL]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestRepeatIR:
    def test_static(self) -> None:
        class RepeatModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.repeat(2, 3)

        ir = get_ir(RepeatModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<4x9xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[REPS:.*]] = coreai.constant dense<[2, 3]> : tensor<2xui32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.tile %[[X]], %[[REPS]] : (tensor<2x3xf32>, tensor<2xui32>) -> tensor<4x9xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<4x9xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestRoundDecimalsIR:
    def test_static(self) -> None:
        class RoundDecimalsModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.round(x, decimals=2)

        ir = get_ir(RoundDecimalsModel().eval(), x=torch.rand(2, 3) * 10)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SCALE:.*]] = coreai.constant dense<1.000000e+02> : tensor<f32>
                // CHECK-NEXT:     %[[MUL:.*]] = coreai.decomposable.broadcasting_mul %[[X]], %[[SCALE]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[ROUND:.*]] = coreai.round %[[MUL]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_divide %[[ROUND]], %[[SCALE]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_negative_decimals(self) -> None:
        class RoundDecimalsModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.round(x, decimals=-1)

        ir = get_ir(RoundDecimalsModel().eval(), x=torch.rand(2, 3) * 100)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SCALE:.*]] = coreai.constant dense<1.000000e-01> : tensor<f32>
                // CHECK-NEXT:     %[[MUL:.*]] = coreai.decomposable.broadcasting_mul %[[X]], %[[SCALE]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[ROUND:.*]] = coreai.round %[[MUL]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_divide %[[ROUND]], %[[SCALE]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestScalarTensorIR:
    def test_static(self) -> None:
        class ScalarTensorModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.scalar_tensor(2.0) + x

        ir = get_ir(ScalarTensorModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_add %[[X]], %[[C]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestScaledDotProductAttentionIR:
    def test_static(self) -> None:
        class SdpaModel(nn.Module):
            def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        ir = get_ir(
            SdpaModel().eval(),
            q=torch.rand(1, 2, 3, 4),
            k=torch.rand(1, 2, 3, 4),
            v=torch.rand(1, 2, 3, 4),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[Q:.*]]: tensor<1x2x3x4xf32> {coreai.name = "q"}, %[[K:.*]]: tensor<1x2x3x4xf32> {coreai.name = "k"}, %[[V:.*]]: tensor<1x2x3x4xf32> {coreai.name = "v"}) -> (tensor<1x2x3x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK:           coreai.decomposable.broadcasting_mul
                // CHECK:           coreai.transpose
                // CHECK:           coreai.decomposable.broadcasting_mul
                // CHECK:           coreai.decomposable.broadcasting_batch_matmul
                // CHECK:           coreai.softmax
                // CHECK:           coreai.decomposable.broadcasting_equal
                // CHECK:           coreai.not
                // CHECK:           coreai.any
                // CHECK:           coreai.not
                // CHECK:           coreai.decomposable.broadcasting_where
                // CHECK:           coreai.decomposable.broadcasting_batch_matmul
                // CHECK:           coreai.output
                // CHECK-NEXT:    }
                // CHECK-NEXT:  }
            """,
        )


class TestScatterIR:
    def test_src(self) -> None:
        class ScatterSrcModel(nn.Module):
            def forward(self, x: Tensor, idx: Tensor, src: Tensor) -> Tensor:
                return torch.scatter(x, 0, idx, src)

        ir = get_ir(
            ScatterSrcModel().eval(),
            x=torch.zeros(3, 5),
            idx=torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.int64),
            src=torch.rand(2, 5),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<3x5xf32> {coreai.name = "x"}, %[[IDX:.*]]: tensor<2x5xsi32> {coreai.name = "idx"}, %[[SRC:.*]]: tensor<2x5xf32> {coreai.name = "src"}) -> (tensor<3x5xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.scatter_along_axis %[[X]], %[[IDX]], %[[SRC]], %[[DIM]] : (tensor<3x5xf32>, tensor<2x5xsi32>, tensor<2x5xf32>, tensor<si32>) -> tensor<3x5xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<3x5xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_value(self) -> None:
        class ScatterValueModel(nn.Module):
            def forward(self, x: Tensor, idx: Tensor) -> Tensor:
                return torch.scatter(x, 0, idx, 5.0)

        ir = get_ir(
            ScatterValueModel().eval(),
            x=torch.zeros(3, 5),
            idx=torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.int64),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<3x5xf32> {coreai.name = "x"}, %[[IDX:.*]]: tensor<2x5xsi32> {coreai.name = "idx"}) -> (tensor<3x5xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[VAL:.*]] = coreai.constant dense<5.000000e+00> : tensor<2x5xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.scatter_along_axis %[[X]], %[[IDX]], %[[VAL]], %[[DIM]] : (tensor<3x5xf32>, tensor<2x5xsi32>, tensor<2x5xf32>, tensor<si32>) -> tensor<3x5xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<3x5xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_reduce(self) -> None:
        class ScatterReduceModel(nn.Module):
            def forward(self, x: Tensor, idx: Tensor, src: Tensor) -> Tensor:
                return torch.scatter(x, 0, idx, src, reduce="add")

        ir = get_ir(
            ScatterReduceModel().eval(),
            x=torch.zeros(3, 5),
            idx=torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.int64),
            src=torch.rand(2, 5),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<3x5xf32> {coreai.name = "x"}, %[[IDX:.*]]: tensor<2x5xsi32> {coreai.name = "idx"}, %[[SRC:.*]]: tensor<2x5xf32> {coreai.name = "src"}) -> (tensor<3x5xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<0> : tensor<si32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.scatter_along_axis %[[X]], %[[IDX]], %[[SRC]], %[[DIM]] scatter_mode = <add> : (tensor<3x5xf32>, tensor<2x5xsi32>, tensor<2x5xf32>, tensor<si32>) -> tensor<3x5xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<3x5xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSelectIntIR:
    def test_static(self) -> None:
        class SelectIntModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x[1]

        ir = get_ir(SelectIntModel().eval(), x=torch.rand(2, 3, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3x4xf32> {coreai.name = "x"}) -> (tensor<3x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[3, 4]> : tensor<2xui32>
                // CHECK-NEXT:     %[[START:.*]] = coreai.constant dense<[1, 0, 0]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[END:.*]] = coreai.constant dense<[2, 2147483647, 2147483647]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[STRIDE:.*]] = coreai.constant dense<1> : tensor<3xsi32>
                // CHECK-NEXT:     %[[S:.*]] = coreai.slice %[[X]], %[[START]], %[[END]], %[[STRIDE]] : (tensor<2x3x4xf32>, tensor<3xsi32>, tensor<3xsi32>, tensor<3xsi32>) -> tensor<1x3x4xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[S]], %[[SHAPE]] : (tensor<1x3x4xf32>, tensor<2xui32>) -> tensor<3x4xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<3x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_inner_dim(self) -> None:
        class SelectIntInnerModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.select(x, dim=1, index=2)

        ir = get_ir(SelectIntInnerModel().eval(), x=torch.rand(2, 3, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3x4xf32> {coreai.name = "x"}) -> (tensor<2x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[2, 4]> : tensor<2xui32>
                // CHECK-NEXT:     %[[START:.*]] = coreai.constant dense<[0, 2, 0]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[END:.*]] = coreai.constant dense<[2147483647, 3, 2147483647]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[STRIDE:.*]] = coreai.constant dense<1> : tensor<3xsi32>
                // CHECK-NEXT:     %[[S:.*]] = coreai.slice %[[X]], %[[START]], %[[END]], %[[STRIDE]] : (tensor<2x3x4xf32>, tensor<3xsi32>, tensor<3xsi32>, tensor<3xsi32>) -> tensor<2x1x4xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[S]], %[[SHAPE]] : (tensor<2x1x4xf32>, tensor<2xui32>) -> tensor<2x4xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSigmoidIR:
    def test_static(self) -> None:
        class SigmoidModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.sigmoid(x)

        ir = get_ir(SigmoidModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.sigmoid %[[X]] : (tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class SigmoidModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.sigmoid(x)

        x = torch.rand(2, 3)
        ir = get_ir(
            SigmoidModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.sigmoid %[[X]] : (tensor<?x?xf32>) -> tensor<?x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSignIR:
    def test_static(self) -> None:
        class SignModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.sign(x)

        ir = get_ir(SignModel().eval(), x=torch.rand(2, 3) - 0.5)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[ZERO:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[POS:.*]] = coreai.decomposable.broadcasting_greater %[[X]], %[[ZERO]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[POSF:.*]] = coreai.cast %[[POS]] : tensor<2x3xi1> to tensor<2x3xf32>
                // CHECK-NEXT:     %[[NEG:.*]] = coreai.decomposable.broadcasting_greater %[[ZERO]], %[[X]] : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[NEGF:.*]] = coreai.cast %[[NEG]] : tensor<2x3xi1> to tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_sub %[[POSF]], %[[NEGF]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSiluIR:
    def test_static(self) -> None:
        class SiluModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.s = nn.SiLU()

            def forward(self, x: Tensor) -> Tensor:
                return self.s(x)

        ir = get_ir(SiluModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SIG:.*]] = coreai.sigmoid %[[X]] : (tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_mul %[[X]], %[[SIG]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSliceIR:
    def test_static(self) -> None:
        class SliceModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x[:, 1:3, :]

        ir = get_ir(SliceModel().eval(), x=torch.rand(2, 4, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x4x3xf32> {coreai.name = "x"}) -> (tensor<2x2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[START:.*]] = coreai.constant dense<[0, 1, 0]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[END:.*]] = coreai.constant dense<[2147483647, 3, 2147483647]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[STRIDE:.*]] = coreai.constant dense<1> : tensor<3xsi32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.slice %[[X]], %[[START]], %[[END]], %[[STRIDE]] : (tensor<2x4x3xf32>, tensor<3xsi32>, tensor<3xsi32>, tensor<3xsi32>) -> tensor<2x2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSliceScatterIR:
    def test_static(self) -> None:
        class SliceScatterModel(nn.Module):
            def forward(self, x: Tensor, src: Tensor) -> Tensor:
                x = x.clone()
                x[:, 1:3, :] = src
                return x

        ir = get_ir(
            SliceScatterModel().eval(),
            x=torch.rand(2, 4, 3),
            src=torch.rand(2, 2, 3),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x4x3xf32> {coreai.name = "x"}, %[[SRC:.*]]: tensor<2x2x3xf32> {coreai.name = "src"}) -> (tensor<2x4x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[START:.*]] = coreai.constant dense<[0, 1, 0]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[END:.*]] = coreai.constant dense<[2147483647, 3, 2147483647]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[STRIDE:.*]] = coreai.constant dense<1> : tensor<3xsi32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.slice_update %[[X]] with %[[SRC]] at (%[[START]], %[[END]], %[[STRIDE]]) : (tensor<2x4x3xf32>, tensor<2x2x3xf32>, tensor<3xsi32>, tensor<3xsi32>, tensor<3xsi32>) to tensor<2x4x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x4x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSplitWithSizesIR:
    def test_static(self) -> None:
        class SplitWithSizesModel(nn.Module):
            def forward(self, x: Tensor):
                return torch.split(x, [1, 2], dim=1)

        ir = get_ir(SplitWithSizesModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x1xf32> {coreai.name = "{{.*}}"}, tensor<2x2xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SIZES:.*]] = coreai.constant dense<[1, 2]> : tensor<2xui32>
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[R:.*]]:2 = coreai.split %[[X]], %[[SIZES]], %[[DIM]] : (tensor<2x3xf32>, tensor<2xui32>, tensor<si32>) -> (tensor<2x1xf32>, tensor<2x2xf32>)
                // CHECK-NEXT:     coreai.output %[[R]]#0, %[[R]]#1 : tensor<2x1xf32>, tensor<2x2xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSqueezeDimsIR:
    def test_static(self) -> None:
        class SqueezeModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.squeeze(x, [0, 2])

        ir = get_ir(SqueezeModel().eval(), x=torch.rand(1, 3, 1, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<1x3x1x4xf32> {coreai.name = "x"}) -> (tensor<3x4xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[3, 4]> : tensor<2xui32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[X]], %[[SHAPE]] : (tensor<1x3x1x4xf32>, tensor<2xui32>) -> tensor<3x4xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<3x4xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSubIR:
    def test_tensor(self) -> None:
        class SubTensorModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.sub(x, y)

        ir = get_ir(SubTensorModel().eval(), x=torch.rand(2, 3), y=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_sub %[[X]], %[[Y]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_scalar(self) -> None:
        class SubScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.sub(x, 2.0)

        ir = get_ir(SubScalarModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[C:.*]] = coreai.constant dense<2.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_sub %[[X]], %[[C]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSumDimIntListIR:
    def test_static(self) -> None:
        class SumDimModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.sum(x, dim=[1])

        ir = get_ir(SumDimModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<2> : tensor<1xui32>
                // CHECK-NEXT:     %[[AXIS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_sum %[[X]], %[[AXIS]] : (tensor<2x3xf32>, tensor<1xsi32>) -> tensor<2x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[REDUCE]], %[[SHAPE]] : (tensor<2x1xf32>, tensor<1xui32>) -> tensor<2xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class SumDimModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.sum(x, dim=[1])

        x = torch.rand(2, 3)
        ir = get_ir(
            SumDimModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[AXIS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[REDUCE:.*]] = coreai.reduce_sum %[[X]], %[[AXIS]] : (tensor<?x?xf32>, tensor<1xsi32>) -> tensor<?x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.shrink_dims %[[REDUCE]], %[[AXIS]] : (tensor<?x1xf32>, tensor<1xsi32>) to tensor<?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestSymSizeIntIR:
    def test_dynamic(self) -> None:
        class SymSizeModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.view(x.size(0) * x.size(1))

        x = torch.rand(2, 3)
        ir = get_ir(
            SymSizeModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[TWO:.*]] = coreai.constant dense<2> : tensor<1xsi32>
                // CHECK-NEXT:     %[[ONE:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[ZERO:.*]] = coreai.constant dense<0> : tensor<1xsi32>
                // CHECK-NEXT:     %[[GS:.*]] = coreai.get_shape %[[X]] : tensor<?x?xf32> -> tensor<2xui32>
                // CHECK-NEXT:     %[[S0:.*]] = coreai.slice %[[GS]], %[[ZERO]], %[[ONE]], %[[ONE]] : (tensor<2xui32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[C0:.*]] = coreai.cast %[[S0]] : tensor<1xui32> to tensor<1xsi32>
                // CHECK-NEXT:     %[[S1:.*]] = coreai.slice %[[GS]], %[[ONE]], %[[TWO]], %[[ONE]] : (tensor<2xui32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xui32>
                // CHECK-NEXT:     %[[C1:.*]] = coreai.cast %[[S1]] : tensor<1xui32> to tensor<1xsi32>
                // CHECK-NEXT:     %[[MUL:.*]] = coreai.decomposable.broadcasting_mul %[[C0]], %[[C1]] : (tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xsi32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[X]], %[[MUL]] : (tensor<?x?xf32>, tensor<1xsi32>) -> tensor<?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestTileIR:
    def test_static(self) -> None:
        class TileModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.tile(x, (2, 3))

        ir = get_ir(TileModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<4x9xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[REPS:.*]] = coreai.constant dense<[2, 3]> : tensor<2xui32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.tile %[[X]], %[[REPS]] : (tensor<2x3xf32>, tensor<2xui32>) -> tensor<4x9xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<4x9xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestToDtypeIR:
    def test_static(self) -> None:
        class ToDtypeModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.to(torch.int32)

        ir = get_ir(ToDtypeModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.cast %[[X]] : tensor<2x3xf32> to tensor<2x3xsi32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestTopkIR:
    def test_static(self) -> None:
        class TopkModel(nn.Module):
            def forward(self, x: Tensor):
                return torch.topk(x, k=2)

        ir = get_ir(TopkModel().eval(), x=torch.rand(2, 5))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x5xf32> {coreai.name = "x"}) -> (tensor<2x2xf32> {coreai.name = "{{.*}}"}, tensor<2x2xsi32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[STRIDE:.*]] = coreai.constant dense<1> : tensor<2xsi32>
                // CHECK-NEXT:     %[[END:.*]] = coreai.constant dense<[2147483647, 2]> : tensor<2xsi32>
                // CHECK-NEXT:     %[[START:.*]] = coreai.constant dense<0> : tensor<2xsi32>
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<1> : tensor<si32>
                // CHECK-NEXT:     %[[TRUE:.*]] = coreai.constant dense<true> : tensor<i1>
                // CHECK-NEXT:     %[[SORT:.*]] = coreai.sort %[[X]], %[[DIM]], %[[TRUE]], %[[TRUE]] : (tensor<2x5xf32>, tensor<si32>, tensor<i1>, tensor<i1>) -> tensor<2x5xf32>
                // CHECK-NEXT:     %[[ARG:.*]] = coreai.argsort %[[X]], %[[DIM]], %[[TRUE]], %[[TRUE]] : (tensor<2x5xf32>, tensor<si32>, tensor<i1>, tensor<i1>) -> tensor<2x5xsi32>
                // CHECK-NEXT:     %[[VAL:.*]] = coreai.slice %[[SORT]], %[[START]], %[[END]], %[[STRIDE]] : (tensor<2x5xf32>, tensor<2xsi32>, tensor<2xsi32>, tensor<2xsi32>) -> tensor<2x2xf32>
                // CHECK-NEXT:     %[[IDX:.*]] = coreai.slice %[[ARG]], %[[START]], %[[END]], %[[STRIDE]] : (tensor<2x5xsi32>, tensor<2xsi32>, tensor<2xsi32>, tensor<2xsi32>) -> tensor<2x2xsi32>
                // CHECK-NEXT:     coreai.output %[[VAL]], %[[IDX]] : tensor<2x2xf32>, tensor<2x2xsi32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestTrueDivideIR:
    def test_static(self) -> None:
        class TrueDivideModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.true_divide(x, y)

        ir = get_ir(
            TrueDivideModel().eval(),
            x=torch.rand(2, 3),
            y=torch.rand(2, 3) + 1.0,
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_divide %[[X]], %[[Y]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestTruedivIR:
    def test_dynamic(self) -> None:
        class TruedivModel(nn.Module):
            def forward(self, x: Tensor):
                return x.size(0) / 2

        x = torch.rand(2, 3)
        ir = get_ir(
            TruedivModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<1xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK:           coreai.get_shape
                // CHECK:           coreai.slice
                // CHECK:           coreai.cast
                // CHECK:           coreai.decomposable.broadcasting_divide
                // CHECK:           coreai.output
                // CHECK-NEXT:    }
                // CHECK-NEXT:  }
            """,
        )


class TestTruncIR:
    def test_static(self) -> None:
        class TruncModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.trunc(x)

        ir = get_ir(TruncModel().eval(), x=torch.rand(2, 3) - 0.5)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[ONE:.*]] = coreai.constant dense<1.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[ZERO:.*]] = coreai.constant dense<0.000000e+00> : tensor<f32>
                // CHECK-NEXT:     %[[POS:.*]] = coreai.decomposable.broadcasting_greater %[[X]], %[[ZERO]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[POSF:.*]] = coreai.cast %[[POS]] : tensor<2x3xi1> to tensor<2x3xf32>
                // CHECK-NEXT:     %[[NEG:.*]] = coreai.decomposable.broadcasting_greater %[[ZERO]], %[[X]] : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                // CHECK-NEXT:     %[[NEGF:.*]] = coreai.cast %[[NEG]] : tensor<2x3xi1> to tensor<2x3xf32>
                // CHECK-NEXT:     %[[SIGN:.*]] = coreai.decomposable.broadcasting_sub %[[POSF]], %[[NEGF]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[ABS:.*]] = coreai.abs %[[X]] : tensor<2x3xf32> -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[FD:.*]] = coreai.decomposable.broadcasting_floor_divide %[[ABS]], %[[ONE]] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_mul %[[SIGN]], %[[FD]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestUnsqueezeIR:
    def test_static(self) -> None:
        class UnsqueezeModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.unsqueeze(x, 1)

        ir = get_ir(UnsqueezeModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x1x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[2, 1, 3]> : tensor<3xui32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[X]], %[[SHAPE]] : (tensor<2x3xf32>, tensor<3xui32>) -> tensor<2x1x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x1x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )

    def test_dynamic(self) -> None:
        class UnsqueezeModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.unsqueeze(x, 1)

        x = torch.rand(2, 3)
        ir = get_ir(
            UnsqueezeModel().eval(),
            x=x,
            dynamic_shapes={"x": _all_dims_dynamic(x)},
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<?x?xf32> {coreai.name = "x"}) -> (tensor<?x1x?xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[AXIS:.*]] = coreai.constant dense<1> : tensor<1xsi32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.expand_dims %[[X]], %[[AXIS]] : (tensor<?x?xf32>, tensor<1xsi32>) to tensor<?x1x?xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<?x1x?xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestUpsampleBilinear2dIR:
    def test_static(self) -> None:
        class UpsampleBilinearModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.interpolate(
                    x, scale_factor=2.0, mode="bilinear"
                )

        ir = get_ir(UpsampleBilinearModel().eval(), x=torch.rand(1, 3, 4, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<1x3x4x4xf32> {coreai.name = "x"}) -> (tensor<1x3x8x8xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[OUT:.*]] = coreai.constant dense<[1, 3, 8, 8]> : tensor<4xsi32>
                // CHECK-NEXT:     %[[SCALE:.*]] = coreai.constant dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>
                // CHECK-NEXT:     %[[OFFSET:.*]] = coreai.constant dense<0.000000e+00> : tensor<4xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.interpolate %[[X]], %[[OUT]], %[[SCALE]], %[[OFFSET]] {interpolation_mode = #coreai.interpolation_mode<linear>} : (tensor<1x3x4x4xf32>, tensor<4xsi32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x3x8x8xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<1x3x8x8xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestUpsampleNearest2dIR:
    def test_static(self) -> None:
        class UpsampleNearestModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.interpolate(
                    x, scale_factor=2.0, mode="nearest"
                )

        ir = get_ir(UpsampleNearestModel().eval(), x=torch.rand(1, 3, 4, 4))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<1x3x4x4xf32> {coreai.name = "x"}) -> (tensor<1x3x8x8xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[OUT:.*]] = coreai.constant dense<[1, 3, 8, 8]> : tensor<4xsi32>
                // CHECK-NEXT:     %[[SCALE:.*]] = coreai.constant dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>
                // CHECK-NEXT:     %[[OFFSET:.*]] = coreai.constant dense<0.000000e+00> : tensor<4xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.interpolate %[[X]], %[[OUT]], %[[SCALE]], %[[OFFSET]] {interpolation_mode = #coreai.interpolation_mode<nearest_neighbor>} : (tensor<1x3x4x4xf32>, tensor<4xsi32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x3x8x8xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<1x3x8x8xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestViewIR:
    def test_static(self) -> None:
        class ViewModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.view(6)

        ir = get_ir(ViewModel().eval(), x=torch.rand(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<6xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<6> : tensor<1xui32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.reshape %[[X]], %[[SHAPE]] : (tensor<2x3xf32>, tensor<1xui32>) -> tensor<6xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<6xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestViewAsComplexIR:
    def test_static(self) -> None:
        class ViewAsComplexModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.view_as_complex(x)

        ir = get_ir(ViewAsComplexModel().eval(), x=torch.rand(2, 3, 2))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3x2xf32> {coreai.name = "x"}) -> (tensor<2x3xcomplex<f32>> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[2, 3]> : tensor<2xui32>
                // CHECK-NEXT:     %[[INF:.*]] = coreai.constant dense<2147483647> : tensor<3xsi32>
                // CHECK-NEXT:     %[[STARTIM:.*]] = coreai.constant dense<[0, 0, 1]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[STARTRE:.*]] = coreai.constant dense<0> : tensor<3xsi32>
                // CHECK-NEXT:     %[[ENDRE:.*]] = coreai.constant dense<[2147483647, 2147483647, 1]> : tensor<3xsi32>
                // CHECK-NEXT:     %[[STRIDE:.*]] = coreai.constant dense<1> : tensor<3xsi32>
                // CHECK-NEXT:     %[[RE:.*]] = coreai.slice %[[X]], %[[STARTRE]], %[[ENDRE]], %[[STRIDE]] : (tensor<2x3x2xf32>, tensor<3xsi32>, tensor<3xsi32>, tensor<3xsi32>) -> tensor<2x3x1xf32>
                // CHECK-NEXT:     %[[IM:.*]] = coreai.slice %[[X]], %[[STARTIM]], %[[INF]], %[[STRIDE]] : (tensor<2x3x2xf32>, tensor<3xsi32>, tensor<3xsi32>, tensor<3xsi32>) -> tensor<2x3x1xf32>
                // CHECK-NEXT:     %[[RE2:.*]] = coreai.reshape %[[RE]], %[[SHAPE]] : (tensor<2x3x1xf32>, tensor<2xui32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[IM2:.*]] = coreai.reshape %[[IM]], %[[SHAPE]] : (tensor<2x3x1xf32>, tensor<2xui32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.create_complex %[[RE2]], %[[IM2]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xcomplex<f32>>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xcomplex<f32>>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestViewAsRealIR:
    def test_static(self) -> None:
        class ViewAsRealModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.view_as_real(x)

        ir = get_ir(
            ViewAsRealModel().eval(),
            x=torch.complex(torch.rand(2, 3), torch.rand(2, 3)),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xcomplex<f32>> {coreai.name = "x"}) -> (tensor<2x3x2xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[SHAPE:.*]] = coreai.constant dense<[2, 3, 1]> : tensor<3xui32>
                // CHECK-NEXT:     %[[DIM:.*]] = coreai.constant dense<2> : tensor<si32>
                // CHECK-NEXT:     %[[RE:.*]] = coreai.real_part %[[X]] : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[IM:.*]] = coreai.imaginary_part %[[X]] : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32>
                // CHECK-NEXT:     %[[RE2:.*]] = coreai.reshape %[[RE]], %[[SHAPE]] : (tensor<2x3xf32>, tensor<3xui32>) -> tensor<2x3x1xf32>
                // CHECK-NEXT:     %[[IM2:.*]] = coreai.reshape %[[IM]], %[[SHAPE]] : (tensor<2x3xf32>, tensor<3xui32>) -> tensor<2x3x1xf32>
                // CHECK-NEXT:     %[[R:.*]] = coreai.concat %[[DIM]], %[[RE2]], %[[IM2]] : (tensor<si32>, tensor<2x3x1xf32>, tensor<2x3x1xf32>) -> tensor<2x3x2xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3x2xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestWhereIR:
    def test_static(self) -> None:
        class WhereModel(nn.Module):
            def forward(self, c: Tensor, x: Tensor, y: Tensor) -> Tensor:
                return torch.where(c, x, y)

        ir = get_ir(
            WhereModel().eval(),
            c=torch.rand(2, 3) > 0.5,
            x=torch.rand(2, 3),
            y=torch.rand(2, 3),
        )
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[C:.*]]: tensor<2x3xi1> {coreai.name = "c"}, %[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}, %[[Y:.*]]: tensor<2x3xf32> {coreai.name = "y"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK-NEXT:     %[[R:.*]] = coreai.decomposable.broadcasting_where %[[C]], %[[X]], %[[Y]] : (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
                // CHECK-NEXT:     coreai.output %[[R]] : tensor<2x3xf32>
                // CHECK-NEXT:   }
                // CHECK-NEXT: }
            """,
        )


class TestWhileLoopIR:
    def test_static(self) -> None:
        from torch._higher_order_ops.while_loop import while_loop

        class WhileLoopModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                def cond_fn(x):
                    return x.sum() < 10

                def body_fn(x):
                    return (x + 1,)

                return while_loop(cond_fn, body_fn, (x,))[0]

        ir = get_ir(WhileLoopModel().eval(), x=torch.zeros(2, 3))
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK-LABEL: module {
                // CHECK-NEXT:   coreai.graph @main(%[[X:.*]]: tensor<2x3xf32> {coreai.name = "x"}) -> (tensor<2x3xf32> {coreai.name = "{{.*}}"}) attributes {__coreai_pure__} {
                // CHECK:           coreai.while
                // CHECK:             coreai.reduce_sum
                // CHECK:             coreai.decomposable.broadcasting_greater
                // CHECK:             coreai.reshape
                // CHECK:             coreai.condition
                // CHECK:           } do {
                // CHECK:             coreai.decomposable.broadcasting_add
                // CHECK:             coreai.yield
                // CHECK:           }
                // CHECK:           coreai.output
                // CHECK-NEXT:    }
                // CHECK-NEXT:  }
            """,
        )
