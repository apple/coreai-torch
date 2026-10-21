# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for ImageType graph inputs/outputs (input_image_types/output_image_types)."""

import pytest
import torch
import torch.nn as nn
from coreai._compiler.dialects.coreai import FourCC
from coreai.authoring import ImageType

from coreai_torch import TorchConverter

from .utils import filecheck_pattern

H, W = 4, 6


class InvertWithScale(nn.Module):
    """Image in -> image out, plus a pass-through float tensor."""

    def forward(self, image: torch.Tensor, scale: torch.Tensor):
        return 255 - image, scale * 2


class StatefulImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros(1))

    def forward(self, image: torch.Tensor):
        self.count.add_(1)
        return 255 - image


def _export(model: nn.Module, *args: torch.Tensor):
    return torch.export.export(model, args=args).run_decompositions()


def _rgb_image() -> torch.Tensor:
    return torch.zeros(H, W, 3, dtype=torch.uint8)


def _invert_program():
    return _export(InvertWithScale(), _rgb_image(), torch.randn(2))


RGB = ImageType(FourCC.RGB24, H, W)


@pytest.mark.ir
class TestImageIO:
    def test_default_is_all_tensors(self):
        program = TorchConverter().add_exported_program(_invert_program()).to_coreai()
        ir = str(program)
        assert "!coreai.image" not in ir
        assert "image_to_tensor" not in ir
        assert "tensor_to_image" not in ir

    def test_image_input_only(self):
        program = (
            TorchConverter()
            .add_exported_program(_invert_program(), input_image_types={"image": RGB})
            .to_coreai()
        )
        filecheck_pattern(
            str(program),
            check_file="""
                // CHECK: coreai.graph @main(%[[IMG:.*]]: !coreai.image<<k24RGB> x 4 : i64 x 6 : i64> {coreai.name = "image"}, %{{.*}}: tensor<2xf32> {coreai.name = "scale"}) -> (tensor<4x6x3xui8> {{.*}}, tensor<2xf32> {{.*}})
                // CHECK: coreai.image_to_tensor %[[IMG]] : <<k24RGB> x 4 : i64 x 6 : i64> -> tensor<4x6x3xui8>
                // CHECK-NOT: coreai.tensor_to_image
            """,
        )

    def test_image_output_only(self):
        program = (
            TorchConverter()
            .add_exported_program(_invert_program(), output_image_types={"sub": RGB})
            .to_coreai()
        )
        filecheck_pattern(
            str(program),
            check_file="""
                // CHECK: coreai.graph @main(%{{.*}}: tensor<4x6x3xui8> {{.*}}, %{{.*}}: tensor<2xf32> {{.*}}) -> (!coreai.image<<k24RGB> x 4 : i64 x 6 : i64> {coreai.name = "sub"}, tensor<2xf32> {{.*}})
                // CHECK-NOT: coreai.image_to_tensor
                // CHECK: %[[OUT:.*]] = coreai.tensor_to_image %{{.*}} : tensor<4x6x3xui8> -> <<k24RGB> x 4 : i64 x 6 : i64>
                // CHECK: coreai.output %[[OUT]], %{{.*}} : !coreai.image<<k24RGB> x 4 : i64 x 6 : i64>, tensor<2xf32>
            """,
        )

    def test_image_input_and_output_with_user_names(self):
        """Keys are the user-provided input_names/output_names."""
        program = (
            TorchConverter()
            .add_exported_program(
                _invert_program(),
                input_names=["photo", "s"],
                output_names=["inverted", "scaled"],
                input_image_types={"photo": RGB},
                output_image_types={"inverted": RGB},
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(program),
            check_file="""
                // CHECK: coreai.graph @main(%[[IMG:.*]]: !coreai.image<{{.*}}> {coreai.name = "photo"}, %{{.*}}: tensor<2xf32> {coreai.name = "s"}) -> (!coreai.image<{{.*}}> {coreai.name = "inverted"}, tensor<2xf32> {coreai.name = "scaled"})
                // CHECK: coreai.image_to_tensor %[[IMG]]
                // CHECK: %[[OUT:.*]] = coreai.tensor_to_image
                // CHECK: coreai.output %[[OUT]], %{{.*}}
            """,
        )

    def test_add_pytorch_module(self):
        example = (_rgb_image(), torch.randn(2))
        program = (
            TorchConverter()
            .add_pytorch_module(
                InvertWithScale(),
                export_fn=lambda m: _export(m, *example),
                input_image_types={"image": RGB},
                output_image_types={"sub": RGB},
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(program),
            check_file="""
                // CHECK: coreai.graph @main(%{{.*}}: !coreai.image<{{.*}}> {coreai.name = "image"}, {{.*}}) -> (!coreai.image<{{.*}}> {coreai.name = "sub"}, {{.*}})
                // CHECK: coreai.image_to_tensor
                // CHECK: coreai.tensor_to_image
            """,
        )

    @pytest.mark.parametrize(
        "fourcc, fourcc_ir, channels",
        [
            (FourCC.RGB24, "k24RGB", 3),
            (FourCC.ARGB32, "k32ARGB", 4),
            (FourCC.BGRA32, "k32BGRA", 4),
        ],
    )
    def test_fourcc_formats(self, fourcc, fourcc_ir, channels):
        image = torch.zeros(H, W, channels, dtype=torch.uint8)
        image_type = ImageType(fourcc, H, W)
        program = (
            TorchConverter()
            .add_exported_program(
                _export(InvertWithScale(), image, torch.randn(2)),
                input_image_types={"image": image_type},
                output_image_types={"sub": image_type},
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(program),
            check_file=f"""
                // CHECK: coreai.image_to_tensor %{{{{.*}}}} : <<{fourcc_ir}> x 4 : i64 x 6 : i64> -> tensor<4x6x{channels}xui8>
                // CHECK: coreai.tensor_to_image %{{{{.*}}}} : tensor<4x6x{channels}xui8> -> <<{fourcc_ir}> x 4 : i64 x 6 : i64>
            """,
        )

    def test_stateful_model_image_io(self):
        """Image IO coexists with mutable-buffer state IO."""
        program = (
            TorchConverter()
            .add_exported_program(
                _export(StatefulImage(), _rgb_image()),
                input_image_types={"image": RGB},
                output_image_types={"sub": RGB},
            )
            .to_coreai()
        )
        filecheck_pattern(
            str(program),
            check_file="""
                // CHECK: coreai.graph @main(%{{.*}} {MutableBuffers.buffer_mutation = {{.*}}}, %{{.*}}: !coreai.image<{{.*}}> {coreai.name = "image"})
                // CHECK-SAME: !coreai.image<{{.*}}> {coreai.name = "sub"}
                // CHECK: coreai.image_to_tensor
                // CHECK: coreai.tensor_to_image
            """,
        )


class TestImageIOErrors:
    def test_non_image_type_value(self):
        with pytest.raises(TypeError, match=r"input_image_types\['image'\]"):
            TorchConverter().add_exported_program(
                _invert_program(), input_image_types={"image": "RGB24"}
            )

    @pytest.mark.parametrize("arg", ["input_image_types", "output_image_types"])
    def test_unknown_name(self, arg):
        converter = TorchConverter().add_exported_program(
            _invert_program(), **{arg: {"missing": RGB}}
        )
        with pytest.raises(ValueError, match=rf"{arg} has unknown name\(s\)"):
            converter.to_coreai()

    def test_fx_name_rejected_when_input_names_given(self):
        converter = TorchConverter().add_exported_program(
            _invert_program(),
            input_names=["photo", "s"],
            input_image_types={"image": RGB},
        )
        with pytest.raises(ValueError, match=r"Valid names: \['photo', 's'\]"):
            converter.to_coreai()

    def test_state_input_rejected(self):
        ep = _export(StatefulImage(), _rgb_image())
        state_name = next(
            spec.arg.name
            for spec in ep.graph_signature.input_specs
            if spec.target == "count"
        )
        converter = TorchConverter().add_exported_program(
            ep, input_image_types={state_name: RGB}
        )
        with pytest.raises(ValueError, match="State inputs and outputs cannot"):
            converter.to_coreai()

    @pytest.mark.parametrize(
        "image",
        [
            torch.zeros(H, W, 4, dtype=torch.uint8),  # channel mismatch
            torch.zeros(W, H, 3, dtype=torch.uint8),  # height/width swapped
            torch.zeros(H, W, 3, dtype=torch.float32),  # dtype mismatch
        ],
    )
    def test_input_tensor_mismatch(self, image):
        converter = TorchConverter().add_exported_program(
            _export(InvertWithScale(), image, torch.randn(2)),
            input_image_types={"image": RGB},
        )
        with pytest.raises(ValueError, match=r"input_image_types\['image'\]"):
            converter.to_coreai()
