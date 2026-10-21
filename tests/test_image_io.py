# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for ImageType graph inputs/outputs (input_image_types/output_image_types)."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
import torch.nn as nn
from coreai import runtime
from coreai._compiler.dialects.coreai import FourCC
from coreai.authoring import ImageType
from coreai.runtime import NDArray
from PIL import Image

from coreai_torch import TorchConverter

from .utils import _get_test_specialization_options, filecheck_pattern

H, W = 4, 6
RGB = ImageType(FourCC.RGB24, H, W)
BGRA = ImageType(FourCC.BGRA32, H, W)


class Invert(nn.Module):
    """Default IO names: inputs ``image``, ``scale``; outputs ``sub``, ``mul``."""

    def forward(self, image: torch.Tensor, scale: torch.Tensor):
        return 255 - image, scale * 2


class StatefulInvert(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros(1))

    def forward(self, image: torch.Tensor):
        self.count.add_(1)
        return 255 - image


def _image(channels: int = 3, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    return torch.zeros(H, W, channels, dtype=dtype)


def _export(model: nn.Module, *args: torch.Tensor):
    return torch.export.export(model, args=args).run_decompositions()


def _convert(image: torch.Tensor | None = None, **kwargs):
    """Export ``Invert`` on ``image`` and convert it with ``kwargs``."""
    image = _image() if image is None else image
    ep = _export(Invert(), image, torch.randn(2))
    return TorchConverter().add_exported_program(ep, **kwargs).to_coreai()


# ---------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------


@pytest.mark.ir
class TestImageIOIR:
    def test_default_is_all_tensors(self):
        ir = str(_convert())
        assert "!coreai.image" not in ir
        assert "image_to_tensor" not in ir
        assert "tensor_to_image" not in ir

    def test_image_input(self):
        filecheck_pattern(
            str(_convert(input_image_types={"image": RGB})),
            check_file="""
                // CHECK: coreai.graph @main(%[[IMG:.*]]: !coreai.image<<k24RGB> x 4 : i64 x 6 : i64> {coreai.name = "image"}, %{{.*}}: tensor<2xf32> {coreai.name = "scale"}) -> (tensor<4x6x3xui8> {{.*}}, tensor<2xf32> {{.*}})
                // CHECK: coreai.image_to_tensor %[[IMG]] : <<k24RGB> x 4 : i64 x 6 : i64> -> tensor<4x6x3xui8>
                // CHECK-NOT: coreai.tensor_to_image
            """,
        )

    def test_image_output(self):
        filecheck_pattern(
            str(_convert(output_image_types={"sub": RGB})),
            check_file="""
                // CHECK: coreai.graph @main(%{{.*}}: tensor<4x6x3xui8> {{.*}}, %{{.*}}: tensor<2xf32> {{.*}}) -> (!coreai.image<<k24RGB> x 4 : i64 x 6 : i64> {coreai.name = "sub"}, tensor<2xf32> {{.*}})
                // CHECK-NOT: coreai.image_to_tensor
                // CHECK: %[[OUT:.*]] = coreai.tensor_to_image %{{.*}} : tensor<4x6x3xui8> -> <<k24RGB> x 4 : i64 x 6 : i64>
                // CHECK: coreai.output %[[OUT]], %{{.*}} : !coreai.image<<k24RGB> x 4 : i64 x 6 : i64>, tensor<2xf32>
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
        image_type = ImageType(fourcc, H, W)
        program = _convert(
            _image(channels),
            input_image_types={"image": image_type},
            output_image_types={"sub": image_type},
        )
        filecheck_pattern(
            str(program),
            check_file=f"""
                // CHECK: coreai.image_to_tensor %{{{{.*}}}} : <<{fourcc_ir}> x 4 : i64 x 6 : i64> -> tensor<4x6x{channels}xui8>
                // CHECK: coreai.tensor_to_image %{{{{.*}}}} : tensor<4x6x{channels}xui8> -> <<{fourcc_ir}> x 4 : i64 x 6 : i64>
            """,
        )

    def test_keys_are_user_names(self):
        program = _convert(
            input_names=["photo", "s"],
            output_names=["inverted", "scaled"],
            input_image_types={"photo": RGB},
            output_image_types={"inverted": RGB},
        )
        filecheck_pattern(
            str(program),
            check_file="""
                // CHECK: coreai.graph @main(%{{.*}}: !coreai.image<{{.*}}> {coreai.name = "photo"}, %{{.*}}: tensor<2xf32> {coreai.name = "s"}) -> (!coreai.image<{{.*}}> {coreai.name = "inverted"}, tensor<2xf32> {coreai.name = "scaled"})
            """,
        )

    def test_add_pytorch_module(self):
        program = (
            TorchConverter()
            .add_pytorch_module(
                Invert(),
                export_fn=lambda m: _export(m, _image(), torch.randn(2)),
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

    def test_with_state(self):
        """Image IO coexists with mutable-buffer state IO."""
        program = (
            TorchConverter()
            .add_exported_program(
                _export(StatefulInvert(), _image()),
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


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestImageIOErrors:
    def test_value_not_image_type(self):
        with pytest.raises(TypeError, match=r"input_image_types\['image'\]"):
            _convert(input_image_types={"image": "RGB24"})

    @pytest.mark.parametrize("arg", ["input_image_types", "output_image_types"])
    def test_unknown_name(self, arg):
        with pytest.raises(ValueError, match=rf"{arg} has unknown name\(s\)"):
            _convert(**{arg: {"missing": RGB}})

    def test_fx_name_after_rename(self):
        with pytest.raises(ValueError, match=r"Valid names: \['photo', 's'\]"):
            _convert(input_names=["photo", "s"], input_image_types={"image": RGB})

    def test_state_input(self):
        ep = _export(StatefulInvert(), _image())
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
            _image(channels=4),
            torch.zeros(W, H, 3, dtype=torch.uint8),
            _image(dtype=torch.float32),
        ],
        ids=["channels", "height-width-swapped", "dtype"],
    )
    def test_input_tensor_mismatch(self, image):
        with pytest.raises(ValueError, match=r"input_image_types\['image'\]"):
            _convert(image, input_image_types={"image": RGB})


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


async def _run(program, **inputs) -> dict:
    """Compile ``program`` and call ``main`` once."""
    with TemporaryDirectory() as temp_directory:
        asset = program.save_asset(Path(temp_directory) / "model.aimodel")
        async with asset.executable(
            specialization_options=_get_test_specialization_options(),
        ) as ai_model:
            return await ai_model.load_function("main")(inputs=inputs)


def _pixels(channels: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (H, W, channels), dtype=np.uint8)


def _rgba_to_bgra(pixels: np.ndarray) -> np.ndarray:
    """Pixel order the runtime uses for an RGB(A) PIL image; alpha defaults to 255."""
    alpha = pixels[..., 3] if pixels.shape[-1] == 4 else np.full((H, W), 255, np.uint8)
    return np.dstack([pixels[..., 2], pixels[..., 1], pixels[..., 0], alpha])


@pytest.mark.skipif(
    not runtime.HAS_IMAGE_SUPPORT, reason="Core AI runtime image support required"
)
class TestImageIONumerics:
    """Run converted models on the Core AI runtime and compare against PyTorch.

    The runtime turns every ``PIL.Image`` input into a BGRA pixel buffer, so
    image inputs are tested with ``FourCC.BGRA32`` only.
    """

    @pytest.mark.parametrize("mode", ["RGB", "RGBA"])
    async def test_image_input(self, mode):
        pixels = _pixels(len(mode))
        program = _convert(_image(4), input_image_types={"image": BGRA})
        outputs = await _run(
            program,
            image=Image.fromarray(pixels, mode),
            scale=NDArray(np.ones(2, np.float32)),
        )
        np.testing.assert_array_equal(
            outputs["sub"].numpy(), 255 - _rgba_to_bgra(pixels)
        )

    @pytest.mark.parametrize(
        "image_type, mode", [(RGB, "RGB"), (BGRA, "RGBA")], ids=["RGB24", "BGRA32"]
    )
    async def test_image_output(self, image_type, mode):
        pixels = _pixels(len(mode))
        scale = np.random.default_rng(1).standard_normal(2).astype(np.float32)
        program = _convert(_image(len(mode)), output_image_types={"sub": image_type})
        outputs = await _run(program, image=NDArray(pixels), scale=NDArray(scale))

        image = outputs["sub"]
        assert isinstance(image, Image.Image) and image.mode == mode
        expected = 255 - pixels
        if mode == "RGBA":  # BGRA tensor comes back as an RGBA PIL image
            expected = expected[..., [2, 1, 0, 3]]
        np.testing.assert_array_equal(np.asarray(image), expected)
        np.testing.assert_allclose(outputs["mul"].numpy(), scale * 2, rtol=1e-5)

    async def test_image_round_trip(self):
        pixels = _pixels(4)
        program = _convert(
            _image(4),
            input_image_types={"image": BGRA},
            output_image_types={"sub": BGRA},
        )
        outputs = await _run(
            program,
            image=Image.fromarray(pixels, "RGBA"),
            scale=NDArray(np.ones(2, np.float32)),
        )
        # BGRA in and out: the channel swaps cancel.
        np.testing.assert_array_equal(np.asarray(outputs["sub"]), 255 - pixels)
