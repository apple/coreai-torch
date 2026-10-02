# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Torch implementation of composite rotary positional embedding op."""

from typing import cast

import torch
from torch import Tensor
from typing_extensions import Self

from ._utils import Version


def _compute_angle(
    freqs: Tensor,
    scale: float,
    input: Tensor,
    position_ids: Tensor,
    use_hf_impl: bool = False,
) -> Tensor:
    """Construct rotation angle with given parameters."""
    # We split this utils out of the method in order to unittest the implementation.
    torch._check(
        position_ids.dtype == torch.float32,
        message="position_ids needs to be in fp32",
    )  # type: ignore[no-untyped-call]
    torch._check(freqs.dtype == torch.float32, message="freqs needs to be in fp32")  # type: ignore[no-untyped-call]

    freqs = freqs * scale

    if use_hf_impl:
        # In HF, the freq is cast to match the input dtype; however, this harms output numerics.
        freqs = freqs.to(input.dtype)

    # when doing the matmul, the position_ids and freq need to both in fp32 precision
    # (..., q_len, half_dim) = (..., q_len, 1) * (half_dim)
    position_ids_expand = position_ids.unsqueeze(-1)
    return position_ids_expand * freqs.float()


def _construct_cos_and_sin(  # noqa: PLR0913
    input: Tensor,
    half_dim: int,
    position_ids: Tensor | None = None,
    freqs: Tensor | None = None,
    offset: int | Tensor = 0,
    scale: float = 1.0,
    base: float = 1e4,
    use_hf_impl: bool = False,
) -> tuple[Tensor, Tensor]:
    """Construct rotation cos and sin with given parameters."""
    if position_ids is not None:
        # position_ids has shape (batch_size, q_len)
        # but needs to be broadcast to (batch_size, num_heads, q_len)
        # so we explicitly create the num_heads dimension
        # unfortunately this will not work for no-head / more-dims cases
        # (batch_size, 1, q_len)
        position_ids = position_ids.unsqueeze(1)
    else:
        if isinstance(offset, Tensor):
            # offset has shape (batch_size,)
            # so we explicitly create the num_heads dimension
            # unfortunately this will not work for no-head / more-dims cases
            # (batch_size, 1, 1)
            offset = offset.unsqueeze(-1).unsqueeze(-1)
        # Tensor offset: (batch_size, 1, q_len)
        # Int offset: (q_len,)
        q_len = input.shape[-2]
        torch._check(q_len >= 0, message="int query length >= 0")  # type: ignore[no-untyped-call]
        position_ids = offset + torch.arange(q_len, device=input.device)

    position_ids = position_ids.float()

    if freqs is None:
        # need f32, but why?
        #     exponent = 0, 1 / half_dim, ..., (half_dim - 1) / half_dim
        #     For half_dim <= 1024 = 2^10, f16 (with 10 fraction bits)
        #     in principle could represent 1 / half_dim exactly
        # in practice f16 gives wrong generated text... even if half_dim = 64 = 2^6
        # anyway, observation is always correct :p let us just stick to f32
        exponent_f32 = (
            torch.arange(half_dim, dtype=torch.float32, device=input.device) / half_dim
        )
        # need f32, since the big base ^ small exponent is numerically fragile
        period_f32 = torch.pow(base, exponent_f32)
        freqs = 1.0 / period_f32

    angle = _compute_angle(
        freqs,
        scale,
        input,
        position_ids,
        use_hf_impl=use_hf_impl,
    )

    # Not until the last step we convert the dtype back to the input dtype
    return angle.cos().to(input.dtype), angle.sin().to(input.dtype)


def _determine_if_partial_rotation(input: Tensor, dims: int | None) -> tuple[bool, int]:
    """Determine if rotation is partial and rotation dims."""
    embedding_dim = input.shape[-1]
    is_partial_rotation = dims is not None and dims < embedding_dim
    rotation_dims = dims if is_partial_rotation else embedding_dim

    # need this cast to convince mypy that rotation dims cannot be None
    # TODO: Remove this cast once mypy can infer type correctly
    rotation_dims = cast("int", rotation_dims)

    torch._check(rotation_dims >= 0, message="int rotation dimension >= 0")  # type: ignore[no-untyped-call]
    torch._check(rotation_dims >= 2, message="int rotation dimension >= 2")  # type: ignore[no-untyped-call]  # noqa: PLR2004
    torch._check(rotation_dims % 2 == 0, message="rotation dimension divisible by 2")  # type: ignore[no-untyped-call]
    return is_partial_rotation, rotation_dims


def _rope_with_cos_and_sin_impl(
    input: Tensor,
    cos: Tensor,
    sin: Tensor,
    dims: int | None = None,
    interleaved: bool = False,
) -> Tensor:
    """Perform rotary positional embedding on input with given cos & sin."""
    is_partial_rotation, rotation_dims = _determine_if_partial_rotation(input, dims)
    half_dim = rotation_dims // 2
    torch._check(half_dim >= 0, message="int embedding dimension / 2 >= 0")  # type: ignore[no-untyped-call]
    # split x
    if interleaved:
        x1 = input[..., :rotation_dims:2]
        x2 = input[..., 1:rotation_dims:2]
    else:
        x1 = input[..., :half_dim]
        x2 = input[..., half_dim:rotation_dims]
    # y = rotate . x
    y1 = cos * x1 - sin * x2
    y2 = sin * x1 + cos * x2
    # concatenate y back to original shape
    if interleaved:
        y1_expand = y1.unsqueeze(-1)
        y2_expand = y2.unsqueeze(-1)
        y_expand = torch.cat((y1_expand, y2_expand), dim=-1)
        if is_partial_rotation:
            y_shape = (*input.shape[:-1], rotation_dims)
        else:
            y_shape = input.shape
        y = y_expand.reshape(y_shape)
    else:
        y = torch.cat((y1, y2), dim=-1)
    if is_partial_rotation:
        y = torch.cat((y, input[..., rotation_dims:]), dim=-1)
    return y


def _rope_impl(  # noqa: PLR0913
    input: Tensor,
    cos: Tensor | None = None,
    sin: Tensor | None = None,
    position_ids: Tensor | None = None,
    freqs: Tensor | None = None,
    offset: int | Tensor = 0,
    scale: float = 1.0,
    base: float = 1e4,
    dims: int | None = None,
    interleaved: bool = False,
    use_hf_impl: bool = False,
) -> Tensor:
    """Perform rotary positional embedding on input."""
    if cos is None or sin is None:
        _, rotation_dims = _determine_if_partial_rotation(input, dims)
        half_dim = rotation_dims // 2
        torch._check(half_dim >= 0, message="int embedding dimension / 2 >= 0")  # type: ignore[no-untyped-call]
        cos, sin = _construct_cos_and_sin(
            input,
            half_dim,
            position_ids=position_ids,
            offset=offset,
            scale=scale,
            freqs=freqs,
            base=base,
            use_hf_impl=use_hf_impl,
        )
    output = _rope_with_cos_and_sin_impl(
        input,
        cos,
        sin,
        dims=dims,
        interleaved=interleaved,
    )
    return output


def rope(  # noqa: PLR0913
    input: Tensor,
    *,
    cos: Tensor | None = None,
    sin: Tensor | None = None,
    position_ids: Tensor | None = None,
    freqs: Tensor | None = None,
    offset: int | Tensor = 0,
    scale: float = 1.0,
    base: float = 1e4,
    dims: int | None = None,
    interleaved: bool = False,
    use_hf_impl: bool = False,
    version: Version = Version.v1,
) -> Tensor:
    """Perform rotary positional embedding on input."""
    if version != Version.v1:
        msg = "For now only support rope v1"
        raise NotImplementedError(msg)

    output = _rope_impl(
        input,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        freqs=freqs,
        offset=offset,
        scale=scale,
        base=base,
        dims=dims,
        interleaved=interleaved,
        use_hf_impl=use_hf_impl,
    )
    return output


class RoPE(torch.nn.Module):
    """Apply the rotary positional embedding function to input tensors, with attributes pre-determined."""

    def __init__(
        self: Self,
        scale: float = 1.0,
        base: float = 1e4,
        dims: int | None = None,
        interleaved: bool = False,
        _use_hf_impl: bool = False,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.base = float(base)
        self.dims = dims
        self.interleaved = interleaved
        self.version = Version.v1
        self._use_hf_impl = _use_hf_impl

    def forward(  # noqa: PLR0913
        self: Self,
        input: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the rotary positional embedding function to input tensors."""
        return rope(
            input,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            freqs=freqs,
            offset=offset if offset is not None else 0,
            scale=self.scale,
            base=self.base,
            dims=self.dims,
            interleaved=self.interleaved,
            use_hf_impl=self._use_hf_impl,
        )
