# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Torch implementation of composite scaled dot product attention op."""

import enum
from typing import cast

import strenum
import torch
from torch import Tensor
from typing_extensions import Self

from ._utils import Version


class CausalVariant(strenum.StrEnum):
    """
    The variant of causal mask.

    When query length = key length, causal mask is simply
    a square lower triangular True matrix, and this enum has no effect

    When query length != key length, causal mask can be considered as
    a slice of that square matrix. This enum specifies how to slice. Let
        q_len = query length
        k_len = key length
        max_len = max(q_len, k_len)
        square_causal_mask = torch.ones((max_len, max_len), dtype=torch.bool).tril()
    * upper_left: causal_mask = square_causal_mask[0 : q_len, 0 : k_len]
    * lower_right: causal_mask = square_causal_mask[max_len - q_len : max_len, max_len - k_len : max_len]

    More concise code to construct these variants
    * upper_left: torch.ones(q_len, k_len, dtype=torch.bool).tril()
    * lower_right: torch.ones(q_len, k_len, dtype=torch.bool).tril(diagonal=k_len - q_len)

    Reference
    https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.bias.CausalVariant.html
    """

    upper_left = enum.auto()
    lower_right = enum.auto()


def _maybe_construct_attn_mask(
    query: Tensor,
    key: Tensor,
    is_causal: bool = False,
    causal_variant: CausalVariant = cast("CausalVariant", CausalVariant.lower_right),
    window_size: int = 0,
) -> Tensor | None:
    attn_mask = None
    q_len = query.shape[-2]
    k_len = key.shape[-2]
    torch._check(q_len >= 0, message="int query length >= 0")  # type: ignore[no-untyped-call]
    torch._check(k_len >= 0, message="int key length >= 0")  # type: ignore[no-untyped-call]
    if window_size <= 0:
        # invalid window size, i.e. no window
        if is_causal and causal_variant == CausalVariant.lower_right:
            # lower-right causal mask, i.e. query being the trailing tokens,
            # is what we need for decoding, where when q_len != k_len
            # we have query being the latest token in sequence
            num_past_tokens = k_len - q_len
            torch._check(
                num_past_tokens >= 0,
                message="int number of past tokens >= 0",
            )  # type: ignore[no-untyped-call]
            # TODO: Simply use tril after PyTorch teammates fix
            #     https://github.com/pytorch/pytorch/issues/165613
            # attn_mask = torch.ones(q_len, k_len, dtype=torch.bool).tril(diagonal=num_past_tokens)
            q_indices = torch.arange(num_past_tokens, k_len, device=query.device)
            k_indices = torch.arange(k_len, device=query.device)
            attn_mask = q_indices[:, None] >= k_indices[None, :]
        # else can simply use torch sdpa: either no mask,
        # or torch internally constructed upper-left causal mask
    else:
        # sliding window attention
        q_idx = torch.arange(q_len, device=query.device).unsqueeze(1)
        k_idx = torch.arange(k_len, device=query.device).unsqueeze(0)
        left_out_of_window = q_idx >= k_idx - (k_len - q_len - window_size)
        if is_causal:
            # causal sliding window mask
            if causal_variant != CausalVariant.lower_right:
                msg = "For now only support lower-right causal sliding window mask"
                raise NotImplementedError(msg)
            causal_mask = q_idx >= k_idx - (k_len - q_len)
            attn_mask = torch.logical_xor(causal_mask, left_out_of_window)
        else:
            # sliding window mask
            right_out_of_window = q_idx <= k_idx - (k_len - q_len + window_size)
            attn_mask = torch.logical_not(
                torch.logical_or(left_out_of_window, right_out_of_window),
            )
    return attn_mask


def _vanilla_repeat_interleave(x: torch.Tensor, reps: int) -> torch.Tensor:
    """Vanilla implementation for repeat interleave used in GQA.

    PyTorch official torch.repeat_interleave has dynamic shape bug
    starting from torch 2.8 and still fails at torch 2.10, e.g.
        min(11*seq_len, 22*seq_len) == 11*seq_len
    In principle this should always hold true for 1 <= seq_len <= 2048,
    in practise it hits
        raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))

    Although the root cause is the incorrect evaluation of the bool expression,
    as a workaround we may dodge that constraint, which comes from
    an underlying reshape. Consider this vanilla
        batch, num_heads, seq_len, head_dim = x.shape
        return (
            x.unsqueeze(2)
            .expand(
                batch,
                num_heads,
                reps,
                seq_len,
                head_dim,
            )
            .contiguous()
            .view(batch, num_heads * reps, seq_len, head_dim)
        )
    The view op is trying to confirm
        num_heads x reps x seq_len == num_heads * reps x seq_len
    resulting in the incorrectly evaluated constraint.

    TODO: Remove this vanilla once PyTorch official repeat interleave gets fixed
    """
    if reps == 1:
        return x

    rank = len(x.shape)
    torch._check(
        rank == 4,
        message="GQA requires query rank == 4",
    )  # type: ignore[no-untyped-call]
    num_heads = x.shape[1]
    indices = torch.arange(num_heads, device=x.device).repeat_interleave(reps)
    return torch.index_select(x, 1, indices)


def _vanilla_sdpa(  # noqa: PLR0913, PLR0915
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float | None = None,
    enable_gqa: bool = True,
    attn_mask: Tensor | None = None,
    is_causal: bool = False,
    sinks: Tensor | None = None,
) -> Tensor:
    """Vanilla implementation for scaled dot product attention.

    Beyond PyTorch official torch.nn.functional.scaled_dot_product_attention
    this vanilla additionally handles:
    1. sinks.
    2. gqa that can torch.export.export with dynamic shape.

    TODO: Remove this vanilla once PyTorch official SDPA has feature pairty.
    """
    if enable_gqa:
        gqa_compatible_rank = 4
        q_rank = len(query.shape)
        torch._check(
            q_rank == gqa_compatible_rank,
            message="GQA requires query rank == 4",
        )  # type: ignore[no-untyped-call]
        k_rank = len(key.shape)
        torch._check(
            k_rank == gqa_compatible_rank,
            message="GQA requires key rank == 4",
        )  # type: ignore[no-untyped-call]
        v_rank = len(value.shape)
        torch._check(
            v_rank == gqa_compatible_rank,
            message="GQA requires value rank == 4",
        )  # type: ignore[no-untyped-call]
        n_q_heads = query.shape[1]
        torch._check(n_q_heads >= 0, message="int number of query heads >= 0")  # type: ignore[no-untyped-call]
        n_k_heads = key.shape[1]
        torch._check(n_k_heads >= 0, message="int number of key heads >= 0")  # type: ignore[no-untyped-call]
        n_v_heads = value.shape[1]
        torch._check(n_v_heads >= 0, message="int number of value heads >= 0")  # type: ignore[no-untyped-call]

        torch._check(
            n_q_heads % n_k_heads == 0,
            message="GQA requires number of query heads divisible by number of key heads",
        )  # type: ignore[no-untyped-call]
        k_group_size = n_q_heads // n_k_heads
        torch._check(k_group_size >= 0, message="int key group size >= 0")  # type: ignore[no-untyped-call]
        key = _vanilla_repeat_interleave(key, k_group_size)

        torch._check(
            n_q_heads % n_v_heads == 0,
            message="GQA requires number of query heads divisible by number of value heads",
        )  # type: ignore[no-untyped-call]
        v_group_size = n_q_heads // n_v_heads
        torch._check(v_group_size >= 0, message="int value group size >= 0")  # type: ignore[no-untyped-call]
        value = _vanilla_repeat_interleave(value, v_group_size)

    if scale is None:
        scale = query.shape[-1] ** -0.5

    scaled_query = scale * query
    transposed_key = key.transpose(-1, -2)
    attn_scores = torch.matmul(scaled_query, transposed_key)

    if attn_mask is None and is_causal:
        # construct bool upper-left causal mask according to
        # torch.nn.functional.scaled_dot_product_attention semantics
        q_len = query.shape[-2]
        torch._check(q_len >= 0, message="int query length >= 0")  # type: ignore[no-untyped-call]
        k_len = key.shape[-2]
        torch._check(k_len >= 0, message="int key length >= 0")  # type: ignore[no-untyped-call]
        q_indices = torch.arange(q_len, device=query.device)
        k_indices = torch.arange(k_len, device=query.device)
        attn_mask = q_indices.unsqueeze(-1) >= k_indices.unsqueeze(0)
    if attn_mask is not None:
        # bool mask to GPU-friendly float mask
        not_attended = torch.logical_not(attn_mask)
        float_mask = -1e4 * not_attended.to(query.dtype)
        attn_scores = attn_scores + float_mask

    if sinks is None:
        attn_weights = torch.softmax(attn_scores, dim=-1)
    else:
        torch._check(len(sinks.shape) == 1, message="sinks must be 1-dimensional")  # type: ignore[no-untyped-call]
        torch._check(
            sinks.shape[0] == query.shape[1],
            message="sinks length must match number of query heads",
        )  # type: ignore[no-untyped-call]
        # 1 x number of query heads x 1 x 1
        sinks_to_broadcast = sinks.reshape(1, -1, 1, 1)
        # batch size x number of query heads x query length x 1
        cat_shape = (*query.shape[:-1], 1)
        sinks_to_cat = torch.broadcast_to(sinks_to_broadcast, cat_shape)
        # batch size x number of query heads x query length x (key length + 1)
        attn_scores_with_sinks = torch.cat((attn_scores, sinks_to_cat), dim=-1)
        attn_weights_with_sinks = torch.softmax(attn_scores_with_sinks, dim=-1)
        # batch size x number of query heads x query length x key length
        k_len = key.shape[-2]
        torch._check(k_len >= 0, message="int key length >= 0")  # type: ignore[no-untyped-call]
        attn_weights = attn_weights_with_sinks.narrow(-1, 0, k_len)

    context_vector = torch.matmul(attn_weights, value)
    return context_vector


def _sdpa_impl(  # noqa: PLR0913
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    sinks: Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = False,
    causal_variant: CausalVariant = cast("CausalVariant", CausalVariant.lower_right),
    window_size: int = 0,
) -> Tensor:
    """
    Perform (sliding window) scaled dot-product attention on inputs.

    No reference op from other frameworks for sliding window case.
    The implementation below is to be verified by integration test:
    whether our own torch source Gemma3 model using this composite op produces
    the same result as Hugging Face Gemma3 model
    """
    torch_sdpa_kwargs = {
        "query": query,
        "key": key,
        "value": value,
        "scale": scale,
        "enable_gqa": True,
    }
    if attn_mask is None:
        attn_mask = _maybe_construct_attn_mask(
            query,
            key,
            is_causal,
            causal_variant,
            window_size,
        )
    if attn_mask is None:
        torch_sdpa_kwargs["is_causal"] = is_causal
    else:
        torch_sdpa_kwargs["attn_mask"] = attn_mask
    if sinks is not None:
        torch_sdpa_kwargs["sinks"] = sinks
    attention = _vanilla_sdpa(**torch_sdpa_kwargs)  # type: ignore[arg-type]
    return attention


def _sdpa_hf_impl(  # noqa: PLR0913
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    sinks: Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = False,
    causal_variant: CausalVariant = cast("CausalVariant", CausalVariant.lower_right),
    window_size: int = 0,
) -> Tensor:
    """
    Perform (sliding window) scaled dot-product attention on inputs.

    No reference op from other frameworks for sliding window case.
    The implementation below is to be verified by integration test:
    whether our own torch source Gemma3 model using this composite op produces
    the same result as Hugging Face Gemma3 model
    """

    def _reshape(x: torch.Tensor, reps: int) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = x.shape
        if reps == 1:
            return x
        return (
            x.unsqueeze(2)
            .expand(
                batch,
                num_heads,
                reps,
                seq_len,
                head_dim,
            )
            .reshape(batch, num_heads * reps, seq_len, head_dim)
        )

    if sinks is not None:
        msg = "Have not implemented HF sinked SDPA"
        raise NotImplementedError(msg)

    # since we don't allow enable_gqa, we need to reshape the key and value
    reps = query.shape[1] // key.shape[1]
    key = _reshape(key, reps)
    value = _reshape(value, reps)

    # need to do contiguous to avoid some bug in the torch sdpa
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    torch_sdpa_kwargs = {
        "query": query,
        "key": key,
        "value": value,
        "scale": scale,
        # enable_gqa MUST BE False in order to match the HF
        "enable_gqa": False,
    }
    torch._check(
        not torch_sdpa_kwargs["enable_gqa"],
        message="Hugging Face GQA produces wrong f16 numerics",
    )  # type: ignore[no-untyped-call]

    if attn_mask is None:
        attn_mask = _maybe_construct_attn_mask(
            query,
            key,
            is_causal,
            causal_variant,
            window_size,
        )

    if attn_mask is None:
        torch_sdpa_kwargs["is_causal"] = is_causal
    else:
        torch_sdpa_kwargs["attn_mask"] = attn_mask

    return torch.nn.functional.scaled_dot_product_attention(
        **torch_sdpa_kwargs,  # type: ignore[arg-type]
    ).contiguous()


def scaled_dot_product_attention(  # noqa: PLR0913
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_mask: Tensor | None = None,
    sinks: Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = False,
    window_size: int = 0,
    use_hf_impl: bool = False,
    version: Version = Version.v1,
) -> Tensor:
    """
    Perform (sliding window) scaled dot-product attention on inputs.

    Note: When is_causal, torch and our causal masks are the same when
    query.shape[-2] == key.shape[-2], but otherwise different
    torch.nn.functional.scaled_dot_product_attention causal mask is upper-left
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
    while our causal mask is lower-right
        1 1 1 0 0
        1 1 1 1 0
        1 1 1 1 1
    Lower-right is the v1 used in language model linear decoding
    """
    if version != Version.v1:
        msg = "For now only support scaled_dot_product_attention v1"
        raise NotImplementedError(msg)

    impl_func = _sdpa_hf_impl if use_hf_impl else _sdpa_impl
    output = impl_func(
        query,
        key,
        value,
        attn_mask=attn_mask,
        sinks=sinks,
        scale=scale,
        is_causal=is_causal,
        # only found lower-right use case for now, so only support it for now
        causal_variant=cast("CausalVariant", CausalVariant.lower_right),
        window_size=window_size,
    )
    return output


class SDPA(torch.nn.Module):
    """Apply scaled dot product attention to input tensors, with attributes pre-determined."""

    def __init__(
        self: Self,
        scale: float | None = None,
        is_causal: bool = False,
        window_size: int = 0,
        _use_hf_impl: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.is_causal = is_causal
        self.window_size = window_size
        self.version = Version.v1
        self._use_hf_impl = _use_hf_impl

    def forward(
        self: Self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply scaled dot product attention to input tensors."""
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            sinks=sinks,
            scale=self.scale,
            is_causal=self.is_causal,
            window_size=self.window_size,
            use_hf_impl=self._use_hf_impl,
        )
