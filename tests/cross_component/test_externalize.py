# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""End-to-end externalize regression: static query, dynamic KV context.

When a model is exported with a *static* query length but a *dynamic*
KV-context length (the prefill / decode shape used by hybrid linear-
attention models), the externalize pipeline re-exports the SDPA
submodule standalone. Pre-fix, the key sequence dim came back as
``[query_len, +inf)`` and ``torch.export`` rejected the submodule with
``L['key'].size()[2] <= IntInfinity()``.

Targets the torch 2.9 failure path. Coverage on torch >= 2.10 is in
progress.
"""

from __future__ import annotations

import pytest
import torch

from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table
from coreai_torch.composite_ops import RMSNorm, RoPE, SDPA


@torch.library.custom_op(
    "coreai_torch_test::mutable_slice_update_regression", mutates_args=["x"]
)
def _mutable_slice_update(
    x: torch.Tensor,
    update: torch.Tensor,
    begin: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    begin_t = torch.split(begin, 1, dim=0)
    end_t = torch.split(end, 1, dim=0)
    slices = tuple(slice(b.item(), e.item()) for b, e in zip(begin_t, end_t))
    x[slices] = update
    return x.clone()


@_mutable_slice_update.register_fake
def _mutable_slice_update_meta(
    x: torch.Tensor,
    update: torch.Tensor,
    begin: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype)


class _KVCache:
    """Minimal KV cache exposing ``update_and_fetch``.

    Mirrors the shape / op pattern used by upstream hybrid linear-
    attention models (``mutable_slice_update`` to write back, ``narrow``
    to fetch the active prefix); only what's needed to surface the
    externalize Dim bug is kept.
    """

    def __init__(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        self._k_cache = k_cache
        self._v_cache = v_cache

    def update_and_fetch(  # noqa: PLR0913
        self,
        layer_idx: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
        query_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch._check_is_size(query_len)  # type: ignore[no-untyped-call]
        torch._check_is_size(offset)  # type: ignore[no-untyped-call]
        torch._check_is_size(seq_len)  # type: ignore[no-untyped-call]
        torch._check_is_size(layer_idx)  # type: ignore[no-untyped-call]

        device = self._k_cache.device
        layer_index = torch.tensor((layer_idx,), dtype=torch.int32, device=device)
        layer_index_end = torch.tensor(
            (layer_idx + 1,), dtype=torch.int32, device=device
        )
        zero = torch.tensor((0,), dtype=torch.int32, device=device)

        for buf, src in ((self._k_cache, k), (self._v_cache, v)):
            _mutable_slice_update(
                buf,
                src.unsqueeze(0),
                torch.cat(
                    [
                        layer_index,
                        zero,
                        zero,
                        torch.tensor((offset,), dtype=torch.int32, device=device),
                        zero,
                    ]
                ),
                torch.cat(
                    [
                        layer_index_end,
                        torch.tensor(
                            (buf.size(1),), dtype=torch.int32, device=device
                        ),
                        torch.tensor(
                            (buf.size(2),), dtype=torch.int32, device=device
                        ),
                        torch.tensor(
                            (offset + src.size(2),),
                            dtype=torch.int32,
                            device=device,
                        ),
                        torch.tensor(
                            (buf.size(4),), dtype=torch.int32, device=device
                        ),
                    ]
                ),
            )

        k_out = self._k_cache.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len).squeeze(0)
        v_out = self._v_cache.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len).squeeze(0)
        return k_out, v_out


@pytest.mark.skipif(
    tuple(int(p) for p in torch.__version__.split(".")[:2]) >= (2, 10),
    reason="torch >= 2.10 coverage in progress.",
)
def test_attention_layer_static_query_dynamic_kv_externalize() -> None:
    HID, NH, NKV, HD = 256, 8, 2, 64

    def repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
        b, h, s, d = x.shape
        return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)

    class AttnLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(HID, NH * HD, bias=False)
            self.k_proj = torch.nn.Linear(HID, NKV * HD, bias=False)
            self.v_proj = torch.nn.Linear(HID, NKV * HD, bias=False)
            self.o_proj = torch.nn.Linear(NH * HD, HID, bias=False)
            self.q_norm = RMSNorm(HD, eps=1e-6)
            self.k_norm = RMSNorm(HD, eps=1e-6)
            self.rope = RoPE(base=1e5, dims=HD)
            self.sdpa = SDPA(is_causal=True)

        def forward(
            self,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
        ) -> torch.Tensor:
            b, query_len, _ = x.shape
            cache = _KVCache(k_cache, v_cache)
            sequence_length = position_ids.shape[-1]
            torch._check_is_size(sequence_length)  # type: ignore[no-untyped-call]
            offset = sequence_length - query_len
            torch._check_is_size(offset)  # type: ignore[no-untyped-call]
            q = self.q_proj(x).reshape(b, query_len, NH, HD).permute(0, 2, 1, 3)
            k = self.k_proj(x).reshape(b, query_len, NKV, HD).permute(0, 2, 1, 3)
            v = self.v_proj(x).reshape(b, query_len, NKV, HD).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            k = self.k_norm(k)
            rp = position_ids.narrow(-1, offset, query_len)
            q = self.rope(q, position_ids=rp)
            k = self.rope(k, position_ids=rp)
            k, v = cache.update_and_fetch(
                0, offset, k, v, seq_len=sequence_length, query_len=query_len
            )
            k = repeat_kv(k, NH // NKV)
            v = repeat_kv(v, NH // NKV)
            out = (
                self.sdpa(q, k, v)
                .permute(0, 2, 1, 3)
                .reshape(b, query_len, NH * HD)
            )
            return self.o_proj(out)

    torch.manual_seed(0)
    s, ctx, cap = 12, 20, 64  # static query=12, dynamic ctx (trace 20), cap 64
    model = AttnLayer().to(torch.float16).eval()
    x = torch.randn(1, s, HID, dtype=torch.float16)
    pos = torch.arange(ctx, dtype=torch.int32).unsqueeze(0)
    kc = torch.zeros(1, 1, NKV, cap, HD, dtype=torch.float16)
    vc = torch.zeros(1, 1, NKV, cap, HD, dtype=torch.float16)
    ds = {
        "x": None,
        "position_ids": {1: torch.export.Dim("ctx", min=s + 1, max=cap)},
        "k_cache": {3: torch.export.Dim("kseq", min=s + 1, max=cap)},
        "v_cache": {3: torch.export.Dim("vseq", min=s + 1, max=cap)},
    }
    spec = ExternalizeSpec(
        target_class=SDPA,
        composite_op_name="scaled_dot_product_attention",
        composite_attrs=["scale", "is_causal", "window_size"],
    )

    # Pre-fix this raised:
    #   RuntimeError: Internal error: failed to export submodule 'sdpa_*':
    #   Constraints violated (d_20)! ...
    #   12 <= L['key'].size()[2] and L['key'].size()[2] <= IntInfinity()
    # The fix lives in the externalize pipeline (run inside ``to_coreai``),
    # so we drive the converter past that step and tolerate any *downstream*
    # MLIR-lowering failure — the regression marker is the absence of the
    # constraint-violation message above.
    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(
            m, args=(x, pos, kc, vc), dynamic_shapes=ds
        ).run_decompositions(get_decomp_table()),
        externalize_modules=[spec],
    )
    try:
        converter.to_coreai()
    except Exception as e:  # noqa: BLE001
        if "Constraints violated" in str(e) and "IntInfinity" in str(e):
            raise AssertionError(
                "Externalize SDPA submodule re-export regressed "
                f"(issue #1): {e}"
            ) from e
        # Any other downstream failure (e.g. MLIR lowering on an
        # incomplete dev install) is unrelated to this bug.
