# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Workflow test: macOS (GPU) style stateful LLM export, end to end.

A minimal, self-contained replica of the macOS LLM export path in
`coreai-models <https://github.com/apple/coreai-models>`_
(``coreai_models.export.macos.export_to_coreai``), so regressions in that
workflow surface here without depending on coreai-models or HF weights.

What is replicated:

- **Authoring**: a tiny Qwen3-style decoder (fused QKV projection, fused
  query/key RMSNorm, RoPE, causal SDPA, gated MLP) built on the
  ``coreai_torch.composite_ops`` primitives.
- **KV cache as state**: 5D ``(layers, 1, kv_heads, max_seq, head_dim)`` K/V
  caches passed as mutated user inputs and updated through a fused
  ``mutable_cache_update_and_fetch`` custom op.
- **Export**: ``torch.export`` with dynamic query / position lengths ->
  ``get_decomp_table()`` decompositions -> defunctionalization (rewriting the
  auto-functionalized cache op into ``immutable_slice_update`` + aten slices)
  -> ``TorchConverter.add_pytorch_module`` with RMSNorm / RoPE / SDPA
  externalized as composite ops and a custom lowering to
  ``coreai.slice_update``.
- **Entrypoints**: ``main`` (decode, returns logits) and ``prefill`` (same
  signature, no outputs; only writes the cache).

The numerical test then drives the interpreter through a short generation
loop -- prefill, a multi-token chunk at a non-zero offset, and a few
single-token decode steps -- with the runtime-owned KV cache carried across
calls, comparing logits and caches against eager PyTorch at every step.
"""

import operator
from typing import Any

import pytest
import torch
import torch.nn as nn
from coreai._compiler.dialects import coreai
from coreai.authoring import AIProgram
from torch import Tensor, fx
from torch._higher_order_ops.auto_functionalize import (
    AutoFunctionalized,
    AutoFunctionalizedV2,
)

import coreai_torch
import coreai_torch.composite_ops
from coreai_torch import ExternalizeSpec, TorchConverter

from ..utils import LLMStep, filecheck_pattern, validate_llm_decode_loop

# Graph / state names match the coreai-models runner contract.
MAIN_GRAPH_NAME = "main"
PREFILL_GRAPH_NAME = "prefill"
KEY_CACHE_NAME = "keyCache"
VALUE_CACHE_NAME = "valueCache"
INPUT_NAMES = ("input_ids", "position_ids")
OUTPUT_NAMES = ("logits",)
STATE_NAMES = (KEY_CACHE_NAME, VALUE_CACHE_NAME)

# Tiny model, small enough to keep conversion + interpreter runs fast.
VOCAB_SIZE = 64
HIDDEN_SIZE = 32
INTERMEDIATE_SIZE = 64
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10000.0
MAX_CONTEXT_LENGTH = 64

# Trace-time shapes (coreai-models' ``TraceSpec``): ``position_ids`` is
# ``query_len + offset`` long, so the traced offset is non-zero.
TRACE_QUERY_LEN = 4
TRACE_OFFSET = 2

# Generation loop: prompt via ``prefill``, a multi-token chunk via ``main``
# (exercises the lower-right causal mask with offset > 0), then decode.
PROMPT_LEN = 8
CHUNK_LEN = 3
NUM_DECODE_STEPS = 4

EXTERNALIZE_SPECS: list[type | ExternalizeSpec] = [
    ExternalizeSpec(
        target_class=coreai_torch.composite_ops.RMSNormImpl,
        composite_op_name="rms_norm",
        composite_attrs=["axes", "eps"],
    ),
    ExternalizeSpec(
        target_class=coreai_torch.composite_ops.RoPE,
        composite_op_name="rope",
        composite_attrs=["scale", "base", "dims", "interleaved"],
    ),
    ExternalizeSpec(
        target_class=coreai_torch.composite_ops.SDPA,
        composite_op_name="scaled_dot_product_attention",
        composite_attrs=["scale", "is_causal", "window_size"],
    ),
]


# ===========================================================================
# KV cache custom ops (mirror coreai_models.primitives._ops / export.mlir_ops)
# ===========================================================================


@torch.library.custom_op(
    "workflow_test::mutable_cache_update_and_fetch", mutates_args=["x"]
)
def mutable_cache_update_and_fetch(
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
    layer_idx: int,
    seq_dim: int,
    seq_len: int | None,
) -> Tensor:
    """Write ``update`` into ``x[begin:end]``, then fetch one layer's prefix.

    ``x`` is the 5D cache, ``update`` the 4D per-layer K or V. Returns
    ``x[layer_idx, ..., :seq_len, :]`` (layer dim squeezed).
    """
    slices = tuple(slice(int(b), int(e)) for b, e in zip(begin.tolist(), end.tolist()))
    x[slices] = update.unsqueeze(0)
    fetched = x.narrow(0, layer_idx, 1)
    if seq_len is not None:
        fetched = fetched.narrow(seq_dim, 0, seq_len)
    # Clone: the fetched slice aliases the mutated cache.
    return fetched.squeeze(0).clone()


@mutable_cache_update_and_fetch.register_fake
def _(
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
    layer_idx: int,
    seq_dim: int,
    seq_len: int | None,
) -> Tensor:
    out_shape = list(x.shape)
    if seq_len is not None:
        out_shape[seq_dim] = seq_len
    out_shape.pop(0)
    return x.new_empty(out_shape)


@torch.library.custom_op("workflow_test::immutable_slice_update", mutates_args=[])
def immutable_slice_update(
    x: Tensor, update: Tensor, begin: Tensor, end: Tensor
) -> Tensor:
    """Out-of-place ``x[begin:end] = update``; lowered to ``coreai.slice_update``."""
    result = x.clone()
    slices = tuple(slice(int(b), int(e)) for b, e in zip(begin.tolist(), end.tolist()))
    result[slices] = update
    return result


@immutable_slice_update.register_fake
def _(x: Tensor, update: Tensor, begin: Tensor, end: Tensor) -> Tensor:
    return torch.empty_like(x)


class KVCache:
    """macOS-layout KV cache: ``(layers, 1, kv_heads, max_seq, head_dim)``."""

    def __init__(self, k_cache: Tensor, v_cache: Tensor) -> None:
        self._k_cache = k_cache
        self._v_cache = v_cache

    @staticmethod
    def create_cache_tensors(
        dtype: torch.dtype, seq_len: int = MAX_CONTEXT_LENGTH
    ) -> tuple[Tensor, Tensor]:
        shape = (NUM_LAYERS, 1, NUM_KV_HEADS, seq_len, HEAD_DIM)
        return torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)

    def _update_and_fetch_one(
        self, cache: Tensor, update: Tensor, layer_idx: int, offset: int, seq_len: int
    ) -> Tensor:
        def idx(v: int) -> Tensor:
            return torch.tensor((v,), dtype=torch.int32)

        begin = torch.cat([idx(layer_idx), idx(0), idx(0), idx(offset), idx(0)])
        end = torch.cat(
            [
                idx(layer_idx + 1),
                idx(cache.size(1)),
                idx(cache.size(2)),
                idx(offset + update.size(-2)),
                idx(cache.size(4)),
            ]
        )
        return mutable_cache_update_and_fetch(
            x=cache,
            update=update,
            begin=begin,
            end=end,
            layer_idx=layer_idx,
            seq_dim=-2,
            seq_len=seq_len,
        )

    def update_and_fetch(
        self,
        layer_idx: int,
        offset: int,
        k: Tensor,
        v: Tensor,
        seq_len: int,
        query_len: int,
    ) -> tuple[Tensor, Tensor]:
        torch._check(query_len >= 0)
        torch._check(query_len <= self._k_cache.size(-2))
        torch._check(offset >= 0)
        torch._check(offset < self._k_cache.size(-2))
        torch._check(seq_len >= 0)
        torch._check(seq_len <= self._k_cache.size(-2))
        k_out = self._update_and_fetch_one(self._k_cache, k, layer_idx, offset, seq_len)
        v_out = self._update_and_fetch_one(self._v_cache, v, layer_idx, offset, seq_len)
        return k_out, v_out


# ===========================================================================
# Model (mirrors coreai_models.models.macos.qwen3)
# ===========================================================================


class Attention(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.qkv_proj = nn.Linear(
            HIDDEN_SIZE, (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, bias=False
        )
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        # Query and key heads share one fused RMSNorm with per-head weights.
        self.qk_norm = coreai_torch.composite_ops.RMSNorm(
            HEAD_DIM, eps=RMS_NORM_EPS, n_heads=NUM_HEADS + NUM_KV_HEADS
        )
        self.rope = coreai_torch.composite_ops.RoPE(base=ROPE_THETA)
        self.sdpa = coreai_torch.composite_ops.SDPA(is_causal=True)

    def forward(self, x: Tensor, position_ids: Tensor, cache: KVCache) -> Tensor:
        batch_size, query_len, _ = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(batch_size, query_len, NUM_HEADS + 2 * NUM_KV_HEADS, HEAD_DIM)
            .permute(0, 2, 1, 3)
        )
        query_key = self.qk_norm(qkv.narrow(1, 0, NUM_HEADS + NUM_KV_HEADS))
        value = qkv.narrow(1, NUM_HEADS + NUM_KV_HEADS, NUM_KV_HEADS)

        seq_len = position_ids.shape[-1]
        torch._check(query_len >= 0)
        torch._check(seq_len >= 0)
        offset = seq_len - query_len
        torch._check(offset >= 0)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        query_key = self.rope(query_key, position_ids=rope_positions)
        query = query_key.narrow(1, 0, NUM_HEADS)
        key = query_key.narrow(1, NUM_HEADS, NUM_KV_HEADS)

        key, value = cache.update_and_fetch(
            self.layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
        )
        output = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, NUM_HEADS * HEAD_DIM)
        )
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.up_proj(x) * nn.functional.silu(self.gate_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(layer_idx)
        self.mlp = MLP()
        self.input_layernorm = coreai_torch.composite_ops.RMSNorm(
            HIDDEN_SIZE, eps=RMS_NORM_EPS
        )
        self.post_attention_layernorm = coreai_torch.composite_ops.RMSNorm(
            HIDDEN_SIZE, eps=RMS_NORM_EPS
        )

    def forward(self, x: Tensor, position_ids: Tensor, cache: KVCache) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), position_ids, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class TinyCausalLM(nn.Module):
    """``forward(input_ids, position_ids, k_cache, v_cache) -> logits``.

    In prefill mode ``forward`` returns ``()``, so the LM head drops out of the
    traced ``prefill`` graph and only the cache writes remain.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = nn.ModuleList([TransformerBlock(i) for i in range(NUM_LAYERS)])
        self.norm = coreai_torch.composite_ops.RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        self.prefill_mode = False
        # Non-unit norm weights so RMSNorm scaling is actually exercised.
        for name, param in self.named_parameters():
            if name.endswith("norm.weight"):
                with torch.no_grad():
                    param.uniform_(0.5, 1.5)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
    ) -> Tensor | tuple[()]:
        cache = KVCache(k_cache, v_cache)
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        if self.prefill_mode:
            # Empty tuple, not None: a bare return leaves a `None` leaf in the trace.
            return ()
        return self.lm_head(self.norm(h))


# ===========================================================================
# Export pipeline (mirrors coreai_models.export.macos / export.mlir_ops)
# ===========================================================================


def _autofunc_base_tensor(node: fx.Node) -> Any:
    if isinstance(node.target, AutoFunctionalizedV2):
        return node.kwargs["_all_bases"][node.kwargs["_x_base_index"]]
    return node.kwargs["x"]


def remove_functionalization(program: torch.export.ExportedProgram) -> None:
    """Rewrite auto-functionalized cache updates into immutable ops in place.

    Core AI can't lower the ``auto_functionalized`` higher-order op, so each
    ``mutable_cache_update_and_fetch`` becomes ``unsqueeze`` ->
    ``immutable_slice_update`` (the new 5D cache, getitem 1) -> layer slice ->
    seq slice -> ``squeeze`` (the fetched 4D K/V, getitem 0).
    """
    graph = program.graph_module.graph
    autofuncs = [
        n
        for n in graph.nodes
        if isinstance(n.target, (AutoFunctionalized, AutoFunctionalizedV2))
    ]
    replacements: dict[str, fx.Node] = {}
    for node in autofuncs:
        op_name = node.args[0].name()
        assert op_name == "workflow_test::mutable_cache_update_and_fetch", op_name
        getitems = {u.args[1]: u for u in node.users if u.target is operator.getitem}
        assert set(node.users) == set(getitems.values())
        fetched_getitem, cache_getitem = getitems.get(0), getitems[1]
        update = node.kwargs["update"]
        layer_idx = node.kwargs["layer_idx"]
        seq_dim = node.kwargs["seq_dim"]
        seq_len = node.kwargs["seq_len"]

        with graph.inserting_before(node):
            unsqueeze = graph.call_function(
                torch.ops.aten.unsqueeze.default, args=(update, 0)
            )
            unsqueeze.meta["val"] = update.meta["val"].unsqueeze(0)
            new_cache = graph.call_function(
                torch.ops.workflow_test.immutable_slice_update.default,
                args=(
                    _autofunc_base_tensor(node),
                    unsqueeze,
                    node.kwargs["begin"],
                    node.kwargs["end"],
                ),
            )
            new_cache.meta["val"] = cache_getitem.meta["val"]
            fetched = graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(new_cache, 0, layer_idx, layer_idx + 1),
            )
            fetched.meta["val"] = new_cache.meta["val"].narrow(0, layer_idx, 1)
            if seq_len is not None:
                seq_len_val = (
                    seq_len.meta["val"] if isinstance(seq_len, fx.Node) else seq_len
                )
                narrowed = graph.call_function(
                    torch.ops.aten.slice.Tensor, args=(fetched, seq_dim, 0, seq_len)
                )
                narrowed.meta["val"] = fetched.meta["val"].narrow(
                    seq_dim, 0, seq_len_val
                )
                fetched = narrowed
            squeeze = graph.call_function(
                torch.ops.aten.squeeze.dims, args=(fetched, [0])
            )
            squeeze.meta["val"] = fetched.meta["val"].squeeze(0)

        for new_node in (unsqueeze, new_cache, fetched, squeeze):
            new_node.meta["nn_module_stack"] = node.meta.get("nn_module_stack", {})
            new_node.meta["stack_trace"] = node.meta.get("stack_trace", "")

        cache_getitem.replace_all_uses_with(new_cache)
        replacements[cache_getitem.name] = new_cache
        graph.erase_node(cache_getitem)
        if fetched_getitem is not None:
            fetched_getitem.replace_all_uses_with(squeeze)
            replacements[fetched_getitem.name] = squeeze
            graph.erase_node(fetched_getitem)
        graph.erase_node(node)

    for spec in program.graph_signature.output_specs:
        if spec.arg.name in replacements:
            spec.arg.name = replacements[spec.arg.name].name
    program.graph_module.recompile()


def _lower_immutable_slice_update(values_map, node, location):  # type: ignore[no-untyped-def]
    x, update, begin, end = (values_map[arg.name] for arg in node.args)
    return coreai.slice_update(x, begin, end, [1] * x.type.rank, update)


def build_reference_inputs(dtype: torch.dtype) -> dict[str, Tensor]:
    """Trace inputs, in ``forward`` signature order (coreai-models ``TraceSpec``)."""
    input_ids = torch.randint(1, VOCAB_SIZE, (1, TRACE_QUERY_LEN), dtype=torch.int32)
    position_ids = torch.arange(
        TRACE_QUERY_LEN + TRACE_OFFSET, dtype=torch.int32
    ).unsqueeze(0)
    k_cache, v_cache = KVCache.create_cache_tensors(dtype)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "k_cache": k_cache,
        "v_cache": v_cache,
    }


def build_dynamic_shapes() -> dict[str, Any]:
    """Dynamic query / position lengths. The caches are traced at the full
    context, so (as in coreai-models) they are static.
    """
    return {
        "input_ids": {1: torch.export.Dim("seq_ids", max=MAX_CONTEXT_LENGTH - 2)},
        "position_ids": {
            1: torch.export.Dim(
                "seq_pos", min=TRACE_QUERY_LEN, max=MAX_CONTEXT_LENGTH - 1
            )
        },
        "k_cache": None,
        "v_cache": None,
    }


def export_macos_style(
    model: TinyCausalLM, dtype: torch.dtype, *, export_prefill_graph: bool = True
) -> AIProgram:
    """``torch.export`` -> decompose -> defunctionalize -> ``TorchConverter``."""
    reference_inputs = build_reference_inputs(dtype)
    dynamic_shapes = build_dynamic_shapes()

    def make_export_fn(prefill: bool):  # type: ignore[no-untyped-def]
        def export_fn(module: nn.Module) -> torch.export.ExportedProgram:
            # Set here, not around staging: the converter re-exports the module
            # during externalization, after both entrypoints are staged.
            module.prefill_mode = prefill
            with torch.no_grad():
                program = torch.export.export(
                    module,
                    args=(),
                    kwargs=reference_inputs,
                    dynamic_shapes=dynamic_shapes,
                )
            program = program.run_decompositions(coreai_torch.get_decomp_table())
            remove_functionalization(program)
            return program

        return export_fn

    model.eval()
    converter = TorchConverter()
    converter.register_torch_lowering("workflow_test::immutable_slice_update.default")(
        _lower_immutable_slice_update
    )
    converter.add_pytorch_module(
        model,
        export_fn=make_export_fn(prefill=False),
        externalize_modules=EXTERNALIZE_SPECS,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        state_names=STATE_NAMES,
        entrypoint_name=MAIN_GRAPH_NAME,
    )
    if export_prefill_graph:
        converter.add_pytorch_module(
            model,
            export_fn=make_export_fn(prefill=True),
            externalize_modules=EXTERNALIZE_SPECS,
            input_names=INPUT_NAMES,
            output_names=(),
            state_names=STATE_NAMES,
            entrypoint_name=PREFILL_GRAPH_NAME,
        )
    try:
        return converter.to_coreai()
    finally:
        model.prefill_mode = False


def _make_model(dtype: torch.dtype) -> TinyCausalLM:
    torch.manual_seed(0)
    return TinyCausalLM().to(dtype).eval()


# ===========================================================================
# Tests
# ===========================================================================


class TestMacOSStyleModelIR:
    """Structure of the exported program: entrypoints, state, composites."""

    @pytest.mark.ir
    def test_entrypoints_state_and_composites(self) -> None:
        """``main`` and ``prefill`` bind the KV caches as handles, update them
        with ``slice_update`` + ``write_handle`` per layer, and call the
        externalized RMSNorm / RoPE / SDPA composites. ``prefill`` declares no
        outputs, so the last layer's attention and the LM head are dropped.
        """
        program = export_macos_style(_make_model(torch.float16), torch.float16)
        filecheck_pattern(
            str(program),
            check_file="""
                // CHECK-DAG: coreai.graph private noinline @[[QK_NORM:layers\\.0\\.self_attn\\.qk_norm\\.rmsnorm_impl_[0-9a-f]+]](%{{.*}}: tensor<1x6x?x8xf16> {coreai.name = "input"}, %{{.*}}: tensor<6x1x8xf16> {coreai.name = "scale"}) {{.*}}composite_decl = #coreai.composite_declaration<"rms_norm"
                // CHECK-DAG: coreai.graph private noinline @[[ROPE:layers\\.0\\.self_attn\\.rope_[0-9a-f]+]](%{{.*}}: tensor<1x6x?x8xf16> {coreai.name = "input"}, %{{.*}}: tensor<1x?xsi32> {coreai.name = "position_ids"}) {{.*}}composite_decl = #coreai.composite_declaration<"rope"
                // CHECK-DAG: coreai.graph private noinline @[[SDPA:layers\\.0\\.self_attn\\.sdpa_[0-9a-f]+]](%{{.*}}: tensor<1x4x?x8xf16> {coreai.name = "query"}, %{{.*}}: tensor<1x2x?x8xf16> {coreai.name = "key"}, %{{.*}}: tensor<1x2x?x8xf16> {coreai.name = "value"}) {{.*}}composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention" = {{.*}}is_causal = true
                // CHECK-DAG: coreai.graph private noinline @[[NORM:layers\\.0\\.input_layernorm\\.rmsnorm_impl_[0-9a-f]+]](%{{.*}}: tensor<1x?x32xf16> {coreai.name = "input"}, %{{.*}}: tensor<32xf16> {coreai.name = "scale"}) {{.*}}composite_decl = #coreai.composite_declaration<"rms_norm"

                // CHECK-LABEL: coreai.graph @main(
                // CHECK-SAME: %[[IDS:.*]]: tensor<1x?xsi32> {coreai.name = "input_ids"}, %{{.*}}: tensor<1x?xsi32> {coreai.name = "position_ids"}
                // CHECK-SAME: %[[K:.*]]: !coreai.handle<tensor<2x1x2x64x8xf16>> {MutableBuffers.buffer_mutation = "keyCache", coreai.name = "keyCache"}
                // CHECK-SAME: %[[V:.*]]: !coreai.handle<tensor<2x1x2x64x8xf16>> {MutableBuffers.buffer_mutation = "valueCache", coreai.name = "valueCache"}
                // CHECK-SAME: -> (!coreai.token {coreai.name = "keyCache"}, !coreai.token {coreai.name = "valueCache"}, tensor<1x?x64xf16> {coreai.name = "logits"})
                // CHECK-DAG: coreai.read_handle %[[K]]
                // CHECK-DAG: coreai.read_handle %[[V]]
                // CHECK: coreai.gather_nd %{{.*}} at %{{.*}} : (tensor<64x32xf16>, tensor<1x?x1xsi32>) to tensor<1x?x32xf16>
                // Layer 0
                // CHECK: coreai.invoke @[[NORM]](
                // CHECK: coreai.invoke @[[QK_NORM]](
                // CHECK: coreai.invoke @[[ROPE]](
                // CHECK: coreai.slice_update %{{.*}} with %{{.*}} : (tensor<2x1x2x64x8xf16>, tensor<1x1x2x?x8xf16>, tensor<5xsi32>, tensor<5xsi32>, tensor<5xsi32>) to tensor<2x1x2x64x8xf16>
                // CHECK: coreai.write_handle %[[K]]
                // CHECK: coreai.slice_update
                // CHECK: coreai.write_handle %[[V]]
                // CHECK: coreai.invoke @[[SDPA]](
                // CHECK: coreai.invoke @[[NORM]](
                // Layer 1
                // CHECK: coreai.invoke @[[NORM]](
                // CHECK: coreai.invoke @[[QK_NORM]](
                // CHECK: coreai.invoke @[[ROPE]](
                // CHECK: coreai.slice_update
                // CHECK: coreai.write_handle %[[K]]
                // CHECK: coreai.slice_update
                // CHECK: coreai.write_handle %[[V]]
                // CHECK: coreai.invoke @[[SDPA]](
                // CHECK: coreai.invoke @[[NORM]](
                // Final norm + LM head
                // CHECK: coreai.invoke @[[NORM]](
                // CHECK: %[[LOGITS:.*]] = coreai.decomposable.broadcasting_batch_matmul %{{.*}} : (tensor<1x?x32xf16>, tensor<32x64xf16>) -> tensor<1x?x64xf16>
                // CHECK: coreai.output %{{.*}}, %{{.*}}, %[[LOGITS]] : !coreai.token, !coreai.token, tensor<1x?x64xf16>

                // CHECK-LABEL: coreai.graph @prefill(
                // CHECK-SAME: -> (!coreai.token {coreai.name = "keyCache"}, !coreai.token {coreai.name = "valueCache"}) attributes
                // CHECK-COUNT-4: coreai.slice_update
                // CHECK-NOT: coreai.invoke
                // CHECK: coreai.output %{{.*}}, %{{.*}} : !coreai.token, !coreai.token
            """,
        )


def _generation_steps() -> list[LLMStep]:
    """Prefill, a multi-token chunk, then single-token decodes (fixed tokens)."""
    generator = torch.Generator().manual_seed(0)
    query_lens = [PROMPT_LEN, CHUNK_LEN] + [1] * NUM_DECODE_STEPS
    steps, seq_len = [], 0
    for idx, query_len in enumerate(query_lens):
        seq_len += query_len
        steps.append(
            LLMStep(
                entrypoint=PREFILL_GRAPH_NAME if idx == 0 else MAIN_GRAPH_NAME,
                inputs={
                    "input_ids": torch.randint(
                        1,
                        VOCAB_SIZE,
                        (1, query_len),
                        dtype=torch.int32,
                        generator=generator,
                    ),
                    "position_ids": torch.arange(seq_len, dtype=torch.int32).unsqueeze(
                        0
                    ),
                },
            )
        )
    return steps


class TestMacOSStyleModelNumerics:
    """Short generation loop on the runtime vs. eager, with a persistent KV cache."""

    @pytest.mark.parametrize(
        "dtype, atol, rtol",
        [
            (torch.float32, 1e-4, 1e-4),
            # fp16 is what macOS exports use; tolerances cover fp16 rounding.
            (torch.float16, 2e-2, 2e-2),
        ],
        ids=["fp32", "fp16"],
    )
    async def test_prefill_extend_decode_loop(
        self, dtype: torch.dtype, atol: float, rtol: float
    ) -> None:
        model = _make_model(dtype)
        program = export_macos_style(model, dtype)

        k_ref, v_ref = KVCache.create_cache_tensors(dtype)

        def reference_fn(step: LLMStep) -> dict[str, Tensor]:
            # The prefill graph is the prefill-mode trace: it only writes the cache.
            model.prefill_mode = step.entrypoint == PREFILL_GRAPH_NAME
            try:
                out = model(**step.inputs, k_cache=k_ref, v_cache=v_ref)
            finally:
                model.prefill_mode = False
            return {} if isinstance(out, tuple) else dict(zip(OUTPUT_NAMES, (out,)))

        await validate_llm_decode_loop(
            program,
            steps=_generation_steps(),
            reference_fn=reference_fn,
            reference_state={KEY_CACHE_NAME: k_ref, VALUE_CACHE_NAME: v_ref},
            atol=atol,
            rtol=rtol,
        )

        # Not vacuous: the generated positions were written, and nothing past them.
        seq_len = PROMPT_LEN + CHUNK_LEN + NUM_DECODE_STEPS
        assert torch.all(k_ref[..., :seq_len, :].abs().sum(-1) != 0)
        assert torch.all(k_ref[..., seq_len:, :] == 0)

    def test_eager_cached_decode_matches_full_forward(self) -> None:
        """Sanity-check the authoring itself: incremental decode through the KV
        cache reproduces a single full-sequence forward. Guards the numerical
        test above from passing on a reference that is consistently wrong.
        """
        model = _make_model(torch.float32)
        ids = torch.randint(
            1,
            VOCAB_SIZE,
            (1, PROMPT_LEN + 2),
            generator=torch.Generator().manual_seed(1),
        ).to(torch.int32)
        positions = torch.arange(PROMPT_LEN + 2, dtype=torch.int32).unsqueeze(0)

        with torch.no_grad():
            full = model(ids, positions, *KVCache.create_cache_tensors(torch.float32))
            k_cache, v_cache = KVCache.create_cache_tensors(torch.float32)
            steps = [
                model(ids[:, :PROMPT_LEN], positions[:, :PROMPT_LEN], k_cache, v_cache)
            ]
            for t in range(PROMPT_LEN, PROMPT_LEN + 2):
                steps.append(
                    model(ids[:, t : t + 1], positions[:, : t + 1], k_cache, v_cache)
                )

        torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-5)
