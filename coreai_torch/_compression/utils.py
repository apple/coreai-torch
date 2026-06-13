# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Compression related utils."""
# ruff: noqa: D205, D401, EM101, EM102, G004, TRY004, D103

import logging
import math
from typing import Any, cast

import torch
from torch.export.exported_program import ExportedProgram

from coreai_torch._compression._intx import (  # type: ignore[attr-defined]
    IntxTensor,
    UintxTensor,
)

logger = logging.getLogger(__name__)

CHAR_BIT = 8  # number of bits in a byte
QUANTIZATION_SUPPORT_NBITS = (2, 4, 8)
PALETTIZATION_SUPPORT_NBITS = (1, 2, 3, 4, 6, 8)

# Sub-byte integer dtypes (int4, uint4, int2, uint2) exist in torch 2.7.0 but
# torch.iinfo does not support them. Map dtype → (is_signed, nbits) for manual
# bounds computation as a fallback.
_SUBBYTE_INT_DTYPES: dict[torch.dtype, tuple[bool, int]] = {}
for _name, _signed, _bits in [
    ("int4", True, 4),
    ("uint4", False, 4),
    ("int2", True, 2),
    ("uint2", False, 2),
]:
    if hasattr(torch, _name):
        _SUBBYTE_INT_DTYPES[getattr(torch, _name)] = (_signed, _bits)


def _int_dtype_min(dtype: torch.dtype) -> int:
    """Get the minimum representable value for an integer dtype, including sub-byte types."""
    try:
        return torch.iinfo(dtype).min
    except TypeError:
        # torch.iinfo does not support sub-byte dtypes in torch 2.7.0.
        signed, bits = _SUBBYTE_INT_DTYPES[dtype]
        return -(1 << (bits - 1)) if signed else 0


def _int_dtype_max(dtype: torch.dtype) -> int:
    """Get the maximum representable value for an integer dtype, including sub-byte types."""
    try:
        return torch.iinfo(dtype).max
    except TypeError:
        # torch.iinfo does not support sub-byte dtypes in torch 2.7.0.
        signed, bits = _SUBBYTE_INT_DTYPES[dtype]
        return (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1


def _int_dtype_bits(dtype: torch.dtype) -> int:
    """Get the bit width of an integer dtype, including sub-byte types."""
    try:
        return torch.iinfo(dtype).bits
    except TypeError:
        # torch.iinfo does not support sub-byte dtypes in torch 2.7.0.
        _, bits = _SUBBYTE_INT_DTYPES[dtype]
        return bits


def repeat_tensor_as(tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Repeat the first K dimensions of tensor to target_shape.

    len(target_shape) == K. It should be guaranteed that each dimension of
    the K dimensions is divisible by the corresponding dimension given by
    the target_shape.
    """
    if len(tensor.shape) < len(target_shape):
        msg = f"tensor rank {len(tensor.shape)} >= target rank {len(target_shape)}"
        raise ValueError(msg)
    repeated_tensor = tensor
    for axis, dim_size in enumerate(target_shape):
        block_num = tensor.shape[axis]
        if dim_size % block_num != 0:
            msg = (
                f"The dim size in each axis must be divisible by the tensor "
                f"dimension. Got invalid tensor shape {tensor.shape} for "
                f"target shape {target_shape} at axis {axis}."
            )
            raise ValueError(msg)
        block_size = dim_size // block_num
        # Can use kron for higher efficiency, but repeat is easier to understand.
        if block_size > 1:
            repeated_tensor = torch.repeat_interleave(
                repeated_tensor,
                block_size,
                dim=axis,
            )
    return repeated_tensor


def wrap_for_parametrization(
    compression_module_class: type[torch.nn.Module],
) -> type[torch.nn.Module]:
    """
    Create a wrapper for Core AI compression modules to make them compatible with PyTorch parametrizations.

    This wrapper adds a forward method that ignores its input parameter and calls the underlying module's
    forward method. This is necessary for using the compression module as a "parametrization" with
    torch.nn.utils.parametrize.register_parametrization, which passes the weight as input.

    Args:
        compression_module_class: The compression module class to be wrapped

    Returns:
        A new class that can be used as a PyTorch parametrization

    """

    class CompressionParametrization(compression_module_class):  # type: ignore[valid-type,misc]
        def forward(self, _: Any) -> Any:
            # Ignore the input parameter and call the original forward method
            return super().forward()

    # Update the class name for better debugging
    CompressionParametrization.__name__ = (
        f"{compression_module_class.__name__}Parametrization"
    )
    return CompressionParametrization


def _inject_subbyte_in_lut(program: ExportedProgram) -> ExportedProgram:
    """
    Inject the sub-byte info into lut op for indices.

    We need this workaround after torch.export since torch.export does not
    support subclass yet. We can get rid of this workaround after the support
    gets added & we can have uint4tensor natively in the pytorch model
    before torch.export.
    """
    name_to_input_spec: dict[str, torch.export.graph_signature.InputSpec] = {}
    for input_spec in program._graph_signature.input_specs:
        name_to_input_spec[input_spec.arg.name] = input_spec

    graph = program.graph
    for node in graph.nodes:
        if hasattr(node.target, "name") and node.target.name().endswith("lut_to_dense"):
            indices_args = [
                arg
                for arg in node.args
                if hasattr(arg, "name") and "indices" in arg.name
            ]
            lut_args = [
                arg for arg in node.args if hasattr(arg, "name") and "lut" in arg.name
            ]
            if len(indices_args) >= 1 or len(lut_args) >= 1:
                indices = indices_args[0]
                lut = lut_args[0]
                indices_name = indices.name
            elif node.args[0].name.startswith("ifp_constant") and node.args[
                1
            ].name.startswith("ifp_constant"):
                # we have some indirection of a ifp node hopefully
                lut, indices = node.args[0:2]
                indices_name = indices.args[0].name
            # The second last dimension or the last dimension of the lut tensor represents the number of palettes, which equals to 2**nbits, we get the
            # nbits from it. We update the annotation of uint8 in the indices
            # tensor to the corresponding torch data type, like torch.uint4
            # or torch.uint2 according to the inferred nbits.
            # LUT shape is (..., num_palettes, cluster_dim), so num_palettes
            # is always at shape[-2].
            num_palettes = lut.meta["val"].shape[-2]
            nbits = int(math.log2(num_palettes))

            if nbits not in PALETTIZATION_SUPPORT_NBITS:
                msg = f"{nbits}-bit palettization is not supported. Supported nbits: {PALETTIZATION_SUPPORT_NBITS}"
                raise RuntimeError(msg)
            if nbits == CHAR_BIT:
                continue

            # We also need to update the indices constant values from uint8
            # to uint4 by constructing our own UintxTensor tensor whose
            # `.elem` would contain the packed bytes.
            input_spec = name_to_input_spec[indices_name]
            if input_spec.target not in program.state_dict:
                msg = "input_spec's target cannot be found in the state_dict."
                raise ValueError(msg)
            program.state_dict[input_spec.target] = UintxTensor.from_unpacked(
                program.state_dict[input_spec.target],
                nbits,
            )

    graph.lint()
    return program


def _inject_subbyte_in_quant(program: ExportedProgram) -> ExportedProgram:  # noqa: C901
    """
    Inject the sub-byte info into quant op for quantized_data and offset (if exists).

    We need this workaround after torch.export since torch.export does not
    support subclass yet. We can get rid of this workaround after the support
    gets added & we can have int4tensor natively in the pytorch model
    before torch.export.
    """
    name_to_input_spec = {
        spec.arg.name: spec for spec in program._graph_signature.input_specs
    }

    def _convert_to_subbyte(subbyte_node: torch.fx.node.Node, nbits: int) -> None:
        """Convert a fx node to use subbyte dtype with specified nbits."""
        if subbyte_node.meta["val"].dtype not in (torch.int8, torch.uint8):
            return

        # We also pack the sub-byte values by using IntxTensor / UintxTensor where
        # the `.elem` field contains the packed bytes.
        input_spec = name_to_input_spec[subbyte_node.name]
        if input_spec.target not in program.state_dict:
            msg = f"input_spec's target ({input_spec.target}) cannot be found in the state_dict."
            raise ValueError(msg)
        is_signed = subbyte_node.meta["val"].is_signed()
        subbyte_class = IntxTensor if is_signed else UintxTensor
        program.state_dict[input_spec.target] = subbyte_class.from_unpacked(
            program.state_dict[input_spec.target],
            nbits,
        )

    graph = program.graph
    for node in graph.nodes:
        if (
            hasattr(node.target, "name")
            and node.target.name() == "coreai::constexpr_blockwise_shift_scale"
        ):
            # Get nbits info for the quantization op.
            quantized_data_node = node.args[0]
            if quantized_data_node.name not in name_to_input_spec:
                continue
            input_spec = name_to_input_spec[quantized_data_node.name]

            # If the model is produced by coremltools compression, there could be nbits info
            # stored in the same path but under `quantization_n_bits`.
            # For example, the `input_spec.target` is "linear1.weight.quantized_data", then
            # the corresponding nbits info is stored in "linear1.weight.quantization_n_bits"
            buffer_path_components = cast("str", input_spec.target).split(".")
            buffer_path_components[-1] = "quantization_n_bits"
            nbits_buffer_name = ".".join(buffer_path_components)

            nbits: int = CHAR_BIT
            # Prefer input_dtype arg (new API) — gives exact nbits directly.
            # torch.export places all parameters (including defaults) in
            # node.args, so input_dtype is at positional index 4.
            input_dtype = node.args[4] if len(node.args) > 4 else None  # noqa: PLR2004
            if input_dtype is not None:
                nbits = _int_dtype_bits(input_dtype)
            elif nbits_buffer_name in program.state_dict:
                nbits = int(program.state_dict[nbits_buffer_name])
            else:
                # When there is no explicit nbits info stored, we infer it by quantized data range.
                quantized_data: torch.Tensor = program.state_dict[input_spec.target]
                if torch.is_floating_point(quantized_data):
                    warning_msg = f'Cannot infer nbits: The quantized data "{input_spec.target}" in state_dict is not integer. Will just use it as-is.'
                    logger.warning(warning_msg)
                    continue
                signed = quantized_data.is_signed()
                for nbits_candidate in sorted(QUANTIZATION_SUPPORT_NBITS):
                    lower_bound = -(2 ** (nbits_candidate - 1)) if signed else 0
                    upper_bound = (
                        2 ** (nbits_candidate - 1) - 1
                        if signed
                        else 2**nbits_candidate - 1
                    )
                    if (
                        quantized_data.min() >= lower_bound
                        and quantized_data.max() <= upper_bound
                    ):
                        nbits = nbits_candidate
                        break

            if nbits == CHAR_BIT:
                continue
            if nbits not in QUANTIZATION_SUPPORT_NBITS:
                msg = f"{nbits}-bit quantization is not supported. Supported nbits: {QUANTIZATION_SUPPORT_NBITS}"
                raise RuntimeError(msg)

            _convert_to_subbyte(node.args[0], nbits)  # quantized_data
            if len(node.args) > 2 and node.args[2] is not None:  # noqa: PLR2004
                _convert_to_subbyte(node.args[2], nbits)  # zero_point

    # --- FP4 injection -------------------------------------------------------
    # Float4Tensor packs 2 fp4 values per uint8 byte, but torch.export loses
    # the subclass.  Detect the pattern: constexpr_blockwise_shift_scale with
    # output_dtype set and uint8 input whose output last-dim is 2× the input's.
    # Inject ``future_dtype`` so that _constant_from_tensor creates the correct
    # IR type (f4E2M1FN with logical shape).
    for node in graph.nodes:
        if not (
            hasattr(node.target, "name")
            and node.target.name() == "coreai::constexpr_blockwise_shift_scale"
        ):
            continue
        quantized_data_node = node.args[0]
        if quantized_data_node.name not in name_to_input_spec:
            continue
        # output_dtype is at arg index 5
        output_dtype = node.args[5] if len(node.args) > 5 else None  # noqa: PLR2004
        if output_dtype is None:
            continue
        input_meta = quantized_data_node.meta.get("val")
        output_meta = node.meta.get("val")
        if input_meta is None or output_meta is None:
            continue
        # FP4 signature: uint8 input, output last dim = 2 × input last dim
        if (
            input_meta.dtype == torch.uint8
            and len(input_meta.shape) == len(output_meta.shape)
            and output_meta.shape[-1] == 2 * input_meta.shape[-1]
        ):
            spec = name_to_input_spec[quantized_data_node.name]
            if spec.target in program.state_dict:
                program.state_dict[spec.target].future_dtype = torch.float4_e2m1fn_x2  # type: ignore[union-attr]

    graph.lint()

    return program


def _inject_subbyte_in_sparse(program: ExportedProgram) -> ExportedProgram:
    """Inject the sub-byte info (uint1) into sparsification op for mask."""
    name_to_input_spec = {
        spec.arg.name: spec for spec in program._graph_signature.input_specs
    }

    graph = program.graph
    for node in graph.nodes:
        if (
            hasattr(node.target, "name")
            and node.target.name() == "coreai::sparse_to_dense"
        ):
            mask = node.args[1]
            input_spec = name_to_input_spec[mask.name]
            if input_spec.target not in program.state_dict:
                msg = "input_spec's target cannot be found in the state_dict."
                raise ValueError(msg)
            program.state_dict[input_spec.target] = UintxTensor.from_unpacked(
                program.state_dict[input_spec.target],
                nbits=1,
            )

    graph.lint()
    return program


def inject_subbyte_tensors(program: ExportedProgram) -> ExportedProgram:
    """Upgrade uint8 weight constants to their proper sub-byte representations.

    After ``torch.export`` the state_dict holds all compressed weights as plain
    ``uint8`` tensors because PyTorch's tracing infrastructure does not yet
    support tensor sub-classes end-to-end.  This function re-runs the three
    injection passes that promote those uint8 tensors to the correct sub-byte
    types so that the converter can emit the right element types:

    - ``_inject_subbyte_in_lut``  → ``UintxTensor`` for LUT indices
    - ``_inject_subbyte_in_quant`` → ``IntxTensor`` / ``UintxTensor`` for blockwise-shift-scale data
    - ``_inject_subbyte_in_sparse`` → ``UintxTensor(nbits=1)`` for sparse masks

    Call this on the ``ExportedProgram`` *before* passing it to
    ``TorchConverter``.
    """
    _inject_subbyte_in_lut(program)
    _inject_subbyte_in_quant(program)
    _inject_subbyte_in_sparse(program)
    return program
