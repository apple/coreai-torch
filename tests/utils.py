# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import platform
import shutil
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from coreai._compiler._transforms import GlobalOptions, PassEntry, apply_passes
from coreai._compiler._transforms.passes import CorePasses
from coreai.authoring import AIProgram
from coreai.runtime import NDArray, StorageKind
from filecheck.matcher import Matcher
from filecheck.options import Options
from torch import Tensor

from coreai_torch import TorchConverter
from coreai_torch._utils import print_graph

from .conftest import dump_optests_enabled, get_current_test_id

if platform.system() == "Darwin":
    import mlx  # type: ignore[import-not-found, unused-ignore]
    import mlx.core  # type: ignore[import-not-found, unused-ignore]
    from coreai.runtime import (  # type: ignore[attr-defined, unused-ignore]
        ComputeUnitKind,
        SpecializationOptions,
    )

_ML_ASSET_EXTENSION = "aimodel"

# Compute unit selection driven by the --compute-unit-kind pytest option (see tests/conftest.py).
# Default is "interpreter" so a plain `pytest` run still works.
_COMPUTE_UNIT_KIND: str = "interpreter"


def set_test_compute_unit_kind(name: str) -> None:
    """Set the compute unit used by validate_numerical_output.

    Called from tests/conftest.py::pytest_configure based on --compute-unit-kind.
    """
    global _COMPUTE_UNIT_KIND
    _COMPUTE_UNIT_KIND = name


def _get_test_specialization_options() -> "SpecializationOptions | None":
    """Translate the configured compute unit into SpecializationOptions (or None).

    On non-macOS platforms only ``interpreter`` is supported — the runtime
    does not expose ``SpecializationOptions`` outside Darwin.
    """
    if _COMPUTE_UNIT_KIND == "interpreter":
        return None
    if platform.system() != "Darwin":
        msg = (
            f"--compute-unit-kind={_COMPUTE_UNIT_KIND} is only supported on macOS; "
            "use --compute-unit-kind=interpreter on this platform."
        )
        raise RuntimeError(msg)
    if _COMPUTE_UNIT_KIND == "cpu":
        return SpecializationOptions.cpu_only()
    if _COMPUTE_UNIT_KIND == "gpu":
        return SpecializationOptions.from_preferred_compute_unit_kind(
            compute_unit_kind=ComputeUnitKind.gpu(),
        )
    if _COMPUTE_UNIT_KIND == "neural_engine":
        return SpecializationOptions.from_preferred_compute_unit_kind(
            compute_unit_kind=ComputeUnitKind.neural_engine(),
        )
    msg = f"Unknown compute unit kind: {_COMPUTE_UNIT_KIND!r}"
    raise ValueError(msg)


async def run_transforms(coreai_program: AIProgram) -> None:
    """Run essential transformation passes."""
    await apply_passes(
        coreai_program._mlir_module,
        passes=[
            PassEntry.get(CorePasses._CORE_OPTIMIZE),
            PassEntry.get(CorePasses._UPDATE_SIGNATURE_TO_HANDLES),
            PassEntry.get(CorePasses._PROPAGATE_HANDLE_UPDATES),
        ],
        options=GlobalOptions(Path()),
    )


def make_dynamic_shapes(
    **arg_specs: list[str | None] | dict[int, str | None],
) -> dict[str, dict[int, torch.export.Dim]]:
    """Build a dynamic_shapes dict with automatic Dim sharing.

    Pass each model argument as a keyword, mapped to either:
      - a list of dim names (index = position), or
      - a dict of {dim_index: dim_name}

    Using the same string name for two positions in different tensors
    produces the *same* Dim object, expressing a shared/constrained
    dimension (e.g. a shared batch or inner-contraction axis).
    Use None to leave a dimension static.

    Examples::

        # Single tensor — all dims independent
        make_dynamic_shapes(x=["batch", "seq", "feat"])

        # Two tensors sharing batch (dim 0) and inner K dim (bmm)
        make_dynamic_shapes(
            mat1=["batch", "M", "K"],
            mat2=["batch", "K", "N"],
        )

        # Sparse: only some dims dynamic
        make_dynamic_shapes(x={0: "batch"}, y={0: "batch"})

        # Mixed: None keeps a dimension static (dim 1 fixed, rest dynamic)
        make_dynamic_shapes(x=["batch", None, "h", "w"])
    """
    named_dims: dict[str, torch.export.Dim] = {}
    result: dict[str, dict[int, torch.export.Dim]] = {}
    for arg_name, spec in arg_specs.items():
        if isinstance(spec, list):
            spec = {i: name for i, name in enumerate(spec) if name is not None}
        arg_dims: dict[int, torch.export.Dim] = {}
        for dim_idx, dim_name in spec.items():
            if dim_name is None:
                continue
            if dim_name not in named_dims:
                named_dims[dim_name] = torch.export.Dim(dim_name, min=1)
            arg_dims[dim_idx] = named_dims[dim_name]
        result[arg_name] = arg_dims
    return result


def _all_dims_dynamic(
    t: torch.Tensor, prefix: str = "d"
) -> dict[int, torch.export.Dim]:
    """Return a dict mapping every dimension of t to a fresh symbolic Dim."""
    return {i: torch.export.Dim(f"{prefix}{i}", min=1) for i in range(t.dim())}


class TemporaryModelAsset(TemporaryDirectory[str]):
    """Create and return a temporary asset package."""

    def __init__(self) -> None:
        """Initialize a temporary model asset directory."""
        super().__init__(suffix=f".{_ML_ASSET_EXTENSION}")


def compare_outputs(
    expected_outputs: dict[str, torch.Tensor | tuple[torch.Tensor]],
    actual_outputs: dict[str, npt.NDArray[np.number[Any]]],
) -> bool:
    """Compare the expected outputs with the actual outputs."""
    for output_name, expected in expected_outputs.items():
        actual = actual_outputs[output_name]
        assert isinstance(expected, torch.Tensor)
        assert isinstance(actual, np.ndarray)
        if not np.allclose(
            expected.detach().numpy().flatten(),
            actual.flatten(),
            # FP16 accuracy is flaky because of random
            # data sometimes causes the result error to be
            # larger.
            # TODO: Update the tests if needed.
            atol=1e-2,
        ):
            print("Torch expected:")  # noqa: T201
            print(expected.detach().numpy().flatten())  # noqa: T201
            print()  # noqa: T201
            print("Core AI actual:")  # noqa: T201
            print(actual.flatten())  # noqa: T201
            print()  # noqa: T201
            return False
    return True


def _narrow_dtype(t: Tensor) -> Tensor:
    """Narrow 64-bit tensors to 32-bit (Core AI has no 64-bit support)."""
    if t.dtype == torch.int64:
        return t.to(torch.int32)
    if t.dtype == torch.float64:
        return t.to(torch.float32)
    return t


def _init_runtime_state(
    desc: Any, sig: Any, sample_inputs: dict[str, Tensor]
) -> tuple[dict[str, NDArray], list[int]]:
    """Allocate runtime state and initialize user-input-mutation states.

    Assumes desc.state_names follows "buffers first, then mutated user inputs"
    ordering — the same invariant asserted in _resolve_io_names.
    """
    state: dict[str, NDArray] = {}
    for name in desc.state_names:
        # `NDArray.from_descriptor` only sizes the buffer; on Linux the backing
        # storage isn't zeroed, so buffer-state reads return garbage on the
        # first call. Allocate a zero-filled numpy array of the right shape and
        # dtype instead so initial state matches the model's `register_buffer`
        # value (assumed zero — same assumption tests already make).
        d = desc.state_descriptor(name=name)
        shape = tuple(s if s is not None else 1 for s in d.shape)
        state[name] = NDArray(np.zeros(shape, dtype=np.dtype(d.dtype)))

    user_mut_names = list(sig.user_inputs_to_mutate.values())
    num_buf_muts = len(sig.buffers_to_mutate)
    mutated_arg_indices: list[int] = []
    user_mut_idx = 0
    for i, arg_name in enumerate(sample_inputs.keys()):
        if arg_name in user_mut_names:
            mutated_arg_indices.append(i)
            if desc.state_names:
                sn = desc.state_names[num_buf_muts + user_mut_idx]
                state[sn] = NDArray(sample_inputs[arg_name].clone().numpy())
            user_mut_idx += 1

    return state, mutated_arg_indices


def _build_rt_inputs(
    desc: Any, sig: Any, sample_inputs: dict[str, Tensor], *, metal_inputs: bool = False
) -> dict[str, NDArray]:
    """Map sample inputs to runtime input names (excludes state inputs)."""
    input_key_order = [k for k in sample_inputs.keys() if k in desc.input_names]
    if not input_key_order:
        mutated = set(sig.user_inputs_to_mutate.values())
        input_key_order = [k for k in sample_inputs.keys() if k not in mutated]

    assert len(desc.input_names) == len(input_key_order), (
        f"Runtime expects {len(desc.input_names)} inputs ({desc.input_names}) "
        f"but got {len(input_key_order)} from sample_inputs ({input_key_order})"
    )

    if metal_inputs:
        return {
            desc_name: NDArray(
                data=_narrow_dtype(sample_inputs[key]), backing=StorageKind.METAL
            )
            for desc_name, key in zip(desc.input_names, input_key_order)
        }
    return {
        desc_name: NDArray(_narrow_dtype(sample_inputs[key]))
        for desc_name, key in zip(desc.input_names, input_key_order)
    }


def _export_and_convert(
    model: torch.nn.Module,
    kwargs: dict[str, Any],
    *,
    dynamic_shapes: Any,
    remove_decomps: list | None,
    prepare_program: Any,
    print_exported_graph: bool,
    state_names: list[str] | None,
    input_names: list[str] | None,
    output_names: list[str] | None,
    run_optimize_passes: bool,
    custom_kernels: list | None = None,
) -> tuple[Any, Any, list[str]]:
    """Export an nn.Module, run decompositions, and convert to a Core AI program.

    Returns ``(coreai_program, exported_program, fx_output_names)``. Optimizer
    passes run when the model has buffer/user-input mutations, the caller
    explicitly requests it, or state_names is supplied.
    """
    model.eval()
    exported_program = torch.export.export(
        model, args=(), kwargs=kwargs, dynamic_shapes=dynamic_shapes
    )
    decomp_table = torch.export.default_decompositions()
    if remove_decomps is not None:
        for decomp in remove_decomps:
            decomp_table.pop(decomp)
    exported_program = exported_program.run_decompositions(decomp_table)
    if prepare_program is not None:
        exported_program = prepare_program(exported_program)
    if print_exported_graph:
        print_graph(exported_program)

    converter = TorchConverter()
    if custom_kernels:
        converter.register_custom_kernels(custom_kernels)
    converter.add_exported_program(
        exported_program,
        state_names=state_names,
        input_names=input_names,
        output_names=output_names,
    )
    coreai_program = converter.to_coreai()

    sig = exported_program.graph_signature
    has_state = bool(sig.buffers_to_mutate) or bool(sig.user_inputs_to_mutate)
    if run_optimize_passes or state_names or has_state:
        coreai_program.optimize()

    output_node = next(
        n for n in exported_program.graph_module.graph.nodes if n.op == "output"
    )
    fx_output_names = [n.name for n in output_node.all_input_nodes]
    return coreai_program, exported_program, fx_output_names


def _compare_sorted(
    coreai_outs: dict[str, Any],
    torch_out: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    rtol: float,
    atol: float,
) -> None:
    """Path B: compare runtime outputs against torch outputs by sorted key."""
    coreai_outs_np = [v.numpy() for _, v in sorted(coreai_outs.items())]
    torch_outs = torch_out if isinstance(torch_out, tuple) else (torch_out,)
    for expected, actual in zip(torch_outs, coreai_outs_np, strict=False):
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(expected),
            actual,  # type: ignore[arg-type]
            rtol=rtol,
            atol=atol,
        )


def _compare_by_name(
    rt_outputs: dict[str, Any],
    torch_out: tuple[torch.Tensor, ...],
    *,
    names: list[str] | None,
    fx_output_names: list[str],
    rtol: float,
    atol: float,
    call_idx: int,
) -> None:
    """Path A: compare runtime outputs against torch outputs by name.

    If ``names`` is provided, every name must exist in ``rt_outputs``. Otherwise
    match by FX node name; missing names are silently skipped — state mutation
    outputs become tokens after optimize and won't appear here.
    """
    if names:
        for i, name in enumerate(names):
            assert name in rt_outputs, (
                f"Expected output '{name}' not found in runtime outputs "
                f"(available: {list(rt_outputs.keys())})"
            )
            np.testing.assert_allclose(
                rt_outputs[name].numpy(),
                _torch_tensor_to_numpy_array(torch_out[i]),
                atol=atol,
                rtol=rtol,
                err_msg=f"Output '{name}' mismatch on call {call_idx + 1}",
            )
        return

    for i, fx_name in enumerate(fx_output_names):
        if fx_name in rt_outputs and i < len(torch_out):
            np.testing.assert_allclose(
                rt_outputs[fx_name].numpy(),
                _torch_tensor_to_numpy_array(torch_out[i]),
                atol=atol,
                rtol=rtol,
                err_msg=f"Output '{fx_name}' mismatch on call {call_idx + 1}",
            )


def _optest_dump_path(test_id: str) -> Path:
    dump_path_str, test = test_id.removeprefix("tests/").split(".py", maxsplit=1)
    test = (
        test.removeprefix("::")
        .replace("::", "_")
        .replace("[", "-params-")
        .replace("]", "")
    )
    return Path(f"op_tests/{dump_path_str}") / test


async def _execute_and_compare(
    rt_func: Any,
    inputs: dict[str, NDArray],
    state: dict[str, NDArray],
    *,
    num_calls: int,
    produce_torch_out: Any,
    compare: Any,
    coreai_program: Any = None,
    dump_path: Path | None = None,
) -> None:
    """The single place we invoke the Core AI runtime.

    Per iteration: build expected torch output, call ``rt_func``, compare. The
    two callbacks let path A (stateful, recompute per call) and path B
    (fixed expected output) share this loop without ad-hoc branching.
    """
    # Stateful tests can't round-trip for now,
    # which has no `--input` JSON path for state — so skip the dump entirely.
    should_dump = dump_optests_enabled() and dump_path is not None and not state
    try:
        for call_idx in range(num_calls):
            torch_out = produce_torch_out(call_idx)
            rt_outputs = await rt_func(inputs=inputs, state=state)
            compare(rt_outputs, torch_out, call_idx)

            if call_idx == 0 and should_dump:
                _dump_optest_artifacts(coreai_program, inputs, rt_outputs, dump_path)

    except Exception:
        # Wipe bytecode and reference IO if the test failed as
        # the dumps cannot be considered valid if the test does not pass
        if dump_path is not None:
            shutil.rmtree(dump_path, ignore_errors=True)
        raise


def _add_npz_entry(io_numpy: dict[str, np.ndarray], key: str, arr: np.ndarray) -> None:
    """Add an array to the npz dict, emitting a bf16 dtype override if needed.

    NumPy has no native bf16, so surfaces it as void16 (``|V2``).
    """
    io_numpy[key] = arr
    if arr.dtype.str == "|V2":
        io_numpy[f"_dtype_{key}"] = np.array("bf16")


def _dump_optest_artifacts(
    coreai_program: Any,
    inputs: dict[str, NDArray],
    rt_outputs: dict[str, NDArray],
    dump_path: Path,
) -> None:
    """Write a `<testname>.aimodel` + `<testname>_test_data.npz` pair.

    Format: aimodel prefix == npz prefix == dump_path
    leaf name; npz holds an ``op_name`` scalar plus ``input_<n>`` /
    ``output_<n>`` keys.
    """
    testname = dump_path.name
    coreai_program.save_asset(dump_path / f"{testname}.aimodel")

    io_numpy: dict[str, np.ndarray] = {"op_name": np.array("main")}
    for name, arr in inputs.items():
        _add_npz_entry(io_numpy, f"input_{name}", arr.numpy())
    for name, arr in rt_outputs.items():
        _add_npz_entry(io_numpy, f"output_{name}", arr.numpy())
    np.savez(dump_path / f"{testname}_test_data.npz", **io_numpy)


async def _run_with_model(
    model: torch.nn.Module,
    rt_func: Any,
    exported_program: Any,
    kwargs: dict[str, Any],
    *,
    output_names: list[str] | None,
    fx_output_names: list[str],
    num_calls: int,
    rtol: float,
    atol: float,
    metal_inputs: bool = False,
    coreai_program: Any = None,
    dump_path: Path | None = None,
) -> None:
    """Path A: stateful, multi-call, name-based matching."""
    sig = exported_program.graph_signature
    desc = rt_func.desc
    state, mutated_arg_indices = _init_runtime_state(desc, sig, kwargs)
    inputs = _build_rt_inputs(desc, sig, kwargs, metal_inputs=metal_inputs)
    current_kwargs = dict(kwargs)

    def produce_torch_out(_call_idx: int) -> tuple[torch.Tensor, ...]:
        cloned = {k: v.clone() for k, v in current_kwargs.items()}
        out = model(**cloned)
        if not isinstance(out, tuple):
            out = (out,)
        for idx, key in enumerate(current_kwargs.keys()):
            if idx in mutated_arg_indices:
                current_kwargs[key] = cloned[key]
        return out

    def compare(
        outs: dict[str, Any],
        expected: tuple[torch.Tensor, ...],
        call_idx: int,
    ) -> None:
        _compare_by_name(
            outs,
            expected,
            names=output_names,
            fx_output_names=fx_output_names,
            rtol=rtol,
            atol=atol,
            call_idx=call_idx,
        )

    await _execute_and_compare(
        rt_func,
        inputs,
        state,
        num_calls=num_calls,
        produce_torch_out=produce_torch_out,
        compare=compare,
        coreai_program=coreai_program,
        dump_path=dump_path,
    )


async def _run_with_program(
    rt_func: Any,
    kwargs: dict[str, Any],
    torch_out: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    rtol: float,
    atol: float,
    metal_inputs: bool = False,
    coreai_program: Any = None,
    dump_path: Path | None = None,
) -> None:
    """Path B: pre-converted program, single call, sorted-key matching."""
    if metal_inputs:
        inputs = {
            k: NDArray(data=_narrow_dtype(v), backing=StorageKind.METAL)
            for k, v in kwargs.items()
        }
    else:
        inputs = {k: NDArray(data=_narrow_dtype(v)) for k, v in kwargs.items()}
    torch_out_tuple = torch_out if isinstance(torch_out, tuple) else (torch_out,)

    def produce_torch_out(_call_idx: int) -> tuple[torch.Tensor, ...]:
        return torch_out_tuple

    def compare(
        outs: dict[str, Any],
        expected: tuple[torch.Tensor, ...],
        _call_idx: int,
    ) -> None:
        _compare_sorted(outs, expected, rtol=rtol, atol=atol)

    await _execute_and_compare(
        rt_func,
        inputs,
        state={},
        num_calls=1,
        produce_torch_out=produce_torch_out,
        compare=compare,
        coreai_program=coreai_program,
        dump_path=dump_path,
    )


async def validate_numerical_output(**kwargs: Any) -> None:
    """Validate that a Core AI program produces the same output as torch.

    Two ways to call:

      1. End-to-end (default): pass ``model=<nn.Module>`` and the named tensor
         inputs. The helper exports, converts, runs, and compares against
         ``model(**inputs)``. Supports stateful models via state_names /
         input_names / output_names / num_calls. Output matching is by FX node
         name (or explicit ``output_names``).

      2. Pre-converted: pass ``coreai_program=<AIProgram>``,
         ``torch_out=<expected>``, and the named tensor inputs. The helper
         only runs and compares. Output matching is by sorted runtime key.
         Stateful machinery is not available on this path.

    Pass ``custom_kernels=[kernel, ...]`` to register ``TorchMetalKernel``
    instances before conversion. Pass ``metal_inputs=True`` to back all
    runtime inputs with ``StorageKind.METAL`` (required for Metal kernels).
    """
    model = kwargs.pop("model", None)
    coreai_program = kwargs.pop("coreai_program", None)
    torch_out = kwargs.pop("torch_out", None)

    assert (model is None) != (coreai_program is None), (
        "validate_numerical_output: pass exactly one of "
        "model=<nn.Module> or coreai_program=<AIProgram>"
    )
    assert coreai_program is None or torch_out is not None, (
        "validate_numerical_output: torch_out=... is required when "
        "passing coreai_program=..."
    )

    dynamic_shapes = kwargs.pop("dynamic_shapes", None)
    print_exported_graph = kwargs.pop("print_exported_graph", False)
    remove_decomps = kwargs.pop("remove_decomps", None)
    prepare_program = kwargs.pop("prepare_program", None)
    run_optimize_passes = kwargs.pop("run_optimize_passes", False)
    state_names: list[str] | None = kwargs.pop("state_names", None)
    input_names: list[str] | None = kwargs.pop("input_names", None)
    output_names: list[str] | None = kwargs.pop("output_names", None)
    num_calls: int = kwargs.pop("num_calls", 1)
    atol: float = kwargs.pop("atol", 1e-2)
    rtol: float = kwargs.pop("rtol", 1e-5)
    custom_kernels: list | None = kwargs.pop("custom_kernels", None)
    metal_inputs: bool = kwargs.pop("metal_inputs", False)

    exported_program = None
    fx_output_names: list[str] = []

    if model is not None:
        coreai_program, exported_program, fx_output_names = _export_and_convert(
            model,
            kwargs,
            dynamic_shapes=dynamic_shapes,
            remove_decomps=remove_decomps,
            prepare_program=prepare_program,
            print_exported_graph=print_exported_graph,
            state_names=state_names,
            input_names=input_names,
            output_names=output_names,
            run_optimize_passes=run_optimize_passes,
            custom_kernels=custom_kernels,
        )

    dump_path: Path | None = None
    if dump_optests_enabled():
        dump_path = _optest_dump_path(get_current_test_id())
        dump_path.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temp_directory:
        aimodel_path = Path(temp_directory) / "model.aimodel"
        asset = coreai_program.save_asset(aimodel_path)
        async with asset.executable(
            specialization_options=_get_test_specialization_options(),
        ) as ai_model:
            rt_func = ai_model.load_function("main")

            if model is not None:
                await _run_with_model(
                    model,
                    rt_func,
                    exported_program,
                    kwargs,
                    output_names=output_names,
                    fx_output_names=fx_output_names,
                    num_calls=num_calls,
                    rtol=rtol,
                    atol=atol,
                    metal_inputs=metal_inputs,
                    coreai_program=coreai_program,
                    dump_path=dump_path,
                )
            else:
                await _run_with_program(
                    rt_func,
                    kwargs,
                    torch_out,
                    rtol=rtol,
                    atol=atol,
                    metal_inputs=metal_inputs,
                    coreai_program=coreai_program,
                    dump_path=dump_path,
                )


def walk_coreai_program(coreai_program):
    def walk_operations(op, indent=0):
        prefix = "  " * indent
        print(f"{prefix}Operation: {op.name}")
        print(f"{prefix}  Location: {op.location}")

        if op.attributes:
            print(f"{prefix}  Attributes:")
            for named_attr in op.attributes:
                print(f"{prefix}    {named_attr.name}: {named_attr.attr}")

        # Recursively walk nested operations
        if hasattr(op, "regions"):
            for region in op.regions:
                for block in region.blocks:
                    for nested_op in block.operations:
                        walk_operations(nested_op, indent + 1)

    for op in coreai_program._mlir_module.operation.regions[0].blocks[0].operations:
        walk_operations(op)


def _torch_tensor_to_numpy_array(torch_tensor: torch.Tensor) -> np.ndarray:
    # TODO: Deprecate when all torch.Tensor can be seamlessly convert to numpy.ndarray
    if torch_tensor.dtype == torch.bfloat16:
        torch_tensor = torch_tensor.to(torch.float32)
    return torch_tensor.detach().cpu().numpy()


def _mlx_array_to_numpy_array(mlx_array: "mlx.core.array") -> np.ndarray:
    if platform.system() != "Darwin":
        raise RuntimeError("_mlx_array_to_numpy_array requires macOS (MLX)")
    if mlx_array.dtype == mlx.core.bfloat16:
        mlx_array = mlx_array.astype(mlx.core.float32)
    return np.array(mlx_array)


def filecheck_pattern(ir_output: str, check_file: str) -> None:
    __tracebackhide__ = True

    with tempfile.NamedTemporaryFile(mode="w", suffix=".check", delete=True) as f:
        f.write(check_file)
        f.flush()

        captured_out, captured_err = StringIO(), StringIO()
        old_stdin = sys.stdin

        try:
            sys.stdin = StringIO(ir_output)

            with redirect_stdout(captured_out), redirect_stderr(captured_err):
                opts = Options(
                    match_filename=f.name, input_file="-", check_prefixes=["CHECK"]
                )
                exit_code = Matcher.from_opts(opts).run()

            if exit_code != 0:
                error_parts = ["FileCheck failed"]
                if out := captured_out.getvalue():
                    error_parts.append(f"\n\nFileCheck output:\n{out}")
                if err := captured_err.getvalue():
                    error_parts.append(f"\n\nFileCheck errors:\n{err}")
                raise RuntimeError("".join(error_parts))
        finally:
            sys.stdin = old_stdin


def get_ir(
    model: torch.nn.Module,
    dynamic_shapes: dict | None = None,
    remove_decomps: list | None = None,
    **kwargs,
) -> str:
    """Export a model and return the Core AI IR string."""
    program = torch.export.export(
        model, args=(), kwargs=kwargs, dynamic_shapes=dynamic_shapes
    )
    decomp_table = torch.export.default_decompositions()
    if remove_decomps is not None:
        for decomp in remove_decomps:
            decomp_table.pop(decomp)
    program = program.run_decompositions(decomp_table)
    coreai_program = TorchConverter().add_exported_program(program).to_coreai()
    coreai_program.optimize()
    return str(coreai_program)
