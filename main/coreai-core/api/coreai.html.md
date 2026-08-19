# coreai

CoreAI high-level Python API.

## coreai.runtime

CoreAI Runtime.

### *final class* coreai.runtime.InferenceFunction

Represents a function in a model.

#### *property* desc *: InferenceFunctionDescriptor*

The descriptor for this function.

### *final class* coreai.runtime.NDArray(data: np.ndarray[Any, Any] | torch.Tensor | list[Any], backing: [StorageKind](#coreai.runtime.StorageKind) = StorageKind.BYTES)

Represent a multi-dimensional array.

#### *property* dtype *: str*

The type of scalars in the array.

#### *classmethod* from_descriptor(descriptor: NDArrayDescriptor) → Self

Allocates an NDArray from an NDArrayDescriptor.

Will have the correct shape, dtype, and alignment.

#### numpy() → ndarray[Any, Any]

Return the array as a NumPy array.

Standard dtypes (e.g. float32, int8) return a zero-copy view that shares the
backing buffer and preserves strides via DLPack. For bfloat16 / float8 dtypes,
which DLPack may not support depending on the NumPy version, it falls back to
the buffer protocol: C-contiguous arrays return a zero-copy view, non-contiguous
arrays (e.g. a transpose) return a copy gathered in C order. Sub-byte dtypes
(int2, int4, …) are always unpacked into a new array.

#### *property* shape *: tuple[int, ...]*

The shape of the array.

#### *property* strides *: tuple[int, ...]*

The strides, in elements (not bytes).

Describes the distance, in elements, between consecutive values along a
given dimension.

### *class* coreai.runtime.SpecializationOptions

Options for configuring model specialization.

#### *property* allowed_compute_unit_kinds *: list[ComputeUnitKind]*

Get the allowed compute unit kinds for this specialization.

#### *classmethod* cpu_only() → [SpecializationOptions](#coreai.runtime.SpecializationOptions)

Specialization options restricted to CPU only.

#### *classmethod* default() → [SpecializationOptions](#coreai.runtime.SpecializationOptions)

Default specialization options.

#### *classmethod* from_preferred_compute_unit_kind(compute_unit_kind: ComputeUnitKind) → [SpecializationOptions](#coreai.runtime.SpecializationOptions)

Create specialization options with a specific preferred compute unit kind.

#### *static* is_supported() → bool

Whether delegate specialization is available on this platform and setup.

Returns `True` on macOS with env var USE_OS_COREAI=1 (because the
in-package Core AI runtime doesn’t support delegates). Unlike the
constructors below, this never
raises, so callers can use it to gate or skip cleanly before attempting
to build options.

#### *property* preferred_compute_unit_kind *: ComputeUnitKind | None*

Get the preferred compute unit for this specialization.

#### with_debug(, enabled: bool) → [SpecializationOptions](#coreai.runtime.SpecializationOptions)

Return a new SpecializationOptions instance with debugging toggled.

### *class* coreai.runtime.StorageKind(value)

Backing buffer type for NDArray.

Python analogue for CoreAI.NDArrayDescriptor.StorageKind.

## coreai.authoring

Authoring utilities for CoreAI.

### *final class* coreai.authoring.AIModelAsset

A model asset.

Represents an unspecialized source model asset on disk. Unlike a loaded
`AIModel`, an asset cannot be used directly for inference — it is used to
query model information such as function signatures, metadata, storage/compute
types, and to manage derived compilation artifacts.
The asset is a directory (typically with an `.aimodel` extension) that
contains an unspecialized model asset, along with any associated metadata.

#### executable(specialization_options: [SpecializationOptions](#coreai.runtime.SpecializationOptions) | None = None) → AsyncGenerator[AIModel]

Async context manager that loads this asset as a runnable `AIModel`.

Loads the model on entry and releases its resources on exit. Use this
instead of constructing an `AIModel` directly to ensure the model
lifetime is scoped to the block:

```default
async with asset.executable() as model:
    outputs = await model.run(inputs)
```

Args:
: specialization_options: Optional specialization options for model
  : compilation. Only available on macOS.

Yields:
: AIModel: A loaded model ready for inference.

#### *static* is_valid(path: Path | str) → bool

Return `True` if the path contains a valid model asset.

Checks that the path is an existing directory with a known model extension
and that it contains either a source program or a derived artifact.

#### *classmethod* load(path: Path | str) → [AIModelAsset](#coreai.authoring.AIModelAsset)

Load an existing model asset from disk.

Args:
: path: Path to the asset directory. The directory must already exist.

Raises:
: RuntimeError: If the asset cannot be read or its metadata is malformed.

#### *property* metadata *: AIModelAssetMetadata*

The asset metadata, loaded from `metadata.json`.

#### *property* path *: Path*

The file-system path to the asset directory.

#### *property* program *: [AIProgram](#coreai.authoring.AIProgram)*

Return an in-memory representation of the on-disk IR bytecode.

Changes to this are not persisted to the on-disk asset.

#### summary(, include_statistics: bool = True) → AIModelAssetSummary | None

Return a summary of the model’s source program.

Args:
: include_statistics: When `True`, also reads storage types, compute
  : types, and the operation distribution. This is considerably slower
    for large models. When `False`, only the function signatures are
    populated.

Returns:
: An `AIModelAssetSummary`, or `None` if no source bytecode exists.

Raises:
: RuntimeError: If the bytecode file cannot be read or parsed.

#### update_metadata(updates: Callable[[AIModelAssetMetadata], None]) → None

Update the asset metadata atomically.

Calls `updates` with the current `AIModelAssetMetadata` object, which
must be mutated **in-place**. Changes are written to `metadata.json`
after the callable returns.

#### WARNING
The callback **must mutate the object it receives**, not reassign the
local variable. Reassigning is silently ignored:

```default
# WRONG — reassignment has no effect; the update is lost
def update(m):
    m = AIModelAssetMetadata()   # rebinds local 'm', does nothing

# CORRECT — mutate the received object in-place
def update(m):
    m.author = "Alice"
```

Args:
: updates: A callable that receives the current metadata and mutates it
  : in-place. Example:
    <br/>
    ```default
    asset.update_metadata(lambda m: setattr(m, "author", "Alice"))
    ```

Raises:
: RuntimeError: If the updated metadata cannot be written to disk.

### *class* coreai.authoring.AIProgram(module: [Module](#coreai.authoring.Module))

A machine-agnostic, deployable CoreAI program.

Holds the CoreAI authoring representation that can be optimized and
serialized into a deployable asset.

#### get_graph(entrypoint: str) → GraphOp | None

Get the GraphOp for the given name. Recurses into udml.namespace ops as required to try and find the graph.

#### optimize() → None

Apply frontend-agnostic optimization passes to the program.

Applies common optimization passes like CSE, DCE, canonicalize,
and CoreAI-specific optimizations. Mutations are made in-place on the
program.

#### save_asset(path: Path, metadata: AIModelAssetMetadata | None = None, minimum_os: [OSVersion](#coreai.authoring.OSVersion) = OSVersion.v27) → [AIModelAsset](#coreai.authoring.AIModelAsset)

Persist an AIProgram in AIModelAsset format.

#### set_hardware_constraints(entrypoint: str, hw_constraint_config: dict[str, [HardwareConstraints](#coreai.authoring.HardwareConstraints)]) → None

Rewrite the named entrypoint with harware constraints on the specified IO.

Args:
: entrypoint
  : Name of the `coreai.graph` to rewrite.
  <br/>
  hw_constraint_config
  : Mapping of tensor name → hardware constraints to apply.
    Names not present in the dict are left with their original types.

#### set_static_shape_config(entrypoint: str, shapes_config: dict[str, dict[str, tuple[int, ...]]]) → None

Attach static input shape configurations to a graph’s argument attributes.

### Parameters

entrypoint
: Name of the `coreai.graph` to configure.

shapes_config
: Mapping of new entrypoint name → (input name → static shape).
  Each input that appears in any entrypoint’s dict will receive a
  `coreai.enumerated_shapes` attribute enumerating all
  provided specializations for that input. Inputs not mentioned in
  any entrypoint dict are left unchanged.

### *class* coreai.authoring.AllocationType(value)

Allocations for hardware constraints.

#### Heap *= 'Heap'*

Use Heap for the allocation.

#### IOSurface *= 'IOSurface'*

Use a IOSurface for the allocation.

#### MTLBuffer *= 'MTLBuffer'*

Use a Metal buffer for the allocation.

### *class* coreai.authoring.ContainerType(value)

Backing allocation.

### *class* coreai.authoring.Context

A CoreAI authoring context.

Holds the runtime state needed to build a CoreAI program. Use it as a
context manager to make it active while constructing a program:

```default
with Context():
    ...
```

### *class* coreai.authoring.CustomMetalKernel(name: str, input_names: list[str], result_names: list[str], src: str, metal_params: list[[MetalParameter](#coreai.authoring.MetalParameter)] | None = None, helper_src: str | None = None, template_dtypes: dict[str, str] | None = None)

Specification class for metal kernels.

Intended for use by coreai-torch. The invocation path (_construct_kernel_op)
is intentionally private in 0.1.0; do not call directly from end-user code.

#### template_kernel_src(kernel_id: str, input_metadata: list[TensorMeta], result_metadata: list[TensorMeta]) → str

Template the kernel signature based on the provided IO types.

#### validate_and_set_dtype_templates(templates: dict[str, str], input_names: list[str]) → None

Validate and set dtype template dict.

### *class* coreai.authoring.FourCC(value)

Four character code denoting image type.

### *class* coreai.authoring.HardwareConstraints(allocation_type: [AllocationType](#coreai.authoring.AllocationType), alignments: list[int], interleave: list[int])

Hardware constraints attribute.

### *class* coreai.authoring.ImageConstraints(four_cc: [FourCC](#coreai.authoring.FourCC), backing: [ContainerType](#coreai.authoring.ContainerType), row_alignments_per_plane: list[int])

Tensor encoding attribute describing an image encoding.

### Parameters

four_cc: FourCC
: - Four-character code denoting image type.

backing: ContainerType
: - Backing buffer type.

row_alignments_per_plane: list[int]
: - Minimum row alignment for each row in the image.

### *class* coreai.authoring.ImageSpec(fourcc: Any, height: int, width: int, name: str | None = None, constraints: [ImageConstraints](#coreai.authoring.ImageConstraints) | None = None)

Image spec describing an image input/output for graph functions.

### Examples

```python
import numpy as np
from typing import Annotated

from coreai.authoring import FourCC, ImageSpec, TensorSpec

# An RGBA image input and a uint8 tensor output, used as
# Annotated metadata on a graph function's signature.
image_in = Annotated[object, ImageSpec(FourCC.RGBA, 224, 224, "input")]
tensor_out = Annotated[object, TensorSpec((224, 224, 3), np.uint8, "output")]
```

### Parameters

fourcc: coreai.authoring.FourCC
: - The image format (e.g. `FourCC.RGBA`)

height: int
: - The height of the image in pixels

width: int
: - The width of the image in pixels

name: str (optional)
: - Name of the image input/output

constraints: ImageConstraints (optional)
: - Any image-specific constraints (CVPixelBuffer vs MTLTexture, etc.)

### *class* coreai.authoring.MetalParameter(name: str, dtype: str, attr: str)

Specification class for metal parameters.

This class is to be used to specify metal parameters that are not the input tensors
and their associated metadata. This should be used to specify parameters like
thread_position_in_grid.

name: variable name to be included in parameters.
dtype: variable type, e.g. uint2, etc.
attr: metal attribute type, e.g. thread_position_in_grid, etc.

### *final class* coreai.authoring.Module

A CoreAI authoring module that holds the IR being constructed.

- Create a `Module` with `Module.create()`; a default context and
  source-line location are bootstrapped for you.
- Use the `Module` as a context manager to append operations to the module body.
- Inspect the constructed IR with `verify`, `dump`, and `as_string`.

#### as_string() → str

Return the IR as a string.

#### *classmethod* create() → Self

Create a Module.

Bootstraps a context and an active location if they do not exist.

#### dump() → None

Print the IR to stderr (useful for debugging).

#### verify() → None

Verify the constructed IR for correctness. Raises on error.

### *class* coreai.authoring.OSVersion(value)

Operating system version to serialize for.

### *class* coreai.authoring.TensorSpec(shape: list[int] | None = None, dtype: type[np.generic] | Type | None = None, name: str | None = None, encoding: [HardwareConstraints](#coreai.authoring.HardwareConstraints) | [ImageConstraints](#coreai.authoring.ImageConstraints) | None = None)

Tensor spec describing a tensor’s shape, dtype, and constraints.

#### *property* is_materialized *: bool*

Check if shape and dtype info is available.

#### *property* rank *: int | None*

Get rank of tensor.

#### set_default_mtl_constraints() → None

Set metal constraints with default alignments and interleave.
