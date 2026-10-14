# coreai

CoreAI high-level Python API.

The supported public API lives in two packages:

* [`coreai.authoring`](#module-coreai.authoring) — build AI Models programmatically.
* [`coreai.runtime`](#module-coreai.runtime) — load and run `.aimodel` files.

The raw MLIR Python bindings stay reachable under `coreai._compiler`
(notably `coreai._compiler.ir`) as an escape hatch for power users, but
they carry **no stability guarantee** and are **unversioned**: they track the
LLVM pin and may change with any bump. Prefer the [`coreai.authoring`](#module-coreai.authoring)
facades, whose surface is versioned independently of that pin.

## coreai.runtime

CoreAI Runtime.

### *final class* coreai.runtime.InferenceFunction

Represents a function in a model.

#### *property* desc *: InferenceFunctionDescriptor*

The descriptor for this function.

### *final class* coreai.runtime.NDArray(data: ndarray[Any, Any] | list[Any], backing: [StorageKind](#coreai.runtime.StorageKind) = StorageKind.BYTES)

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

#### *property* is_debug_enabled *: bool*

Whether debugging is enabled for this specialization.

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

Authoring facades for CoreAI.

Everything in `__all__` is the supported, versioned public surface: facade
objects that accept and return other facades, never raw MLIR bindings. The raw
MLIR Python bindings stay reachable under `coreai._compiler.ir`, but that
path is unsupported and unversioned (see [`coreai`](#module-coreai)).

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

Holds the CoreAI authoring representation that can be serialized into a
deployable asset. The representation is optimized as it is built: exiting a
`coreai.authoring.Module` `with` block applies the frontend-agnostic
pre-compilation rewrite passes in place.

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

Specialize a graph into one statically-shaped entrypoint per configuration.

Each input named in `shapes_config` receives a
`coreai.enumerated_shapes` attribute, then the pre-compilation rewrite
runs and consumes it: the original dynamic graph is erased and replaced
by one statically-shaped clone per configuration key, named
`<entrypoint>_<key>`. Existing argument attributes — including
`coreai.name` and any hardware-constraint encoding on the argument
type — are carried over to every clone.

#### WARNING
`entrypoint` no longer exists once this returns. Later calls that
name it — including a second call to this method — raise, because only
the `<entrypoint>_<key>` graphs remain.
`set_hardware_constraints` may be applied either before or after
this method, but afterwards it has to name the generated graphs.

### Parameters

entrypoint
: Name of the `coreai.graph` to specialize.

shapes_config
: Mapping of new entrypoint name → (input name → static shape).
  Inputs not mentioned in any entrypoint dict keep their original
  type in every specialization.

### *class* coreai.authoring.AllocationType(value)

Allocation backing for hardware constraints.

#### Heap *= 'Heap'*

Use Heap for the allocation.

#### IOSurface *= 'IOSurface'*

Use a IOSurface for the allocation.

#### MTLBuffer *= 'MTLBuffer'*

Use a Metal buffer for the allocation.

### *class* coreai.authoring.CompileUnitScope(file: [FileInfo](#coreai.authoring.FileInfo), source_language: str = 'DW_LANG_Python', producer: str = 'ODIE', optimized: bool = False, flags: str = '')

A DWARF compilation unit scope over a [`FileInfo`](#coreai.authoring.FileInfo).

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

### *class* coreai.authoring.DType

`numpy.dtype`-like facade over an MLIR scalar type.

A dtype pairs the deferred MLIR-type thunk (inherited from [`Type`](#coreai.authoring.Type))
with its canonical MLIR name and an optional numpy dtype. Instances are the
pre-registered module singletons; direct construction is disallowed
(inherited from `IRObject`) — use e.g.
`float32`.

Equality and hashing come from
`ValueEquality` keyed on
`_eq_key` (the canonical name), so they never require an active
`Context`.

#### *property* numpy *: np.dtype[Any] | None*

The corresponding numpy dtype, or `None` (e.g. for `index`).

### *class* coreai.authoring.DebugInfo

Frozen facade over an MLIR debug-info `Location`.

Build one with a factory ([`file()`](#coreai.authoring.DebugInfo.file), [`unknown()`](#coreai.authoring.DebugInfo.unknown), [`current()`](#coreai.authoring.DebugInfo.current),
[`metadata_only()`](#coreai.authoring.DebugInfo.metadata_only)) and refine it with the functional updates
([`with_scope()`](#coreai.authoring.DebugInfo.with_scope), [`with_subprogram()`](#coreai.authoring.DebugInfo.with_subprogram), [`with_op_id()`](#coreai.authoring.DebugInfo.with_op_id),
[`with_metadata()`](#coreai.authoring.DebugInfo.with_metadata), [`with_inlined_at()`](#coreai.authoring.DebugInfo.with_inlined_at)). Direct construction is
disallowed (inherited from `IRObject`); every constructor routes
through `_make()`.

#### activate() → Generator[None, None, None]

Activate this location for the current thread as a context manager.

Ops built inside the `with` block inherit this location. Implemented
as a generator (not `__enter__`/`__exit__`) because the facade is
frozen and cannot stash the push token on the instance; the generator
frame holds the raw `Location` per activation, so nesting is correct.
Must run with a `Context` active on the current thread.

#### *property* col *: int | None*

The source column, or `None`.

#### *classmethod* current() → [DebugInfo](#coreai.authoring.DebugInfo) | None

Wrap the `Location` active on the current thread, or `None`.

#### *property* end *: tuple[int, int] | None*

The `(end_line, end_col)` of a ranged location, or `None`.

#### *classmethod* file(filename: str, line: int, col: int = 0, , end: tuple[int, int] | None = None, scope: [CompileUnitScope](#coreai.authoring.CompileUnitScope) | [SubprogramScope](#coreai.authoring.SubprogramScope) | [SymbolRefScope](#coreai.authoring.SymbolRefScope) | None = None) → Self

Build a source-coordinate location, optionally with a range and scope.

#### *property* filename *: str | None*

The source file name, or `None` when there is no file.

#### *property* inlined_at *: [DebugInfo](#coreai.authoring.DebugInfo) | None*

The location this was inlined at, or `None`.

#### *property* is_unknown *: bool*

Whether this carries no file, scope, metadata, or raw spelling.

#### *property* line *: int | None*

The source line, or `None`.

#### *classmethod* metadata_only(names: Sequence[str]) → Self

Build a metadata-only location from one or more entry names.

Rejects an empty sequence (mirroring the raw builder), so a degenerate
empty-metadata location cannot masquerade as [`unknown()`](#coreai.authoring.DebugInfo.unknown).

#### *property* scope *: [CompileUnitScope](#coreai.authoring.CompileUnitScope) | [SubprogramScope](#coreai.authoring.SubprogramScope) | [SymbolRefScope](#coreai.authoring.SymbolRefScope) | None*

The lexical scope, or `None`.

#### *classmethod* unknown() → Self

Build the unknown location (no file, scope, metadata, or coordinates).

#### with_inlined_at(loc: [DebugInfo](#coreai.authoring.DebugInfo)) → Self

Return a copy marked as inlined at `loc`.

#### with_metadata(name: str) → Self

Return a copy with a named (payload-free) metadata entry appended.

#### with_op_id(dialect: str, value: int) → Self

Return a copy with an `op_id` metadata entry appended.

#### with_scope(scope: [CompileUnitScope](#coreai.authoring.CompileUnitScope) | [SubprogramScope](#coreai.authoring.SubprogramScope) | [SymbolRefScope](#coreai.authoring.SymbolRefScope)) → Self

Return a copy with the lexical scope replaced.

#### with_subprogram(name: str, , mlir_type_str: str | None = None, linkage_name: str = '') → Self

Return a copy scoped to a subprogram named `name`.

The subprogram’s file and parent compile unit are derived from this
location’s file, so this location must already have one (call
[`file()`](#coreai.authoring.DebugInfo.file) first). `mlir_type_str` is the function type’s
context-free spelling (required to later materialize the scope via
`_to_mlir()`).

### *class* coreai.authoring.FileInfo(filename: str, directory: str = '', sha256_sum: str = '')

A source file: base name plus optional directory and SHA-256 checksum.

### *class* coreai.authoring.FourCC(value)

Four character code denoting image type.

### *class* coreai.authoring.HandleType(inner_type: [Type](#coreai.authoring.Type))

Frozen facade over the CoreAI `HandleType` wrapping an inner `Type`.

#### *property* inner_type *: [Type](#coreai.authoring.Type)*

The wrapped inner type facade.

### *class* coreai.authoring.HardwareConstraints(allocation_type: [AllocationType](#coreai.authoring.AllocationType), alignments: tuple[int, ...], interleave: tuple[int, ...])

Hardware constraints tensor-encoding attribute (a frozen `IRObject`).

Describes hardware-specific layout requirements — the backing allocation
(`allocation_type`) plus per-dimension `alignments` and `interleave`.
Frozen so it is hashable and participates in `TensorType`’s equality key.

### *class* coreai.authoring.ImageConstraints(backing: [ContainerType](#coreai.authoring.ContainerType), row_alignments_per_plane: tuple[int, ...], , four_cc: [FourCC](#coreai.authoring.FourCC) | None = None)

Image encoding tensor-encoding attribute (a frozen `IRObject`).

Carries the `backing` buffer type (`CVPixelBuffer`/`MTLTexture`) and
the minimum `row_alignments_per_plane` for each plane of the image.

`four_cc` is an optional, keyword-only authoring sidecar kept for backward
compatibility: it is **not** carried by the `#coreaix.image_constraints`
attribute — the CoreAI `ImageType` carries the FourCC separately — so it is
excluded from equality/hashing (`compare=False`) and is not recovered by
`_from_mlir()`. It is keyword-only so the two serialized fields stay the
positional interface.

### coreai.authoring.ImageSpec

alias of [`ImageType`](#coreai.authoring.ImageType)

### *class* coreai.authoring.ImageType(fourcc: [FourCC](#coreai.authoring.FourCC), height: int, width: int, name: str | None = None, constraints: [ImageConstraints](#coreai.authoring.ImageConstraints) | None = None)

Frozen facade over the CoreAI `ImageType` (also exported as `ImageSpec`).

Identity is `(fourcc, height, width, constraints, name)`; `name` is
included because an image spec is a named I/O (like `TensorType`). An
image type is always materialized, so `_to_mlir()` never has an
unmaterialized case. `fourcc` is the CoreAI dialect `FourCC` binding
enum, passed straight through to `FourCCAttr`.

#### *property* constraints *: [ImageConstraints](#coreai.authoring.ImageConstraints) | None*

The image constraints encoding, if any.

#### *property* fourcc *: [FourCC](#coreai.authoring.FourCC)*

The CoreAI-dialect `FourCC` enum value.

#### *property* height *: int*

Image height in pixels.

#### *property* is_materialized *: bool*

An image type is always materialized.

#### *property* name *: str | None*

Authoring I/O name; part of identity (`None` for a derived type).

#### *property* width *: int*

Image width in pixels.

#### with_constraints(constraints: [ImageConstraints](#coreai.authoring.ImageConstraints) | None) → [ImageType](#coreai.authoring.ImageType)

Return a copy with the image constraints replaced.

Mirrors [`TensorType.with_encoding()`](#coreai.authoring.TensorType.with_encoding); `ImageType` names its
updater `with_constraints` because it has no `encoding` alias.

#### with_name(name: str | None) → [ImageType](#coreai.authoring.ImageType)

Return a copy with `name` replaced (a distinct type: name is identity).

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
  On a clean exit from the `with` block, the frontend-agnostic
  pre-compilation rewrite passes are applied to the IR in place.
- Inspect the constructed IR with `verify`, `dump`, and `as_string`.

The rewrite passes run over the *whole* module on each clean exit, so
building a module across several `with` blocks re-optimizes it once per
block. A handle obtained inside a block survives only if the rewrite keeps
its operation — reading one that was folded or DCE’d away is a
use-after-free, and will crash rather than raise.

#### as_string() → str

Return the IR as a string.

#### *classmethod* create() → Self

Create a Module.

Bootstraps a context and an active location if they do not exist.

#### *property* debug_info *: [DebugInfo](#coreai.authoring.DebugInfo)*

The location where the Module was created, as a `DebugInfo` facade.

#### dump() → None

Print the IR to stderr (useful for debugging).

#### verify() → None

Verify the constructed IR for correctness. Raises on error.

### *class* coreai.authoring.OSVersion(value)

Operating system version to serialize for.

### *class* coreai.authoring.Operation

Frozen facade over an MLIR `Operation`.

Wraps a live MLIR op; instances are produced by `_from_mlir()`, never by
direct construction. The `_operands`/`_results` slots memoize the lazy
operand/result views. Mixes in
`AttributeManager` for dict-like attribute
access (`op["name"]` / `op["name"] = value`) over `_attr_dict()`.

#### *property* debug_info *: [DebugInfo](#coreai.authoring.DebugInfo)*

The op’s location, as a `DebugInfo` facade (cached on first access).

#### *property* name *: str*

The fully qualified op name (e.g. `"coreai.add"`).

#### *property* operands *: Sequence[[Value](#coreai.authoring.Value)]*

The op’s operands as a lazy, cached sequence of `Value` facades.

#### *property* results *: Sequence[[Value](#coreai.authoring.Value)]*

The op’s results as a lazy, cached sequence of `Value` facades.

### *class* coreai.authoring.SubprogramScope(name: str, file: [FileInfo](#coreai.authoring.FileInfo), compile_unit: [CompileUnitScope](#coreai.authoring.CompileUnitScope) | str, linkage_name: str = '', mlir_type_str: str | None = None)

A DWARF subprogram (function) scope.

`compile_unit` is either a full [`CompileUnitScope`](#coreai.authoring.CompileUnitScope) or a bare symbol
name referencing one. `mlir_type_str` is the subprogram’s MLIR function
type as a context-free textual spelling, re-parsed under a `Context` in
`DebugInfo._to_mlir()`; it is excluded from equality (`compare=False`)
because it is a context-bound implementation detail, not identity.

### *class* coreai.authoring.SymbolRefScope(symbol: str)

A bare `@symbol` reference to a scope (`symbol` stored without `@`).

`symbol` must be a valid MLIR bare identifier (no spaces or characters
that would require `@"..."` quoting, and no nested `a::b` path): it is
emitted and recovered via `@`-prefixed string spelling, matching the
incumbent authoring builders. Op/graph/func names are Python identifiers,
so this always holds for the locations authoring produces.

### coreai.authoring.TensorSpec

alias of [`TensorType`](#coreai.authoring.TensorType)

### *class* coreai.authoring.TensorType(shape: Sequence[int] | None = None, dtype: \_DTypeLike | None = None, name: str | None = None, encoding: \_Encoding | None = None)

Frozen facade over `RankedTensorType` (also exported as `TensorSpec`).

Carries `(shape, dtype, constraints)` plus an authoring `name`.
Equality/hashing (via `ValueEquality`) are over all four fields,
`name` included: the merged object is also an authoring spec, so two
specs differing only by name are distinct I/O declarations (and must stay
distinct as `Annotated` metadata). A `Value`-derived type always has
`name=None`, so anonymous types still compare equal.

#### *property* constraints *: \_Encoding | None*

The hardware/image constraints encoding, if any.

#### *property* dtype *: [DType](#coreai.authoring.DType) | Type | None*

The element type as a `DType` (or raw MLIR type passthrough).

#### *property* encoding *: \_Encoding | None*

Backwards-compatible alias of [`constraints`](#coreai.authoring.TensorType.constraints).

#### *property* is_materialized *: bool*

Whether both shape and dtype are known (context-free).

#### *property* is_static *: bool*

Whether the tensor is materialized with only static dimensions.

Returns `False` for an unmaterialized type as well as for one with a
dynamic dimension; callers distinguishing *dynamic* from *unknown* must
check [`is_materialized`](#coreai.authoring.TensorType.is_materialized) first.

#### *property* name *: str | None*

Authoring I/O name; part of identity (`None` for a derived type).

#### *property* rank *: int | None*

The tensor rank, or `None` when the shape is unknown.

#### *property* shape *: tuple[int, ...] | None*

The tensor shape, or `None` when unmaterialized.

#### with_default_mtl_constraints() → [TensorType](#coreai.authoring.TensorType)

Return a copy carrying default contiguous Metal-buffer constraints.

#### with_encoding(encoding: \_Encoding | None) → [TensorType](#coreai.authoring.TensorType)

Return a copy with the constraints/encoding replaced.

#### with_name(name: str | None) → [TensorType](#coreai.authoring.TensorType)

Return a copy with `name` replaced (a distinct type: name is identity).

### *class* coreai.authoring.TokenType

Frozen facade over the CoreAI async `TokenType` (stateless).

### *class* coreai.authoring.Type

Abstract base facade for all MLIR types.

The deferred-construction thunk in `_inner` lets a facade be created
(e.g. a module-level dtype singleton) without an active MLIR `Context`;
the thunk is only invoked by `_to_mlir()`, which must run with a
`Context` active on the current thread. Concrete subclasses populate
`_inner` via `_make()`.

`Type` is never instantiated directly: it does not implement
`__str__()`, so it remains abstract and `_make` raises for
it. Concrete subclasses supply `__str__` and register themselves in
`TYPE_REGISTRY` so `_from_mlir()` can route to them.

### *class* coreai.authoring.Value

Frozen facade over an MLIR `Value`.

Wraps a live MLIR value; instances are produced by `_from_mlir()` (e.g.
from [`Operation.operands`](#coreai.authoring.Operation.operands)/[`results`](#coreai.authoring.Operation.results)), never by direct
construction.

#### *property* defining_op *: [Operation](#coreai.authoring.Operation) | None*

The [`Operation`](#coreai.authoring.Operation) that defines this value.

Returns the defining op for an op result, or `None` for a block
argument (which has no defining op).

#### *property* type *: [Type](#coreai.authoring.Type)*

The value’s type, as a [`Type`](#coreai.authoring.Type) facade.

### coreai.authoring.abs_(x: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

AbsOp constructor.

Operation that accepts one tensor and calculates the absolute value
of each element.

### coreai.authoring.acos(x: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

AcosOp constructor.

Applies element-wise inverse cosine (arccosine) to the input tensor.

Computes acos(x) for each element x, returning the angle in radians whose
cosine is x. Valid input domain is [-1, 1]; output range is [0, π]. Follows
IEEE 754 semantics: acos(1.0)=0.0, acos(0.0)=π/2, acos(-1.0)=π,
acos(x) for 

```
|x|
```

>1=NaN, acos(NaN)=NaN.

### coreai.authoring.acosh(x: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

AcoshOp constructor.

Operation that accepts one tensor and calculates the arc
hyperbolic cosine for each element.

### coreai.authoring.add(x: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, y: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

AddOp constructor.

Operation that accepts two tensors of the same shape
and type and returns a + b for each corresponding element.

### coreai.authoring.all_(x: [Value](#coreai.authoring.Value) | ConstLike, axes: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

AllOp constructor.

### coreai.authoring.and_(x: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, y: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

AndOp constructor.

Operation that accepts two tensors of the same shape
and type and returns the logical AND of each corresponding element.

### coreai.authoring.any_(x: [Value](#coreai.authoring.Value) | ConstLike, axes: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

AnyOp constructor.

### coreai.authoring.argmax(x: [Value](#coreai.authoring.Value) | ConstLike, axis: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ArgmaxOp constructor.

### coreai.authoring.argsort(x: [Value](#coreai.authoring.Value) | ConstLike, dim: [Value](#coreai.authoring.Value) | ConstLike, descending: [Value](#coreai.authoring.Value) | ConstLike, stable: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ArgSortOp constructor.

### coreai.authoring.asin(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

AsinOp constructor.

### coreai.authoring.asinh(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

AsinhOp constructor.

### coreai.authoring.asymmetric_quant_stats(x: [Value](#coreai.authoring.Value) | ConstLike, value_range: [Value](#coreai.authoring.Value) | ConstLike, axis: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → list[[Value](#coreai.authoring.Value)]

AsymmetricQuantStatsOp constructor.

### coreai.authoring.atan(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

AtanOp constructor.

### coreai.authoring.atanh(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

AtanhOp constructor.

### coreai.authoring.batch_matmul(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Batch matrix multiplication of two tensors.

### coreai.authoring.bitwise_and(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BitwiseAndOp constructor.

### coreai.authoring.bitwise_or(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BitwiseOrOp constructor.

### coreai.authoring.bitwise_xor(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BitwiseXorOp constructor.

### coreai.authoring.blockwise_shift_scale(data: [Value](#coreai.authoring.Value) | ConstLike, scale: [Value](#coreai.authoring.Value) | ConstLike, offset1: [Value](#coreai.authoring.Value) | ConstLike, offset2: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BlockwiseShiftScaleOp constructor.

### coreai.authoring.broadcast_in_dims(x: [Value](#coreai.authoring.Value) | ConstLike, dim_sizes: [Value](#coreai.authoring.Value) | ConstLike, axes: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastInDimsOp constructor.

### coreai.authoring.broadcast_shapes(shape_1: [Value](#coreai.authoring.Value) | ConstLike, shape_2: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastShapesOp constructor.

### coreai.authoring.broadcast_to(x: [Value](#coreai.authoring.Value) | ConstLike, shape: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastToOp constructor.

### coreai.authoring.broadcasting_add(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingAdd constructor.

### coreai.authoring.broadcasting_and(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingAndOp constructor.

### coreai.authoring.broadcasting_batch_matmul(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BatchMatmul op constructor.

### coreai.authoring.broadcasting_bitwise_and(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingBitwiseAnd constructor.

### coreai.authoring.broadcasting_bitwise_or(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingBitwiseOr constructor.

### coreai.authoring.broadcasting_bitwise_xor(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingBitwiseXor constructor.

### coreai.authoring.broadcasting_divide(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingDivide constructor.

### coreai.authoring.broadcasting_equal(x: [Value](#coreai.authoring.Value) | ConstLike, other: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingEqualOp constructor.

### coreai.authoring.broadcasting_floor_divide(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingFloorDivide constructor.

### coreai.authoring.broadcasting_greater(x: [Value](#coreai.authoring.Value) | ConstLike, other: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingGreaterOp constructor.

### coreai.authoring.broadcasting_maximum(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingMaximum constructor.

### coreai.authoring.broadcasting_minimum(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingMinimum constructor.

### coreai.authoring.broadcasting_modulo(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingModulo constructor.

### coreai.authoring.broadcasting_mul(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingMul constructor.

### coreai.authoring.broadcasting_not_equal(x: [Value](#coreai.authoring.Value) | ConstLike, other: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingNotEqual constructor.

### coreai.authoring.broadcasting_or(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingOrOp constructor.

### coreai.authoring.broadcasting_pow(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingPow constructor.

### coreai.authoring.broadcasting_sub(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingSub constructor.

### coreai.authoring.broadcasting_where(condition: [Value](#coreai.authoring.Value) | ConstLike, x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingWhere constructor.

### coreai.authoring.broadcasting_xor(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BroadcastingXorOp constructor.

### coreai.authoring.build_sparse_with_bitmask(values: [Value](#coreai.authoring.Value) | ConstLike, bitmask: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

BuildSparseWithBitmaskOp constructor.

### coreai.authoring.cast(x: [Value](#coreai.authoring.Value) | ConstLike, dtype: [DType](#coreai.authoring.DType) | [type](#coreai.authoring.Value.type)[np.generic] | [TensorType](#coreai.authoring.TensorType), , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Cast *x* to *dtype* (a facade `DType`/`TensorType` or a numpy type).

### coreai.authoring.concat(dimension: [Value](#coreai.authoring.Value) | ConstLike, inputs: list[[Value](#coreai.authoring.Value) | ConstLike], , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Concatenate *inputs* along *dimension*.

### coreai.authoring.conv2d(x: [Value](#coreai.authoring.Value) | ConstLike, weight: [Value](#coreai.authoring.Value) | ConstLike, stride: [Value](#coreai.authoring.Value) | ConstLike, dilation: [Value](#coreai.authoring.Value) | ConstLike, groups: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Perform a 2D convolution of `input` with `weight`.

### coreai.authoring.conv3d(x: [Value](#coreai.authoring.Value) | ConstLike, weight: [Value](#coreai.authoring.Value) | ConstLike, stride: [Value](#coreai.authoring.Value) | ConstLike, dilation: [Value](#coreai.authoring.Value) | ConstLike, groups: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Perform a 3D convolution of `input` with `weight`.

### coreai.authoring.conv_transpose2d(x: [Value](#coreai.authoring.Value) | ConstLike, weight: [Value](#coreai.authoring.Value) | ConstLike, stride: [Value](#coreai.authoring.Value) | ConstLike, padding: [Value](#coreai.authoring.Value) | ConstLike, dilation: [Value](#coreai.authoring.Value) | ConstLike, output_pad: [Value](#coreai.authoring.Value) | ConstLike, groups: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ConvTranspose2dOp constructor.

### coreai.authoring.conv_transpose3d(x: [Value](#coreai.authoring.Value) | ConstLike, weight: [Value](#coreai.authoring.Value) | ConstLike, stride: [Value](#coreai.authoring.Value) | ConstLike, padding: [Value](#coreai.authoring.Value) | ConstLike, dilation: [Value](#coreai.authoring.Value) | ConstLike, output_pad: [Value](#coreai.authoring.Value) | ConstLike, groups: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ConvTranspose3dOp constructor.

### coreai.authoring.cos(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

CosOp constructor.

### coreai.authoring.cosh(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

CoshOp constructor.

### coreai.authoring.create_complex(real: [Value](#coreai.authoring.Value) | ConstLike, imaginary: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

CreateComplexOp constructor.

### coreai.authoring.dequantize(x: [Value](#coreai.authoring.Value) | ConstLike, scale: [Value](#coreai.authoring.Value) | ConstLike, offset1: [Value](#coreai.authoring.Value) | ConstLike, offset2: [Value](#coreai.authoring.Value) | ConstLike, axis: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Dequantize a lower-precision tensor back to a floating-point representation.

### coreai.authoring.divide(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

DivideOp constructor.

### coreai.authoring.equal(x: [Value](#coreai.authoring.Value) | ConstLike, other: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

EqualOp constructor.

### coreai.authoring.erf(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ErfOp constructor.

### coreai.authoring.exp(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ExpOp constructor.

### coreai.authoring.expand_dims(x: [Value](#coreai.authoring.Value) | ConstLike, axes: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ExpandDims constructor.

### coreai.authoring.floor_divide(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

FloorDivideOp constructor.

### coreai.authoring.gather_along_axis(x: [Value](#coreai.authoring.Value) | ConstLike, indices: [Value](#coreai.authoring.Value) | ConstLike, axis: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

GatherAlongAxisOp constructor.

### coreai.authoring.gather_nd(x: [Value](#coreai.authoring.Value) | ConstLike, indices: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

GatherNdOp constructor.

### coreai.authoring.gelu(x: [Value](#coreai.authoring.Value) | ConstLike, , approximate: str = 'none', loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

GELUOp constructor.

### coreai.authoring.get_shape(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

GetShapeOp constructor.

### coreai.authoring.graph(fn: Callable[..., Any], , private: bool = False, externalize: bool = False, no_inline: bool = False, provided_name: str | None = None, composite_decl: Attribute | None = None) → \_FacadeGraph

### coreai.authoring.graph(fn: None = None, , private: bool = False, externalize: bool = False, no_inline: bool = False, provided_name: str | None = None, composite_decl: Attribute | None = None) → Callable[[Callable[..., Any]], \_FacadeGraph]

Build a `coreai.graph` from a facade-authored function.

The decorated function takes a variable number of
`Annotated[Value, TensorType(...)]` (or `ImageType`) parameters and
returns one such `Value` or a tuple of them; its body receives and returns
facade [`Value`](#coreai.authoring.Value) objects. Un-annotated parameters
defer construction until the graph is first invoked (shape inference).

Usable both bare (`@graph`) and called (`@graph(externalize=True)`); the
result is a `_FacadeGraph`. `composite_decl` accepts a raw MLIR
`Attribute` for composite-op declaration — an advanced escape hatch outside
the stable facade surface.

### coreai.authoring.greater(x: [Value](#coreai.authoring.Value) | ConstLike, other: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

GreaterOp constructor.

### coreai.authoring.image_to_tensor(source: [Value](#coreai.authoring.Value), , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ImageToTensorOp constructor.

### coreai.authoring.imaginary_part(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Retrieve the imaginary part of a complex tensor.

### coreai.authoring.interpolate(x: [Value](#coreai.authoring.Value) | ConstLike, output_shape: [Value](#coreai.authoring.Value) | ConstLike, scale: [Value](#coreai.authoring.Value) | ConstLike, offset: [Value](#coreai.authoring.Value) | ConstLike, interpolation_mode: str, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Interpolate *x* onto *output_shape* using *scale*/*offset* transforms.

*interpolation_mode* selects the algorithm: `"linear"` or
`"nearest_neighbor"`.

### coreai.authoring.interpolate_coordinates(x: [Value](#coreai.authoring.Value) | ConstLike, coords: [Value](#coreai.authoring.Value) | ConstLike, interpolation_mode: str, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Interpolate values of *x* at the per-point *coords*.

*interpolation_mode* selects the algorithm: `"linear"` or
`"nearest_neighbor"`.

### coreai.authoring.log(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

LogOp constructor.

### coreai.authoring.lut_to_dense(indices: [Value](#coreai.authoring.Value) | ConstLike, lut: [Value](#coreai.authoring.Value) | ConstLike, axis: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

LutToDenseOp constructor.

### coreai.authoring.lut_to_sparse_with_bitmask(sparse_indices: [Value](#coreai.authoring.Value) | ConstLike, lut: [Value](#coreai.authoring.Value) | ConstLike, axis: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

LutToSparseWithBitmaskOp constructor.

### coreai.authoring.maximum(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

MaximumOp constructor.

### coreai.authoring.maxpool2d(output: [TensorType](#coreai.authoring.TensorType), x: [Value](#coreai.authoring.Value) | ConstLike, kernel_size: [Value](#coreai.authoring.Value) | ConstLike, stride: [Value](#coreai.authoring.Value) | ConstLike, dilation: [Value](#coreai.authoring.Value) | ConstLike, ceil_mode: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

2-D max pooling with an explicit *output* tensor type.

### coreai.authoring.maxpool3d(output: [TensorType](#coreai.authoring.TensorType), x: [Value](#coreai.authoring.Value) | ConstLike, kernel_size: [Value](#coreai.authoring.Value) | ConstLike, stride: [Value](#coreai.authoring.Value) | ConstLike, dilation: [Value](#coreai.authoring.Value) | ConstLike, ceil_mode: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

3-D max pooling with an explicit *output* tensor type.

### coreai.authoring.minimum(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

MinimumOp constructor.

### coreai.authoring.modulo(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ModuloOp constructor.

### coreai.authoring.mul(x: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, y: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

MulOp constructor.

Operation that accepts two tensors of the same shape and
type and returns a \* b for each corresponding element.

### coreai.authoring.non_zero(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

NonZeroOp constructor.

### coreai.authoring.not_(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

NotOp constructor.

### coreai.authoring.not_equal(x: [Value](#coreai.authoring.Value) | ConstLike, other: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

NotEqualOp constructor.

### coreai.authoring.or_(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

OrOp constructor.

### coreai.authoring.pad(x: [Value](#coreai.authoring.Value) | ConstLike, padding: [Value](#coreai.authoring.Value) | ConstLike, padding_value: [Value](#coreai.authoring.Value) | ConstLike, padding_mode: str = 'constant', , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

PadOp constructor.

### coreai.authoring.pow_(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

PowOp constructor.

### coreai.authoring.quantize(x: [Value](#coreai.authoring.Value) | ConstLike, scale: [Value](#coreai.authoring.Value) | ConstLike, offset1: [Value](#coreai.authoring.Value) | ConstLike, offset2: [Value](#coreai.authoring.Value) | ConstLike, axis: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Quantize a real-valued tensor to a lower-precision representation.

### coreai.authoring.range_(start: [Value](#coreai.authoring.Value) | ConstLike, stop: [Value](#coreai.authoring.Value) | ConstLike, step: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

RangeOp constructor.

### coreai.authoring.real_part(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Retrieve the real part of a complex tensor.

### coreai.authoring.reduce_max(x: [Value](#coreai.authoring.Value) | ConstLike, axes: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ReduceMaxOp constructor.

### coreai.authoring.reduce_mean(x: [Value](#coreai.authoring.Value) | ConstLike, axes: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ReduceMeanOp constructor.

### coreai.authoring.reduce_min(x: [Value](#coreai.authoring.Value) | ConstLike, axes: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ReduceMinOp constructor.

### coreai.authoring.reduce_product(x: [Value](#coreai.authoring.Value) | ConstLike, axes: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ReduceProductOp constructor.

### coreai.authoring.reduce_sum(x: [Value](#coreai.authoring.Value) | ConstLike, axes: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ReduceSumOp constructor.

### coreai.authoring.relu(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ReLUOp constructor.

### coreai.authoring.reshape(x: [Value](#coreai.authoring.Value) | ConstLike, new_shape: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ReshapeOp constructor.

### coreai.authoring.reverse(x: [Value](#coreai.authoring.Value) | ConstLike, dims: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ReverseOp constructor.

### coreai.authoring.round_(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

RoundOp constructor.

### coreai.authoring.rsqrt(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

RsqrtOp constructor.

### coreai.authoring.scan(x: [Value](#coreai.authoring.Value) | ConstLike, axis: [Value](#coreai.authoring.Value) | ConstLike, reverse: [Value](#coreai.authoring.Value) | ConstLike, combiner: str = 'sum', , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ScanOp constructor.

### coreai.authoring.scatter_along_axis(output: [TensorType](#coreai.authoring.TensorType), x: [Value](#coreai.authoring.Value) | ConstLike, indices: [Value](#coreai.authoring.Value) | ConstLike, updates: [Value](#coreai.authoring.Value) | ConstLike, axis: [Value](#coreai.authoring.Value) | ConstLike, , scatter_mode: str | None = None, loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Scatter *updates* into *x* at *indices* along *axis*.

*scatter_mode* selects the combine mode (e.g. `"update"` (default),
`"add"`, `"sub"`, `"mul"`); `None` uses the op’s default.

### coreai.authoring.scatter_nd(x: [Value](#coreai.authoring.Value) | ConstLike, indices: [Value](#coreai.authoring.Value) | ConstLike, updates: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ScatterNdOp constructor.

### coreai.authoring.shrink_dims(x: [Value](#coreai.authoring.Value) | ConstLike, axes: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

ShrinkDimsOp constructor.

### coreai.authoring.sigmoid(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SigmoidOp constructor.

### coreai.authoring.silu(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SiluOp constructor.

### coreai.authoring.sin(x: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SinOp constructor.

Operation that accepts one tensor and returns sin(a) for
each element.

### coreai.authoring.sinh(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SinhOp constructor.

### coreai.authoring.slice_(x: [Value](#coreai.authoring.Value) | ConstLike, start_indices: [Value](#coreai.authoring.Value) | ConstLike, end_indices: [Value](#coreai.authoring.Value) | ConstLike, strides: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SliceOp constructor.

### coreai.authoring.slice_update(x: [Value](#coreai.authoring.Value) | ConstLike, start_indices: [Value](#coreai.authoring.Value) | ConstLike, end_indices: [Value](#coreai.authoring.Value) | ConstLike, strides: [Value](#coreai.authoring.Value) | ConstLike, update: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SliceUpdateOp constructor.

### coreai.authoring.softmax(x: [Value](#coreai.authoring.Value) | ConstLike, axis: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SoftmaxOp constructor.

### coreai.authoring.sort(x: [Value](#coreai.authoring.Value) | ConstLike, dim: [Value](#coreai.authoring.Value) | ConstLike, descending: [Value](#coreai.authoring.Value) | ConstLike, stable: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SortOp constructor.

### coreai.authoring.sparse_with_bitmask_to_dense(sparse_value: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SparseWithBitmaskToDenseOp constructor.

### coreai.authoring.split(x: [Value](#coreai.authoring.Value) | ConstLike, split_size_or_sections: [Value](#coreai.authoring.Value) | ConstLike, dim: [Value](#coreai.authoring.Value) | ConstLike, , result_types: Sequence[[TensorType](#coreai.authoring.TensorType)] | None = None, loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → list[[Value](#coreai.authoring.Value)]

SplitOp constructor.

*result_types*, if given, sets the output types explicitly via
`build_generic`, bypassing shape inference — needed when inference cannot
resolve dynamic-shape splits.

### coreai.authoring.sqrt(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SqrtOp constructor.

### coreai.authoring.sub(x: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, y: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SubOp constructor.

Operation that accepts two tensors of the same shape and
type and returns a - b for each corresponding element.

### coreai.authoring.sumpool2d(x: [Value](#coreai.authoring.Value) | ConstLike, kernel_size: [Value](#coreai.authoring.Value) | ConstLike, strides: [Value](#coreai.authoring.Value) | ConstLike, dilation: [Value](#coreai.authoring.Value) | ConstLike | None = None, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SumPool2dOp constructor.

### coreai.authoring.sumpool3d(x: [Value](#coreai.authoring.Value) | ConstLike, kernel_size: [Value](#coreai.authoring.Value) | ConstLike, strides: [Value](#coreai.authoring.Value) | ConstLike, dilation: [Value](#coreai.authoring.Value) | ConstLike | None = None, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SumPool3dOp constructor.

### coreai.authoring.symmetric_quant_stats(x: [Value](#coreai.authoring.Value) | ConstLike, value_range: [Value](#coreai.authoring.Value) | ConstLike, axis: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

SymmetricQuantStatsOp constructor.

### coreai.authoring.tan(x: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

TanOp constructor.

### coreai.authoring.tanh(x: [Value](#coreai.authoring.Value) | bool | int | float | range | list[bool | int | float] | ndarray[tuple[Any, ...], dtype[number[Any]]] | YUVFrame | Image, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

TanhOp constructor.

Applies element-wise hyperbolic tangent to the input tensor.

Computes tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) for each element.
The domain is all real numbers and the output range is (-1, 1). Follows
IEEE 754 semantics: tanh(0.0)=0.0, tanh(-x)=-tanh(x) (odd function),
tanh(+inf)=1.0, tanh(-inf)=-1.0, tanh(NaN)=NaN.

### coreai.authoring.tensor_to_image(result: [ImageType](#coreai.authoring.ImageType), source: [Value](#coreai.authoring.Value), , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

Encode *source* tensor into an image of type *result*.

### coreai.authoring.tile(x: [Value](#coreai.authoring.Value) | ConstLike, dims: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

TileOp constructor.

### coreai.authoring.transpose(x: [Value](#coreai.authoring.Value) | ConstLike, permutation: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

TransposeOp constructor.

### coreai.authoring.where(condition: [Value](#coreai.authoring.Value) | ConstLike, x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

WhereOp constructor.

### coreai.authoring.xor_(x: [Value](#coreai.authoring.Value) | ConstLike, y: [Value](#coreai.authoring.Value) | ConstLike, , loc: [DebugInfo](#coreai.authoring.DebugInfo) | None = None) → [Value](#coreai.authoring.Value)

XorOp constructor.
