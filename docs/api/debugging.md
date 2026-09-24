# Debugging

Find NaN/infinity issues, compare model implementations, and analyze structural differences with the debugging module.

:::{note}
During the current preview, set the following environment variables to ensure
operation-level debug metadata is preserved and available to these tools:

```bash
export USE_LOCAL_COREAI=1
export ENABLE_DEBUG_INFO=1
```
:::

## Quick start

```python
from coreai_torch.debugging.validator import create_validator_for_exported_program

# Find NaN/inf issues in PyTorch models
model = MyModel().eval()
exported = torch.export.export(model, args=(torch.randn(1, 10),))

validator = create_validator_for_exported_program(exported)
result = await validator.check_for_nans(inputs=(torch.randn(1, 10),))

if result.failed_nodes:
    print(f"NaN detected at: {result.failed_nodes[0]}")
```

## Finding NaN/infinity issues

**Use when:** Your model produces NaN or infinity values and you need to find which operation caused the issue.

### PyTorch models

```python
from coreai_torch.debugging.validator import create_validator_for_exported_program

# Export your model
exported_program = torch.export.export(model, args=example_input)

# Create validator
validator = create_validator_for_exported_program(exported_program)

# Check for numerical issues
nan_result = await validator.check_for_nans(inputs=example_input)
inf_result = await validator.check_for_infs(inputs=example_input)

# Get first failing operation
if nan_result.failed_nodes:
    print(f"First NaN at: {nan_result.failed_nodes[0]}")
```

### Core AI programs

```python
from coreai_torch.debugging.validator import create_validator_for_coreai_program

# Convert to Core AI
converter = TorchConverter().add_exported_program(exported_program)
coreai_program = converter.to_coreai()

# Create validator
validator = await create_validator_for_coreai_program(coreai_program, "main")

# Check for issues
result = await validator.check_for_nans(inputs={"x": torch.randn(2, 4)})
```

## Comparing model implementations

**Use when:** You need to verify that PyTorch and Core AI models produce the same outputs after conversion.

### Cross-framework comparison

Compare PyTorch vs Core AI to verify conversion correctness:

```python
from coreai_torch.debugging.comparator import create_comparator_for_programs

# Create comparator between PyTorch and Core AI
comparator = await create_comparator_for_programs(
    source_program=exported_program,
    target_program=coreai_program,
    target_entry_point="main",
)

# Compare outputs with tolerance
result = await comparator.compare_with_tolerance(
    inputs={"x": example_input}, rtol=1e-5, atol=1e-8
)

# Check for differences
if result.failed_nodes:
    for source_op, target_op in result.failed_nodes:
        print(f"Mismatch: {source_op} vs {target_op}")
```


## Core AI inspector

**Use when:** You need to examine intermediate values from specific operations in a deployed Core AI model.

Capture intermediate values from deployed Core AI models:

```python
from coreai_torch.debugging.inspector import CoreAIInspector
from coreai.runtime import AIModel

# Load deployed Core AI model
asset_path = Path("my_model.aimodel")
ai_model = await AIModel.load(asset_path)

# Create inspector
inspector = CoreAIInspector(model=ai_model, function_name="main")

# Get operation IDs to inspect (from debug info)
coreai_op_ids = [1, 5, 10, 15]

# Capture intermediate values
results = await inspector.get_intermediates_for_ops(
    coreai_op_ids, inputs={"x": np.random.randn(2, 4).astype(np.float32)}
)

# Check results
for op_id, outputs in results.items():
    print(f"Op {op_id}: {len(outputs) if outputs else 0} outputs")
```

## Structural graph analysis

**Use when:** You want to understand how model structure changes between different versions or after optimization passes.

### Graph difference analysis

Analyze structural differences between model implementations using graph isomorphism:

```python
from coreai_torch.debugging.graph_diff import (
    compute_exported_program_diff,
    compute_coreai_program_diff,
    write_diff,
)

# Compare two PyTorch programs
source_program = torch.export.export(model_v1, example_input)
target_program = torch.export.export(model_v2, example_input)

diff = compute_exported_program_diff(source_program, target_program)

# Check structural compatibility
if diff.is_isomorphic:
    print("✓ Graphs have identical structure")
else:
    print(f"✗ Found {diff.summary.unmapped_source_node_count} structural differences")

    # Write detailed diff report to stdout
    write_diff(diff, diff.source_graph, diff.target_graph, max_items=20)
```

### Which layer changed

**Use when:** A structural diff told you *something* changed and you need to know **where** — which layer, and which line of your model.

`compute_changes` takes the same two programs and reports each differing operation
against the module instance and source line it came from, so a diff of thousands of
operations reads as a handful of layers:

```python
from coreai_torch.debugging.changes import compute_changes

result = compute_changes(before_program, after_program)

# Totals plus per-module counts, hotspot first.
result.write_summary()
```

```text
        9 change(s): 7 added, 0 removed, 2 modified
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Module                     ┃ Added ┃ Removed ┃ Modified ┃ Total ┃ Share ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ ExtraLayerModel$1/Linear$3 │     6 │       0 │        0 │     6 │   67% │
│ ExtraLayerModel$1/Linear$4 │     0 │       0 │        2 │     2 │   22% │
│ ExtraLayerModel$1          │     1 │       0 │        0 │     1 │   11% │
└────────────────────────────┴───────┴─────────┴──────────┴───────┴───────┘
```

Module names identify *instances* (`Linear$3`, not `Linear`), so two structurally
identical layers are told apart — which operation names cannot do.

Then drill into one layer once you know which to ask about:

```python
for change in result.changes_in("ExtraLayerModel$1/Linear$4"):
    where = change.attribution.source
    print(f"{change.change.value}: {change.op_name} at {where.filename}:{where.line}")
```

A module matches everything under it, so `"Block$2"` includes `"Block$2/Linear$1"`.

For a programmatic caller, `to_dict()` gives the same report as plain values — counts
first, then the module tree with each change nested under the module it happened in:

```python
import json

print(json.dumps(result.to_dict(), indent=2))
```

:::{note}
Attribution comes from the stack traces `TorchConverter.Mode.DEBUG` records, which is
the default mode. Converted with stack traces off, changes are still reported but every
one files under `<unknown>`.
:::

### Grouping any finding by module

`compute_changes` is one consumer of a general rollup. Any tool holding op ids can name
the layer they came from:

```python
from coreai_torch.debugging.modules import attributions_by_op_id, build_module_tree

attributions = attributions_by_op_id(program)
print(attributions[12].label)  # 'Model$1/Block$2/Linear$1'
print(attributions[12].source)  # LocationInfo(filename='model.py', line=41, col=8)

# Group your own findings the same way every report does.
tree = build_module_tree(
    (attributions[op_id].module, finding) for op_id, finding in my_findings.items()
)
for root in tree:
    print(root.name, len(root.all_items()))
```


## Performance profiling

**Use when:** You need to identify slow operations and performance bottlenecks in your Core AI model.

Profile operation timing in Core AI programs:

```python
from coreai_torch.debugging.benchmarker import benchmark_coreai_program

# Run benchmark
result = await benchmark_coreai_program(
    coreai_program=coreai_program, inputs={"x": torch.randn(2, 4)}, num_runs=50
)

# Show timing summary
result.write_summary(sys.stdout)

# Get module-level timing
module_timings = result.get_module_timings()
for name, module in module_timings.items():
    print(f"{name}: {module.aggregated_op_stats.average:.3f}ms avg")
```

## Every report groups the same way

`by_module()` is on the comparison reports too, so "which layer" is one call whatever
you were measuring — and always the same shape:

```python
# Which layer moved off a device
diff = await compare_compute_plans(before_program, after_program)
for node in diff.by_module():
    print(f"{node.name}: {len(node.all_items())} operation(s) moved")

# Which layer diverges numerically
report = await compare_program_intermediates(exported_program, program, inputs)
for node in report.by_module():
    print(f"{node.name}: {len(node.all_items())} failing")

# Which layer changed structurally
changes = compute_changes(before_program, after_program)
changes.write_summary()
```

Each entry also carries its own `module` path, so you can group or filter without the
tree:

```python
for entry in report.failures:
    print("/".join(entry.module), entry.max_diff)
```

:::{note}
`ComputePlanDiff.by_module()` covers *moves*, and `IntermediatesReport.by_module()`
covers *failures* — the entries each report exists to draw attention to. Additions and
removals are `compute_changes`' question, and a report over a real model is mostly
passing entries that would bury the handful that matter. Group `changes` or
`comparisons` yourself for the full picture.
:::

## Custom validation

**Use when:** You need to check for specific conditions beyond NaN/infinity (e.g., value ranges, specific patterns).

Create custom checks beyond NaN/infinity:

```python
def check_large_values(outputs):
    """Check if any output has values > threshold"""
    return any(abs(arr).max() > 1000.0 if arr is not None else False for arr in outputs)


# Use custom check
result = await validator.check(check_large_values, inputs=example_input)
```

## Configuration

### Search strategies

Choose how to search through operations:

```python
from coreai_torch.debugging.search_strategy import (
    ExhaustiveStrategy,
    LevelOrderStrategy,
)

# Exhaustive (default) - one batch, one model execution per side, reports every issue
strategy = ExhaustiveStrategy(graph)

# Bisection - narrows by depth to the first failing level
strategy = LevelOrderStrategy.bisection(graph, batch_size=10)

# Top-down (systematic from inputs to outputs)
strategy = LevelOrderStrategy.top_down(graph)

# Adaptive (automatically selects best approach)
strategy = LevelOrderStrategy.auto(graph)
```

Every batch a strategy yields costs a full model execution on *both* sides, so
narrowing only pays when capturing a value is more expensive than a run — very large
intermediates, or an early exit that skips most of the graph. Otherwise the default
`ExhaustiveStrategy` is both cheaper and more complete.

### Batch size
```python
# Control batch size for memory efficiency
strategy = LevelOrderStrategy.bisection(graph, batch_size=5)  # Smaller batches
strategy = LevelOrderStrategy.bisection(graph, batch_size=20)  # Larger batches
validator = create_validator_for_exported_program(exported)
```

## Torch utilities

**Use when:** You need to save intermediate values to disk for later analysis or share debug data.

### Saving intermediate values

Save all intermediate tensor values from PyTorch model execution:

```python
from coreai_torch.debugging.torch_utils import save_intermediates, load_intermediates
from pathlib import Path

# Export your PyTorch model
exported_program = torch.export.export(model, args=example_input)

# Save intermediate values to disk
metadata_path = save_intermediates(
    program=exported_program, inputs=example_input, output_dir=Path("./debug_output")
)

print(f"Intermediates saved to: {metadata_path}")
```

### Loading intermediate values

Load saved intermediate values for analysis:

```python
# Load intermediate values from disk
debug_trace = load_intermediates(Path("./debug_output/main.aimodelintermediates"))

# Access saved values
print(f"Inputs: {list(debug_trace.inputs.keys())}")
print(f"Outputs: {list(debug_trace.outputs.keys())}")
print(f"Intermediates: {len(debug_trace.intermediates)} operations")

# Analyze specific intermediate values
for node_name, tensor in debug_trace.intermediates.items():
    print(f"{node_name}: shape {tensor.shape}, mean {tensor.mean():.3f}")
```

### Custom value filtering

Filter which intermediate values to save:

```python
def custom_filter(node, result):
    """Only save convolution and linear layer outputs"""
    return any(op in str(node.target).lower() for op in ["conv", "linear", "matmul"])


# Save only filtered operations
metadata_path = save_intermediates(
    program=exported_program,
    inputs=example_input,
    output_dir=Path("./debug_output"),
    node_filter=custom_filter,
)
```

The debugging module provides tools for validating model correctness, analyzing structural changes, and identifying performance issues.

## See also

- {doc}`../guides/conversion-workflows` — understand the full pipeline from export to `DeployableProgram`.
- {doc}`supported-aten-ops` — check which ATen ops have built-in lowering rules if you hit a conversion error.
- {doc}`TorchConverter` — the main conversion class; `to_coreai()` produces the program you validate here.
