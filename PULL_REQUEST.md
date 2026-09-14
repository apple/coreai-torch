# PR: Fused MSL Sub-byte INT4 Quantization Kernels, Composite Ops Lowering, and LayerNorm+GELU Graph Fusion Pass

**Target Repository**: `apple/coreai-torch` (from fork `stefanutc1/coreai-torch`)  
**Branch**: `feat/metal-kernel-fusion-and-composite-ops`  
**Related Components**: `coreai_torch.kernels`, `coreai_torch.composite_ops`, `coreai_torch._aten_to_core`, `coreai_torch.passes`

---

## 1. Summary of Changes

This pull request introduces three major performance, lower-level IR lowering, and compiler pass enhancements to `coreai-torch` specifically engineered to maximize throughput and minimize unified memory bandwidth consumption on Apple Silicon (M-series) hardware:

1. **Optimized Sub-byte Quantization Metal Kernels (`coreai_torch/kernels/quantization.py`)**:
   - Implemented high-performance Metal Shading Language (MSL) custom kernels for 4-bit integer quantization (`fused_quantize_int4_kernel`) and dequantization (`fused_dequantize_int4_kernel`).
   - Bit-packing and unpacking operations (`>> 4`, `& 0x0F`, branchless sign-extension `(val ^ 0x08) - 0x08`) and affine scale/zero-point transformations are fused entirely in GPU thread registers.
   - Eliminates intermediate tensor allocations, reducing UMA traffic by 8× compared to FP32 and 4× compared to FP16.

2. **Composite Ops & Native Core AI IR Lowering (`coreai_torch/composite_ops/_swiglu.py`, `coreai_torch/_aten_to_core.py`, `coreai_torch/_decomp.py`)**:
   - Added `SwiGLU` and `SwiGLUImpl` composite operators conforming to modern transformer architectures (LLaMA 3/4, Mistral, Gemma 3, Qwen).
   - Added native ATen lowerings for `torch.ops.aten.glu.default`, `torch.ops.aten.softplus.default`, `torch.ops.aten.mish.default`, and `torch.ops.aten.elu.default` into the Core AI MLIR dialect (`coreai.glu`, `coreai.softplus`, `coreai.mish`, `coreai.elu`).
   - Excluded these ops from decomposition tables (`_decomp.py`), retaining high-level structural semantics and eliminating expensive CPU fallbacks during PyTorch export.

3. **Execution Graph Optimization (LayerNorm + GELU Fusion Pass) (`coreai_torch/passes/fusion.py`)**:
   - Implemented an FX graph fusion pass `fuse_layernorm_gelu` that scans exported models for adjacent normalization and activation patterns.
   - Paired with a fused MSL kernel `fused_layernorm_gelu_kernel` utilizing two-pass reduction and threadgroup shared memory to compute mean, variance, affine transform, and GELU activation in a single dispatch.
   - Eliminates the round-trip memory read/write barrier between LayerNorm output and GELU input, cutting memory traffic for normalization-activation blocks by ~50%.

---

## 2. Motivation & Apple Silicon Architecture Impact

Apple Silicon chips (M1/M2/M3/M4) rely on a Unified Memory Architecture (UMA) shared between CPU, GPU, and the Neural Engine (NPU). While UMA offers extraordinary bandwidth (up to 800+ GB/s on Max/Ultra configurations), deep learning inference and training at low batch sizes (batch size = 1 to 8) remain strictly **memory bandwidth bound**.

### Memory Round-Trip Bottlenecks
In conventional multi-kernel pipelines:
1. **LayerNorm** reads the activations from unified memory, calculates mean/variance, applies scale/bias, and writes the normalized tensor back to unified memory.
2. **GELU** subsequently reads that normalized tensor from memory, computes the activation function, and writes the output back to unified memory.

By fusing LayerNorm and GELU into a single MSL kernel and rewriting the FX execution graph:
- Activations remain inside the GPU's register file and threadgroup cache.
- Eliminates 1 intermediate memory write and 1 intermediate memory read.
- Decreases memory bandwidth pressure, lowers thermal throttling, and noticeably reduces latency in transformer models.

### Sub-byte INT4 Quantization Efficiencies
Standard PyTorch implementations often convert INT4 tensors through multiple unpack and cast operations, materializing temporary FP16/FP32 arrays. Our custom MSL kernel:
- Decodes two 4-bit nibbles per byte in SIMD vector lanes.
- Directly executes fused multiply-add ($x_{fp} = (x_{int4} - zp) \times scale$) in registers.
- Enables memory-bandwidth-bound LLM decoding kernels to run near theoretical peak memory transfer speeds.

---

## 3. Technical Implementation Details

### A. Sub-byte Quantization (`coreai_torch/kernels/quantization.py`)
- Vectorized MSL string template utilizing Core AI's `TorchMetalKernel` abstraction.
- Handles bit extraction without conditionals to prevent thread divergence within 32-wide SIMD execution groups (Apple GPU warps):
  ```metal
  int raw_low = static_cast<int>(packed_val & 0x0F);
  int raw_high = static_cast<int>((packed_val >> 4) & 0x0F);
  int low_val = (raw_low ^ 0x08) - 0x08;
  int high_val = (raw_high ^ 0x08) - 0x08;
  ```
- Exposes `dequantize_int4_metal` and `quantize_int4_metal` for explicit model building, alongside PyTorch fake tensors for tracing and export.

### B. Composite Operators & Lowering (`coreai_torch/composite_ops/_swiglu.py`)
- Implements `SwiGLU(dim, dim_out, bias)` and `SwiGLUImpl` with dual invocation semantics (single interleaved tensor vs. split gate and value tensors).
- In `coreai_torch/_decomp.py`, added:
  - `torch.ops.aten.glu.default`
  - `torch.ops.aten.softplus.default`
  - `torch.ops.aten.mish.default`
  - `torch.ops.aten.elu.default`
  to `_COMPOSITE_OPS`.
- In `coreai_torch/_aten_to_core.py`, added handlers:
  - `replace_glu`: extracts split dimension and maps directly to `coreai.glu`.
  - `replace_softplus`: converts `beta` and `threshold` attributes to `coreai.softplus`.
  - `replace_mish`: lowers to `coreai.mish`.
  - `replace_elu`: maps `alpha`, `scale`, and `input_scale` attributes to `coreai.elu`.

### C. Graph Fusion Pass (`coreai_torch/passes/fusion.py`)
- Detects the pattern:
  $$\text{Input} \longrightarrow \text{aten.native\_layer\_norm / aten.layer\_norm} \longrightarrow \text{aten.gelu} \longrightarrow \dots$$
- Validates that the LayerNorm output is consumed solely by GELU (or safely substitutes references).
- Replaces the subgraph with a single call to `fused_layernorm_gelu_kernel.torch_custom_op(x, weight, bias, eps)`.
- Eliminates dead nodes and recompiles the FX graph.

---

## 4. Verification & Testing

All code adheres strictly to PEP 8, formatted and linted with `ruff`:
- `python -m ruff check coreai_torch tests` $\rightarrow$ **All checks passed! (0 errors)**
- `python -m ruff format --check coreai_torch tests` $\rightarrow$ **126 files inspected, 0 formatting issues**

### Added Tests:
1. `tests/dsl/test_quantize_metal_kernel.py`:
   - Validates INT4 quant/dequant roundtrip accuracy, negative number representation, zero-point alignment, and MSL source code generation.
2. `tests/composite_ops/test_swiglu.py`:
   - Validates single-tensor and dual-tensor inputs against reference `F.silu(gate) * val`.
   - Validates eager vs. `torch.export.export` parity with static and dynamic shapes.
   - Validates numerical parity against MLX (`mlx.nn.silu`) on macOS.
3. `tests/api/test_glu_lowering.py`:
   - Validates resolver registrations for `glu`, `softplus`, `mish`, and `elu`.
   - Validates preservation in FX graph when using `get_decomp_table()`.
4. `tests/passes/test_fusion_pass.py`:
   - Validates LayerNorm + GELU pattern recognition, node replacement, and dead code elimination.
   - Verifies that non-adjacent or interleaved nodes are preserved without unintended mutations.

---

## 5. Checklist

- [x] Code adheres to repository style guidelines (`ruff check` and `ruff format` passed).
- [x] All new public functions and classes include full type annotations and Google-style docstrings.
- [x] Tested with unit tests for each component.
- [x] Compatible with PyTorch 2.5+ export pipeline and Core AI dialect specifications.
- [x] No breaking changes to existing public APIs (`coreai_torch.TorchConverter`, `coreai_torch.get_decomp_table`).
