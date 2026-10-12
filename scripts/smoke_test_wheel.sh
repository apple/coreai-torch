#!/usr/bin/env bash
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# Smoke-test a built wheel across supported Python versions.
#
# Installs the wheel into a fresh venv (runtime deps only, no [test]/[docs]
# extras) for each target Python version, then verifies the public API
# imports cleanly. Catches missing dependency declarations before publish.
#
# Usage:
#   ./scripts/smoke_test_wheel.sh                       # build + smoke test
#   ./scripts/smoke_test_wheel.sh --no-build            # skip build, use existing dist/*.whl
#   ./scripts/smoke_test_wheel.sh --python 3.11,3.12   # restrict versions
#   ./scripts/smoke_test_wheel.sh --help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Versions tested by default. pyproject.toml declares `requires-python = ">=3.11"`
# with no upper bound, so this list must be bumped when a new Python ships
# (e.g. when 3.15 final lands and torch + coreai-core have wheels for it).
# Override at invocation: --python 3.11
PYTHON_VERSIONS="3.11,3.12,3.13,3.14"
BUILD=true

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD=false; shift ;;
        --python)   PYTHON_VERSIONS="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

cd "$REPO_ROOT"

if $BUILD; then
    echo "==> Building wheel"
    rm -rf dist
    uv build --wheel
fi

WHEELS=( dist/coreai_torch-*.whl )
if [[ ! -e "${WHEELS[0]}" ]]; then
    echo "ERROR: No wheel found in dist/" >&2
    exit 1
fi
if [[ ${#WHEELS[@]} -gt 1 ]]; then
    echo "ERROR: Multiple wheels in dist/ — refusing to guess which to test:" >&2
    printf '  %s\n' "${WHEELS[@]}" >&2
    echo "Run 'rm -rf dist && uv build --wheel' (or pass --no-build after cleaning)." >&2
    exit 1
fi
WHEEL="${WHEELS[0]}"
echo "==> Smoke testing wheel: $WHEEL"

SMOKE_TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$SMOKE_TEST_ROOT"' EXIT

IFS=',' read -ra VERSIONS <<< "$PYTHON_VERSIONS"
FAILED=()

for PYVER in "${VERSIONS[@]}"; do
    VENV="$SMOKE_TEST_ROOT/py$PYVER"
    echo
    echo "==> [Python $PYVER] Creating clean venv"

    if ! uv venv --python "$PYVER" "$VENV"; then
        echo "FAIL [Python $PYVER]: could not create venv"
        FAILED+=("$PYVER (venv)")
        continue
    fi

    echo "==> [Python $PYVER] Installing wheel (runtime deps only)"
    if ! VIRTUAL_ENV="$VENV" uv pip install \
            --prerelease=allow \
            "$WHEEL"; then
        echo "FAIL [Python $PYVER]: install failed"
        FAILED+=("$PYVER (install)")
        continue
    fi

    echo "==> [Python $PYVER] Verifying imports"
    # Run from /tmp so the source tree at $REPO_ROOT isn't on sys.path —
    # we want to exercise the installed wheel, not the working copy.
    if ! (cd /tmp && "$VENV/bin/python" -c "
import importlib
import pkgutil

# 1. Explicit imports of the documented public surface. These are the names
#    users actually import — if any are renamed or removed, this fails loudly
#    with a 'cannot import name X from coreai_torch' traceback.
from coreai_torch import (
    __version__,
    ExternalizeSpec,
    MetalParameter,
    TorchConverter,
    TorchMetalKernel,
    generate_composite_decl,
    get_decomp_table,
)
from coreai_torch.composite_ops import (
    GatedDeltaUpdate,
    GatherMM,
    RMSNorm,
    RMSNormImpl,
    RoPE,
    SDPA,
)

# 2. Cross-check __all__ matches what's actually exported (catches __all__
#    drift from real attributes — e.g. a symbol listed but not bound).
import coreai_torch
for name in coreai_torch.__all__:
    assert hasattr(coreai_torch, name), f'missing public symbol: coreai_torch.{name}'

import coreai_torch.composite_ops as composite_ops
for name in composite_ops.__all__:
    assert hasattr(composite_ops, name), f'missing public symbol: composite_ops.{name}'

# 3. Walk every non-private submodule and import it. Catches dep-declaration
#    regressions in lazily-imported modules (e.g. debugging.graph_diff/networkx).
def _walk(package):
    for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + '.'):
        if any(part.startswith('_') for part in info.name.split('.')):
            continue
        importlib.import_module(info.name)

_walk(coreai_torch)

# 4. Construct the main entry point — catches init-time issues in
#    registries / pass-managers that pure imports can miss.
TorchConverter()

print(f'  coreai_torch=={__version__} from {coreai_torch.__file__}')
"); then
        echo "FAIL [Python $PYVER]: import failed"
        FAILED+=("$PYVER (import)")
        continue
    fi

    echo "PASS [Python $PYVER]"
done

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Smoke test FAILED on: ${FAILED[*]}"
    exit 1
fi
echo "Smoke test PASSED on: $PYTHON_VERSIONS"
