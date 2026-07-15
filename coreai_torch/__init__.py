# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""coreai-torch: Convert PyTorch models to Core AI format."""

import warnings as _warnings

# Re-export MetalParameter so users don't need a separate coreai import.
from coreai.authoring import MetalParameter
from packaging.version import Version as _Version
from torch import __version__ as _torch_version

from .__version__ import __version__
from ._composite_declaration import generate_composite_decl
from ._decomp import get_decomp_table
from ._torch_metal_kernel import TorchMetalKernel
from .converter import TorchConverter
from .externalize import ExternalizeSpec

__all__ = [
    "__version__",
    "ExternalizeSpec",
    "MetalParameter",
    "TorchConverter",
    "TorchMetalKernel",
    "get_decomp_table",
    "generate_composite_decl",
]

_TORCH_MAX_VERSION = "2.13.0"

if _Version(_torch_version) > _Version(_TORCH_MAX_VERSION):
    _warnings.warn(
        f"coreai-torch has only been validated with torch<={_TORCH_MAX_VERSION}; "
        f"found torch {_torch_version}. Some functionality may not work as expected.",
        stacklevel=2,
    )
