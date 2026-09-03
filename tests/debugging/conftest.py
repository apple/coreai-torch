# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test configuration for the debugging suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from ..utils import _get_test_specialization_options

if TYPE_CHECKING:
    from coreai.runtime import SpecializationOptions

# Verified non-degenerate for the models in `test_model.py` -- see the fixture for what
# a degenerate draw does and how to recognise one.
_SEED = 0


@pytest.fixture(autouse=True)
def seed_example_inputs() -> None:
    """
    Draw the same example inputs every run, so a failure here can be reproduced.
    """
    torch.manual_seed(_SEED)


@pytest.fixture
def specialization_options() -> "SpecializationOptions | None":
    """
    The options for whatever ``--compute-unit-kind`` the run selected.

    ``None`` under ``interpreter``, which is what the bundled runtime wants, so the
    debug flag can only be set once there is something to set it on.
    """
    options = _get_test_specialization_options()
    return None if options is None else options.with_debug(enabled=True)
