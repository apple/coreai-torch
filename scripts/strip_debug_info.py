# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Command line tool to strip debug info from a ``.aimodel`` asset.

Usage::

    python strip_debug_info.py model.aimodel                  # overwrite in place
    python strip_debug_info.py model.aimodel -o stripped.aimodel
    python strip_debug_info.py model.aimodel --no-op-ids      # all locations unknown
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coreai_torch.debugging.debug_info import strip_debug_info_from_aimodel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="Input .aimodel asset.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .aimodel path (default: overwrite the input).",
    )
    parser.add_argument(
        "--no-op-ids",
        action="store_true",
        help="Replace every location with a plain unknown location (no op IDs).",
    )
    args = parser.parse_args()
    saved = strip_debug_info_from_aimodel(
        args.path, output_path=args.output, include_op_ids=not args.no_op_ids
    )
    print(f"Saved stripped asset to {saved}")


if __name__ == "__main__":
    main()
