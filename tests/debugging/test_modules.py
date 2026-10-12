# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Grouping findings by module instance.

`build_module_tree` is a pure function of its input, so these need nothing converted
and nothing compiled -- which is why the tree is tested here rather than only through
the reports that use it.
"""

from __future__ import annotations

import json

from coreai_torch.debugging.modules import (
    UNATTRIBUTED,
    ModuleNode,
    build_module_tree,
)


def test_nests_by_path() -> None:
    """A path becomes a chain of nodes, one per frame."""
    roots = build_module_tree(
        [
            (("Model$1", "Block$1", "Linear$1"), "a"),
            (("Model$1", "Block$1"), "b"),
        ]
    )

    assert [node.name for node in roots] == ["Model$1"]
    model = roots[0]
    assert [child.name for child in model.children] == ["Block$1"]
    block = model.children[0]
    assert block.items == ["b"]
    assert [child.name for child in block.children] == ["Linear$1"]
    assert block.children[0].items == ["a"]


def test_items_are_own_not_inherited() -> None:
    """A finding is filed against its own module, and counted by its ancestors."""
    roots = build_module_tree(
        [
            (("Model$1", "Block$1"), "a"),
            (("Model$1", "Block$2"), "b"),
            (("Model$1", "Block$2"), "c"),
        ]
    )

    model = roots[0]
    assert model.items == []
    assert len(model.all_items()) == 3
    assert sorted(child.name for child in model.children) == ["Block$1", "Block$2"]


def test_ancestors_exist_without_items() -> None:
    """
    A module nothing is filed against still appears.

    The honest reading of a module whose work always fuses with a sibling's: the
    structure is there and nothing is attributed to it alone.
    """
    roots = build_module_tree([(("Model$1", "Block$1", "Linear$1"), "a")])

    model = roots[0]
    assert model.items == []
    assert model.children[0].items == []
    assert model.children[0].children[0].items == ["a"]


def test_ordered_by_weight_then_name() -> None:
    """The branch holding the most is first, and ties break on name so runs agree."""
    roots = build_module_tree(
        [
            (("Small$1",), "a"),
            (("Big$1",), "b"),
            (("Big$1",), "c"),
            (("Also$1",), "d"),
        ]
    )

    assert [node.name for node in roots] == ["Big$1", "Also$1", "Small$1"]


def test_empty_path_is_bucketed() -> None:
    """A finding with no module path lands in one named bucket, not at the root."""
    roots = build_module_tree([((), "a"), (("Model$1",), "b")])

    assert {node.name for node in roots} == {UNATTRIBUTED, "Model$1"}


def test_type_and_instance_split() -> None:
    """A frame names an instance; the type is what groups instances together."""
    node: ModuleNode[str] = ModuleNode(name="Linear$3")
    assert (node.type_name, node.instance) == ("Linear", 3)

    unnumbered: ModuleNode[str] = ModuleNode(name=UNATTRIBUTED)
    assert unnumbered.instance is None


def test_find_by_name_and_by_path() -> None:
    """A bare name finds any instance; a path finds the one under that parent."""
    roots = build_module_tree(
        [
            (("Model$1", "Block$1", "Linear$1"), "a"),
            (("Model$1", "Block$2", "Linear$1"), "b"),
        ]
    )
    model = roots[0]

    assert model.find("Block$2") is not None
    assert model.find("Block$2/Linear$1") is not None
    assert model.find("Block$2/Linear$1").items == ["b"]  # type: ignore[union-attr]
    assert model.find("Block$9") is None


def test_to_dict_is_json_serialisable() -> None:
    """A tree must survive `json.dumps`, which is how a caller not in Python reads it."""
    roots = build_module_tree(
        [
            (("Model$1", "Block$1"), 1),
            (("Model$1", "Block$1"), 2),
        ]
    )

    plain = json.loads(json.dumps(roots[0].to_dict(lambda item: item)))
    assert plain["name"] == "Model$1"
    assert plain["type_name"] == "Model"
    assert plain["instance"] == 1
    # The count is the subtree's, so a reader sees the branch total without descending.
    assert plain["count"] == 2
    assert plain["items"] == []
    assert plain["children"][0]["items"] == [1, 2]
