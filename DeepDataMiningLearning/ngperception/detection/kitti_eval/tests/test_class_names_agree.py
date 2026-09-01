"""The class index -> name maps in eval.py have to agree with the filter.

`clean_data` decides which annotations belong to a class index, from

    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']

and two other maps in the same file turn indices into the names that appear in
the results and accept class names from callers: one in
`get_official_eval_result`, one in `get_coco_eval_result`. If those disagree
with `CLASS_NAMES`, a result is filed under a class it was not measured on, and
a caller naming a class gets a different one -- or a KeyError.

Reads the source with `ast` rather than importing it: eval.py compiles its
kernels with `@numba.jit` at import, and this makes no claim about numba.
Standard library only, no dataset, no GPU.

    pytest DeepDataMiningLearning/ngperception/detection/kitti_eval/tests
"""
import ast
import io
from pathlib import Path

import pytest

EVAL_PY = Path(__file__).resolve().parents[1] / "eval.py"


def _tree():
    return ast.parse(io.open(EVAL_PY, encoding="utf-8", errors="replace").read())


def _class_names():
    """CLASS_NAMES as assigned inside clean_data."""
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == "clean_data":
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign)
                        and getattr(stmt.targets[0], "id", None) == "CLASS_NAMES"):
                    return ast.literal_eval(stmt.value)
    pytest.fail("clean_data no longer assigns CLASS_NAMES")


def _index_maps():
    """Every `class_to_name = {...}` literal, keyed by the function holding it."""
    maps = {}
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.FunctionDef):
            continue
        for stmt in ast.walk(node):
            if (isinstance(stmt, ast.Assign)
                    and getattr(stmt.targets[0], "id", None) == "class_to_name"
                    and isinstance(stmt.value, ast.Dict)):
                try:
                    maps[node.name] = ast.literal_eval(stmt.value)
                except ValueError:
                    pass
    return maps


def test_the_file_is_where_it_is_expected():
    assert EVAL_PY.is_file()


def test_clean_data_still_defines_the_filter_list():
    names = _class_names()
    assert names[:3] == ["car", "pedestrian", "cyclist"]
    assert len(names) == 6


def test_both_index_maps_were_found():
    maps = _index_maps()
    assert "get_official_eval_result" in maps
    assert "get_coco_eval_result" in maps


@pytest.mark.parametrize("func", ["get_official_eval_result", "get_coco_eval_result"])
def test_every_index_names_the_class_clean_data_filters_for(func):
    names = _class_names()
    mapping = _index_maps()[func]
    mismatches = [
        f"{i}: filter={names[i]!r} label={label!r}"
        for i, label in sorted(mapping.items())
        if i < len(names) and label.lower() != names[i]
    ]
    assert mismatches == []


def test_the_two_maps_agree_with_each_other():
    a = _index_maps()["get_official_eval_result"]
    b = _index_maps()["get_coco_eval_result"]
    shared = set(a) & set(b)
    assert shared, "the two maps share no index"
    assert [i for i in sorted(shared) if a[i] != b[i]] == []


@pytest.mark.parametrize("func", ["get_official_eval_result", "get_coco_eval_result"])
def test_a_caller_can_name_every_class_the_filter_accepts(func):
    """`name_to_class` is built by inverting the map; a caller passes a name."""
    names = _class_names()
    mapping = _index_maps()[func]
    name_to_class = {v: k for k, v in mapping.items()}
    for i in sorted(mapping):
        if i >= len(names):
            continue
        assert any(k.lower() == names[i] for k in name_to_class), (
            f"no caller-facing name resolves to {names[i]!r} (index {i})"
        )


def test_no_index_map_carries_a_class_kitti_does_not_have():
    """`Sign` is a Waymo class. waymo2kitti_async.py:87 says so: 'not in kitti'."""
    names = {n.lower() for n in _class_names()}
    strays = []
    for func, mapping in _index_maps().items():
        for i, label in sorted(mapping.items()):
            if i < len(_class_names()) and label.lower() not in names:
                strays.append(f"{func}[{i}] = {label!r}")
    assert strays == []


def test_the_inverted_map_has_no_duplicate_names():
    """Two indices sharing a name would make the caller's choice ambiguous."""
    for func, mapping in _index_maps().items():
        labels = [v for _, v in sorted(mapping.items())]
        assert len(labels) == len(set(labels)), f"{func} repeats a name"
