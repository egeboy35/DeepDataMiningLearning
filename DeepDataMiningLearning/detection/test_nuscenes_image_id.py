"""The COCO image id built for a nuScenes sample has to survive a new process.

`get_target` puts an image id on every target, and `cocoevaluator.py` keys both
the ground truth (`convert_to_coco_api`, line 300) and the predictions
(`res[image_id] = out`, line 370) on it. Two properties matter and neither is
about the model: the id must not change between runs, and two samples must not
share one.

No dataset download, no GPU. `cv2` is stubbed only if it is genuinely missing,
because dataset_nuscenes imports it at module scope and none of the code under
test touches it -- the stub is reported by test_cv2_stub_is_declared so a run
that used one cannot be mistaken for one that did not.

    pytest DeepDataMiningLearning/detection/test_nuscenes_image_id.py
"""
import os
import subprocess
import sys
import types

import pytest

# --------------------------------------------------------------- import setup
_CV2_STUBBED = False
try:                                             # pragma: no cover - env probe
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")
    _CV2_STUBBED = True

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from DeepDataMiningLearning.detection.dataset_nuscenes import (  # noqa: E402
    stable_image_id,
)

# A placeholder in the shape of a nuScenes sample token: 32 lowercase hex.
# Deliberately low-entropy -- a realistic-looking random token reads as a
# credential to secret scanners, and nothing here depends on its entropy.
TOKEN = "0" * 31 + "1"
JS_SAFE_MAX = 2 ** 53 - 1


def _tokens(n, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    return [rng.bytes(16).hex() for _ in range(n)]


def test_cv2_stub_is_declared():
    """Fails nothing; records in the report whether cv2 was real."""
    assert _CV2_STUBBED in (True, False)
    if _CV2_STUBBED:
        print("\n[note] cv2 was absent and stubbed; it is not on any path under test")


# ------------------------------------------------------------- determinism
def test_the_same_token_gives_the_same_id_within_a_process():
    assert stable_image_id(TOKEN) == stable_image_id(TOKEN)


def test_the_same_token_gives_the_same_id_in_a_fresh_interpreter():
    """The property `hash()` does not have: PEP 456 salts str hashing."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import types; sys.modules.setdefault('cv2', types.ModuleType('cv2'))\n"
        "from DeepDataMiningLearning.detection.dataset_nuscenes import stable_image_id\n"
        # the module prints environment warnings on import, so tag the answer
        "print('ID=%%d' %% stable_image_id(%r))" % (_REPO, TOKEN)
    )
    seen = set()
    for salt in ("0", "1", "2", "random", "random", "random"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, encoding="utf-8")
        assert r.returncode == 0, r.stderr
        tagged = [l for l in r.stdout.splitlines() if l.startswith("ID=")]
        assert len(tagged) == 1, f"expected one tagged line, got {r.stdout!r}"
        seen.add(tagged[0][3:])
    assert len(seen) == 1, f"id moved between processes: {seen}"
    assert int(seen.pop()) == stable_image_id(TOKEN)


def test_the_builtin_hash_really_is_unstable_here():
    """Guards the premise: if hash() were stable this change would be pointless."""
    seen = set()
    for salt in ("0", "1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        r = subprocess.run([sys.executable, "-c", f"print(hash({TOKEN!r}) % 1000000)"],
                           capture_output=True, text=True, env=env, encoding="utf-8")
        seen.add(r.stdout.strip())
    assert len(seen) > 1, "hash() was stable across seeds; the premise needs rechecking"


# -------------------------------------------------------------- collisions
@pytest.mark.parametrize("n", [1_000, 34_149])
def test_no_collisions_at_nuscenes_scale(n):
    """34,149 is the keyframe count of nuScenes v1.0-trainval."""
    ids = [stable_image_id(t) for t in _tokens(n, seed=n)]
    assert len(set(ids)) == n


def test_the_previous_id_function_would_have_collided():
    """The comparison the change is justified by, measured rather than asserted."""
    import zlib
    n = 34_149
    toks = _tokens(n, seed=n)
    old = {zlib.crc32(t.encode()) % 1_000_000 for t in toks}   # stands in for hash()%1e6
    new = {stable_image_id(t) for t in toks}
    assert len(old) < n, "the 1e6 range collided in no sample; recheck the range"
    assert len(new) == n


# ------------------------------------------------------------- id shape
def test_ids_are_exact_javascript_integers():
    """COCO JSON is routinely read by JS tooling, which loses precision past 2^53."""
    for t in _tokens(2_000, seed=1):
        i = stable_image_id(t)
        assert 0 <= i <= JS_SAFE_MAX


def test_ids_are_plain_python_ints():
    assert isinstance(stable_image_id(TOKEN), int)
    assert not isinstance(stable_image_id(TOKEN), bool)


def test_non_hex_and_unicode_tokens_are_accepted():
    """The simplified dataset layout does not promise hex tokens."""
    for t in ("sample-0001", "", "ünïcode-token", "0" * 64):
        assert isinstance(stable_image_id(t), int)


def test_different_tokens_give_different_ids():
    assert stable_image_id("a") != stable_image_id("b")


# --------------------------------------------------- the call site uses it
def test_get_target_builds_its_image_id_from_this_function():
    """A regression guard on the line this change exists for."""
    import ast
    import inspect
    from DeepDataMiningLearning.detection import dataset_nuscenes

    src = inspect.getsource(dataset_nuscenes)
    tree = ast.parse(src)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "stable_image_id" in calls
    # and nothing in the module builds an id out of the salted builtin any more
    assert "hash(sample_token)" not in src
