# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""The two ways a pass could destroy an index it cannot justify, and the guard
that now stops both:

* A walk that cannot list a directory. `os.walk` calls `onerror` and otherwise
  continues as if the directory were empty, so its result used to look exactly
  like a genuinely empty root and a build pruned the index to nothing. `_walk`
  now reports every unlistable directory; a build refuses a partial walk and
  `index_state()` surfaces the failure instead of reporting `ready`. An
  empty-but-successful walk over an empty root is still a valid corpus.
* A rebuild driven by a subset manifest. The prune guard used to run only on the
  incremental path, so `build_index(rebuild=True)` with a one-row manifest cut a
  three-document index to one. The guard now runs on both paths, applies before
  the store is cleared, and leaves the store byte-identical unless the operator
  raises `SILICA_INVENTORY_MAX_DROP`; the refusal carries the index state so a
  search that triggered it still serves the existing store.

The walk stays the authority whenever it can run: a complete manifest still
prunes and indexes normally.
"""
from __future__ import annotations

import json
import os

import pytest


@pytest.fixture
def root(tmp_path, monkeypatch):
    from silica_core.config import CONFIG
    from silica_core.kernel.recall import paths

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "lsm.md").write_text(
        "# LSM trees\n\n## 1 Write path\n\nWrites land in a memtable.\n\n"
        "## 2 Compaction\n\nCompaction merges sorted runs; leveled compaction bounds read amplification.\n",
        encoding="utf-8")
    (tmp_path / "docs" / "btree.md").write_text("# B-trees\n\nA B-tree keeps pages sorted.\n",
                                                encoding="utf-8")
    (tmp_path / "docs" / "ctree.md").write_text("# C-trees\n\nA C-tree bounds write amplification.\n",
                                                encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(paths, "index_dir_for",
                        lambda vault, _d=tmp_path.parent / (tmp_path.name + "-idx"): _d)
    import silica_core.core as core
    core._section_cache.clear()
    return core


@pytest.fixture
def empty(tmp_path, monkeypatch):
    """A genuinely empty root: a walk that succeeds over nothing."""
    from silica_core.config import CONFIG
    from silica_core.kernel.recall import paths

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(paths, "index_dir_for",
                        lambda vault, _d=tmp_path.parent / (tmp_path.name + "-idx"): _d)
    import silica_core.core as core
    core._section_cache.clear()
    return core


def _unlistable(top, onerror=None, **kwargs):
    """A stand-in for `os.walk` that lists nothing and reports the directory as
    unlistable — a permission failure or a dead mount, from `_walk`'s side."""
    if onerror is not None:
        onerror(PermissionError(13, "Permission denied", str(top)))
    return
    yield  # pragma: no cover - makes this a generator, like os.walk


def _boom(*_a, **_k):
    """A `_walk` that raises outright: the walk cannot run at all."""
    raise RuntimeError("the walk cannot run")


def _manifest(core):
    """A manifest describing the root as a real walk sees it. `SILICA_INVENTORY`
    must be unset when this is called."""
    stamps, unextracted = core._inventory()
    return {"version": 1, "root": str(core._root()), "count": len(stamps),
            "stamps": stamps, "unextracted": list(unextracted)}


def _write(tmp_path, data):
    p = tmp_path / "inventory.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _index_bytes(core):
    idx = core._paths.index_dir()
    return {str(p.relative_to(idx)): p.read_bytes()
            for p in sorted(idx.rglob("*")) if p.is_file()}


def _subset(core):
    stamps = core._load_meta()["stamps"]
    return {"version": 1, "root": str(core._root()), "count": 1,
            "stamps": {"docs/btree.md": stamps["docs/btree.md"]}}


# ---------------------------------------------------------------------------
# a. a walk that cannot list a directory is not an empty corpus
# ---------------------------------------------------------------------------

def test_a_walk_that_cannot_list_a_directory_refuses_and_keeps_the_index(root, monkeypatch):
    root.build_index()
    assert root.index_state()["docs"] == 3
    before = _index_bytes(root)

    monkeypatch.setattr(os, "walk", _unlistable)
    out = root.build_index()

    assert out["error"]["code"] == "inventory_walk_failed"
    assert out["failures"], "the unlistable directory is named"
    assert _index_bytes(root) == before, "a failed walk writes nothing"
    assert root._load_meta()["stamps"], "no stored row was pruned"

    state = root.index_state()
    assert state["state"] != "ready", "a failed walk is never reported ready"
    assert state["error"]["code"] == "inventory_walk_failed"
    assert state["docs"] == 3


def test_an_empty_root_walks_successfully_and_is_not_a_failure(empty):
    out = empty.build_index()
    assert "error" not in out and out["docs"] == 0

    state = empty.index_state()
    assert "error" not in state
    assert state["state"] == "ready" and state["docs"] == 0


# ---------------------------------------------------------------------------
# b. the drop guard runs on the rebuild path too
# ---------------------------------------------------------------------------

def test_a_rebuild_with_a_subset_manifest_leaves_the_store_byte_identical(root, tmp_path, monkeypatch):
    """A one-row manifest and a rebuild used to cut three documents to one. The
    walk is unavailable, so the guard is the only thing between the manifest and
    the store: past the threshold the pass aborts with nothing written."""
    root.build_index()
    assert root.index_state()["docs"] == 3
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, _subset(root))))
    monkeypatch.setattr(root, "_walk", _boom)
    before = _index_bytes(root)

    out = root.build_index(rebuild=True)

    assert out["error"]["code"] == "inventory_prune_refused"
    assert _index_bytes(root) == before, "the store is byte-identical"
    assert root._load_meta()["stamps"], "the stored rows survive the refused rebuild"


def test_the_refusal_carries_the_index_state(root, tmp_path, monkeypatch):
    """The error hands back the index state, so a search that triggered the
    build serves the store it already has instead of failing."""
    root.build_index()
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, _subset(root))))
    monkeypatch.setattr(root, "_walk", _boom)

    out = root.build_index(rebuild=True)

    assert out["error"]["code"] == "inventory_prune_refused"
    assert out["index"]["docs"] == 3 and out["index"]["state"] in {"stale", "ready"}
    assert out["removed"] == 2 and out["stored"] == 3


def test_a_raised_threshold_honours_the_smaller_manifest_on_a_rebuild(root, tmp_path, monkeypatch):
    """An operator who knows the corpus shrank while the share is down says so
    explicitly; the rebuild then follows the smaller manifest."""
    root.build_index()
    stamps = root._load_meta()["stamps"]
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, _subset(root))))
    monkeypatch.setattr(root, "_walk", _boom)
    monkeypatch.setenv("SILICA_INVENTORY_MAX_DROP", "1.0")

    out = root.build_index(rebuild=True)

    assert "error" not in out
    assert root._load_meta()["stamps"] == {"docs/btree.md": stamps["docs/btree.md"]}
    assert root.index_state()["docs"] == 1


# ---------------------------------------------------------------------------
# c. the intended path is intact: the walk stays the authority
# ---------------------------------------------------------------------------

def test_a_complete_manifest_still_prunes_and_indexes_normally(root, tmp_path, monkeypatch):
    """A manifest that matches the walk is not refused: it prunes the row the
    corpus really lost and indexes what remains."""
    root.build_index()
    (root._root() / "docs" / "ctree.md").unlink()
    data = _manifest(root)  # the current root: two rows
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, data)))

    out = root.build_index()

    assert "error" not in out
    stamps = root._load_meta()["stamps"]
    assert set(stamps) == {"docs/lsm.md", "docs/btree.md"}
    assert root.index_state()["docs"] == 2
