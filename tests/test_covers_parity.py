# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""Manifest coverage is a claim the manifest has to earn, and the build reply
carries the `inventory` block TOOLS.md documents.

Two faults this pins:

* `_manifest_covers` used to check only that `sizes` (and `mtimes` for `since=`)
  named every stamp. A one-row manifest passed that and `silica_files` served
  its single row as if it were the corpus, never walking, so every other
  document vanished from the listing with no error. Coverage now also requires
  the manifest to account for the rows the index already holds, up to the prune
  guard's `SILICA_INVENTORY_MAX_DROP` ceiling — a manifest that omits more of
  them than that is a subset, not the corpus, and the walk answers.
* A successful `build_index` reply built its `index` block by hand and left out
  `inventory`, which TOOLS.md requires on every `index` block. It is added, and
  no other key is reshaped.

The intended path is unchanged: a complete manifest still serves the listing
with no traversal, and with no manifest nothing changes.
"""
from __future__ import annotations

import json

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
    (tmp_path / "docs" / "report.docx").write_bytes(b"PK\3\4not really a docx")
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=tmp_path.parent / (tmp_path.name + "-idx"): _d)
    import silica_core.core as core
    core._section_cache.clear()
    return core


class _Walked(Exception):
    """Raised by the rigged `_walk`: seeing it proves the walk ran."""


def _boom(*_a, **_k):
    raise _Walked("the walk ran")


def _covering(root):
    """A manifest describing the root the way a real walk does, with the sizes
    and mtimes a listing needs. `SILICA_INVENTORY` must be unset when called."""
    stamps, unextracted = root._inventory()
    r = root._root()
    return {"version": 1, "root": str(r), "generated_at": "2026-11-25T22:14:00-05:00",
            "count": len(stamps), "stamps": stamps,
            "sizes": {rel: (r / rel).stat().st_size for rel in stamps},
            "mtimes": {rel: (r / rel).stat().st_mtime for rel in stamps},
            "unextracted": list(unextracted)}


def _subset(root, keep="docs/btree.md"):
    """One row of the three the index holds: the manifest that listed a single
    document while the rest vanished."""
    stamp = root._load_meta()["stamps"][keep]
    r = root._root()
    return {"version": 1, "root": str(r), "generated_at": "2026-11-25T22:14:00-05:00",
            "count": 1, "stamps": {keep: stamp},
            "sizes": {keep: (r / keep).stat().st_size},
            "mtimes": {keep: (r / keep).stat().st_mtime}}


def _write(tmp_path, data, name="inventory.json"):
    # Outside the root: a manifest written under it would be walked and listed.
    p = tmp_path.parent / (tmp_path.name + "-" + name)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _by_path(reply):
    return {row["path"] for row in reply["files"]}


# ---------------------------------------------------------------------------
# a. a subset manifest is not served as coverage
# ---------------------------------------------------------------------------

def test_a_subset_manifest_is_not_covering(root):
    root.build_index()
    assert root._manifest_covers(_subset(root), None, *_known(root)) is False


def _known(root):
    meta = root._load_meta()
    return meta.get("stamps", {}), meta.get("no_text", {})


def test_a_subset_manifest_makes_the_listing_walk_and_serve_the_corpus(root, tmp_path, monkeypatch):
    root.build_index()
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, _subset(root))))
    walked = []
    real_walk = root._walk

    def counting(base=None):
        walked.append(1)
        yield from real_walk(base)

    monkeypatch.setattr(root, "_walk", counting)

    reply = root.silica_files()

    assert walked, "a subset manifest is not coverage: the walk answers"
    assert _by_path(reply) == {"docs/lsm.md", "docs/btree.md", "docs/ctree.md", "docs/report.docx"}, \
        "the whole corpus is listed, not the one row the manifest named"
    assert reply["total"] == 4


# ---------------------------------------------------------------------------
# b. a complete manifest still serves from the manifest, no traversal
# ---------------------------------------------------------------------------

def test_a_complete_manifest_serves_the_listing_with_no_walk(root, tmp_path, monkeypatch):
    root.build_index()
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, _covering(root))))
    monkeypatch.setattr(root, "_walk", _boom)  # any traversal is now an error

    reply = root.silica_files()

    assert _by_path(reply) == {"docs/lsm.md", "docs/btree.md", "docs/ctree.md", "docs/report.docx"}
    assert reply["total"] == 4 and reply["next_cursor"] is None


def test_a_split_below_the_ceiling_still_serves_from_the_manifest(root, tmp_path, monkeypatch):
    """The ceiling is not zero: a manifest a hair short of the stored rows — a
    corpus that genuinely shrank — is still served rather than walked."""
    root.build_index()
    data = _covering(root)
    del data["stamps"]["docs/ctree.md"]        # 1 of 3 omitted, 0.33 > default 0.1
    data["count"] = len(data["stamps"])
    monkeypatch.setenv("SILICA_INVENTORY_MAX_DROP", "0.5")
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, data)))
    monkeypatch.setattr(root, "_walk", _boom)

    reply = root.silica_files()

    assert _by_path(reply) == {"docs/lsm.md", "docs/btree.md", "docs/report.docx"}


# ---------------------------------------------------------------------------
# c. a successful build reply carries the inventory block
# ---------------------------------------------------------------------------

def test_a_build_reply_from_the_manifest_carries_the_inventory_block(root, tmp_path, monkeypatch):
    root.build_index()
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, _covering(root))))

    out = root.build_index()

    assert "error" not in out
    inv = out["index"]["inventory"]
    assert inv == {"source": "manifest", "root": str(root._root()),
                   "generated_at": "2026-11-25T22:14:00-05:00", "count": 3}
    # every old key is still there
    assert set(out) >= {"docs", "changed", "failed", "index"}
    assert set(out["index"]) == {"state", "docs", "built_at", "pending", "inventory"}


def test_a_build_reply_from_the_walk_carries_the_inventory_block(root, monkeypatch):
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)

    out = root.build_index()

    assert "error" not in out
    assert out["index"]["inventory"] == {"source": "walk", "root": str(root._root()),
                                         "generated_at": None, "count": 3}


# ---------------------------------------------------------------------------
# d. no manifest: nothing changes
# ---------------------------------------------------------------------------

def test_no_manifest_walks_and_the_reply_shape_is_unchanged(root, monkeypatch):
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    root.build_index()

    reply = root.silica_files()

    assert set(reply) == {"root", "total", "files", "counts", "excluded_dirs",
                          "truncated", "next_cursor", "index"}
    assert _by_path(reply) == {"docs/lsm.md", "docs/btree.md", "docs/ctree.md", "docs/report.docx"}
    assert reply["total"] == 4
    assert reply["index"]["inventory"]["source"] == "walk"
