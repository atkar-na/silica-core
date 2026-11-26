# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""`SILICA_INVENTORY`: the inventory comes from a manifest instead of a walk.

A root on a network filesystem pays the whole traversal on every search, and
on a mount that reports a parent-directory mtime the walk's stamps cannot tell
a replaced file from an untouched one. Both are properties of where the corpus
lives, not of the index, so the caller that already knows what is there — a
listing, a sync, a database — can hand the answer in.

The contract these tests pin:

* `SILICA_INVENTORY` unset: nothing changes, the walk is the inventory.
* set to a readable manifest: the inventory is the manifest, and the query
  path performs no traversal at all.
* set to a missing, unreadable or malformed manifest: fall back to the walk
  rather than fail — an inventory source must never be able to break search.

A manifest is `{"stamps": {rel: <number>}, "unextracted": [rel, ...]}`. The
stamp is opaque: the index only ever compares it for equality against the one
it stored, so a content hash is as valid as an mtime, and that is how a corpus
whose mtimes cannot be trusted gets real change detection.
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
    (tmp_path / "docs" / "scan.pdf").write_bytes(b"%PDF-1.4\0binary")
    (tmp_path / "docs" / "report.docx").write_bytes(b"PK\3\4not really a docx")
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=tmp_path / ".idx": _d)
    import silica_core.core as core
    return core


def _manifest(tmp_path, stamps, unextracted=()):
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"stamps": stamps, "unextracted": list(unextracted)}), encoding="utf-8")
    return path


def _boom(*_a, **_k):
    raise AssertionError("the walk ran: the query path must not traverse the root")


def test_unset_leaves_the_walk_as_the_inventory(root, monkeypatch):
    walked = []
    real = root._walk

    def counting(base=None):
        walked.append(base)
        yield from real(base)

    monkeypatch.setattr(root, "_walk", counting)
    stamps, unextracted = root._inventory()
    assert walked, "with no manifest the inventory is the walk"
    # A PDF with no `.md` beside it is a document the index reads, not an
    # unconverted one; an office file nothing extracted is the latter.
    assert "docs/lsm.md" in stamps and "docs/scan.pdf" in stamps
    assert unextracted == ["docs/report.docx"]


def test_a_manifest_is_the_inventory_and_nothing_walks(root, tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, {"docs/lsm.md": 11.0, "docs/btree.md": 12.0}, ["docs/scan.pdf"])
    monkeypatch.setenv("SILICA_INVENTORY", str(manifest))
    monkeypatch.setattr(root, "_walk", _boom)

    stamps, unextracted = root._inventory()
    assert stamps == {"docs/lsm.md": 11.0, "docs/btree.md": 12.0}
    assert unextracted == ["docs/scan.pdf"]


def test_search_answers_from_the_store_without_traversing(root, tmp_path, monkeypatch):
    """The reason this exists: a search over a served index touches no
    directory listing, so a root that is slow, remote or unreachable at the
    moment of the query still answers."""
    root.build_index()
    manifest = _manifest(tmp_path, {"docs/lsm.md": 11.0, "docs/btree.md": 12.0})
    monkeypatch.setenv("SILICA_INVENTORY", str(manifest))
    monkeypatch.setattr(root, "_walk", _boom)

    hits = root.silica_search("leveled compaction", k=3)
    assert hits["hits"], "the store answers with no traversal of the root"
    assert any(h["path"] == "docs/lsm.md" for h in hits["hits"])


def test_a_missing_or_malformed_manifest_falls_back_to_the_walk(root, tmp_path, monkeypatch):
    monkeypatch.setenv("SILICA_INVENTORY", str(tmp_path / "nope.json"))
    stamps, _un = root._inventory()
    assert "docs/lsm.md" in stamps

    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("SILICA_INVENTORY", str(bad))
    stamps, _un = root._inventory()
    assert "docs/lsm.md" in stamps

    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("SILICA_INVENTORY", str(empty))
    assert "docs/lsm.md" in root._inventory()[0]


def test_the_stamp_is_the_change_key(root, tmp_path, monkeypatch):
    """`pending` is what the index has not caught up with, and it is decided
    by comparing the stored stamp with the manifest's — that is what makes a
    content hash as good as an mtime."""
    root.build_index()
    stamps = dict(root._inventory()[0])
    manifest = _manifest(tmp_path, stamps)
    monkeypatch.setenv("SILICA_INVENTORY", str(manifest))
    monkeypatch.setattr(root, "_walk", _boom)
    assert root.index_state()["state"] == "ready"

    changed = dict(stamps)
    changed["docs/btree.md"] = stamps["docs/btree.md"] + 1.0
    _manifest(tmp_path, changed)
    state = root.index_state()
    assert state["state"] == "stale" and state["pending"] == 1
