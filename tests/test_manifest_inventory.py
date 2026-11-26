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

import hashlib
import json
import os
import sys
import types

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


def test_same_size_same_mtime_replacement_invalidates_section_cache(root, tmp_path, monkeypatch):
    note = tmp_path / "replace.md"
    note.write_text("old", encoding="utf-8")
    old = note.stat()
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    root._section_cache.clear()

    assert "old" in root._sections_of("replace.md")[0]
    note.write_text("new", encoding="utf-8")
    os.utime(note, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert note.stat().st_size == old.st_size
    assert "new" in root._sections_of("replace.md")[0]


def test_same_size_same_mtime_replacement_is_reindexed_next_pass(root, tmp_path, monkeypatch):
    note = tmp_path / "fresh.md"
    note.write_text("old text", encoding="utf-8")
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    root.build_index()

    old = note.stat()

    note.write_text("new text", encoding="utf-8")
    os.utime(note, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert note.stat().st_size == old.st_size
    assert root.index_state()["pending"] == 1
    assert root.build_index()["changed"] == 1
    assert root.index_state()["state"] == "ready"


def test_a_manifest_digest_reindexes_a_same_size_same_mtime_replacement(root, tmp_path, monkeypatch):
    note = tmp_path / "manifest-fresh.md"
    note.write_text("old text", encoding="utf-8")
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    root.build_index()
    old = note.stat()

    note.write_text("new text", encoding="utf-8")
    os.utime(note, ns=(old.st_atime_ns, old.st_mtime_ns))
    listing = []
    for full in tmp_path.rglob("*"):
        if full.is_file() and full.name not in ("stamps.json", "failures.json"):
            rel = full.relative_to(tmp_path).as_posix()
            listing.append({
                "path": rel,
                "digest": hashlib.sha256(full.read_bytes()).hexdigest(),
                "size": full.stat().st_size,
            })
    manifest_path = tmp_path / "fresh-inventory.json"
    root._produce_inventory_manifest(tmp_path, listing, manifest_path)
    monkeypatch.setenv("SILICA_INVENTORY", str(manifest_path))

    assert root.index_state()["pending"] >= 1
    assert root.build_index()["changed"] >= 1


def test_replaced_pdf_does_not_reuse_stale_extraction(root, tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"old-pdf")
    old = pdf.stat()
    body = ["old text"]

    class TextPage:
        def get_text_range(self):
            return body[0]

        def close(self):
            pass

    class Page:
        def get_textpage(self):
            return TextPage()

        def close(self):
            pass

    class PdfDocument:
        def __init__(self, _path):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return Page()

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "pypdfium2", types.SimpleNamespace(PdfDocument=PdfDocument))
    root._section_cache.clear()
    text, _pages, _reason = root._pdf_text("paper.pdf", pdf)
    assert text == "old text\n\n"

    pdf.write_bytes(b"new-pdf")
    os.utime(pdf, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert pdf.stat().st_size == old.st_size
    body[0] = "new text"
    text, _pages, _reason = root._pdf_text("paper.pdf", pdf)
    assert text == "new text\n\n"


def test_unreadable_and_empty_object_manifests_fall_back(root, tmp_path, monkeypatch):
    unreadable = tmp_path / "inventory-dir"
    unreadable.mkdir()
    monkeypatch.setenv("SILICA_INVENTORY", str(unreadable))
    assert "docs/lsm.md" in root._inventory()[0]

    empty = tmp_path / "empty-object.json"
    empty.write_text(json.dumps({"stamps": {}, "unextracted": []}), encoding="utf-8")
    monkeypatch.setenv("SILICA_INVENTORY", str(empty))
    assert "docs/lsm.md" in root._inventory()[0]


def test_producer_matches_the_walk_across_document_rules(root, tmp_path, monkeypatch):
    from silica_core.config import CONFIG

    monkeypatch.setattr(CONFIG, "index_code", True)
    fixture = tmp_path / "listing-fixture"
    fixture.mkdir()
    (fixture / "docs").mkdir()
    (fixture / "docs" / "prose.md").write_text("# Prose\n", encoding="utf-8")
    (fixture / "docs" / "code.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    (fixture / "docs" / "paper.pdf").write_bytes(b"%PDF-paper")
    (fixture / "docs" / "paper.md").write_text("# Paper sidecar\n", encoding="utf-8")
    (fixture / "docs" / "scan.pdf").write_bytes(b"%PDF-scan")
    (fixture / "docs" / "report.docx").write_bytes(b"not really docx")
    (fixture / "ignored.md").write_text("ignored", encoding="utf-8")
    (fixture / ".hidden.md").write_text("hidden", encoding="utf-8")
    (fixture / ".hidden-dir").mkdir()
    (fixture / ".hidden-dir" / "hidden.md").write_text("hidden", encoding="utf-8")
    (fixture / "node_modules").mkdir()
    (fixture / "node_modules" / "package.py").write_text("hidden", encoding="utf-8")
    (fixture / ".silicaignore").write_text("ignored.md\n", encoding="utf-8")

    monkeypatch.setattr(CONFIG, "vault_path", str(fixture))
    import silica_core.core as core
    listing = []
    for full in fixture.rglob("*"):
        if full.is_file():
            rel = full.relative_to(fixture).as_posix()
            listing.append({
                "path": rel,
                "digest": hashlib.sha256(full.read_bytes()).hexdigest(),
                "size": full.stat().st_size,
            })

    manifest_path = tmp_path / "inventory.json"
    manifest = core._produce_inventory_manifest(fixture, listing, manifest_path)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    walked, unextracted = core._inventory()
    assert set(manifest["stamps"]) == set(walked)
    assert set(manifest["unextracted"]) == set(unextracted)
    assert "docs/prose.md" in manifest["stamps"]
    assert "docs/code.py" in manifest["stamps"]
    assert "docs/paper.md" in manifest["stamps"]
    assert "docs/paper.pdf" not in manifest["stamps"]
    assert "docs/scan.pdf" in manifest["stamps"]
    assert "docs/report.docx" in manifest["unextracted"]
    assert "ignored.md" not in manifest["stamps"]
    assert ".hidden.md" not in manifest["stamps"]
    assert ".hidden-dir/hidden.md" not in manifest["stamps"]
    assert "node_modules/package.py" not in manifest["stamps"]


def test_a_subset_manifest_cannot_prune_the_index(root, tmp_path, monkeypatch):
    root.build_index()
    meta = tmp_path / ".idx" / "stamps.json"
    before = json.loads(meta.read_text(encoding="utf-8"))["stamps"]
    subset = _manifest(tmp_path, {"docs/lsm.md": 11.0})
    monkeypatch.setenv("SILICA_INVENTORY", str(subset))

    root.build_index()
    after = json.loads(meta.read_text(encoding="utf-8"))["stamps"]
    assert set(after) >= set(before)


def test_a_subset_manifest_never_prunes_when_the_walk_is_unavailable(root, tmp_path, monkeypatch):
    root.build_index()
    meta = tmp_path / ".idx" / "stamps.json"
    before = json.loads(meta.read_text(encoding="utf-8"))["stamps"]
    subset = _manifest(tmp_path, {"docs/lsm.md": 11.0})
    monkeypatch.setenv("SILICA_INVENTORY", str(subset))
    monkeypatch.setattr(root, "_walk", _boom)

    root.build_index()
    after = json.loads(meta.read_text(encoding="utf-8"))["stamps"]
    assert set(after) == set(before)
