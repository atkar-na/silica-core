# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""`SILICA_INVENTORY` is the authority for an index pass: a pass never walks.

A manifest exists so a large remote corpus can be indexed from a listing of
paths and content hashes without traversing the tree — a walk on a network
mount costs minutes, and on a mount that reports no mtimes its `st_mtime` is
0.0 for every file. So the manifest's integer digest stamps ARE the change keys
written into the store, and a pass that stores anything else for a document it
processed is a bug: it makes that document look changed on every later pass and
re-queues the whole corpus for ever.

What this pins:

* no traversal — a build over a manifest never calls the walk, and every
  manifest row is indexed and stamped with its digest;
* a row the filesystem does not list is still in the pass, with its digest:
  the listing is the claim, and pruning it because a mount cannot see it is
  the empty-inventory defect in the other direction;
* the empty-inventory protection without a walk: a manifest that would remove
  more of the stored rows than `SILICA_INVENTORY_MAX_DROP` allows is refused
  with `inventory_prune_refused` and the store is left byte-identical;
* a second pass over an unchanged manifest changes nothing, reports pending 0
  and does not rewrite the store.
"""
from __future__ import annotations

import hashlib
import json

import pytest


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Three notes under a temporary root, with an isolated index dir."""
    from silica_core.config import CONFIG
    from silica_core.kernel.recall import paths

    root_dir = tmp_path / "corpus"
    root_dir.mkdir()
    for i in range(3):
        (root_dir / f"n{i}.md").write_text(
            f"# Note {i}\n\ncompaction merges sorted runs; note {i} alpha beta.\n",
            encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(root_dir))
    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=root_dir / ".idx": _d)
    import silica_core.core as core
    core._section_cache.clear()
    return core, root_dir, tmp_path


def _digest(path):
    return int(hashlib.sha1(path.read_bytes()).hexdigest()[:12], 16)


def _manifest(tmp_path, root_dir, rows=None, *, sizes=True, name="inventory.json"):
    """A manifest for `rows` (root-relative paths; all notes by default), with
    48-bit content digests as stamps — what a listing of an object store
    carries, and the only change key a `--no-modtime` mount can offer."""
    from silica_core.core import _root

    rows = sorted(p.relative_to(root_dir).as_posix() for p in root_dir.rglob("*.md")) \
        if rows is None else list(rows)
    stamps = {rel: _digest(root_dir / rel) for rel in rows}
    data = {"version": 1, "root": str(_root()), "count": len(stamps), "stamps": stamps,
            "mtimes": {rel: 0.0 for rel in stamps}, "unextracted": []}
    if sizes:
        data["sizes"] = {rel: (root_dir / rel).stat().st_size for rel in stamps}
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, stamps


def _no_traversal(core, monkeypatch):
    """Any walk at all is an error: a manifest pass does not traverse."""
    seen = []

    def boom(*_a, **_k):
        seen.append(1)
        raise AssertionError("a manifest pass walked the corpus")

    monkeypatch.setattr(core, "_walk", boom)
    return seen


def _index_bytes(root_dir):
    return {p.name: p.read_bytes() for p in (root_dir / ".idx").rglob("*") if p.is_file()}


def _store(root_dir):
    return json.loads((root_dir / ".idx" / "lexical.json").read_text(encoding="utf-8"))


def test_a_manifest_pass_never_walks_and_stores_the_manifest_digests(corpus, monkeypatch):
    core, root_dir, tmp = corpus
    path, digests = _manifest(tmp, root_dir)
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    walked = _no_traversal(core, monkeypatch)

    out = core.build_index()

    assert walked == [], "the pass traversed the corpus"
    assert out["failed"] == {}
    assert out["docs"] == 3
    assert core._load_meta()["stamps"] == digests, "the walk's stamp reached the store"
    assert set(_store(root_dir)["docs"]) == set(digests)


def test_a_manifest_row_the_filesystem_does_not_list_is_still_in_the_pass(corpus, monkeypatch):
    """The listing is the claim. A mount that does not list a directory — or a
    walk that could not run at all — is not evidence that a row was deleted."""
    core, root_dir, tmp = corpus
    hidden = root_dir / "n2.md"
    path, digests = _manifest(tmp, root_dir)
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    _no_traversal(core, monkeypatch)

    core.build_index()

    assert core._load_meta()["stamps"] == digests
    assert hidden.relative_to(root_dir).as_posix() in _store(root_dir)["docs"]


def test_the_guard_keeps_a_manifest_row_absent_from_disk_and_from_the_walk():
    """A row the manifest names but nothing on disk confirms: kept with its
    digest, never dropped from the pass."""
    from silica_core import core

    manifest = {"stamps": {"a.md": 1234, "gone.md": 5678}, "sizes": {"a.md": 1, "gone.md": 1},
                "unextracted": [], "mtimes": {}}
    rows, unextracted = core._guard_live({"a.md": 0.0}, [], {"stamps": {}, "no_text": {}}, [],
                                         manifest)
    assert rows == {"a.md": 1234, "gone.md": 5678}
    assert unextracted == []


def test_a_manifest_smaller_than_the_store_is_refused_and_the_store_is_untouched(corpus, monkeypatch):
    """A manifest that names fewer rows than the index holds is a truncated or
    subset write, not a corpus that shrank: refused, nothing written."""
    core, root_dir, tmp = corpus
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    core.build_index()
    before = _index_bytes(root_dir)
    stored = core._load_meta()["stamps"]
    assert len(stored) == 3

    path, _digests = _manifest(tmp, root_dir, rows=["n0.md"])  # 1 of 3 stored rows
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    out = core.build_index()

    assert out["error"]["code"] == "inventory_prune_refused"
    assert out["removed"] == 2 and out["stored"] == 3
    assert _index_bytes(root_dir) == before, "the store changed under a refused pass"
    assert core.index_state()["docs"] == 3


def test_a_second_pass_over_an_unchanged_manifest_writes_nothing(corpus, monkeypatch):
    core, root_dir, tmp = corpus
    path, digests = _manifest(tmp, root_dir)
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    assert core.build_index()["changed"] == 3

    import silica_core.kernel.recall.lexical as lex
    calls = []
    real = lex.LexicalStore._serialize
    monkeypatch.setattr(lex.LexicalStore, "_serialize",
                        lambda self, snap: (calls.append(1), real(self, snap))[1])
    before = (root_dir / ".idx" / "lexical.json").read_bytes()

    out = core.build_index()

    assert out["changed"] == 0
    assert out["index"]["pending"] == 0 and out["index"]["state"] == "ready"
    assert calls == [], "the whole store was rewritten for nothing"
    assert (root_dir / ".idx" / "lexical.json").read_bytes() == before
    assert core._load_meta()["stamps"] == digests


def test_a_digest_stamp_does_not_drift_into_a_walk_mtime_on_the_next_pass(corpus, monkeypatch):
    """The regression: three passes over the same manifest must not re-index
    anything, whatever a no-mtime mount would report."""
    core, root_dir, tmp = corpus
    path, digests = _manifest(tmp, root_dir)
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    core.build_index()
    for _ in range(3):
        out = core.build_index()
        assert out["changed"] == 0 and out["index"]["pending"] == 0
        assert core._load_meta()["stamps"] == digests


def test_verification_is_the_only_way_a_manifest_pass_walks(corpus, monkeypatch):
    """The opt-in cross-check may add a row the manifest omits; it may never
    restamp one, and it does not run unless asked."""
    core, root_dir, tmp = corpus
    path, digests = _manifest(tmp, root_dir)
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    (root_dir / "fresh.md").write_text("# Fresh\n\nalpha beta gamma\n", encoding="utf-8")

    out = core.build_index()
    assert "fresh.md" not in core._load_meta()["stamps"], "the default pass traversed"

    monkeypatch.setenv("SILICA_INVENTORY_VERIFY_WALK", "1")
    out = core.build_index()
    stamps = core._load_meta()["stamps"]
    assert all(stamps[rel] == st for rel, st in digests.items()), "verification restamped"
    assert "fresh.md" in stamps, "verification may add a row the manifest omits"
    assert out["failed"] == {}