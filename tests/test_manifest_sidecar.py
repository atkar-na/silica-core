# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""`SILICA_INVENTORY` sidecars: a manifest pass honours a conversion's note.

A manifest is built from a listing of the corpus — a bucket listing, a sync, a
database — and a `silica-core import` writes its converted `.md` beside the
original, into a view (a writable overlay over the mount) that the listing
never covered. So the manifest names the PDF and not the note beside it, and a
manifest-driven pass used to read the PDF, find no text layer, and index
nothing: the conversion was invisible.

The walk has no such gap. `_inventory` skips a PDF whose `.md` sits beside it
and indexes that `.md` instead, carrying `source_state` from `_source_state`.
A manifest pass must do exactly the same, and this file pins it:

* the converted text enters the store, under the sidecar's own path, with the
  `source_state` the read and the listing already report;
* the row's change key is the sidecar's own content (`digest_stamp` of its
  sha1), so a conversion that changes re-indexes and one that does not, does
  not — the original's byte-identity is irrelevant to it;
* a document the manifest names twice (itself and its sidecar) appears once;
* a missing, unreadable or `stale` sidecar, or a missing original, leaves the
  row its ordinary handling — never a crash, never a dropped row;
* with no sidecars at all the pass is byte-for-byte the pass it always was.

Fixtures are generic (`alpha/`, `beta/`, `paper.pdf`): nothing here names a
real corpus.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A root with one scanned document and a folder of notes, isolated index."""
    from silica_core.config import CONFIG
    from silica_core.kernel.recall import paths

    root = tmp_path / "corpus"
    (root / "alpha").mkdir(parents=True)
    (root / "notes").mkdir()
    # A document with no text layer of its own: the conversion is the only way
    # its text can reach the index.
    (root / "alpha" / "paper.pdf").write_bytes(b"%PDF-1.4\nscan, no text layer\n")
    for i in range(12):
        (root / "notes" / f"n{i:02d}.md").write_text(
            f"# Note {i}\n\nalpha beta gamma note number {i} content line\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(root))
    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=root / ".idx": _d)
    import silica_core.core as core
    from silica_core.kernel.recall import lexical
    core._section_cache.clear()
    core._sha_cache.clear()
    lexical.clear()
    return core, root, tmp_path


def _notes(root) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in (root / "notes").glob("*.md"))


def _digest(path) -> int:
    return int(hashlib.sha1(path.read_bytes()).hexdigest()[:12], 16)


def _sidecar(pdf, body: str, *, stale: bool = False, hash_field: bool = True):
    """Write the `.md` a conversion would leave beside `pdf`.

    `stale` records a hash that disagrees with the original's bytes; without
    `hash_field` the note carries no `source_sha256` at all (an old conversion,
    or a hand-written note), which `_source_state` calls `unverifiable`.
    """
    lines = [f'source_file: "{pdf}"\n']
    if hash_field:
        dig = "0" * 64 if stale else hashlib.sha256(pdf.read_bytes()).hexdigest()
        lines.append(f'source_sha256: "{dig}"\n')
    text = "---\n" + "".join(lines) + 'converter: "test"\n---\n\n' + body.strip() + "\n"
    pdf.with_suffix(".md").write_text(text, encoding="utf-8")
    return pdf.with_suffix(".md")


def _manifest(tmp, root, rows, *, unextracted=(), overrides=None, name="inventory.json"):
    """A manifest over `rows`: content digests as stamps (as a listing carries),
    unless `overrides` pins one row's stamp (to prove the sidecar's own digest
    wins over a row the listing also names)."""
    from silica_core.core import _root

    rows = list(rows)
    stamps = {}
    for rel in rows:
        stamps[rel] = (overrides or {}).get(rel, _digest(root / rel))
    sizes = {}
    for rel in stamps:
        try:
            sizes[rel] = (root / rel).stat().st_size
        except OSError:
            sizes[rel] = 0
    data = {"version": 1, "root": str(_root()), "count": len(stamps), "stamps": stamps,
            "mtimes": {rel: 0.0 for rel in stamps}, "sizes": sizes,
            "unextracted": list(unextracted)}
    path = tmp / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, stamps


def _no_traversal(core, monkeypatch):
    """Any walk at all is an error: a manifest pass does not traverse."""
    def boom(*_a, **_k):
        raise AssertionError("a manifest pass walked the corpus")

    monkeypatch.setattr(core, "_walk", boom)


def _wipe(core, root):
    """Fresh index: remove it and drop the in-process caches that key on it."""
    import shutil
    from silica_core.kernel.recall import lexical
    shutil.rmtree(root / ".idx", ignore_errors=True)
    core._section_cache.clear()
    core._sha_cache.clear()
    lexical.clear()


def _store(root) -> dict:
    return json.loads((root / ".idx" / "lexical.json").read_text(encoding="utf-8"))


def _store_bytes(root) -> bytes:
    return (root / ".idx" / "lexical.json").read_bytes()


def _docs(root) -> dict:
    return _store(root)["docs"]


def test_a_manifest_row_is_indexed_from_its_sidecar_and_reports_source_state(corpus, monkeypatch):
    """The defect: the manifest names the PDF, the conversion wrote the note."""
    core, root, tmp = corpus
    pdf = root / "alpha" / "paper.pdf"
    _sidecar(pdf, "# Paper\n\nzephyr quintessence alpha beta gamma delta\n")
    path, digests = _manifest(tmp, root, ["alpha/paper.pdf", *_notes(root)])
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    _no_traversal(core, monkeypatch)

    out = core.build_index()

    assert out["failed"] == {}
    docs = _docs(root)
    assert "alpha/paper.md" in docs, "the converted text never reached the store"
    assert "alpha/paper.pdf" not in docs
    assert "zephyr" in docs["alpha/paper.md"] and "quintessence" in docs["alpha/paper.md"]
    # The row's change key is the sidecar's own content digest.
    assert core._load_meta()["stamps"]["alpha/paper.md"] == _digest(root / "alpha" / "paper.md")

    hits = core.silica_search("zephyr quintessence", k=3)
    converted = [h for h in hits["hits"] if h["path"] == "alpha/paper.md"]
    assert converted, "a search did not reach the converted note"
    assert converted[0]["source_state"] == "current", "the conversion's verdict is not reported"


def test_a_changed_conversion_reindexes_without_the_original_moving(corpus, monkeypatch):
    """The stamp is the sidecar's content, not the original's mtime or hash: a
    same-size rewrite of the note re-indexes though the PDF never moved."""
    core, root, tmp = corpus
    pdf = root / "alpha" / "paper.pdf"
    sidecar = _sidecar(pdf, "# Paper\n\nzephyr bravo alpha beta gamma\n")
    stamp_mtime = 1_600_000_000
    os.utime(sidecar, (stamp_mtime, stamp_mtime))
    path, _ = _manifest(tmp, root, ["alpha/paper.pdf", *_notes(root)])
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    _no_traversal(core, monkeypatch)

    assert core.build_index()["changed"] >= 1
    assert "bravo" in _docs(root)["alpha/paper.md"]

    # Same length, same mtime, same original — only the note's bytes differ.
    _sidecar(pdf, "# Paper\n\nzephyr delta alpha beta gamma\n")
    os.utime(sidecar, (stamp_mtime, stamp_mtime))
    out = core.build_index()

    assert out["changed"] == 1, "a changed conversion did not re-index"
    docs = _docs(root)
    assert "delta" in docs["alpha/paper.md"] and "bravo" not in docs["alpha/paper.md"]


def test_an_unchanged_conversion_is_not_reindexed(corpus, monkeypatch):
    core, root, tmp = corpus
    _sidecar(root / "alpha" / "paper.pdf", "# Paper\n\nzephyr quintessence alpha beta\n")
    path, _ = _manifest(tmp, root, ["alpha/paper.pdf", *_notes(root)])
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    _no_traversal(core, monkeypatch)
    core.build_index()
    assert "alpha/paper.md" in _docs(root), "the conversion was never indexed"

    before_bytes = _store_bytes(root)
    before_mtime = (root / ".idx" / "lexical.json").stat().st_mtime_ns
    before_stamps = dict(core._load_meta()["stamps"])

    out = core.build_index()

    assert out["changed"] == 0
    assert out["index"]["pending"] == 0 and out["index"]["state"] == "ready"
    assert _store_bytes(root) == before_bytes, "the store was rewritten for nothing"
    assert (root / ".idx" / "lexical.json").stat().st_mtime_ns == before_mtime
    assert core._load_meta()["stamps"] == before_stamps


def test_a_deleted_sidecar_reverts_without_dropping_the_row(corpus, monkeypatch):
    core, root, tmp = corpus
    pdf = root / "alpha" / "paper.pdf"
    sidecar = _sidecar(pdf, "# Paper\n\nzephyr quintessence alpha beta\n")
    path, _ = _manifest(tmp, root, ["alpha/paper.pdf", *_notes(root)])
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    _no_traversal(core, monkeypatch)
    core.build_index()
    assert "alpha/paper.md" in _docs(root)

    sidecar.unlink()
    out = core.build_index()

    assert "error" not in out, "a vanished conversion tripped a guard instead of reverting"
    assert out["failed"] == {}, "a scanned PDF with no sidecar is unconverted, not failed"
    assert "alpha/paper.md" not in _docs(root), "the stale sidecar row was not pruned"
    # The original keeps its ordinary handling: accounted for, never dropped.
    assert "alpha/paper.pdf" in core._load_meta()["no_text"]


def test_a_document_the_manifest_names_twice_appears_once(corpus, monkeypatch):
    """The PDF row and an explicit sidecar row collapse to one entry: the
    sidecar's own content digest is the stamp either way."""
    core, root, tmp = corpus
    sidecar = _sidecar(root / "alpha" / "paper.pdf", "# Paper\n\nzephyr quintessence alpha\n")
    rows = ["alpha/paper.pdf", *_notes(root), "alpha/paper.md"]
    path, _ = _manifest(tmp, root, rows, overrides={"alpha/paper.md": 12345.0})
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    _no_traversal(core, monkeypatch)

    core.build_index()

    docs = _docs(root)
    assert set(docs) == set(_notes(root)) | {"alpha/paper.md"}
    assert len(docs) == len(_notes(root)) + 1, "the document appears more than once"
    assert core._load_meta()["stamps"]["alpha/paper.md"] == _digest(sidecar), \
        "the sidecar's content digest did not win over the listing's own stamp"


def test_a_stale_sidecar_falls_back_to_the_original(corpus, monkeypatch):
    """`_source_state` semantics: the conversion's hash disagrees with the bytes
    beside it, so its text is not adopted — but the read still reports `stale`."""
    core, root, tmp = corpus
    pdf = root / "alpha" / "paper.pdf"
    _sidecar(pdf, "# Paper\n\nzephyr quintessence alpha beta\n", stale=True)
    path, _ = _manifest(tmp, root, ["alpha/paper.pdf", *_notes(root)])
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    _no_traversal(core, monkeypatch)

    out = core.build_index()

    assert out["failed"] == {}
    assert "alpha/paper.md" not in _docs(root), "a stale conversion was adopted as current"
    assert "alpha/paper.pdf" in core._load_meta()["no_text"]
    assert core.silica_read("alpha/paper.pdf")["source_state"] == "stale"


def test_an_unhashable_or_unreadable_sidecar_and_a_missing_original(corpus, monkeypatch):
    """Every unusable sidecar, and an original gone from the mount, leaves the
    row its ordinary handling: no crash, no drop. A hand-written note with no
    hash is `unverifiable` and IS adopted, as the walk adopts it."""
    core, root, tmp = corpus
    pdf = root / "alpha" / "paper.pdf"

    # A sidecar path that is not a readable file (here: a directory).
    (root / "alpha" / "paper.md").mkdir()
    path, _ = _manifest(tmp, root, ["alpha/paper.pdf", *_notes(root)])
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    _no_traversal(core, monkeypatch)
    out = core.build_index()
    assert out["failed"] == {} and "alpha/paper.pdf" in core._load_meta()["no_text"]
    assert "alpha/paper.md" not in _docs(root)

    # A hand-written note: no `source_sha256`, so `unverifiable` and adopted.
    (root / "alpha" / "paper.md").rmdir()
    _sidecar(pdf, "# Paper\n\nzephyr quintessence alpha beta\n", hash_field=False)
    _wipe(core, root)
    out = core.build_index()
    assert "alpha/paper.md" in _docs(root)
    assert core.silica_read("alpha/paper.pdf")["source_state"] == "unverifiable"

    # An original the mount no longer lists: recorded as a failure, never dropped.
    _wipe(core, root)
    path, _ = _manifest(tmp, root, ["alpha/paper.pdf", *_notes(root)])
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    (root / "alpha" / "paper.pdf").unlink()
    out = core.build_index()
    assert "alpha/paper.pdf" in out["failed"], "a missing original was silently dropped"


def test_freshness_recognises_a_document_indexed_through_its_sidecar(corpus, monkeypatch):
    """A manifest row answered through its sidecar is not pending, and a changed
    conversion makes the index stale again — the freshness check has to know the
    store's key differs from the manifest's on purpose."""
    core, root, tmp = corpus
    pdf = root / "alpha" / "paper.pdf"
    sidecar = _sidecar(pdf, "# Paper\n\nzephyr bravo alpha beta gamma\n")
    stamp_mtime = 1_600_000_000
    os.utime(sidecar, (stamp_mtime, stamp_mtime))
    path, _ = _manifest(tmp, root, ["alpha/paper.pdf", *_notes(root)])
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    _no_traversal(core, monkeypatch)
    core.build_index()

    state = core.index_state()
    assert state["state"] == "ready" and state["pending"] == 0

    _sidecar(pdf, "# Paper\n\nzephyr delta alpha beta gamma\n")
    os.utime(sidecar, (stamp_mtime, stamp_mtime))
    state = core.index_state()
    assert state["state"] == "stale" and state["pending"] == 1, "a changed conversion read as current"


def test_manifest_and_walk_index_the_same_text(corpus, monkeypatch):
    """Parity: on a corpus with a sidecar, a manifest pass and a walk pass
    produce the identical store — same documents, same text, same keys."""
    core, root, tmp = corpus
    _sidecar(root / "alpha" / "paper.pdf", "# Paper\n\nzephyr quintessence alpha beta gamma\n")

    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    core.build_index()
    walk_bytes = _store_bytes(root)
    _wipe(core, root)

    path, _ = _manifest(tmp, root, ["alpha/paper.pdf", *_notes(root)])
    monkeypatch.setenv("SILICA_INVENTORY", str(path))
    _no_traversal(core, monkeypatch)
    core.build_index()

    assert _store_bytes(root) == walk_bytes, "manifest mode drifted from the walk"
    assert "alpha/paper.md" in _docs(root)


def test_a_manifest_pass_without_sidecars_is_byte_identical(corpus, monkeypatch):
    """The invariant: with no sidecar anywhere, the pass changes nothing. The
    baseline is the same pass with the sidecar step neutralised — i.e. the
    behaviour before this change."""
    core, root, tmp = corpus
    path, _ = _manifest(tmp, root, ["alpha/paper.pdf", *_notes(root)])
    monkeypatch.setenv("SILICA_INVENTORY", str(path))

    real = core._sidecar_documents
    monkeypatch.setattr(core, "_sidecar_documents",
                        lambda stamps, unextracted, root: (dict(stamps), list(unextracted)))
    _no_traversal(core, monkeypatch)
    core.build_index()
    baseline = _store_bytes(root)
    _wipe(core, root)

    monkeypatch.setattr(core, "_sidecar_documents", real)
    core.build_index()

    assert _store_bytes(root) == baseline, "a sidecar-less pass was not byte-for-byte identical"
    assert "alpha/paper.pdf" not in _docs(root)
