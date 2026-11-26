# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""The producer side of `SILICA_INVENTORY`: a corpus listing → a manifest.

The reader (`core._manifest_inventory`, landed) answers a search from a JSON
file that names the documents and their change stamps. Nothing produced that
file, so the fix was inert. A calling system that already holds a listing —
one row per object, each carrying a size and a content digest — can produce it
without walking anything.

The contract these tests pin:

* **Parity is the acceptance test.** The manifest's `stamps` and `unextracted`
  must be *exactly* the sets `_walk()` + `_inventory()` derive from the same
  tree: one extra or missing row and the index set drifts while reporting
  `ready`. The document set is decided by importing upstream's own symbols,
  never by re-typing a suffix list.
* **The stamp rule.** A content digest maps to `int(digest[:12], 16)` — 48 bits,
  exactly representable in a float. A missing, non-hex or short digest raises.
* **Fail closed.** A candidate document row with no size or no stamp raises,
  unless the caller opts in, and then the manifest records how many it dropped.
  A manifest that silently omits documents is worse than no manifest.
* **Atomic writes.** A reader never sees a half-written file, and a failure
  mid-write leaves the previous manifest intact (or none at all).
* **Round-trip.** The written file loads through the landed reader unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# The fixture tree exercises every rule in the document set at once.
def _make_tree(root: Path) -> None:
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "prose.md").write_text("# Prose\n\nA note in prose.\n", encoding="utf-8")

    # A `.md` beside a converted PDF: the sidecar is the note, the PDF is not read.
    (root / "papers").mkdir()
    (root / "papers" / "paper.md").write_text("# Paper\n\nextracted text\n", encoding="utf-8")
    (root / "papers" / "paper.pdf").write_bytes(b"%PDF-1.4\0binary")

    # A PDF with no `.md` beside it: the index reads it as a document.
    (root / "loose").mkdir()
    (root / "loose" / "manual.pdf").write_bytes(b"%PDF-1.4\0binary")

    # An office file nothing extracted: a document the index holds no text for.
    (root / "office").mkdir()
    (root / "office" / "report.docx").write_bytes(b"PK\x03\x04not really a docx")

    # A file matched by `.silicaignore`, a hidden file, a noise directory, an image.
    (root / ".silicaignore").write_text("# a comment\nignored.md\n", encoding="utf-8")
    (root / "ignored.md").write_text("# ignored\n", encoding="utf-8")
    (root / ".hidden.md").write_text("# hidden\n", encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "note.md").write_text("# build output\n", encoding="utf-8")
    (root / "media").mkdir()
    (root / "media" / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # A code file over the size cap (never read) and one under it.
    (root / "src").mkdir()
    (root / "src" / "big.py").write_text("x = 1\n" * 100_000, encoding="utf-8")
    (root / "src" / "small.py").write_text("y = 2\n", encoding="utf-8")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    from silica_core.config import CONFIG

    root = tmp_path / "corpus"
    root.mkdir()
    _make_tree(root)
    monkeypatch.setattr(CONFIG, "vault_path", str(root))
    monkeypatch.setattr(CONFIG, "index_code", True)  # so the code lane is exercised
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    import silica_core.core as core
    return core, root


def _listing(core, root):
    """The rows a listing hands a producer: `(rel, size, mtime)` for every file
    `_walk` enters — never the excluded ones, which a real listing would not
    carry either."""
    rows = []
    for rel, full in core._walk():
        if full is None:
            continue
        st = full.stat()
        rows.append((rel, st.st_size, st.st_mtime))
    return rows


def _boom(*_a, **_k):
    raise AssertionError("a traversal ran where the manifest should have answered")


# --------------------------------------------------------------------------
# (a) PARITY — the acceptance test
# --------------------------------------------------------------------------

def test_manifest_matches_the_walk_exactly(tree):
    from silica_core.inventory import build_manifest

    core, root = tree
    walk_stamps, walk_unextracted = core._inventory()
    manifest = build_manifest(root, _listing(core, root))

    # EXACTLY the walk's sets, values and all: one extra, one missing, or one
    # differing stamp and the index set would drift while reporting ready.
    assert manifest["stamps"] == walk_stamps, "stamps differ from the walk"
    assert sorted(manifest["unextracted"]) == sorted(walk_unextracted), \
        "unextracted differs from the walk"
    assert len(manifest["unextracted"]) == len(walk_unextracted)

    # And the document-set rules themselves, spelled out.
    assert "docs/prose.md" in manifest["stamps"]                 # prose
    assert "papers/paper.md" in manifest["stamps"]               # the sidecar, read
    assert "papers/paper.pdf" not in manifest["stamps"]          # converted PDF, skipped
    assert "loose/manual.pdf" in manifest["stamps"]              # PDF with no sidecar
    assert manifest["unextracted"] == ["office/report.docx"]     # office, no text held
    assert "src/big.py" not in manifest["stamps"]                # code over the cap
    for excluded in ("ignored.md", ".hidden.md", "build/note.md", "media/pic.png"):
        assert excluded not in manifest["stamps"]
        assert excluded not in manifest["unextracted"]


def test_one_extra_row_breaks_parity(tree):
    """The parity comparison must FAIL LOUDLY on drift — here, a row the walk
    never produces (a document the listing invented) is not silently dropped."""
    from silica_core.inventory import build_manifest

    core, root = tree
    walk_stamps, _walk_unextracted = core._inventory()
    rows = _listing(core, root) + [("docs/ghost.md", 10, 1.0)]
    manifest = build_manifest(root, rows)
    assert manifest["stamps"] != walk_stamps, "an invented document must not vanish"


# --------------------------------------------------------------------------
# (b) the digest → stamp rule
# --------------------------------------------------------------------------

def test_digest_stamp_is_the_first_12_hex_digits_as_an_int():
    from silica_core.inventory import digest_stamp

    assert digest_stamp("a" * 40) == int("a" * 12, 16)
    assert digest_stamp("0123456789ab" + "cd" * 14) == int("0123456789ab", 16)
    assert isinstance(digest_stamp("0" * 40), int)
    # 48 bits is exactly a float: the round-trip through JSON loses nothing.
    assert float(digest_stamp("f" * 40)) == int("f" * 12, 16)


def test_a_missing_or_non_hex_or_short_digest_raises():
    from silica_core.inventory import digest_stamp

    for bad in (None, "", 12345, {"a": 1}, "zzzzzzzzzzzzzz"):
        with pytest.raises(ValueError):
            digest_stamp(bad)
    with pytest.raises(ValueError):
        digest_stamp("abc")            # shorter than 12 characters


def test_a_row_with_a_missing_digest_raises_unless_the_caller_opts_in(tmp_path):
    from silica_core.inventory import build_manifest

    root = tmp_path
    (root / "note.md").write_text("# n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_manifest(root, [("note.md", 4, None)])          # no stamp
    with pytest.raises(ValueError):
        build_manifest(root, [("note.md", None, "a" * 40)])   # no size

    manifest = build_manifest(root, [("note.md", 4, None), ("ok.md", 2, "a" * 40)],
                              allow_incomplete=True)
    assert manifest["count"] == len(manifest["stamps"]) == 1
    assert manifest["skipped"] == 1
    assert "note.md" not in manifest["stamps"]


def test_a_digest_round_trips_as_a_stamp(tmp_path):
    from silica_core.inventory import build_manifest

    root = tmp_path
    (root / "note.md").write_text("# n\n", encoding="utf-8")
    digest = "deadbeef" * 5
    manifest = build_manifest(root, [("note.md", 4, digest)])
    assert manifest["stamps"] == {"note.md": int("deadbeefdead", 16)}
    assert "mtimes" not in manifest            # the rows carried digests, not mtimes


# --------------------------------------------------------------------------
# (c) atomicity
# --------------------------------------------------------------------------

def test_a_failure_mid_write_leaves_no_manifest_behind(tmp_path, monkeypatch):
    from silica_core import inventory

    out = tmp_path / "manifest.json"

    def boom(_src, _dst):
        raise OSError("disk full")

    monkeypatch.setattr(inventory.os, "replace", boom)
    with pytest.raises(OSError):
        inventory.write_manifest(out, {"version": 1, "stamps": {}})
    assert not out.exists()
    assert list(tmp_path.glob("manifest.json*")) == [], "a temp file was left behind"


def test_a_failure_mid_write_leaves_the_previous_manifest_intact(tmp_path, monkeypatch):
    from silica_core import inventory

    out = tmp_path / "manifest.json"
    inventory.write_manifest(out, {"version": 1, "stamps": {"a": 1}})
    good = out.read_bytes()

    def boom(_src, _dst):
        raise OSError("disk full")

    monkeypatch.setattr(inventory.os, "replace", boom)
    with pytest.raises(OSError):
        inventory.write_manifest(out, {"version": 1, "stamps": {"a": 2}})
    assert out.read_bytes() == good


# --------------------------------------------------------------------------
# (d) round-trip through the landed reader
# --------------------------------------------------------------------------

def test_the_written_file_loads_through_the_landed_reader(tree, tmp_path, monkeypatch):
    from silica_core.inventory import build_manifest, write_manifest

    core, root = tree
    manifest = build_manifest(root, _listing(core, root))
    out = tmp_path / "inventory.json"
    write_manifest(out, manifest)

    monkeypatch.setenv("SILICA_INVENTORY", str(out))
    monkeypatch.setattr(core, "_walk", _boom)          # the manifest must answer alone
    stamps, unextracted = core._inventory()
    assert stamps == manifest["stamps"]
    assert sorted(unextracted) == sorted(manifest["unextracted"])


# --------------------------------------------------------------------------
# the module entry point
# --------------------------------------------------------------------------

def test_cli_builds_a_manifest_from_a_listing(tmp_path, monkeypatch):
    from silica_core import inventory

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("# a\n", encoding="utf-8")
    (root / "b.docx").write_bytes(b"PK\x03\x04x")
    digest = "ab" * 20
    listing = tmp_path / "rows.jsonl"
    listing.write_text(
        '{"rel": "a.md", "size": 4, "sha1": "%s"}\n'
        '{"rel": "b.docx", "size": 6, "mtime": 1764111240.0}\n' % digest,
        encoding="utf-8")
    out = tmp_path / "out.json"

    inventory.main(["--root", str(root), "--listing", str(listing), "--out", str(out)])
    assert out.is_file()
    import orjson
    data = orjson.loads(out.read_bytes())
    assert data["stamps"] == {"a.md": int("ab" * 6, 16)}
    assert data["unextracted"] == ["b.docx"]
    assert data["count"] == 1
    assert data["version"] == 1
    assert data["root"] == str(root.resolve())
