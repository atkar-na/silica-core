# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""The in-process caches key on a document's *identity stamp*, not on the
filesystem's ``st_mtime``.

On a mount whose mtime is constant (``--no-modtime``) the key ``(path, mtime)``
never changes, so a file replaced in place at the same path — same size, same
mtime — is served as stale text and stale page bounds for the life of the
process. When ``SILICA_INVENTORY`` supplies a manifest, the stamp is the value
the manifest carries for the path (a content digest is the one that gives real
change detection); the caches compare that, so a replaced document misses the
cache and is re-read. With no manifest the stamp is the filesystem's own
``st_mtime_ns``, so caching is exactly as it was.

Three sites share one helper — the section cache, the source-hash cache and the
PDF extraction cache — and the contract pinned here is:

* a manifest whose stamp changed for a path invalidates that path's cache entry;
* a manifest whose stamp is unchanged still reuses it (no needless re-read);
* with no manifest, an unchanged file is still served from the cache;
* a PDF archive whose bytes were replaced in place is re-extracted, not served
  from the extraction cache.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def root(tmp_path, monkeypatch):
    from silica_core.config import CONFIG
    from silica_core.kernel.recall import paths

    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=tmp_path / ".idx": _d)
    import silica_core.core as core
    core._section_cache.clear()
    core._sha_cache.clear()
    return core


def _manifest(tmp_path, stamps, unextracted=()):
    from silica_core.core import _root

    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"version": 1, "root": str(_root()), "count": len(stamps),
                                "stamps": stamps, "unextracted": list(unextracted)}), encoding="utf-8")
    return path


def _replace_in_place(full, new_bytes):
    """The fault: swap the bytes at the same path, keep size, reset mtime so the
    filesystem reports nothing different (a ``--no-modtime`` mount does this by
    itself)."""
    mtime_ns = full.stat().st_mtime_ns
    assert len(new_bytes) == full.stat().st_size, "the replacement must keep the size"
    full.write_bytes(new_bytes)
    import os
    os.utime(full, ns=(mtime_ns, mtime_ns))
    assert full.stat().st_mtime_ns == mtime_ns


def _counting_reads(core, monkeypatch):
    real = core._read
    seen = []

    def counting(full):
        seen.append(str(full))
        return real(full)

    monkeypatch.setattr(core, "_read", counting)
    return seen


def test_a_manifest_stamp_change_serves_the_new_text(root, tmp_path, monkeypatch):
    """Same path, same size, unchanged mtime: only the manifest's stamp says the
    document changed, and the section cache must honour it."""
    note = tmp_path / "docs" / "lsm.md"
    note.write_text("# Note\n\nAAAA compaction\n", encoding="utf-8")
    monkeypatch.setenv("SILICA_INVENTORY", str(_manifest(tmp_path, {"docs/lsm.md": 100.0})))

    text, _secs = root._sections_of("docs/lsm.md")
    assert "AAAA" in text

    _replace_in_place(note, b"# Note\n\nBBBB compaction\n")
    # The producer re-hashed the file, so the manifest's stamp for the path moved.
    monkeypatch.setenv("SILICA_INVENTORY", str(_manifest(tmp_path, {"docs/lsm.md": 200.0})))

    text, _secs = root._sections_of("docs/lsm.md")
    assert "BBBB" in text and "AAAA" not in text, "the replaced document must not be served from the cache"


def test_an_unchanged_manifest_stamp_still_reuses_the_cache(root, tmp_path, monkeypatch):
    """The other half: a manifest that says nothing changed must not force a re-read."""
    (tmp_path / "docs" / "lsm.md").write_text("# Note\n\nAAAA compaction\n", encoding="utf-8")
    monkeypatch.setenv("SILICA_INVENTORY", str(_manifest(tmp_path, {"docs/lsm.md": 100.0})))
    reads = _counting_reads(root, monkeypatch)

    root._sections_of("docs/lsm.md")
    root._sections_of("docs/lsm.md")
    assert len(reads) == 1, "an unchanged stamp must be served from the cache"


def test_without_a_manifest_the_cache_still_serves_a_second_call(root, tmp_path, monkeypatch):
    """No manifest: the cache behaves as it always did (a read counter proves the
    second call did not touch the file)."""
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    (tmp_path / "docs" / "lsm.md").write_text("# Note\n\nAAAA compaction\n", encoding="utf-8")
    reads = _counting_reads(root, monkeypatch)

    root._sections_of("docs/lsm.md")
    root._sections_of("docs/lsm.md")
    assert len(reads) == 1, "the normal case must still be cached"


def test_a_pdf_replaced_in_place_is_re_extracted(root, tmp_path, monkeypatch):
    """The extraction cache records the source's identity stamp inside the
    artifact, so a PDF whose bytes were swapped at the same path is re-extracted
    rather than served from the text beside the index."""
    pdfium = pytest.importorskip("pypdfium2")
    extractions = []

    class _TP:
        def __init__(self, text):
            self._text = text

        def get_text_range(self):
            return self._text

        def close(self):
            pass

    class _Page:
        def __init__(self, text):
            self._text = text

        def get_textpage(self):
            return _TP(self._text)

        def close(self):
            pass

    class _Doc:
        def __init__(self, path):
            extractions.append(str(path))
            self._body = open(path, "rb").read().decode("utf-8", "replace")

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return _Page(self._body)

        def close(self):
            pass

    monkeypatch.setattr(pdfium, "PdfDocument", _Doc)

    pdf = tmp_path / "docs" / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 alpha")
    full = tmp_path / "docs" / "scan.pdf"
    monkeypatch.setenv("SILICA_INVENTORY", str(_manifest(tmp_path, {"docs/scan.pdf": 1.0})))

    text, _pages, _reason = root._pdf_text("docs/scan.pdf", full)
    assert "alpha" in text

    _replace_in_place(pdf, b"%PDF-1.4 bravo")
    monkeypatch.setenv("SILICA_INVENTORY", str(_manifest(tmp_path, {"docs/scan.pdf": 2.0})))

    text, _pages, _reason = root._pdf_text("docs/scan.pdf", full)
    assert "bravo" in text and "alpha" not in text, "the replaced PDF must be re-extracted"
    assert len(extractions) == 2, "extraction must run again for a replaced source"


def test_a_pdf_with_an_unchanged_stamp_is_served_from_the_cache(root, tmp_path, monkeypatch):
    """A manifest whose stamp for a PDF is unchanged must reuse the extraction."""
    pdfium = pytest.importorskip("pypdfium2")
    extractions = []

    class _TP:
        def get_text_range(self):
            return "alpha"

        def close(self):
            pass

    class _Page:
        def get_textpage(self):
            return _TP()

        def close(self):
            pass

    class _Doc:
        def __init__(self, path):
            extractions.append(str(path))

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return _Page()

        def close(self):
            pass

    monkeypatch.setattr(pdfium, "PdfDocument", _Doc)
    pdf = tmp_path / "docs" / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 alpha")
    monkeypatch.setenv("SILICA_INVENTORY", str(_manifest(tmp_path, {"docs/scan.pdf": 1.0})))

    first = root._pdf_text("docs/scan.pdf", pdf)[0]
    second = root._pdf_text("docs/scan.pdf", pdf)[0]
    assert first == second and "alpha" in first
    assert len(extractions) == 1, "an unchanged stamp must reuse the extraction cache"
