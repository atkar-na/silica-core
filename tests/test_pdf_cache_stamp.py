# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""The PDF extraction cache keys a legacy bare-list artifact on the manifest's
identity stamp too, not on the filesystem mtime alone.

An entry written before the stamp field existed is a JSON bare array of page
records beside the text (`.txt` + `.json`). Both files live under the index and
are written locally, so on a `--no-modtime` mount the cache's own mtime is real
and always newer than the source's constant one: the old reuse test
(``cache mtime >= source mtime``) can never fire a miss, and a PDF whose
manifest stamp changed kept serving its old text and page bounds for the life
of the process. The stamped form this build writes records the source's
identity beside the pages and compares it, so it already misses; the bare list
records nothing, so when ``SILICA_INVENTORY`` has a stamp for the document it
cannot be held to it and is a miss — the re-extraction that follows rewrites it
in the stamped form. With no manifest stamp the identity is the filesystem's
own and the artifact is honoured by the mtime comparison it was written under,
exactly as upstream. A directory holding both forms serves both.

The converter is stubbed: the test environment has no PDF lane, and the point
is which artifact `_pdf_text` trusts, not how PDFium reads a page.
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
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=tmp_path / ".idx": _d)
    import silica_core.core as core
    core._section_cache.clear()
    core._sha_cache.clear()
    return core


def _manifest(tmp_path, stamps):
    """A manifest in the frozen shape (`INVENTORY.md`), over this tmp root."""
    from silica_core.core import _root

    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"version": 1, "root": str(_root()), "count": len(stamps),
                                "stamps": stamps, "unextracted": []}), encoding="utf-8")
    return path


def _converter(monkeypatch):
    """A stub PDFium: a page's text is the source file's bytes, and every open
    is counted, so a test can tell a reuse from a re-extraction without a real
    PDF. `_pdf_text` imports the module inside the call, so patch its attribute."""
    pdfium = pytest.importorskip("pypdfium2")
    extractions: list[str] = []

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
            with open(path, "rb") as fh:
                self._body = fh.read().decode("utf-8", "replace")

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return _Page(self._body)

        def close(self):
            pass

    monkeypatch.setattr(pdfium, "PdfDocument", _Doc)
    return extractions


def _legacy_artifact(core, rel, text, pages):
    """The artifact an older build left: a text file beside a bare page array
    (a JSON list, no `stamp` field)."""
    cache = core._extract_cache(rel)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    cache.with_suffix(".json").write_bytes(json.dumps(pages).encode("utf-8"))
    return cache


def _replace_in_place(full, new_bytes):
    """The fault: swap the bytes at the same path, keep the size, reset mtime so
    the filesystem reports nothing different (a `--no-modtime` mount does this
    by itself)."""
    mtime_ns = full.stat().st_mtime_ns
    assert len(new_bytes) == full.stat().st_size, "the replacement must keep the size"
    full.write_bytes(new_bytes)
    os.utime(full, ns=(mtime_ns, mtime_ns))
    assert full.stat().st_mtime_ns == mtime_ns


def test_a_legacy_bare_list_with_a_changed_stamp_is_re_extracted(root, tmp_path, monkeypatch):
    """The reported fault: a bare-list artifact is newer than the source, so the
    mtime reuse would serve it, yet the manifest says the PDF changed."""
    extractions = _converter(monkeypatch)
    pdf = tmp_path / "docs" / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 alpha")
    _legacy_artifact(root, "docs/scan.pdf", "alpha\n\n", [{"page": 1, "line": 1, "offset": 0}])
    monkeypatch.setenv("SILICA_INVENTORY", str(_manifest(tmp_path, {"docs/scan.pdf": 100.0})))

    _replace_in_place(pdf, b"%PDF-1.4 bravo")
    monkeypatch.setenv("SILICA_INVENTORY", str(_manifest(tmp_path, {"docs/scan.pdf": 200.0})))

    text, _pages, _reason = root._pdf_text("docs/scan.pdf", pdf)
    assert "bravo" in text and "alpha" not in text, "the changed stamp must not serve the old text"
    assert len(extractions) == 1, "the bare list must be re-extracted, not reused on mtime"
    payload = json.loads(root._extract_cache("docs/scan.pdf").with_suffix(".json").read_bytes())
    assert isinstance(payload, dict) and payload.get("stamp") == 200.0, \
        "the re-extraction rewrites the artifact in the stamped form"


def test_a_legacy_bare_list_is_migrated_on_read_then_reused_while_the_stamp_holds(root, tmp_path, monkeypatch):
    """A bare list records no stamp, so it is adopted once (re-extracted, then
    rewritten stamped); from then on, while the manifest stamp is unchanged, it
    is served from the cache with no further extraction."""
    extractions = _converter(monkeypatch)
    pdf = tmp_path / "docs" / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 alpha")
    _legacy_artifact(root, "docs/scan.pdf", "stale\n\n", [{"page": 1, "line": 1, "offset": 0}])
    monkeypatch.setenv("SILICA_INVENTORY", str(_manifest(tmp_path, {"docs/scan.pdf": 100.0})))

    first, _pages, _reason = root._pdf_text("docs/scan.pdf", pdf)
    assert "alpha" in first and "stale" not in first
    assert len(extractions) == 1, "the unstamped artifact is re-extracted once"

    second, _pages, _reason = root._pdf_text("docs/scan.pdf", pdf)
    assert second == first
    assert len(extractions) == 1, "an unchanged stamp reuses the now-stamped artifact"


def test_a_legacy_bare_list_with_no_manifest_keeps_the_mtime_reuse(root, tmp_path, monkeypatch):
    """Requirement 2: with `SILICA_INVENTORY` unset the upstream rule stands —
    a cache as new as the source is reused and the converter never runs."""
    extractions = _converter(monkeypatch)
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    pdf = tmp_path / "docs" / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 alpha")
    pages = [{"page": 1, "line": 1, "offset": 0}]
    _legacy_artifact(root, "docs/scan.pdf", "cached\n\n", pages)

    text, got_pages, _reason = root._pdf_text("docs/scan.pdf", pdf)
    assert text.strip() == "cached" and got_pages == pages
    assert extractions == [], "no manifest: an unchanged mtime still reuses the old artifact"


def test_the_current_form_still_detects_a_changed_stamp(root, tmp_path, monkeypatch):
    """The stamped form this build writes keeps detecting a moved stamp."""
    extractions = _converter(monkeypatch)
    pdf = tmp_path / "docs" / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 alpha")
    monkeypatch.setenv("SILICA_INVENTORY", str(_manifest(tmp_path, {"docs/scan.pdf": 1.0})))
    assert "alpha" in root._pdf_text("docs/scan.pdf", pdf)[0]

    _replace_in_place(pdf, b"%PDF-1.4 bravo")
    monkeypatch.setenv("SILICA_INVENTORY", str(_manifest(tmp_path, {"docs/scan.pdf": 2.0})))

    text, _pages, _reason = root._pdf_text("docs/scan.pdf", pdf)
    assert "bravo" in text and "alpha" not in text
    assert len(extractions) == 2


def test_a_mixed_extract_cache_serves_both_forms(root, tmp_path, monkeypatch):
    """A directory holding a stamped entry and a legacy one: the stamped entry
    is reused, the bare list is re-extracted, and both are served correctly."""
    extractions = _converter(monkeypatch)
    stamped = tmp_path / "docs" / "stamped.pdf"
    stamped.write_bytes(b"%PDF-1.4 keep")
    legacy = tmp_path / "docs" / "legacy.pdf"
    legacy.write_bytes(b"%PDF-1.4 newtext")
    monkeypatch.setenv("SILICA_INVENTORY",
                       str(_manifest(tmp_path, {"docs/stamped.pdf": 5.0, "docs/legacy.pdf": 7.0})))

    root._pdf_text("docs/stamped.pdf", stamped)  # this build writes the stamped form
    assert len(extractions) == 1
    _legacy_artifact(root, "docs/legacy.pdf", "oldtext\n\n", [{"page": 1, "line": 1, "offset": 0}])

    assert "keep" in root._pdf_text("docs/stamped.pdf", stamped)[0]
    assert len(extractions) == 1, "the stamped entry is reused"
    assert "newtext" in root._pdf_text("docs/legacy.pdf", legacy)[0]
    assert len(extractions) == 2, "the bare list is re-extracted"
