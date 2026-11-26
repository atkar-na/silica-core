# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""A pathological document must not stall or poison an index pass, and a pass
must not rewrite the whole store while it runs.

What this pins:

* per-document crash isolation: a worker that dies no longer fails the rest of
  its generation — the document is retried alone in a fresh interpreter and the
  pass carries on;
* quarantine: when the isolated retry dies or outlives `SILICA_INDEX_TIMEOUT`
  the document is recorded with a machine-readable reason and never retried in
  the same pass, within a budget (`SILICA_INDEX_MAX_QUARANTINE`);
* the oversize guard: a document past `SILICA_INDEX_MAX_MB` (or
  `SILICA_INDEX_MAX_PAGES`) is sampled, and a sample that carries almost no text
  is recorded as `oversize_low_text` with what it measured, never indexed as if
  it were whole;
* a sampled read that does carry text is indexed from the pages read, and the
  pass says so (`truncated`) instead of claiming the document was read whole;
* write amplification: the store is serialised at most once per
  `SILICA_INDEX_BATCH` generation (plus the pass's own final flush), never once
  per document and never on a timer.
"""
from __future__ import annotations

import json
import math
import os
import signal
import sys
import time

import orjson
import pytest

from silica_core.core import (
    _QUARANTINE_BUDGET,
    _QUARANTINE_CRASHED,
    _QUARANTINE_STALLED,
    _index_max_mb,
    _index_max_pages,
    _index_max_quarantine,
)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Six notes under a temporary root, with an isolated index dir."""
    from silica_core.config import CONFIG
    from silica_core.kernel.recall import paths

    root_dir = tmp_path / "corpus"
    root_dir.mkdir()
    for i in range(6):
        (root_dir / f"n{i}.md").write_text(
            f"# Note {i}\n\ncompaction merges runs; note {i} alpha beta gamma delta.\n",
            encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(root_dir))
    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=root_dir / ".idx": _d)
    import silica_core.core as core
    core._section_cache.clear()
    return core, root_dir, tmp_path


def _documents(root_dir, count, body="alpha beta gamma delta"):
    for i in range(count):
        (root_dir / f"d{i}.md").write_text(f"# Doc {i}\n\n{body} {i}\n", encoding="utf-8")


def _run(core, monkeypatch, *, procs=None, batch=None, timeout=None, quarantine=None):
    if procs is not None:
        monkeypatch.setenv("SILICA_INDEX_PROCESSES", str(procs))
    if batch is not None:
        monkeypatch.setenv("SILICA_INDEX_BATCH", str(batch))
    if timeout is not None:
        monkeypatch.setenv("SILICA_INDEX_TIMEOUT", str(timeout))
    if quarantine is not None:
        monkeypatch.setenv("SILICA_INDEX_MAX_QUARANTINE", str(quarantine))
    import silica_core.kernel.recall.lexical as lex
    lex._STORE_CACHE.clear()
    return core.build_index()


def _store(index_dir):
    return json.loads((index_dir / "lexical.json").read_text(encoding="utf-8"))


# An isolated retry that reproduces a native crash for the documents named in
# SILICA_TEST_CRASH_PREFIX, and behaves like the real retry for everything else.
_ISOLATE_SCRIPT = r"""
import os, sys
from silica_core import _isolate
if sys.argv[2].startswith(os.environ.get("SILICA_TEST_CRASH_PREFIX", "\0")):
    raise SystemExit(3)
sys.exit(_isolate.main(sys.argv[1:]))
"""


def _crashing_isolate(core, monkeypatch, prefix=""):
    """The isolated retry dies for documents under `prefix` (all of them when
    it is empty), exactly as a native PDFium crash would."""
    monkeypatch.setenv("SILICA_TEST_CRASH_PREFIX", prefix if prefix else "")
    monkeypatch.setattr(
        core, "_isolate_cmd",
        lambda rel, root: [sys.executable, "-c", _ISOLATE_SCRIPT, str(root), rel])


def _stalling_isolate(core, monkeypatch, seconds=60):
    monkeypatch.setattr(
        core, "_isolate_cmd",
        lambda rel, root: [sys.executable, "-c", f"import time; time.sleep({seconds})"])


# ---------------------------------------------------------------------------
# a. the guard's limits
# ---------------------------------------------------------------------------

def test_the_guard_limits_default_to_bounded_values(monkeypatch):
    for name in ("SILICA_INDEX_MAX_MB", "SILICA_INDEX_MAX_PAGES",
                 "SILICA_INDEX_MAX_QUARANTINE"):
        monkeypatch.delenv(name, raising=False)
    assert _index_max_mb() == 64.0
    assert _index_max_pages() == 400
    assert _index_max_quarantine() == 200
    for name in ("SILICA_INDEX_MAX_MB", "SILICA_INDEX_MAX_PAGES"):
        monkeypatch.setenv(name, "0")
    assert _index_max_mb() == 0.0 and _index_max_pages() == 0, "0 disables a guard"
    monkeypatch.setenv("SILICA_INDEX_MAX_MB", "nonsense")
    monkeypatch.setenv("SILICA_INDEX_MAX_PAGES", "nonsense")
    assert _index_max_mb() == 64.0 and _index_max_pages() == 400


# ---------------------------------------------------------------------------
# b. per-document crash isolation
# ---------------------------------------------------------------------------

def test_a_dying_worker_is_retried_alone_and_the_pass_finishes(corpus, monkeypatch):
    """One document kills its worker; the rest of the generation is not failed
    for it. The victim is re-run alone in a fresh interpreter and indexed."""
    core, root_dir, tmp = corpus
    _documents(root_dir, 6)
    real = core._doc_result

    def killer(rel, root):
        # Only the pool workers see this: the isolated retry is a fresh
        # interpreter that imports the package, not this monkeypatched module.
        if rel == "d2.md":
            os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(0.02)
        return real(rel, root)

    monkeypatch.setattr(core, "_doc_result", killer)
    out = _run(core, monkeypatch, procs=4, batch=4, timeout=20)
    assert out["failed"] == {}, f"the generation was poisoned: {out['failed']}"
    assert out["docs"] == 12 and out["index"]["state"] == "ready"
    assert set(_store(root_dir / ".idx")["docs"]) == (
        {f"n{i}.md" for i in range(6)} | {f"d{i}.md" for i in range(6)})


def test_a_document_whose_retry_also_dies_is_quarantined(corpus, monkeypatch):
    core, root_dir, tmp = corpus
    _documents(root_dir, 2)
    real = core._doc_result
    victim = "d1.md"

    def killer(rel, root):
        if rel == victim:
            os.kill(os.getpid(), signal.SIGKILL)
        return real(rel, root)

    monkeypatch.setattr(core, "_doc_result", killer)
    _crashing_isolate(core, monkeypatch, prefix="d1.md")
    out = _run(core, monkeypatch, procs=2, batch=1, timeout=20)

    assert out["failed"] == {victim: _QUARANTINE_CRASHED}, out["failed"]
    assert out["index"]["state"] == "stale", "a quarantined document is never reported ready"
    docs = _store(tmp / "corpus" / ".idx")["docs"]
    assert victim not in docs, "a document that crashed is not indexed from a partial read"
    assert f"d0.md" in docs and "n0.md" in docs, sorted(docs)


def test_a_retry_that_outlives_the_timeout_is_quarantined(corpus, monkeypatch):
    core, root_dir, _tmp = corpus
    _documents(root_dir, 1)
    real = core._doc_result
    victim = "d0.md"

    def killer(rel, root):
        if rel == victim:
            os.kill(os.getpid(), signal.SIGKILL)
        return real(rel, root)

    monkeypatch.setattr(core, "_doc_result", killer)
    _stalling_isolate(core, monkeypatch, seconds=60)
    started = time.monotonic()
    out = _run(core, monkeypatch, procs=2, batch=1, timeout=2)
    assert time.monotonic() - started < 40, "the pass waited on the isolated retry"
    assert out["failed"] == {victim: _QUARANTINE_STALLED}
    assert out["index"]["state"] == "stale"


def test_the_quarantine_budget_stops_a_corpus_wide_crash(corpus, monkeypatch):
    """Past `SILICA_INDEX_MAX_QUARANTINE` the pass records the reason without
    spending another subprocess, and still finishes the rest."""
    core, root_dir, _tmp = corpus
    _documents(root_dir, 6)
    real = core._doc_result

    def killer(rel, root):
        if rel.startswith("d"):
            os.kill(os.getpid(), signal.SIGKILL)
        return real(rel, root)

    monkeypatch.setattr(core, "_doc_result", killer)
    _crashing_isolate(core, monkeypatch)
    retries = []
    cmd = core._isolate_cmd
    monkeypatch.setattr(core, "_isolate_cmd",
                        lambda rel, root: (retries.append(rel), cmd(rel, root))[1])
    out = _run(core, monkeypatch, procs=2, batch=1, timeout=20, quarantine=2)

    reasons = sorted(out["failed"].values())
    assert reasons.count(_QUARANTINE_CRASHED) == 2, reasons
    assert reasons.count(_QUARANTINE_BUDGET) == 4, reasons
    assert len(retries) == 2, f"{len(retries)} isolated retries past a budget of 2"
    assert out["docs"] == 6, "the healthy documents are still indexed"
    assert out["index"]["state"] == "stale"


# ---------------------------------------------------------------------------
# c. oversize / fruitless-document guard
# ---------------------------------------------------------------------------

def _drawing_pdf(path, pages=6, text="x"):
    from tests.doc_factory import pdf_bytes
    path.write_bytes(pdf_bytes([text] * pages))


def test_an_oversize_drawing_set_is_skipped_with_its_measurement(corpus, monkeypatch):
    core, root_dir, tmp = corpus
    _drawing_pdf(root_dir / "set.pdf", pages=6, text="x")
    monkeypatch.setenv("SILICA_INDEX_MAX_MB", "0.000001")  # every PDF is oversize
    monkeypatch.setenv("SILICA_INDEX_MAX_PAGES", "3")
    out = _run(core, monkeypatch, procs=1)

    meta = core._load_meta()
    entry = meta["no_text"]["set.pdf"]
    assert entry["reason"].startswith("oversize_low_text:"), entry
    assert "chars from 3 of 6 pages" in entry["reason"], entry
    assert "set.pdf" not in meta["stamps"]
    assert "set.pdf" not in _store(root_dir / ".idx")["docs"]
    assert out["index"]["state"] in ("stale", "ready")
    row = {f["path"]: f for f in core.files()["files"]}["set.pdf"]
    assert row["status"] == "unconverted" and row["reason"].startswith("oversize_low_text")


def test_a_page_cap_alone_can_skip_a_fruitless_document(corpus, monkeypatch):
    core, root_dir, _tmp = corpus
    _drawing_pdf(root_dir / "many.pdf", pages=5, text="x")
    monkeypatch.setenv("SILICA_INDEX_MAX_MB", "0")
    monkeypatch.setenv("SILICA_INDEX_MAX_PAGES", "2")
    _run(core, monkeypatch, procs=1)

    entry = core._load_meta()["no_text"]["many.pdf"]
    assert entry["reason"].startswith("page_cap_low_text:"), entry
    assert "chars from 2 of 5 pages" in entry["reason"]


def test_a_sampled_read_with_text_is_indexed_and_reported_as_truncated(corpus, monkeypatch):
    core, root_dir, _tmp = corpus
    from tests.doc_factory import pdf_bytes
    page = "compaction merges sorted runs; leveled compaction bounds read amplification. " * 4
    (root_dir / "big.pdf").write_bytes(pdf_bytes([page] * 4))
    monkeypatch.setenv("SILICA_INDEX_MAX_MB", "0.000001")
    monkeypatch.setenv("SILICA_INDEX_MAX_PAGES", "2")
    out = _run(core, monkeypatch, procs=1)

    assert "big.pdf" in _store(root_dir / ".idx")["docs"], "text found is text indexed"
    assert out["truncated"]["big.pdf"] == {
        "bytes": (root_dir / "big.pdf").stat().st_size, "pages_read": 2, "pages_total": 4}
    # the partial page map is not written where `read` and `page_map` would serve it
    assert not (root_dir / ".idx" / "extract").glob("*.json") or all(
        "big" not in p.read_text(encoding="utf-8", errors="replace")
        for p in (root_dir / ".idx" / "extract").glob("*.json"))


# ---------------------------------------------------------------------------
# d. write amplification
# ---------------------------------------------------------------------------

def test_the_store_is_written_once_per_generation(corpus, monkeypatch):
    """N documents through a batch of B: at most ceil(N/B) writes for the
    generations plus the pass's own final flush — not one per document, and not
    one per timer tick."""
    core, root_dir, tmp = corpus
    import silica_core.kernel.recall.lexical as lex

    n, batch = 12, 3
    _documents(root_dir, n)
    calls = []
    real = lex.LexicalStore._serialize

    def counting(self, snapshot):
        calls.append(len(snapshot[0]))
        return real(self, snapshot)

    monkeypatch.setattr(lex.LexicalStore, "_serialize", counting)
    out = _run(core, monkeypatch, procs=1, batch=batch)

    total = 6 + n  # the fixture's notes plus the new ones
    assert out["docs"] == total
    assert 1 <= len(calls) <= math.ceil(total / batch) + 1, \
        f"{len(calls)} serialisations for {total} documents at batch {batch}"


def test_the_parallel_pass_writes_at_the_same_boundary(corpus, monkeypatch):
    core, root_dir, _tmp = corpus
    import silica_core.kernel.recall.lexical as lex

    n, batch = 8, 2
    _documents(root_dir, n)
    calls = []
    real = lex.LexicalStore._serialize

    def counting(self, snapshot):
        calls.append(1)
        return real(self, snapshot)

    monkeypatch.setattr(lex.LexicalStore, "_serialize", counting)
    out = _run(core, monkeypatch, procs=2, batch=batch)
    total = 6 + n
    assert out["docs"] == total
    assert len(calls) <= math.ceil(total / batch) + 1, len(calls)


def test_an_unchanged_pass_does_not_rewrite_the_store(corpus, monkeypatch):
    core, root_dir, _tmp = corpus
    import silica_core.kernel.recall.lexical as lex

    _run(core, monkeypatch, procs=1)
    calls = []
    real = lex.LexicalStore._serialize
    monkeypatch.setattr(lex.LexicalStore, "_serialize",
                        lambda self, snapshot: (calls.append(1), real(self, snapshot))[1])
    out = _run(core, monkeypatch, procs=1)
    assert out["changed"] == 0
    assert calls == [], "an index with nothing to change is not rewritten"
