# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""`SILICA_INDEX_BATCH` and `SILICA_INDEX_TIMEOUT`: a long pass that leaves
progress behind, and a pass that cannot be held hostage by one document.

What this pins:

* a generation boundary flushes the store, so an interrupted run loses one
  batch rather than the whole pass, and progress is observable between batches;
* a worker stuck on a document is abandoned after the timeout and the pass
  still finishes the other documents in a fresh pool — without the limit the
  parent waits on that future forever;
* the abandoned document is named in `failed`, so the next pass re-runs it and
  nobody has to guess which file stalled the run;
* both limits parse sanely and default off/on as documented.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from silica_core.core import _WORKER_STALLED, _index_batch, _index_timeout


def _stalling_worker(item):
    """Module-level so the process pool can pickle it by reference.

    The document to stall is named in an env var: a forked worker inherits it."""
    from silica_core import core

    rel = item[0]
    if os.environ.get("SILICA_TEST_STALL_REL") and rel.endswith(
            os.environ["SILICA_TEST_STALL_REL"]):
        time.sleep(120)  # a parser caught in a loop
    return core._real_index_worker(item)


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


def _manifest(tmp_path, root_dir):
    from silica_core.core import _root

    stamps = {p.relative_to(root_dir).as_posix(): int(p.stat().st_mtime_ns)
              for p in sorted(root_dir.rglob("*.md"))}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"version": 1, "root": str(_root()),
                                "count": len(stamps), "stamps": stamps}), encoding="utf-8")
    return path


def _run(core, monkeypatch, index_dir, *, procs=2, batch=2, timeout=0, manifest=None):
    monkeypatch.setenv("SILICA_INDEX_PROCESSES", str(procs))
    monkeypatch.setenv("SILICA_INDEX_BATCH", str(batch))
    monkeypatch.setenv("SILICA_INDEX_TIMEOUT", str(timeout))
    if manifest is not None:
        monkeypatch.setenv("SILICA_INVENTORY", str(manifest))
    started = time.monotonic()
    result = core.build_index()
    return result, time.monotonic() - started


def _docs(index_dir):
    store = json.loads((index_dir / "lexical.json").read_text(encoding="utf-8"))
    return store.get("docs", {})


def test_the_limits_default_to_on_and_bounded(monkeypatch):
    monkeypatch.delenv("SILICA_INDEX_BATCH", raising=False)
    monkeypatch.delenv("SILICA_INDEX_TIMEOUT", raising=False)
    assert _index_batch() == 256
    assert _index_timeout() == 300.0
    monkeypatch.setenv("SILICA_INDEX_BATCH", "0")
    assert _index_batch() == 256, "a nonsensical batch falls back to the default"
    monkeypatch.setenv("SILICA_INDEX_TIMEOUT", "0")
    assert _index_timeout() == 0.0, "0 disables the timeout"
    monkeypatch.setenv("SILICA_INDEX_TIMEOUT", "not-a-number")
    assert _index_timeout() == 300.0


def test_each_generation_flushes_the_store(corpus, monkeypatch):
    core, root_dir, tmp = corpus
    manifest = _manifest(tmp, root_dir)
    index_dir = root_dir / ".idx"
    flushes = []
    real_flush = core._flush

    def counting_flush(*args, **kwargs):
        flushes.append(1)
        return real_flush(*args, **kwargs)

    monkeypatch.setattr(core, "_flush", counting_flush)
    result, _wall = _run(core, monkeypatch, index_dir, batch=2, manifest=manifest)
    assert len(_docs(index_dir)) == 6
    # three generations of two, each flushed, plus the pass's own final flush
    assert len(flushes) >= 4, f"only {len(flushes)} flushes; generations did not close"
    assert result["index"]["state"] in ("ready", "stale")


def test_a_stalled_document_is_abandoned_and_the_pass_finishes(corpus, monkeypatch):
    core, root_dir, tmp = corpus
    manifest = _manifest(tmp, root_dir)
    index_dir = root_dir / ".idx"
    monkeypatch.setattr(core, "_real_index_worker", core._index_worker, raising=False)
    monkeypatch.setattr(core, "_index_worker", _stalling_worker)
    monkeypatch.setenv("SILICA_TEST_STALL_REL", "n3.md")
    result, wall = _run(core, monkeypatch, index_dir, batch=2, timeout=3, manifest=manifest)
    assert wall < 30, f"the pass waited {wall:.1f}s on the stalled document"
    assert "n3.md" in result["failed"], result["failed"]
    assert result["failed"]["n3.md"] == _WORKER_STALLED
    docs = _docs(index_dir)
    assert set(docs) == {"n0.md", "n1.md", "n2.md", "n4.md", "n5.md"}, sorted(docs)
