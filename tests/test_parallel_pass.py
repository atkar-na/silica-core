# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""`SILICA_INDEX_PROCESSES`: an opt-in, process-parallel first index pass.

The first pass over a large corpus reads, extracts and tokenizes one document at
a time on a single core; the box has more. `SILICA_INDEX_PROCESSES=N` fans that
work out across N worker processes (stdlib multiprocessing; threads are the wrong
tool — the tokenizer is GIL-bound). The mode is opt-in and OFF at 1.

What this pins:

* byte identity — for one root, one manifest and one starting store, a
  4-process pass writes a byte-identical store to a 1-process pass, so a
  worker's finish order can never leak into the file;
* unset == 1 — the flag simply absent is exactly the single-process pass;
* crash safety — a worker killed mid-pass is surfaced as a failure, leaves the
  store valid, and a re-run finishes every document with none lost, none doubled;
* both answering sources — the pass works with `SILICA_INVENTORY` set and with
  the walk;
* the fan-out is real — the parallel mode reports the worker pids it used, which
  an implementation that ignored the flag could not.
"""
from __future__ import annotations

import json
import os
import signal
import time

import orjson
import pytest

from silica_core.core import _WORKER_DIED


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A small root of six notes, with an isolated index dir under tmp_path."""
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


def _write_manifest(tmp_path, root_dir):
    """A SILICA_INVENTORY manifest for `root_dir`: the six notes, stamped by mtime."""
    from silica_core.core import _root

    stamps = {p.relative_to(root_dir).as_posix(): int(p.stat().st_mtime_ns)
              for p in sorted(root_dir.rglob("*.md"))}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"version": 1, "root": str(_root()),
                                "count": len(stamps), "stamps": stamps}), encoding="utf-8")
    return path


def _pass(core, monkeypatch, index_dir, processes=None, manifest=None):
    """Run one `build_index()` into `index_dir` under the given flag state and
    return `(result, lexical.json bytes)`."""
    from silica_core.kernel.recall import paths
    import silica_core.kernel.recall.lexical as lex

    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=index_dir: _d)
    if processes is None:
        monkeypatch.delenv("SILICA_INDEX_PROCESSES", raising=False)
    else:
        monkeypatch.setenv("SILICA_INDEX_PROCESSES", str(processes))
    if manifest is None:
        monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    else:
        monkeypatch.setenv("SILICA_INVENTORY", str(manifest))
    lex._STORE_CACHE.clear()   # the store singleton keys on the path; start fresh
    core._section_cache.clear()
    out = core.build_index()
    return out, (index_dir / "lexical.json").read_bytes()


def _stable(out):
    """The result without the one field that is a clock reading, so two passes
    can be compared exactly."""
    o = dict(out)
    idx = dict(o.pop("index"))
    idx.pop("built_at", None)
    o["index"] = idx
    return o


def test_processes_env_is_parsed_and_bounded(monkeypatch):
    from silica_core.core import _index_processes

    monkeypatch.delenv("SILICA_INDEX_PROCESSES", raising=False)
    assert _index_processes() == 1
    for raw, want in [("1", 1), ("4", 4), (" 2 ", 2), ("0", 1), ("-3", 1), ("x", 1), ("", 1)]:
        monkeypatch.setenv("SILICA_INDEX_PROCESSES", raw)
        assert _index_processes() == want, raw


@pytest.mark.parametrize("with_manifest", [True, False])
def test_store_is_byte_identical_across_process_counts(corpus, tmp_path, monkeypatch, with_manifest):
    """(a) and (d): N=1 and N=4 write the same bytes, with the manifest set and
    with the walk."""
    core, root_dir, _ = corpus
    manifest = _write_manifest(tmp_path, root_dir) if with_manifest else None
    out1, b1 = _pass(core, monkeypatch, tmp_path / "idx-1", processes=1, manifest=manifest)
    out4, b4 = _pass(core, monkeypatch, tmp_path / "idx-4", processes=4, manifest=manifest)
    assert out1["docs"] == out4["docs"] == 6
    assert b1 == b4, "the store must be byte-identical across process counts"


def test_the_flag_unset_is_exactly_one_process(corpus, tmp_path, monkeypatch):
    """(b): an absent flag and `=1` are the same pass, byte for byte and reply
    for reply."""
    core, root_dir, _ = corpus
    out_unset, b_unset = _pass(core, monkeypatch, tmp_path / "idx-u", processes=None)
    out_one, b_one = _pass(core, monkeypatch, tmp_path / "idx-1", processes=1)
    assert b_unset == b_one
    assert _stable(out_unset) == _stable(out_one)
    assert "parallel" not in out_unset and "parallel" not in out_one


def test_the_parallel_mode_really_fans_out(corpus, tmp_path, monkeypatch):
    """(e): the flag engages real worker processes. A no-op implementation that
    ignored it would report a single worker (or no `parallel` block at all)."""
    core, root_dir, _ = corpus
    real = core._doc_result

    def slow(rel, root_dir):
        time.sleep(0.15)   # widen the window so the workers overlap
        return real(rel, root_dir)

    monkeypatch.setattr(core, "_doc_result", slow)
    out, _ = _pass(core, monkeypatch, tmp_path / "idx", processes=4)
    assert out["parallel"]["processes"] == 4
    assert out["parallel"]["workers"] >= 2, "the flag must actually engage worker processes"


def test_a_killed_worker_leaves_the_store_consistent_and_a_rerun_completes(corpus, tmp_path, monkeypatch):
    """(c): a worker killed mid-pass is surfaced, the store stays a valid
    document, and a re-run finishes every document — none lost, none duplicated."""
    core, root_dir, _ = corpus
    real = core._doc_result
    victim = "n3.md"

    def killer(rel, root_dir):
        if rel == victim:
            os.kill(os.getpid(), signal.SIGKILL)   # a worker dies mid-pass
        time.sleep(0.02)   # keep a few documents in flight across workers
        return real(rel, root_dir)

    monkeypatch.setattr(core, "_doc_result", killer)
    idx = tmp_path / "idx"
    out, blob = _pass(core, monkeypatch, idx, processes=4)
    assert out["failed"], "a killed worker must be reported, never a silent success"
    assert out["index"]["state"] == "stale"
    assert victim in out["failed"] or _WORKER_DIED in out["failed"].values()
    # the store on disk is still a valid, consistent document with no partial row
    loaded = orjson.loads(blob)
    assert set(loaded["docs"]) <= {f"n{i}.md" for i in range(6)}
    assert set(loaded["len"]) == set(loaded["docs"])

    # the re-run finishes the documents the dead worker never delivered
    monkeypatch.setattr(core, "_doc_result", real)
    out2, blob2 = _pass(core, monkeypatch, idx, processes=4)
    assert out2["failed"] == {}
    assert out2["index"]["state"] == "ready"
    assert out2["docs"] == 6
    loaded2 = orjson.loads(blob2)
    assert set(loaded2["docs"]) == {f"n{i}.md" for i in range(6)}
    assert len(loaded2["docs"]) == 6   # none lost, none duplicated
    assert set(loaded2["len"]) == set(loaded2["docs"])
