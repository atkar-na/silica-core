# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""Adversarial acceptance tests for the process-parallel first index pass (S4).

The frozen interface is one environment variable:

* ``SILICA_INDEX_PROCESSES`` — an integer, default ``1``. Unset or ``1`` is
  today's single-process behaviour. Greater than ``1`` distributes the
  document work of the *first* index pass across that many worker processes.
* The resulting store file (``lexical.json``) must be **byte-identical** to the
  ``N=1`` result for the same root + manifest + starting store.
* A worker killed mid-pass must leave the store consistent, must be reported as
  a **failure** (never a silent success), and must leave the pass re-runnable
  to completion with no document lost and none duplicated.

Nothing here reads the implementation's internals: every pass runs in its own
interpreter with the ``SILICA_*`` environment already set (see
``_parallel_pass_driver``), and the suite asserts on the store bytes, the
document/stamp sets, the reported result, and the worker processes the pass
actually spawned. So the same suite runs unchanged against whatever branch
lands the implementation.

Byte-identity is a real constraint, not decoration: the store is serialised by
``orjson.dumps`` without key sorting, so the *insertion order* of ``docs`` /
``len`` / ``name`` is part of the bytes. A pass that merges worker results in
completion order produces different bytes run to run; the ``N=1`` order is the
walk order. Test 1 and test 3 pin that.

These tests are EXPECTED TO FAIL until the implementation lands — that is
correct. Do not weaken an assertion to make one pass.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DRIVER = Path(__file__).resolve().parent / "_parallel_pass_driver.py"

# The flag string the implementation must honour; the frozen interface.
_FLAG = "SILICA_INDEX_PROCESSES"


# --------------------------------------------------------------------------
# corpus fixtures — built in tmp_path, identical bytes for every run
# --------------------------------------------------------------------------

# Uneven on purpose: reading + tokenizing is pure Python, so a 400 KB note is
# two orders of magnitude slower than a 2 KB one. That spread is what makes a
# completion-order merge nondeterministic (test 3).
_UNEVEN_SIZES: tuple[int, ...] = (
    1_500, 2_000, 3_000, 5_000, 7_000, 8_000, 12_000, 18_000,
    25_000, 40_000, 60_000, 90_000, 140_000, 200_000, 280_000, 400_000,
)

# The crash corpus: mid-sized documents, enough of them that a pass takes long
# enough to SIGKILL a worker inside it.
_CRASH_SIZES: tuple[int, ...] = tuple(30_000 + 7_000 * i for i in range(24))


def _text(i: int, target: int) -> str:
    """A note of at least ``target`` bytes whose tokens are unique per ``i``."""
    parts = [f"# note{i}\n\n"]
    size = len(parts[0])
    n = 0
    while size < target:
        n += 1
        chunk = f"tok{i}_{n} freq{i}_{n % 11} lorem ipsum dolor sit amet "
        parts.append(chunk)
        size += len(chunk)
    return "".join(parts)


def _write_corpus(root: Path, sizes: tuple[int, ...]) -> None:
    """A corpus exercising prose, an unextracted document, and exclusions."""
    (root / "notes").mkdir(parents=True, exist_ok=True)
    for i, size in enumerate(sizes):
        (root / "notes" / f"n{i:03d}.md").write_text(_text(i, size), encoding="utf-8")
    # An office document nothing extracted: a document the index holds no text
    # for (the `unextracted` side of the set).
    (root / "office").mkdir()
    (root / "office" / "report.docx").write_bytes(b"PK\x03\x04not really a docx")
    # Exclusions the walk never enters, so they are in neither set.
    (root / ".silicaignore").write_text("skipme.md\n", encoding="utf-8")
    (root / "skipme.md").write_text("# skipped\n", encoding="utf-8")
    (root / ".hidden.md").write_text("# hidden\n", encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "out.md").write_text("# build output\n", encoding="utf-8")


# --------------------------------------------------------------------------
# running a pass in its own interpreter
# --------------------------------------------------------------------------

def _run_pass(
    tmp_path: Path,
    vault: Path,
    *,
    processes: int | None,
    tag: str,
    inventory: Path | None = None,
    no_walk: bool = False,
    kill: bool = False,
    kill_delay: float = 0.2,
    flush_s: float | None = None,
    timeout: float = 240,
) -> dict:
    """Run one first index pass over ``vault`` and return the probe report.

    Every run gets its own ``SILICA_HOME`` (hence its own index dir), so two
    runs never share a store file and the store bytes are comparable.
    """
    home = tmp_path / f"home-{tag}"
    report = tmp_path / f"report-{tag}.json"
    env = dict(os.environ)
    env["SILICA_HOME"] = str(home)
    env["SILICA_VAULT"] = str(vault)
    env["PROBE_REPORT"] = str(report)
    env.pop("SILICA_INVENTORY", None)
    for key in ("PROBE_NO_WALK", "PROBE_KILL", "PROBE_KILL_DELAY", "PROBE_FLUSH_S"):
        env.pop(key, None)
    if inventory is not None:
        env["SILICA_INVENTORY"] = str(inventory)
    if processes is None:
        env.pop(_FLAG, None)
    else:
        env[_FLAG] = str(processes)
    if no_walk:
        env["PROBE_NO_WALK"] = "1"
    if kill:
        env["PROBE_KILL"] = "1"
        env["PROBE_KILL_DELAY"] = str(kill_delay)
    if flush_s is not None:
        env["PROBE_FLUSH_S"] = str(flush_s)

    try:
        proc = subprocess.run(
            [sys.executable, str(_DRIVER)],
            env=env, cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"the index pass (processes={processes}, tag={tag}) did not terminate "
            f"within {timeout}s; it hung rather than reporting the dead worker\n"
            f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e

    if not report.is_file():
        raise AssertionError(
            f"the probe wrote no report (rc={proc.returncode}, tag={tag}); the pass "
            f"crashed the interpreter\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    rep = json.loads(report.read_text(encoding="utf-8"))
    rep["_rc"] = proc.returncode
    rep["_stdout"] = proc.stdout
    rep["_stderr"] = proc.stderr
    assert proc.returncode == 0, f"probe exited {proc.returncode}: {proc.stderr}"
    return rep


def _store_bytes(rep: dict) -> bytes:
    return Path(rep["store"]).read_bytes()


def _store_docs(rep: dict) -> dict:
    return json.loads(_store_bytes(rep))["docs"]


def _stamps(rep: dict) -> dict:
    return json.loads((Path(rep["index_dir"]) / "stamps.json").read_bytes())["stamps"]


def _used_more_than_one_process(rep: dict) -> bool:
    """True when the pass genuinely ran work in more than one process."""
    return (
        len(rep["descendants"]) >= 2
        or rep["max_concurrent"] >= 2
        or rep["forks"] >= 2
    )


def _failure_note(rep: dict) -> str:
    return (
        f"\n  result={rep.get('result')!r}\n  error={rep.get('error')!r}\n"
        f"  descendants={rep.get('descendants')} forks={rep.get('forks')} "
        f"killed={rep.get('killed')}\n  stderr:\n{rep.get('_stderr', '')}"
    )


def _reported_as_failure(result) -> bool:
    """Whether a returned result admits the pass did not complete cleanly."""
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return True
    if result.get("failed"):
        return True
    index = result.get("index") or {}
    if index.get("state") not in ("ready",):
        return True
    if index.get("pending"):
        return True
    return False


@pytest.fixture
def uneven_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    _write_corpus(root, _UNEVEN_SIZES)
    return root


@pytest.fixture
def crash_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    _write_corpus(root, _CRASH_SIZES)
    return root


# --------------------------------------------------------------------------
# 1. byte-identity: N=1 vs N=4
# --------------------------------------------------------------------------

def test_the_store_is_byte_identical_across_process_counts(tmp_path, uneven_corpus):
    r1 = _run_pass(tmp_path, uneven_corpus, processes=1, tag="bi-n1")
    r4 = _run_pass(tmp_path, uneven_corpus, processes=4, tag="bi-n4")
    assert r1["ok"] and r4["ok"], "a pass failed" + _failure_note(r1) + _failure_note(r4)

    b1, b4 = _store_bytes(r1), _store_bytes(r4)
    assert b4 == b1, (
        "the N=4 store is not byte-identical to N=1; the store bytes carry the "
        "insertion order of docs/len/name, so a different order is a different file"
    )
    # The N=4 pass must actually have been parallel, or this compares N=1 to N=1.
    assert _used_more_than_one_process(r4), (
        "N=4 never ran a second process: the flag was accepted and ignored" + _failure_note(r4)
    )


# --------------------------------------------------------------------------
# 2. off-path identity: unset == 1, and neither forks
# --------------------------------------------------------------------------

def test_unset_flag_is_identical_to_one_and_changes_nothing(tmp_path, uneven_corpus):
    unset = _run_pass(tmp_path, uneven_corpus, processes=None, tag="off-unset")
    one = _run_pass(tmp_path, uneven_corpus, processes=1, tag="off-one")
    assert unset["ok"] and one["ok"], "a pass failed" + _failure_note(unset) + _failure_note(one)

    assert _store_bytes(unset) == _store_bytes(one), "unset must equal N=1 exactly"
    # Off the path there must be no parallel machinery at all: today's
    # single-process behaviour, no children.
    assert not _used_more_than_one_process(unset), (
        "the flag was unset yet more than one process ran" + _failure_note(unset)
    )
    assert not _used_more_than_one_process(one), (
        "N=1 spawned a worker; single-process behaviour changed" + _failure_note(one)
    )


# --------------------------------------------------------------------------
# 3. determinism vs completion order
# --------------------------------------------------------------------------

def test_two_n4_runs_agree_despite_an_uneven_corpus(tmp_path, uneven_corpus):
    """Two separate N=4 passes over a corpus where some documents are far slower
    than others must produce identical bytes: the result cannot depend on the
    order workers happen to finish."""
    a = _run_pass(tmp_path, uneven_corpus, processes=4, tag="det-a")
    b = _run_pass(tmp_path, uneven_corpus, processes=4, tag="det-b")
    assert a["ok"] and b["ok"], "a pass failed" + _failure_note(a) + _failure_note(b)

    assert _store_bytes(a) == _store_bytes(b), (
        "two N=4 runs disagree: the store depends on worker completion order"
    )
    assert _used_more_than_one_process(a) and _used_more_than_one_process(b), (
        "a determinism run was not parallel" + _failure_note(a) + _failure_note(b)
    )


# --------------------------------------------------------------------------
# 4. crash safety: kill a worker mid-pass
# --------------------------------------------------------------------------

def test_a_killed_worker_is_reported_and_the_pass_resumes_to_the_full_set(tmp_path, crash_corpus):
    reference = _run_pass(tmp_path, crash_corpus, processes=1, tag="kill-ref")
    assert reference["ok"], "the reference pass failed" + _failure_note(reference)
    ref_docs = _store_docs(reference)
    assert ref_docs, "the reference store is empty"

    # A flush after every document, so the killed pass leaves a partial store to
    # inspect rather than an empty one.
    killed = _run_pass(
        tmp_path, crash_corpus, processes=4, tag="kill-run",
        kill=True, kill_delay=0.15, flush_s=0.0,
    )
    assert killed["killed"], (
        "no worker process was observed to kill mid-pass; the suite cannot prove "
        "crash safety from a run that never went parallel" + _failure_note(killed)
    )
    # A killed worker is a failure, never a silent success.
    assert killed["error"] or _reported_as_failure(killed["result"]), (
        "the pass reported success after a worker was killed" + _failure_note(killed)
    )

    # The store on disk after the kill still parses and holds no partial or
    # duplicated row: every row present equals its row in the reference store.
    assert killed["store_parses"], "the store left by the killed pass does not parse"
    docs = _store_docs(killed)
    assert len(docs) == len(set(docs)), "the store holds a duplicated document"
    for rel, row in docs.items():
        assert rel in ref_docs, f"the store holds a document that is not in the corpus: {rel}"
        assert row == ref_docs[rel], (
            f"the store holds a partial/torn row for {rel}: a row must be written "
            "whole or not at all"
        )

    # And the pass is re-runnable to completion, with no document lost or
    # duplicated: a fresh full pass over the same corpus equals the reference.
    done = _run_pass(tmp_path, crash_corpus, processes=4, tag="kill-done")
    assert done["ok"], "the pass was not re-runnable to completion" + _failure_note(done)
    assert _store_bytes(done) == _store_bytes(reference), (
        "the re-run does not reproduce the N=1 store: a document was lost or doubled"
    )
    assert set(_store_docs(done)) == set(ref_docs)


# --------------------------------------------------------------------------
# 5. no lost or duplicated documents — set equality on keys
# --------------------------------------------------------------------------

def test_the_document_and_stamp_sets_equal_n1_exactly(tmp_path, uneven_corpus):
    r1 = _run_pass(tmp_path, uneven_corpus, processes=1, tag="set-n1")
    r4 = _run_pass(tmp_path, uneven_corpus, processes=4, tag="set-n4")
    assert r1["ok"] and r4["ok"], "a pass failed" + _failure_note(r1) + _failure_note(r4)

    s1, s4 = _stamps(r1), _stamps(r4)
    d1, d4 = _store_docs(r1), _store_docs(r4)

    assert set(s4) == set(s1), "the stamp set drifted from N=1 (a lost or invented key)"
    assert set(d4) == set(d1), "the stored document set drifted from N=1"
    # The stamp set and the stored set are the same documents, both runs.
    assert set(s1) == set(d1), "N=1 stamps and stored documents disagree"
    assert set(s4) == set(d4), "N=4 stamps and stored documents disagree"
    assert len(s4) == len(set(s4)), "a stamp key is duplicated"
    # Set equality is only meaningful if N=4 actually ran parallel; otherwise it
    # compares N=1 to itself.
    assert _used_more_than_one_process(r4), (
        "the N=4 document-set run was not parallel" + _failure_note(r4)
    )


# --------------------------------------------------------------------------
# 6. manifest mode
# --------------------------------------------------------------------------

def _manifest_for(corpus: Path, tmp_path: Path) -> Path:
    """A valid SILICA_INVENTORY manifest for ``corpus``, from the walk's own
    listing (so it covers exactly the documents the walk derives)."""
    from silica_core.config import CONFIG
    from silica_core.inventory import build_manifest, write_manifest
    import silica_core.core as core

    saved = CONFIG.vault_path
    try:
        CONFIG.vault_path = str(corpus)
        rows = []
        for rel, full in core._walk():
            if full is None:
                continue
            st = full.stat()
            rows.append((rel, st.st_size, st.st_mtime))
        manifest = build_manifest(corpus, rows)
    finally:
        CONFIG.vault_path = saved
    out = tmp_path / "inventory.json"
    write_manifest(out, manifest)
    return out


def test_a_manifest_pass_answers_without_a_walk(tmp_path, uneven_corpus):
    manifest = _manifest_for(uneven_corpus, tmp_path)
    expected = set(json.loads(manifest.read_bytes())["stamps"])

    # `PROBE_NO_WALK` makes the walk raise; a pass that still indexes the
    # manifest's documents proves the manifest answered the document set.
    rep = _run_pass(
        tmp_path, uneven_corpus, processes=4, tag="manifest",
        inventory=manifest, no_walk=True,
    )
    assert rep["ok"], "the manifest pass failed" + _failure_note(rep)
    assert set(_store_docs(rep)) == expected, (
        "the manifest pass indexed a different document set than the manifest holds"
    )
    index = (rep["result"] or {}).get("index", {})
    assert (index.get("inventory") or {}).get("source") == "manifest", (
        "the manifest pass did not report the manifest as its source" + _failure_note(rep)
    )
    assert _used_more_than_one_process(rep), "the manifest pass was not parallel"


def test_a_walk_pass_still_works_when_the_manifest_variable_is_unset(tmp_path, uneven_corpus):
    rep = _run_pass(tmp_path, uneven_corpus, processes=4, tag="walk")
    assert rep["ok"], "the walk pass failed" + _failure_note(rep)
    index = (rep["result"] or {}).get("index", {})
    assert (index.get("inventory") or {}).get("source") == "walk"
    assert set(_store_docs(rep)) == set(_stamps(rep))
    assert _used_more_than_one_process(rep), (
        "the walk pass was not parallel" + _failure_note(rep)
    )


# --------------------------------------------------------------------------
# 7. the flag must not be a no-op
# --------------------------------------------------------------------------

def test_the_flag_genuinely_spawns_more_than_one_process(tmp_path, uneven_corpus):
    four = _run_pass(tmp_path, uneven_corpus, processes=4, tag="noop-n4")
    assert four["ok"], "the parallel pass failed" + _failure_note(four)
    assert _used_more_than_one_process(four), (
        "the pass accepted SILICA_INDEX_PROCESSES and never ran a second process; "
        "a stub that ignores the flag must fail here"
        + _failure_note(four)
    )
    # More than one *distinct* child PID, not merely more than one fork event.
    assert len(four["descendants"]) >= 2 or four["max_concurrent"] >= 2, (
        f"only {len(four['descendants'])} distinct worker pid(s) observed"
        + _failure_note(four)
    )

    one = _run_pass(tmp_path, uneven_corpus, processes=1, tag="noop-n1")
    assert one["ok"], "the N=1 pass failed" + _failure_note(one)
    assert not _used_more_than_one_process(one), (
        "N=1 ran more than one process: single-process behaviour changed"
        + _failure_note(one)
    )


# --------------------------------------------------------------------------
# 8. dependency hygiene
# --------------------------------------------------------------------------

def _implementation_modules() -> list[Path]:
    """Every module under silica_core that mentions the frozen flag."""
    pkg = _REPO_ROOT / "silica_core"
    hits = []
    for path in sorted(pkg.rglob("*.py")):
        try:
            if _FLAG in path.read_text(encoding="utf-8", errors="replace"):
                hits.append(path)
        except OSError:
            continue
    return hits


def _top_level_imports(tree: ast.AST) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


# Third-party process/parallel libraries the pass must not reach for: the
# process handling has to be the standard library.
_FORBIDDEN_PARALLEL = frozenset({
    "pebble", "joblib", "dask", "distributed", "ray", "loky", "pathos",
    "mpi4py", "celery", "billiard", "gevent", "eventlet", "ipyparallel",
    "ppft", "cloudpickle", "multiprocess", "concurrent_log_handler",
})
_NETWORK_CLIENTS = frozenset({
    "requests", "httpx", "aiohttp", "urllib3", "http", "socket",
    "ftplib", "smtplib", "websockets", "zmq",
})
# Names that must, when imported for process handling, be standard library.
_PROCESS_NAMES = frozenset({
    "multiprocessing", "concurrent", "subprocess", "os", "sys", "signal",
    "threading", "queue", "pickle", "selectors", "ctypes", "socket",
})


def test_the_parallel_pass_uses_only_the_standard_library():
    stdlib = set(sys.stdlib_module_names)
    modules = _implementation_modules()
    assert modules, (
        "no module under silica_core mentions SILICA_INDEX_PROCESSES: the frozen "
        "interface is not implemented"
    )

    for path in modules:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
        imports = _top_level_imports(tree)
        rel = path.relative_to(_REPO_ROOT)

        # Process handling must be standard library only.
        for name in imports & _PROCESS_NAMES:
            assert name in stdlib, f"{rel} imports non-stdlib process module {name!r}"
        forbidden = imports & _FORBIDDEN_PARALLEL
        assert not forbidden, f"{rel} uses a third-party parallel library: {sorted(forbidden)}"

        # No network calls: no network client import, no socket connect in source.
        clients = imports & _NETWORK_CLIENTS
        assert not clients, f"{rel} imports a network client: {sorted(clients)}"
        assert "urlopen(" not in source, f"{rel} opens a URL"
        assert ".connect(" not in source, f"{rel} opens a network connection"
        assert "socket.socket(" not in source, f"{rel} opens a raw socket"
