# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""Subprocess probe for the process-parallel first index pass (S4 acceptance).

A test cannot observe worker processes it fabricates itself without racing the
pass for `/proc`; so the pass runs HERE, in its own interpreter, with the
`SILICA_*` environment already set at import time (so a `config` field that
reads the flag at import sees it), and this probe watches its own descendants
from a thread for the whole pass.

It is a support module, never collected as a test. Its one contract: write a
JSON report at `PROBE_REPORT` and exit 0 whatever the build did, so the driver
reports a failure the test can read rather than swallowing it in a traceback.

Environment:
  PROBE_REPORT     path to write the JSON report (required)
  SILICA_HOME      the runtime dir (set by the test: one per run)
  SILICA_VAULT     the corpus root
  SILICA_INDEX_PROCESSES   the flag under test (unset == default)
  SILICA_INVENTORY optional manifest path
  PROBE_NO_WALK=1  make `core._walk` raise, to prove the manifest answered
  PROBE_KILL=1     SIGKILL the first observed descendant after PROBE_KILL_DELAY
  PROBE_KILL_DELAY seconds before the kill (default 0.2)
  PROBE_FLUSH_S    value to pin `core._FLUSH_S` to (a flush per document at 0)
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path


_FORK_COUNT = [0]


def _on_fork() -> None:
    # Runs in the forking process just before each fork: a second signal that
    # workers were spawned, independent of whether they lived long enough for
    # the /proc watch to catch them.
    _FORK_COUNT[0] += 1


os.register_at_fork(before=_on_fork)


def _children(pid: int) -> list[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children", encoding="ascii") as fh:
            return [int(x) for x in fh.read().split()]
    except (OSError, ValueError):
        return []


def _descendants(pid: int) -> set[int]:
    out: set[int] = set()
    stack = _children(pid)
    while stack:
        c = stack.pop()
        if c in out:
            continue
        out.add(c)
        stack.extend(_children(c))
    return out


def _report_path() -> Path:
    return Path(os.environ["PROBE_REPORT"])


def main() -> int:
    report = _report_path()
    self_pid = os.getpid()
    stop = threading.Event()
    seen: set[int] = set()
    peak = [0]
    killed: list[int] = []
    kill = os.environ.get("PROBE_KILL") == "1"
    kill_delay = float(os.environ.get("PROBE_KILL_DELAY", "0.2"))
    started = time.monotonic()

    def watch() -> None:
        while not stop.is_set():
            live = _descendants(self_pid)
            if live:
                seen.update(live)
                peak[0] = max(peak[0], len(live))
                if kill and not killed and (time.monotonic() - started) >= kill_delay:
                    victim = sorted(live)[0]
                    try:
                        os.kill(victim, signal.SIGKILL)
                        killed.append(victim)
                    except OSError:
                        pass
            time.sleep(0.002)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    out: dict = {
        "ok": False, "result": None, "error": None, "traceback": None,
        "descendants": [], "max_concurrent": 0, "killed": [], "forks": 0,
        "index_dir": None, "store": None, "store_parses": False,
        "doc_keys": [], "store_sha256": None, "store_size": None,
    }

    import silica_core.core as core

    if os.environ.get("PROBE_FLUSH_S") is not None:
        core._FLUSH_S = float(os.environ["PROBE_FLUSH_S"])
    if os.environ.get("PROBE_NO_WALK") == "1":
        def _boom(*_a, **_k):
            raise RuntimeError("a traversal ran where the manifest must answer")
        core._walk = _boom

    forks_before = _FORK_COUNT[0]
    try:
        out["result"] = core.build_index(rebuild=True)
    except BaseException as e:  # noqa: BLE001 - the probe reports, never raises
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
    out["forks"] = _FORK_COUNT[0] - forks_before

    stop.set()
    watcher.join(timeout=2)
    out["descendants"] = sorted(seen)
    out["max_concurrent"] = peak[0]
    out["killed"] = killed

    try:
        from silica_core.kernel.recall import paths as _paths
        idx = _paths.index_dir()
        out["index_dir"] = str(idx)
        sp = idx / "lexical.json"
        out["store"] = str(sp)
        if sp.is_file():
            blob = sp.read_bytes()
            out["store_sha256"] = hashlib.sha256(blob).hexdigest()
            out["store_size"] = len(blob)
            try:
                out["doc_keys"] = sorted(json.loads(blob).get("docs", {}))
                out["store_parses"] = True
            except ValueError:
                out["store_parses"] = False
    except Exception as e:  # noqa: BLE001
        out["store_error"] = f"{type(e).__name__}: {e}"

    out["ok"] = out["error"] is None
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
