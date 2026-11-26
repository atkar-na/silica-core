# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""One document, one fresh interpreter: the isolated retry of an index pass.

`python -m silica_core._isolate ROOT REL` runs the same read/extract/tokenize
half of the pass a worker process runs (`core._doc_result`) and writes one
tagged JSON line to stdout. It exists because a document can kill the process
that touches it: a malformed PDF crashes PDFium natively, and no Python
exception handler, no `try/except` around a `Future`, and no thread can survive
that. The pass catches the broken pool, retries the one document here, and only
quarantines it when this process dies too.

Why a subprocess and not a fork: PDFium and MuPDF are not thread-safe and
`pypdfium2` initialises the library at import and registers its destructor at
exit, so a forked child inherits a library the dead worker may have left half
torn down — the crash would be reproduced by the retry's own setup. A spawned
interpreter starts clean.

A `_doc_result` that raises a normal exception is reported as `error` with exit
code 0: the caller then records the message exactly as it would from a worker,
and only a non-zero exit means the process died.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Kept in step with `silica_core.core._ISOLATE_MARKER`; duplicated because this
# module is spawned by path and must not import the parent's package state.
MARKER = "SILICA_ISOLATE_RESULT "


def _encode(result: tuple) -> list:
    """A `_doc_result` tuple as JSON. The `units` payload is a list of tuples,
    which JSON would turn into lists anyway; a `prose` tuple may carry its
    truncation note as the last element."""
    if result and result[0] == "units":
        return [result[0], result[1], [list(unit) for unit in result[2]]]
    return list(result)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        sys.stderr.write("usage: python -m silica_core._isolate ROOT REL\n")
        return 2
    root, rel = Path(args[0]), args[1]
    # The extraction cache and the index dir are keyed on the build's root; the
    # worker path pins it too (`core._init_index_worker`), so the retry reads and
    # writes the same cache a worker would.
    from silica_core.kernel.recall import paths
    paths._PINNED.vault = str(root)
    from silica_core.core import _doc_result
    try:
        result = _doc_result(rel, root)
    except Exception as e:  # a document that failed in a way Python can report
        payload = {"kind": "error", "message": f"{type(e).__name__}: {e}"}
    else:
        payload = {"kind": "result", "value": _encode(result)}
    sys.stdout.write(MARKER + json.dumps(payload) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())