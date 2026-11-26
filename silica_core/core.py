# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""The five tools of TOOLS.md as plain functions: files, search, read,
code_pack, write_note. The CLI and the MCP server wrap these. Nothing here
calls a model; the root is the folder Silica was started in (or --vault)."""
from __future__ import annotations

import contextvars
import functools
import hashlib
import math
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Callable

import orjson
import yaml
from pydantic import Field

from silica_core.config import CONFIG
from silica_core.kernel.code.codeast.base import language_for
from silica_core.kernel.code.codeunits import outline as _code_outline
from silica_core.kernel.code.codeunits import units as _code_units
from silica_core.kernel.link.ast import parse_headings
from silica_core.kernel.recall import paths as _paths
from silica_core.kernel.recall.lexical import TOKENIZER_VERSION, _prepare, _tokens, get_store
from silica_core.kernel.recall.rerank import _query_terms, best_window_spans
from silica_core.tools import tool

K1, B = 1.5, 0.75
# candidates the section stage scores: k hits from k*POOL_FACTOR documents
# (never fewer than 10). scripts/bench_beir.py --pool-factor measures it.
POOL_FACTOR = 3
# Documents a search refreshes inline before answering. Past it the index is
# served as it is and `index.pending` says how far behind, so a bulk drop of
# files never turns one search into a rebuild: `silica-core index` catches up.
# ~13 ms a document on the 254-paper corpus (2026-09-08), so 50 stays under
# a second. ponytail: a count, not bytes; one 40 MB PDF still blocks.
STALE_BUDGET = 50
# The lexical store's mutators are not thread-safe (lexical.py): one thread
# in them or in a search at a time, so the warm-up thread never fills the
# store under a search. ponytail: searches serialise too; a reader-writer
# lock if a client ever fans them out.
_STORE_RW = threading.RLock()
# Directories a walk could not list — permission denied, a dead mount, an
# unreadable subdirectory. `os.walk` calls `onerror` and otherwise continues as
# if the directory were empty, so a failed traversal is indistinguishable from a
# genuinely empty root; `_walk` appends every unlistable directory here. A
# build or a state report sets a fresh list for the duration of one walk and
# refuses to treat a result with entries as the corpus. A ContextVar, not a
# module global, because the MCP server serves requests on a thread pool: one
# request must never read another request's failures. Unset outside such a
# scope, where a partial walk has no consumer to tell.
_walk_failures: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "silica_walk_failures", default=None)
# Seconds between partial saves during a build, so Ctrl+C at 90% of a cold
# build keeps the 90% and the next run does the rest (save() is hundreds of
# ms on a big index, so not per document).
_FLUSH_S = 5.0
_NO_EMBEDDER = ("needs SILICA_EMBEDDING_BASE_URL (an OpenAI-compatible /v1/embeddings endpoint) "
                "or SILICA_EMBEDDING_MODEL=model2vec/<id> with the [dense] extra")
# The index this engine owns. The one line silica-internal changes when it
# vendors this file: build_index() drops every path its own walk did not
# produce, so sharing a store with a lane that indexes a different set would
# have each build prune the other's entries.
INDEX = "lexical"
_HEADING = re.compile(r"(?m)^(?=#{1,4} )")
_PAGE = re.compile(r"p\.?\s*(\d+)", re.I)
_WIKILINK = re.compile(r"\[\[([^\]|#]+)")
_TEXT_SUFFIXES = frozenset({".md", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml", ".toml", ".xml", ".html"})
# What the index reads as code when SILICA_INDEX_CODE is set: the languages
# codeast parses, one unit per symbol, and these as windows of lines.
# prose is always indexed, a heading section at a time; the code lane adds
# source languages (codeast's, or 80-line windows for the rest) and the
# config and data files that only mean something beside code
_PROSE_SUFFIXES = frozenset({".md", ".txt", ".rst"})
_CODE_SUFFIXES = frozenset({".rs", ".go", ".rb", ".php", ".kt", ".swift", ".scala", ".sh", ".sql", ".cfg",
                            ".ini", ".yaml", ".yml", ".toml", ".json"})
_CODE_MAX_BYTES = 512 * 1024  # a lockfile, a minified bundle, a fixture dump: not read


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pool(k: int) -> int:
    return max(k * POOL_FACTOR, 10)


def _rrf(rankings: list[list[Any]], k: int = 60) -> dict[Any, float]:
    """Reciprocal-rank fusion (Cormack 2009, k=60): one score per key over
    several rankings, each best first. Rank is what fuses; the raw scores
    stay what they were, in the reply."""
    fused: dict[Any, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
    return fused


_ROOT: tuple[str, Path] | None = None


def _root() -> Path:
    """The configured root, resolved once per value: `resolve()` on every
    call was 2.3 s of a 3 s search over 5k files (profiled 2026-09-09).
    ponytail: a root that is a symlink re-resolves only when the setting
    changes."""
    global _ROOT
    pinned = _paths.pinned_vault()  # a build's own, while it runs (paths.pinned)
    key = (pinned if pinned is not None else CONFIG.vault_path) or os.getcwd()
    if _ROOT is None or _ROOT[0] != key:
        _ROOT = (key, Path(key).resolve())
    return _ROOT[1]


def _version(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogateescape")).hexdigest()[:12]


def _error(code: str, hint: str, **extra: Any) -> dict:
    return {"error": {"code": code, "hint": hint}, **extra}


def _is_code(rel: str) -> bool:
    if not CONFIG.index_code:
        return False
    name = rel.rsplit("/", 1)[-1]
    suffix = name[name.rfind("."):].lower() if "." in name else ""
    return suffix in _CODE_SUFFIXES or (language_for(rel) is not None and suffix not in (".html", ".css"))


def _split_key(key: str) -> tuple[str, int | None]:
    """A unit key `path#offset` -> (path, offset); a document key -> (path, None).
    A code file is indexed one unit per symbol, each its own document to
    the store, so BM25 ranks units directly and idf is over units."""
    rel, sep, off = key.rpartition("#")
    return (rel, int(off)) if sep and rel and off.isdigit() else (key, None)


def _rel(path: str) -> str:
    """Root-relative posix path. ValueError when `path` escapes the root
    through `..` or a symlink (contain_in_vault resolves both)."""
    return _paths.contain_in_vault(path, _root())


def _read(full: Path) -> str:
    return full.read_text(encoding="utf-8", errors="replace")


def _source_state(original: Path, sidecar_text: str) -> str:
    """Whether a sidecar `.md` still describes the bytes beside it: `current`
    when the original hashes to the `source_sha256` the conversion recorded,
    `stale` when it does not (the original was replaced, the sidecar kept),
    `unverifiable` when the sidecar records no hash (a conversion older than
    the field, or a `.md` someone wrote by hand). The text's `version` says
    nothing about this: a stale sidecar is textually stable. The hash is
    kept in this process per (path, mtime, size) since search hits carry
    the verdict too (2026-11-04); a listing's first call still hashes every
    original once.
    """
    from silica_core.kernel.write.frontmatter import split

    data, _raw, _body = split(sidecar_text)
    want = data.get("source_sha256") if isinstance(data, dict) else None
    if not isinstance(want, str) or not want:
        return "unverifiable"
    try:
        return "current" if _sha256_of(original) == want else "stale"
    except OSError:
        return "unverifiable"


_sha_cache: dict[tuple[str, float | int, int], str] = {}


def _sha256_of(full: Path) -> str:
    """The file's sha256, kept per (path, identity stamp, size): every search
    hit on a converted note asks for its original's, and a 40 MB PDF is a 40 MB
    read plus 16 ms of hashing (measured 2026-11-04) per hit. The stamp is the
    document's identity (a manifest's, when one is set), so a replaced file that
    the filesystem reports as unchanged — same mtime on a `--no-modtime` mount —
    still misses here rather than serving a stale verdict."""
    st = full.stat()
    try:
        stamp: float | int = _identity_stamp(_rel(str(full)), full)
    except ValueError:  # outside the root: fall back to the filesystem's own stamp
        stamp = st.st_mtime_ns
    key = (str(full), stamp, st.st_size)
    if key not in _sha_cache:
        if len(_sha_cache) > 512:
            _sha_cache.clear()
        _sha_cache[key] = hashlib.sha256(full.read_bytes()).hexdigest()
    return _sha_cache[key]


def _conversion_of(md_full: Path, text: str) -> tuple[str | None, str | None]:
    """(root-relative original, source_state) for a `.md` a conversion wrote,
    (None, None) for any other note. Search returns the note's path, not the
    PDF's, so the freshness check has to start from the note: the original is
    the `source_file` its frontmatter names when that is a file under the
    root, else a sibling with the same stem and a convertible suffix (a vault
    moved since the conversion); neither found is `unverifiable`."""
    from silica_core.kernel.write.frontmatter import split
    from silica_core.sources.convert import DOC_EXTS

    # Only a conversion writes the key, and a key is in the text before it is
    # in the YAML: parsing every note's frontmatter was 0.45 of the 0.50 s a
    # listing of 2,259 notes took (cProfile, alex_second_brain, 2026-11-04).
    if "source_sha256" not in text:
        return None, None
    data, _raw, _body = split(text)
    if not isinstance(data, dict) or not data.get("source_sha256"):
        return None, None
    named = data.get("source_file")
    root = _root()
    for cand in ([Path(named)] if isinstance(named, str) and named else []) + [md_full.with_suffix(e) for e in DOC_EXTS]:
        try:
            rel = _rel(str(cand))
        except ValueError:
            continue
        if (root / rel).is_file():
            return rel, _source_state(root / rel, text)
    return None, "unverifiable"


def _head(full: Path, limit: int = 65536) -> str:
    """The first `limit` characters: enough for any frontmatter, and a listing
    must not read every note whole to ask whether it is a conversion."""
    try:
        with full.open(encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def _extract_cache(rel: str) -> Path:
    return _paths.index_dir() / "extract" / f"{hashlib.sha1(rel.encode()).hexdigest()}.txt"


def _pdf_text(rel: str, full: Path) -> tuple[str, list[dict], str]:
    """A PDF's own text layer, the page map, and the reason the layer is empty
    when it is. Cached beside the index (`.txt` with the text, `.json` with the
    map) and re-extracted whenever the source's identity stamp moves — the
    manifest's stamp when `SILICA_INVENTORY` supplies one, else the PDF's mtime.

    The page is the section: search ranks pages, `silica_read` serves one by
    `p. N`, `page_map` carries a slice back to its pages. The map is kept
    beside the text rather than marked inside it — a `## p. N` marker splits
    the pages of any paper that prints markdown (olmOCR does), and a form feed
    is a line boundary to `str.splitlines`, which would shift every line
    number the tool reports.

    ponytail: the text layer as PDFium hands it over — no OCR, no column
    reordering, no table reconstruction. A scan carries no text layer, stays
    `unconverted`, and `silica-core import` (mineru, docling) is the upgrade path.
    """
    cache = _extract_cache(rel)
    pages_file = cache.with_suffix(".json")
    stamps = _manifest_stamps()
    stamp = _identity_stamp(rel, full, stamps)
    # A manifest's stamp is the source's real identity. A bare-list artifact
    # written before the stamp field existed carries none of its own, so it
    # cannot be held to that stamp: its cache file is written locally, so on a
    # `--no-modtime` mount its mtime looks newer however constant the source's
    # is, and reusing it would serve a replaced PDF's old text and page bounds.
    # When the manifest has a stamp for the document the artifact is therefore
    # a miss, and the re-extraction that follows rewrites it in the stamped
    # form. With no manifest stamp the identity is the filesystem's own, and
    # the artifact is honoured by the mtime comparison it was written under —
    # upstream's behaviour, unchanged.
    legacy_held_to_stamp = bool(stamps) and rel in stamps
    try:
        payload = orjson.loads(pages_file.read_bytes())
        if isinstance(payload, dict) and "pages" in payload:
            # A cache written with this build: the source's identity is beside
            # the pages, so a replaced file misses however constant its mtime is.
            if payload.get("stamp") == stamp:
                text = cache.read_text(encoding="utf-8", errors="replace")
                return text, payload["pages"], "" if text else "no text layer"
        elif isinstance(payload, list) and not legacy_held_to_stamp \
                and cache.stat().st_mtime >= full.stat().st_mtime:
            # A cache written before the field existed (a bare page array):
            # fall back to the mtime comparison it was written under, but only
            # when no manifest stamp exists to hold it to. A stamp the bare
            # list cannot record makes it a miss rather than a stale read.
            text = cache.read_text(encoding="utf-8", errors="replace")
            return text, payload, "" if text else "no text layer"
    except (OSError, ValueError):
        pass
    try:
        import pypdfium2 as pdfium
    except ImportError:  # pragma: no cover - a base dependency
        return "", [], "pypdfium2 not installed"
    try:
        pdf = pdfium.PdfDocument(str(full))
        try:
            bodies = []
            for i in range(len(pdf)):
                page = pdf[i]
                textpage = page.get_textpage()
                try:
                    bodies.append(textpage.get_text_range().strip())
                finally:
                    textpage.close()
                    page.close()
        finally:
            pdf.close()
    except Exception as e:  # encrypted, truncated, not a PDF: a status, not a crash
        return "", [], f"unreadable: {e}"
    parts: list[str] = []
    pages: list[dict] = []
    line, offset = 1, 0
    for n, body in enumerate(bodies, 1):
        chunk = body + "\n\n"
        pages.append({"page": n, "line": line, "offset": offset})
        parts.append(chunk)
        line += chunk.count("\n")
        offset += len(chunk)
    text = "".join(parts) if any(bodies) else ""
    cache.parent.mkdir(parents=True, exist_ok=True)
    _paths.atomic_write_bytes(cache, text.encode("utf-8"))
    _paths.atomic_write_bytes(pages_file, orjson.dumps({"stamp": stamp, "pages": pages if text else []}))
    return text, (pages if text else []), "" if text else "no text layer"


def _doc_text(rel: str) -> str:
    """What the index reads for one document: a note's bytes, a PDF's text layer."""
    full = _root() / rel
    return _pdf_text(rel, full)[0] if full.suffix.lower() == ".pdf" else _read(full)


def _is_text(full: Path) -> bool:
    if full.suffix.lower() in _TEXT_SUFFIXES:
        return True
    try:
        with full.open("rb") as fh:
            return b"\0" not in fh.read(4096)
    except OSError:
        return False


def _split(text: str):
    """(offset, section text) per heading-delimited section; the preamble
    before the first heading is a section too."""
    off = 0
    for part in _HEADING.split(text):
        if part.strip():
            yield off, part
        off += len(part)


# ---------------------------------------------------------------------------
# index: whole-document BM25, mtime stamps, cold/stale/ready
# ---------------------------------------------------------------------------

def _meta_path() -> Path:
    return _paths.index_dir() / "stamps.json"


def _load_meta() -> dict:
    """The stamps file; an index built by another tokenizer reads as never built."""
    try:
        meta = orjson.loads(_meta_path().read_bytes())
    except (OSError, ValueError):
        meta = {}
    if meta.get("tokenizer") != TOKENIZER_VERSION:
        return {"built_at": None, "stamps": {}, "tokenizer": TOKENIZER_VERSION}
    return meta


def _failures() -> dict[str, str]:
    try:
        return orjson.loads((_paths.index_dir() / "failures.json").read_bytes())
    except (OSError, ValueError):
        return {}


def _walk(base: Path | None = None):
    """(rel, Path) for every regular file under the root, plus what the walk
    did not enter or list as (rel + "/", None) for a directory and (rel, None)
    for a file: hidden names, NOISE_DIRS, and `.silicaignore`, on files and
    directories alike. String paths and one relpath per directory: pathlib
    per file was 5 s of a 3 s search over 5k files (profiled 2026-09-09).

    A directory `os.walk` cannot list — permission denied, a dead mount, an
    unreadable subdirectory — is reported to `_walk_failures` and not silently
    skipped: without this the traversal looks exactly like an empty root and a
    build prunes the index to nothing. A caller that needs the corpus (a build,
    a state report) sets that list and refuses a result the walk could not
    complete; a caller listing files ignores it and sees what the walk saw."""
    root = _root()
    root_s = str(root)
    skip = _paths.ignore_path_matcher(root)

    def onerror(exc: OSError) -> None:
        sink = _walk_failures.get()
        if sink is not None:
            where = getattr(exc, "filename", None) or "?"
            sink.append(f"{where}: {exc}")

    for dirpath, dirnames, filenames in os.walk(base or root, onerror=onerror):
        rel_dir = os.path.relpath(dirpath, root_s).replace(os.sep, "/")
        prefix = "" if rel_dir == "." else rel_dir + "/"
        keep = []
        for d in sorted(dirnames):
            rel = prefix + d
            if d.startswith(".") or skip(rel):
                yield rel + "/", None
            else:
                keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            rel = prefix + name
            yield rel, (None if skip(rel) else Path(dirpath, name))


def _notes() -> dict[str, float]:
    """rel -> mtime for every document the index reads: notes, and PDFs with
    no extracted `.md` beside them — that sidecar is the note, and indexing
    both would double it. The one stat per file happens here, so
    `index_state` and `build_index` given this dict never walk again."""
    return _inventory()[0]


def _manifest_rel(key) -> str | None:
    """A manifest key as a root-relative, normalised posix path, or None when it
    is not one. A key is a path a consumer joins to the root, so it must not be
    able to name anything outside it: an empty key, an absolute path, a
    backslash, or any `.`/`..` or empty component is refused. Every manifest map
    (`stamps`, `sizes`, `mtimes`, `unextracted`) is validated through this, so a
    manifest can never point a read at a path outside the root."""
    if not isinstance(key, str) or not key or key.startswith("/") or "\\" in key:
        return None
    if any(part in ("", ".", "..") for part in key.split("/")):
        return None
    return key


def _manifest_number(value) -> float | None:
    """A stamp, size or mtime as a float, or None when it is not a real number.
    `bool` is an `int` in Python and the schema forbids it as a number, so it is
    refused explicitly — the same reason `version`/`count` use `type(x) is int`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


_manifest_reason_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "silica_manifest_reason", default=None)


def _manifest_fail(reason: str):
    """Record why a manifest was refused and yield None: the reader falls back
    to the walk, and `index_state` can then say why the walk answered."""
    _manifest_reason_var.set(reason)
    return None


def _manifest_reason() -> str | None:
    """Why the last manifest parse refused the file, or None when no manifest is
    configured or the one there was accepted. Read after `_manifest_inventory`,
    which resolves the parse (once per public call) and records the reason; the
    value lives in a ContextVar so one thread's refusal never leaks to another."""
    return _manifest_reason_var.get()


def _manifest_inventory() -> dict | None:
    """`_manifest_inventory_uncached()`, or the parse one public call already did.

    See `_manifest_scoped`: a search or a listing resolves the manifest once and
    every document it reads reuses that parse instead of re-reading the file.
    """
    scoped = _call_manifest.get()
    return scoped[0] if scoped is not None else _manifest_inventory_uncached()


def _manifest_inventory_uncached() -> dict | None:
    """The `SILICA_INVENTORY` manifest, parsed and checked against the frozen
    interface, or None when the variable is unset, unreadable or fails a rule —
    and then the walk answers instead, because where a corpus lives is worth an
    optimisation, never a failure. Every rule is strict, so a malformed manifest
    cannot smuggle a value past a type check:

    * `version` is exactly the integer `1` (`type(x) is int`, so `true`/`1.0` are
      not `1`);
    * `root` equals the configured root;
    * `count` is a real int equal to `len(stamps)` — never a bool, and a
      truncated write's short count is caught;
    * every `stamps`/`sizes`/`mtimes`/`unextracted` key is a root-relative,
      normalised path that cannot escape the root (no absolute path, no `.`/`..`
      or empty component, so `../../outside.md` is refused);
    * every `stamps`/`sizes`/`mtimes` value is a real number, never a bool.

    The result carries `stamps` (`rel -> number`, sorted), `unextracted`,
    `sizes`, `mtimes` (each `rel -> number` or `{}` when absent) and
    `generated_at`. The stamp is the number the index compares for equality
    against the one it stored, so it may be an mtime or a digest of the content —
    only the latter gives real change detection behind a mount that reports a
    parent-directory mtime. A caller hashing content keeps the value exact in a
    float by using the first 12 hex digits (48 bits, well inside a double). The
    order of `stamps` is not significant: the rows are sorted here, as
    `_inventory` sorts its own.
    """
    raw = os.environ.get("SILICA_INVENTORY", "").strip()
    if not raw:
        _manifest_reason_var.set(None)
        return None
    try:
        blob = Path(raw).read_bytes()
    except OSError as e:
        return _manifest_fail(f"unreadable: {type(e).__name__}")
    try:
        data = orjson.loads(blob)
    except ValueError as e:
        return _manifest_fail(f"not JSON: {type(e).__name__}")
    if not isinstance(data, dict):
        return _manifest_fail("not a JSON object")

    version = data.get("version")
    if type(version) is not int or version != 1:
        return _manifest_fail(f"version {version!r} is not the supported integer 1")
    if str(data.get("root")) != str(_root()):
        return _manifest_fail("root does not match the configured root")

    raw_stamps = data.get("stamps")
    if not isinstance(raw_stamps, dict):
        return _manifest_fail("stamps is missing or not an object")
    count = data.get("count")
    if type(count) is not int or count != len(raw_stamps):
        return _manifest_fail(
            f"count {count!r} is not an int equal to the {len(raw_stamps)} stamps")

    stamps: dict[str, float] = {}
    for rel, mt in raw_stamps.items():
        if _manifest_rel(rel) is None:
            return _manifest_fail(f"stamps key {rel!r} is not a root-relative path")
        n = _manifest_number(mt)
        if n is None:
            return _manifest_fail(f"stamp for {rel!r} is not a real number")
        stamps[rel] = n

    numbers: dict[str, dict[str, float]] = {}
    for field in ("sizes", "mtimes"):
        if field not in data:
            numbers[field] = {}
            continue
        block = data[field]
        if not isinstance(block, dict):
            return _manifest_fail(f"{field} is not an object")
        out: dict[str, float] = {}
        for rel, v in block.items():
            if _manifest_rel(rel) is None:
                return _manifest_fail(f"{field} key {rel!r} is not a root-relative path")
            n = _manifest_number(v)
            if n is None:
                return _manifest_fail(f"{field} value for {rel!r} is not a real number")
            out[rel] = n
        numbers[field] = out

    raw_unextracted = data.get("unextracted")
    if raw_unextracted is not None and not isinstance(raw_unextracted, list):
        return _manifest_fail("unextracted is not an array")
    unextracted: list[str] = []
    for rel in raw_unextracted or ():
        if _manifest_rel(rel) is None:
            return _manifest_fail(f"unextracted entry {rel!r} is not a root-relative path")
        unextracted.append(rel)

    _manifest_reason_var.set(None)
    return {"stamps": dict(sorted(stamps.items())),
            "unextracted": unextracted,
            "sizes": numbers["sizes"],
            "mtimes": numbers["mtimes"],
            "generated_at": data.get("generated_at")}


_call_manifest: contextvars.ContextVar[tuple[dict | None] | None] = contextvars.ContextVar(
    "silica_call_manifest", default=None)


def _manifest_scoped(fn):
    """Resolve the manifest once for one public call, and reuse that parse.

    A search or a listing reads many documents, and every read asks for a
    document's identity stamp; without this each one would re-read and
    re-validate the manifest — tens of megabytes of JSON on the real corpus,
    parsed once per hit. The scope is a ContextVar rather than a module global
    because the MCP server serves requests on a thread pool: one request must
    never read another request's manifest. The one-tuple is the sentinel —
    `(None,)` means "resolved, and there is no manifest", which is not the same
    as "not resolved yet".
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        token = _call_manifest.set((_manifest_inventory_uncached(),))
        try:
            return fn(*args, **kwargs)
        finally:
            _call_manifest.reset(token)

    return wrapper


def _manifest_stamps() -> dict[str, float] | None:
    """The manifest's `rel -> stamp`, or None when no manifest is set."""
    manifest = _manifest_inventory()
    return manifest["stamps"] if manifest is not None else None


def _identity_stamp(rel: str, full: Path, stamps: dict[str, float] | None = None) -> float | int:
    """A document's identity for the in-process caches: the manifest's stamp for
    `rel` when `SILICA_INVENTORY` supplies one, else the filesystem's own
    `st_mtime_ns`. Every cache that used to key on `stat().st_mtime` calls this,
    so the three agree on what "the same document" means. The stamp is compared
    for equality only; on a mount whose mtime is constant (`--no-modtime`) the
    filesystem's stamp never moves, so only a manifest that carries a content
    digest can tell a replaced file from an untouched one. With no manifest the
    value is the one the caches already used, so behaviour is unchanged.

    Inside a public call that resolved the manifest once (`_manifest_scoped`),
    the stamp comes from that parse; outside one, this resolves it per call. A
    caller that already has the manifest's `stamps` map (the extraction cache,
    which also has to know whether the document carries a stamp at all) passes
    it as `stamps` so the manifest is read once, not twice.
    """
    if stamps is None:
        stamps = _manifest_stamps()
    if stamps:
        stamp = stamps.get(rel)
        if stamp is not None:
            return stamp
    return full.stat().st_mtime_ns


def _inventory(base: Path | None = None) -> tuple[dict[str, float], list[str]]:
    """`_notes()` and, from the same walk, the documents the index holds no
    text for because nothing extracted it: an office file, an e-book, a
    recording with no `.md` beside it, which `silica_files` calls
    `unconverted`. A search needs both, and a second walk for the count was
    paid on every call (0.032 s on 3,202 files, silica-agent's own walk on
    alex_second_brain, 2026-11-04). `SILICA_INVENTORY` replaces the walk with
    a manifest when the caller already knows what the corpus holds; a `base`
    is always walked, since a manifest describes a whole root."""
    from silica_core.sources.convert import DOC_EXTS, IMG_EXTS

    if base is None:
        manifest = _manifest_inventory()
        if manifest is not None:
            return manifest["stamps"], manifest["unextracted"]

    out: dict[str, float] = {}
    unextracted: list[str] = []
    for rel, full in _walk(base):
        if full is None:
            continue
        suffix = full.suffix.lower()
        if suffix in _PROSE_SUFFIXES or (suffix == ".pdf" and not full.with_suffix(".md").is_file()):
            try:
                out[rel] = full.stat().st_mtime
            except OSError:
                continue
        elif _is_code(rel):
            try:
                st = full.stat()
            except OSError:
                continue
            if st.st_size <= _CODE_MAX_BYTES:
                out[rel] = st.st_mtime
        elif suffix in DOC_EXTS and suffix not in IMG_EXTS and suffix != ".pdf" \
                and not full.with_suffix(".md").is_file():
            unextracted.append(rel)
    return dict(sorted(out.items())), unextracted


def walk(folder: str = ""):
    """(root-relative path, Path) for every file under `folder` (the whole
    root when empty) that the index's walk enters, and (path, None) for what
    it does not: a hidden name, a noise folder, a `.silicaignore` rule. The
    walk every tool here starts from, public for a harness that lists files
    on its own terms (silica-agent's changes since a time) instead of
    reaching for the private one. A folder outside the root raises
    ValueError, as every root-relative path here does."""
    base = _root() / _rel(folder) if folder.strip() else None
    return _walk(base)


def unreadable(folder: str = "") -> dict:
    """The files under `folder` whose text the index does not hold, with the
    rows `silica_files` gives them: `failed` (the read failed) and
    `unconverted` (a PDF with no text layer, or another document with no
    `.md` extracted beside it), each with its `reason`. A relevant document
    among them never ranks, so a harness that reports its own searches names
    them beside a miss; this classifies those two statuses alone, without
    reading every note's head the way a full listing does."""
    try:
        scope = _rel(folder) if folder.strip() else ""
    except ValueError as e:
        return _error("out_of_root", str(e))
    base = _root() / scope if scope else None
    if base is not None and not base.is_dir():
        return _error("not_found", f"{scope} is not a folder under {_root()}")
    rows = _unread_rows(scope, _inventory(base)[1])
    return {"root": str(_root()), "folder": scope, "files": rows,
            "counts": {s: sum(1 for r in rows if r["status"] == s) for s in ("failed", "unconverted")}}


def _unread_rows(scope: str, unextracted: list[str]) -> list[dict]:
    """Failed and unconverted files under `scope`, failed first, then by
    path: what a search under that folder could not rank."""
    failed = _failures()
    rows = [{"path": p, "status": "failed", "reason": failed[p]}
            for p in sorted(failed) if _paths.in_folder(p, scope)]
    skipped = _load_meta().get("no_text", {})
    unconverted = [(p, skipped[p].get("reason") or "no text layer") for p in skipped if p not in failed]
    unconverted += [(p, _NO_SIDECAR) for p in unextracted if p not in failed]
    rows += [{"path": p, "status": "unconverted", "reason": why}
             for p, why in sorted(unconverted) if _paths.in_folder(p, scope)]
    return rows


_NO_SIDECAR = "no extracted .md beside it"


def _note_paths() -> list[str]:
    return list(_notes())


def _mtime(rel: str) -> float | None:
    try:
        return (_root() / rel).stat().st_mtime
    except OSError:
        return None


def _inventory_block(live: dict[str, float] | None = None) -> dict:
    """Which inventory answered a pass, for `index_state`: the manifest that
    stood in for the walk, with `generated_at` so a deployment can see how old
    it is, or the walk. A stale or subset manifest is surfaced here, never
    rejected — otherwise a frozen or partial index reports `ready` with nothing
    saying why. `count` is the documents the answering source holds: the
    manifest's own, or the walk's when the caller already walked (None for a
    cold index that has not)."""
    manifest = _manifest_inventory()
    if manifest is not None:
        return {"source": "manifest", "root": str(_root()),
                "generated_at": manifest.get("generated_at"), "count": len(manifest["stamps"])}
    block = {"source": "walk", "root": str(_root()), "generated_at": None,
             "count": None if live is None else len(live)}
    reason = _manifest_reason()
    if reason is not None:
        # `SILICA_INVENTORY` was set but the manifest was refused: say why, so a
        # malformed or unsafe manifest is visible instead of silently walking.
        block["reason"] = reason
    return block


def index_state(meta: dict | None = None, live: dict[str, float] | None = None) -> dict:
    """{state: cold|stale|ready, docs, built_at, pending, inventory}. `cold` =
    never built. `pending` counts what the index has not caught up with:
    documents changed or new since their stamp, and stamped documents gone from
    disk. `inventory` says which source answered — a manifest or the walk — and
    how old it is. `live` is `_notes()` when the caller already walked.

    A walk that could not list a directory is surfaced, never reported `ready`:
    the state is `stale` and an `error` block names the directories. Otherwise a
    failed mount would read as a corpus with everything deleted, which is what a
    search or a build then acts on."""
    meta = meta or _load_meta()
    if not _paths.index_file(INDEX).is_file() or not meta.get("built_at"):
        return {"state": "cold", "docs": 0, "built_at": None, "inventory": _inventory_block()}
    failures: list[str] = []
    walk_failed = False
    if live is None:
        token = _walk_failures.set(failures)
        try:
            live = _notes()
        except Exception as e:  # a walk that raised outright, not just skipped a directory
            failures.append(str(e))
            live = {}
        finally:
            _walk_failures.reset(token)
        walk_failed = bool(failures)
    stamps = meta.get("stamps", {})
    skipped = meta.get("no_text", {})
    pending = sum(1 for r, mt in live.items()
                  if stamps.get(r) != mt and skipped.get(r, {}).get("mtime") != mt)
    pending += sum(1 for r in stamps if r not in live)
    out = {"state": "stale" if (pending or walk_failed) else "ready", "docs": len(stamps),
           "built_at": meta.get("built_at"), "pending": pending,
           "inventory": _inventory_block(live)}
    if walk_failed:
        out["error"] = {"code": "inventory_walk_failed", "failures": failures,
                        "hint": f"the walk under {_root()} could not list {len(failures)} "
                                "directory(ies); the state is not this corpus"}
    return out


def _flush(store, meta: dict, failed: dict) -> None:
    """Everything the build knows so far, on disk: the store, the stamps with
    a `built_at` so the next run is incremental, the failures."""
    store.save()
    meta["built_at"] = time.time()
    _paths.atomic_write_bytes(_meta_path(), orjson.dumps(meta))
    _paths.atomic_write_bytes(_paths.index_dir() / "failures.json", orjson.dumps(failed))
    # The dir's name is a one-way digest of the root, so without this nothing
    # can tell which folder an index describes or that the folder is gone:
    # 343 of 549 dirs under ~/.silica/index on 2026-10-27. Not `vault.json`:
    # silica-agent routes to every vault whose dir carries that one.
    marker = _paths.index_dir() / "root.json"
    if not marker.is_file():
        _paths.atomic_write_bytes(marker, orjson.dumps({"path": str(_root())}))


def build_index(rebuild: bool = False, embed: bool = False, live: dict[str, float] | None = None,
                allow_remote: bool = False, wait: bool = True,
                progress: Callable[[str, int, int], None] | None = None) -> dict:
    """`_build_index` under one lock per index: the stamps, the store and the
    vectors land as one, so a second server building the same folder waits
    and then finds the work done rather than interleaving with it. With
    `wait` off a busy lock is not waited for: the reply carries the index
    as it is on disk and `busy`, for a search that must answer now.
    `progress(stage, done, total)` after each document the build works on,
    `stage` "index" then "vectors": a library's first build is minutes of
    reading and hours of embedding, and an app showing it needs both. The
    build stays on the root it started on (paths.pinned) when CONFIG moves."""
    with _paths.pinned(CONFIG.vault_path or ""), \
            _paths.index_lock(_paths.index_dir() / "build", blocking=wait) as held:
        if not held:
            state = index_state(live=live)
            return {"docs": state["docs"], "changed": 0, "failed": {}, "index": state, "busy": True}
        return _build_index(rebuild, embed, live, allow_remote, progress)


def _max_drop() -> float:
    """`SILICA_INVENTORY_MAX_DROP`: the fraction of stored rows a single
    manifest pass may remove before it aborts, so a manifest that covers a
    subset of the root cannot empty an index. A fraction in [0, 1], default
    0.1; an unset or unparseable value is the default."""
    raw = os.environ.get("SILICA_INVENTORY_MAX_DROP", "").strip()
    if not raw:
        return 0.1
    try:
        frac = float(raw)
    except ValueError:
        return 0.1
    return frac if 0.0 <= frac <= 1.0 else 0.1


def _guard_live(live: dict[str, float], unextracted: list[str], meta: dict,
                errors: list[str] | None = None):
    """The walk is the truth; a manifest is trusted only where it agrees with it.

    A manifest that omits rows the walk can see is not a smaller corpus, it is a
    broken or partial read of one, and a pass that removed its absent rows would
    silently shrink the index (the empty-inventory defect). So on a
    manifest-driven pass the walk is taken once more as the authority whenever
    the two disagree, and the build then follows the walk.

    A walk is usable only if it ran *and* listed every directory: `errors` (the
    same list `_walk` appends to) holds the directories it could not list, and a
    non-empty addition means the result is not a corpus — exactly as when the
    walk raises and cannot run at all. `errors` defaults to the ambient list.

    When the walk is unusable — a manifest-only deployment, a share that is
    down — the manifest is honoured only while it removes at most
    `SILICA_INVENTORY_MAX_DROP` of the stored rows; past that the pass returns an
    `inventory_prune_refused` error without writing, so the store is left
    byte-identical and the caller sees the counts. The error carries `index`, so
    a search that triggered this build serves the store it already has and
    reports it stale rather than failing.
    """
    if errors is None:
        errors = _walk_failures.get() or []
    mark = len(errors)
    stamps: dict[str, float] = meta.get("stamps", {})
    skipped: dict[str, dict] = meta.get("no_text", {})
    walk_unextracted: list[str] = []
    try:
        walk_live, walk_unextracted = _inventory(base=_root())
    except Exception:
        walk_live = None
    if walk_live is not None and len(errors) == mark:
        if set(live) == set(walk_live) and set(unextracted) == set(walk_unextracted):
            return live, unextracted
        return walk_live, walk_unextracted
    stored = len(stamps) + len(skipped)
    drop = sum(1 for p in list(stamps) + list(skipped) if p not in live)
    max_drop = _max_drop()
    if stored and drop > max_drop * stored:
        return _error("inventory_prune_refused",
                      f"the walk is unavailable and the manifest would remove "
                      f"{drop} of {stored} stored rows (over {max_drop:.0%}); the index is left untouched",
                      docs=len(stamps), changed=0, removed=drop, stored=stored,
                      index=index_state(meta, live=live))
    return live, unextracted


def _index_processes() -> int:
    """`SILICA_INDEX_PROCESSES`: worker processes for the first index pass.

    Unset or 1 — the default — is a single process, today's behaviour. A value
    above 1 distributes the read/extract/tokenize half of the pass across that
    many worker processes (stdlib multiprocessing; threads are the wrong tool,
    the tokenizer is GIL-bound). An unparseable or below-1 value is 1, so a typo
    can never fan a build out."""
    raw = os.environ.get("SILICA_INDEX_PROCESSES", "").strip()
    if not raw:
        return 1
    try:
        n = int(raw)
    except ValueError:
        return 1
    return n if n >= 1 else 1


# Reason a document is reported failed when its worker process did not deliver
# it (killed mid-pass, or the pool broke): the document keeps its old stamp or
# none, so the next pass re-runs exactly it.
_WORKER_DIED = "worker process died before this document was indexed"
_WORKER_STALLED = "worker stalled past SILICA_INDEX_TIMEOUT and was abandoned"


def _doc_result(rel: str, root: Path) -> tuple:
    """The read, extract and tokenize half of the first pass for one document,
    with no store mutation: a tagged tuple the caller applies.

    Pure with respect to the store and the stamps, so a worker process produces
    the same tuple the parent would, and the parent — the single writer — can
    apply results in `todo` order whatever order the workers finish in.

    Tags: `("prose", rel, name, tf, length)` a note or a PDF with a text layer;
    `("units", rel, [(key, title, tf, length), ...])` a code file's symbols;
    `("skip", rel, reason)` a PDF with no text layer; `("fail", rel, message)`
    a document that could not be read."""
    if rel.lower().endswith(".pdf"):
        text, _pages, reason = _pdf_text(rel, root / rel)
        if not text:  # a scan, or a PDF PDFium could not open: `unconverted`, never indexed
            return ("skip", rel, reason)
        tf, length = _prepare(Path(rel).stem, text)
        return ("prose", rel, Path(rel).stem, tf, length)
    if _is_code(rel):
        try:
            parts = _sections_of(rel)[1]
        except OSError as e:
            return ("fail", rel, str(e))
        units = []
        for off, part, _tf, _dl, title in parts:
            u_tf, u_len = _prepare(title, part)
            units.append((f"{rel}#{off}", title, u_tf, u_len))
        return ("units", rel, units)
    try:
        text = _read(root / rel)
    except OSError as e:
        return ("fail", rel, str(e))
    tf, length = _prepare(Path(rel).stem, text)
    return ("prose", rel, Path(rel).stem, tf, length)


def _apply_doc(store, result: tuple, mt: float, stamps: dict, skipped: dict,
               units: dict) -> str:
    """Apply one `_doc_result` to the store, the stamps and the unit map.

    The single writer both the sequential and the parallel pass share, so the
    two cannot drift: given the same documents in the same order the store lands
    byte for byte the same. Returns `"changed"`, `"skipped"` or `"failed"` (the
    failure message stays in `result`; the caller records it)."""
    tag, rel = result[0], result[1]
    if tag == "prose":
        store.upsert_prepared(rel, result[2], result[3], result[4])
        stamps[rel] = mt
        skipped.pop(rel, None)
        return "changed"
    if tag == "units":
        payload = result[2]
        keys = [key for key, *_rest in payload]
        for old in units.get(rel, []):
            if old not in keys:
                store.remove(old)
        for key, title, tf, length in payload:
            store.upsert_prepared(key, title, tf, length)
        units[rel] = keys
        stamps[rel] = mt
        return "changed"
    if tag == "skip":
        skipped[rel] = {"mtime": mt, "reason": result[2]}
        if stamps.pop(rel, None) is not None:
            store.remove(rel)
        return "skipped"
    return "failed"


def _index_worker(item: tuple[str, str]) -> tuple[int, tuple]:
    """One document in a worker process: `(pid, _doc_result(...))`. The pid
    rides back so the caller can see the fan-out actually happened."""
    rel, root_str = item
    return os.getpid(), _doc_result(rel, Path(root_str))


def _init_index_worker(root_str: str) -> None:
    """Pin the build's root in a worker process. A `fork` worker inherits it
    already; pinning it here too keeps a worker's index dir, extraction cache
    and manifest resolution tied to the same root under any start method."""
    from silica_core.kernel.recall import paths as _rpaths
    _rpaths._PINNED.vault = root_str


def _index_batch() -> int:
    """Documents per pool generation: `SILICA_INDEX_BATCH`, default 256.

    A batch ends with a store flush, so a long pass leaves progress on disk as
    it goes and an interrupted run loses one batch, not the pass."""
    try:
        n = int(os.environ.get("SILICA_INDEX_BATCH", "256"))
    except ValueError:
        return 256
    return n if n > 0 else 256


def _index_timeout() -> float:
    """Seconds one document may occupy a worker: `SILICA_INDEX_TIMEOUT`, default
    300; 0 disables the limit.

    A malformed PDF can send a parser into a loop that never returns. Without a
    limit the parent waits on that result forever and the pass neither finishes
    nor reports why."""
    try:
        t = float(os.environ.get("SILICA_INDEX_TIMEOUT", "300"))
    except ValueError:
        return 300.0
    return t if t > 0 else 0.0


def _index_parallel(todo: list[tuple[str, float]], root: Path, store, stamps: dict,
                    skipped: dict, units: dict, failed: dict, meta: dict,
                    progress) -> tuple[int, set[int]]:
    """The first pass across `SILICA_INDEX_PROCESSES` worker processes.

    Workers only read, extract and tokenize; every store mutation and every
    stamp stays in this process, applied in `todo` order as results arrive in
    that order, so no worker's finish time can reach the store. A window of
    futures is kept in flight rather than the whole `todo`, so a large corpus
    does not hold one Future per document.

    The work is cut into generations of `SILICA_INDEX_BATCH` documents, each
    flushed before the next opens, so progress survives a crash at batch
    granularity and a long pass is observable while it runs.

    Crash safety: a worker that dies mid-pass makes the pool report
    `BrokenProcessPool`; a worker stuck on one document is cut off after
    `SILICA_INDEX_TIMEOUT` seconds. Either way the pool is torn down (a stalled
    worker is terminated — a cancelled Future would leave it spinning and the
    interpreter would wait on it at exit), the document is reported in `failed`
    and the next generation starts in a fresh pool. Every document not fully
    applied keeps its previous stamp, or none, so nothing is written for a
    document the parent did not fully receive: no row is partial, none is
    duplicated, and the next pass re-runs exactly those documents.

    Returns `(changed, worker_pids)`, mutating `failed` in place."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout
    from concurrent.futures.process import BrokenProcessPool

    n = _index_processes()
    batch = _index_batch()
    timeout = _index_timeout() or None
    root_str = str(root)
    changed = 0
    pids: set[int] = set()
    last_flush = time.monotonic()
    window = max(4, n * 4)
    # fork, explicitly: a worker inherits this build's root, config, index paths
    # and the SILICA_INVENTORY parse through copied memory, so nothing has to be
    # re-imported or re-resolved, and the tokenizer needs no warm-up.
    ctx = multiprocessing.get_context("fork")
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        items = [(rel, root_str) for rel, _mt in chunk]
        n_items = len(items)
        broken = False
        submitted = 0
        in_flight: dict[int, Any] = {}
        pool = ProcessPoolExecutor(max_workers=n, mp_context=ctx,
                                   initializer=_init_index_worker, initargs=(root_str,))
        try:
            for local, (rel, mt) in enumerate(chunk):
                if not broken:
                    try:
                        while submitted < min(local + window, n_items):
                            in_flight[submitted] = pool.submit(_index_worker, items[submitted])
                            submitted += 1
                    except BrokenProcessPool:
                        broken = True
                if progress:
                    progress("index", start + local, len(todo))
                fut = in_flight.pop(local, None)
                if not broken and fut is not None:
                    try:
                        pid, result = fut.result(timeout=timeout)
                    except FutureTimeout:
                        broken = True
                        failed[rel] = _WORKER_STALLED
                        continue
                    except BrokenProcessPool:
                        broken = True
                        failed[rel] = _WORKER_DIED
                        continue
                    except Exception as e:  # this document alone failed to read
                        failed[rel] = f"{type(e).__name__}: {e}"
                        continue
                else:
                    failed[rel] = _WORKER_DIED
                    continue
                pids.add(pid)
                outcome = _apply_doc(store, result, mt, stamps, skipped, units)
                if outcome == "changed":
                    changed += 1
                elif outcome == "failed":
                    failed[rel] = result[2]
                if time.monotonic() - last_flush > _FLUSH_S:
                    _flush(store, meta, failed)
                    last_flush = time.monotonic()
        finally:
            if broken:
                pool.shutdown(wait=False, cancel_futures=True)
                for child in multiprocessing.active_children():
                    child.terminate()
                    child.join(1)
            else:
                pool.shutdown(wait=True)
        _flush(store, meta, failed)
        last_flush = time.monotonic()
    return changed, pids


def _build_index(rebuild: bool = False, embed: bool = False, live: dict[str, float] | None = None,
                 allow_remote: bool = False, progress: Callable[[str, int, int], None] | None = None) -> dict:
    """Build or refresh the document index. Incremental by mtime; `rebuild`
    re-reads everything. Never raises for one unreadable file: it lands in
    `failed` and in the failures file `silica_files` reports from. Saves
    every `_FLUSH_S` seconds, so an interrupted build resumes where it
    stopped. `live` is `_notes()` when the caller already walked."""
    meta = _load_meta()
    rebuild = rebuild or not meta.get("built_at")
    manifest = _manifest_inventory()
    root = _root()
    # Resolve the live set and reconcile it with the walk before anything is
    # written and before a rebuild clears the store, so a rebuild driven by a
    # subset manifest, or a walk that could not list a directory, aborts with
    # the store still on disk. `errors` is where `_walk` reports the directories
    # it could not list for the walks this pass runs.
    errors: list[str] = []
    token = _walk_failures.set(errors)
    try:
        if live is None:
            # `_inventory()` answers from the manifest when one is set, and
            # walks otherwise: a manifest-driven pass does not traverse.
            try:
                live, unextracted = _inventory()
            except Exception as e:
                return _error("inventory_walk_failed",
                              f"the walk failed under {root}: {e}; the index is left untouched",
                              docs=len(meta.get("stamps", {})), changed=0, index=index_state(meta))
        else:
            unextracted = manifest["unextracted"] if manifest is not None else []
        if manifest is not None:
            # Prune guard, on the incremental *and* the rebuild path: the walk is
            # the authority, and a manifest that disagrees with it — a subset, a
            # truncated write, a stale file — is replaced by the walk's own set
            # rather than pruning the rows it omits; when the walk cannot run, one
            # pass may remove at most `SILICA_INVENTORY_MAX_DROP` of the stored
            # rows and otherwise aborts before anything is written.
            errors.clear()  # only the guard's own walk counts
            guarded = _guard_live(live, unextracted, meta, errors)
            if isinstance(guarded, dict):
                return guarded
            live, unextracted = guarded
        elif errors:
            # No manifest to fall back on and the walk could not list a
            # directory: the result is not a corpus, so nothing is pruned.
            return _error("inventory_walk_failed",
                          f"the walk could not list {len(errors)} directory(ies) under {root}; "
                          "refusing to treat a partial walk as the corpus",
                          docs=len(meta.get("stamps", {})), changed=0,
                          failures=list(errors), index=index_state(meta))
    finally:
        _walk_failures.reset(token)
    if rebuild:
        meta = {"built_at": None, "stamps": {}, "no_text": {}, "tokenizer": TOKENIZER_VERSION}
    with _STORE_RW:  # the store is filled here; no search reads it meanwhile
        store = get_store(INDEX)
        if rebuild:  # a unit key of a rebuilt file would otherwise outlive the offset it names
            for p in store.paths():
                store.remove(p)
        stamps: dict[str, float] = meta["stamps"]
        skipped: dict[str, dict] = meta.setdefault("no_text", {})
        units: dict[str, list[str]] = meta.setdefault("units", {})  # code file -> its unit keys
        changed, failed = 0, {}
        last_flush = time.monotonic()
        todo = [(rel, mt) for rel, mt in live.items()
                if rebuild or not (stamps.get(rel) == mt or skipped.get(rel, {}).get("mtime") == mt)]
        n_proc = _index_processes()
        worker_pids: set[int] = set()
        if n_proc > 1 and len(todo) > 1:
            # The opt-in fan-out: workers read/extract/tokenize, this process
            # applies every result in `todo` order as the single writer, so a
            # worker's finish time cannot reorder the store.
            changed, worker_pids = _index_parallel(
                todo, root, store, stamps, skipped, units, failed, meta, progress)
        else:
            for done, (rel, mt) in enumerate(todo, 1):
                if progress:
                    progress("index", done - 1, len(todo))
                result = _doc_result(rel, root)
                outcome = _apply_doc(store, result, mt, stamps, skipped, units)
                if outcome == "changed":
                    changed += 1
                elif outcome == "failed":
                    failed[rel] = result[2]
                if time.monotonic() - last_flush > _FLUSH_S:
                    _flush(store, meta, failed)
                    last_flush = time.monotonic()
        if progress and todo:
            progress("index", len(todo), len(todo))
        for gone in [p for p in list(stamps) + list(skipped) if p not in live]:
            stamps.pop(gone, None)
            skipped.pop(gone, None)
            units.pop(gone, None)
            _extract_cache(gone).unlink(missing_ok=True)  # a deleted PDF leaves no extraction behind
        for gone in [p for p in store.paths() if _split_key(p)[0] not in live]:
            store.remove(gone)
        _flush(store, meta, failed)
        # Every live document is now stamped, skipped or failed, so the state is
        # known without another walk: only the failures are still pending.
        state = {"state": "stale" if failed else "ready", "docs": len(stamps),
                 "built_at": meta["built_at"], "pending": len(failed),
                 "inventory": _inventory_block(live)}
        out = {"docs": len(stamps), "changed": changed, "failed": failed, "index": state}
        if worker_pids:
            # Present only when the fan-out ran: a single-process pass carries no
            # such key, so `SILICA_INDEX_PROCESSES` unset and `=1` stay identical.
            out["parallel"] = {"processes": n_proc, "workers": len(worker_pids)}
    if embed:
        from silica_core import embeddings
        if not embeddings.enabled():
            return _error("bad_argument", _NO_EMBEDDER, **out)
        host = embeddings.consent_needed()
        if host and allow_remote:
            embeddings.grant(host)
        elif host:
            return _error("consent_required", f"the sections would leave this machine for {host}: "
                          "run `silica-core index --embed --allow-remote` once to allow it", **out)
        to_embed = [(rel, stamps[rel], functools.partial(_embed_units, rel)) for rel in live if rel in stamps]
        try:
            out["embeddings"] = embeddings.build(
                to_embed, rebuild=rebuild,
                progress=(lambda done, total: progress("vectors", done, total)) if progress else None)
        except Exception as e:
            return _error("bad_argument", f"embeddings failed: {e}", **out)
    return out


def _embed_units(rel: str) -> list[tuple[int, str]]:
    """What a document is embedded as, read only when it is (embeddings.build);
    a section under 3 tokens, a bare heading, is left out. [] for a file gone
    or unreadable since the walk: it gets no rows and the next build tries again."""
    from silica_core import embeddings
    try:
        parts = _sections_of(rel)[1]
    except OSError:
        return []
    stem = rel[:-len(Path(rel).suffix)] if _is_code(rel) else Path(rel).stem
    return [(off, embeddings.unit_text(stem, title, part)) for off, part, _tf, dl, title in parts if dl >= 3]


# ---------------------------------------------------------------------------
# 1. files
# ---------------------------------------------------------------------------

def _manifest_covers(manifest: dict, cut: float | None,
                     known: dict[str, float] | None = None,
                     skipped: dict[str, dict] | None = None) -> bool:
    """Whether a manifest can answer a listing rather than the walk.

    Coverage of the fields: `sizes` must name every stamp, and, for a `since=`
    listing, `mtimes` must too — a row with no date cannot be compared to the
    cut, and answering without one would drop what changed.

    Coverage of the corpus: the manifest must describe the whole root, and a
    file cannot prove that about itself. So the claim is earned against the rows
    the index already holds (`known`, the stored stamps, and `skipped`, the
    documents it read and held no text for): the manifest's own documents must
    account for the stored rows up to the same ceiling the prune guard allows
    (`SILICA_INVENTORY_MAX_DROP`). A manifest that omits more of the stored rows
    than that is a subset of the corpus, not the corpus, and the walk answers
    instead — a one-row manifest that listed a single document while the rest
    vanished with no error is exactly that case. The ceiling is not zero so a
    manifest written just after the corpus genuinely shrank is still served; an
    operator whose corpus shrank past it says so with the same variable the
    prune guard reads. With no stored rows there is nothing to disagree with, so
    the field check alone decides."""
    stamps = manifest["stamps"]
    if not set(stamps) <= set(manifest["sizes"]):
        return False
    if cut is not None and not set(stamps) <= set(manifest["mtimes"]):
        return False
    if known:
        stored = list(known) + list(skipped or {})
        omitted = sum(1 for rel in stored if rel not in stamps)
        if stored and omitted > _max_drop() * len(stored):
            return False
    return True


def _walk_entries(base, cut: float | None, stamps: dict, skipped: dict, failed: dict,
                  DOC_EXTS, IMG_EXTS) -> tuple[list[dict], int]:
    """Today's listing: one walk of `base`, every file stat'd, and the reads a
    row needs (a `.md` head, a sidecar, a source hash) run as it goes. Used
    whenever no manifest covers the request."""
    root = _root()
    entries: list[dict] = []
    excluded_dirs = 0
    for rel, full in _walk(base):
        if full is None:  # a directory not entered, or a file `.silicaignore` names
            excluded_dirs += rel.endswith("/")
            if cut is None:  # what the walk never entered has no date to compare
                entries.append({"path": rel, "status": "excluded",
                                "reason": "hidden" if rel.rstrip("/").rsplit("/", 1)[-1].startswith(".") else "ignore rule"})
            continue
        try:
            st = full.stat()
        except OSError:
            continue
        if cut is not None and st.st_mtime <= cut:
            continue
        row: dict = {"path": rel, "bytes": st.st_size}  # `status` says whether it changed; a float mtime said nothing more
        if cut is not None:
            row["modified"] = _iso(st.st_mtime)
        suffix = full.suffix.lower()
        if rel in failed:
            row.update(status="failed", reason=failed[rel])
        elif suffix in _PROSE_SUFFIXES:
            row["status"] = "indexed" if stamps.get(rel) == st.st_mtime else "changed"
            src, state = _conversion_of(full, _head(full)) if suffix == ".md" else (None, None)
            if state:  # the note a conversion wrote carries the verdict too: it is the path search returns
                row.update(source=src, source_state=state)
        elif suffix in DOC_EXTS and suffix not in IMG_EXTS:
            sidecar = full.with_suffix(".md")
            skip = skipped.get(rel, {})
            if sidecar.is_file():
                row.update(status="excluded", reason=f"converted: {sidecar.relative_to(root).as_posix()}",
                           source_state=_source_state(full, _read(sidecar)))
            elif suffix != ".pdf":
                row.update(status="unconverted", reason=_NO_SIDECAR)
            elif stamps.get(rel) == st.st_mtime:
                row["status"] = "indexed"
            elif skip.get("mtime") == st.st_mtime:
                # the index read this PDF and found no text to index; `reason` says why
                row.update(status="unconverted", reason=skip.get("reason") or "no text layer")
            else:
                row["status"] = "changed"
        elif _is_code(rel):
            if st.st_size > _CODE_MAX_BYTES:
                row.update(status="excluded", reason=f"not indexed: over {_CODE_MAX_BYTES // 1024} KB")
            else:
                row["status"] = "indexed" if stamps.get(rel) == st.st_mtime else "changed"
        else:
            row.update(status="excluded", reason=f"not indexed: {suffix or 'no extension'}")
        entries.append(row)
    return entries, excluded_dirs


def _manifest_entries(manifest: dict, stamps: dict, skipped: dict, failed: dict,
                      scope: str, cut: float | None) -> tuple[list[dict], int]:
    """The listing rows a manifest implies: a stamp is `indexed` or `changed`
    by comparing it to the index's stored one, an `unextracted` document is
    `unconverted`, and `sizes` gives each row its bytes. A row the manifest does
    not name — a type the index never reads, a hidden or ignored name — is
    absent: the manifest does not carry it. No file is read here; the rows of
    the returned page are finished by `_manifest_page_row`."""
    sizes = manifest["sizes"]
    mtimes = manifest["mtimes"]
    entries: list[dict] = []
    for rel, stamp in manifest["stamps"].items():
        if scope and not _paths.in_folder(rel, scope):
            continue
        if cut is not None:
            mt = mtimes.get(rel)
            if mt is None or mt <= cut:
                continue
        row: dict = {"path": rel, "bytes": sizes.get(rel)}
        if cut is not None:
            row["modified"] = _iso(mtimes[rel])
        if rel in failed:
            row.update(status="failed", reason=failed[rel])
        else:
            skip = skipped.get(rel, {})
            if skip.get("mtime") == stamp:  # a PDF whose text layer the index could not read
                row.update(status="unconverted", reason=skip.get("reason") or "no text layer")
            else:
                row["status"] = "indexed" if stamps.get(rel) == stamp else "changed"
        entries.append(row)
    for rel in manifest["unextracted"]:
        if scope and not _paths.in_folder(rel, scope):
            continue
        if cut is not None:
            continue  # an unextracted document carries no date in the manifest
        entries.append({"path": rel, "bytes": sizes.get(rel),
                        "status": "unconverted", "reason": _NO_SIDECAR})
    return entries, 0


def _manifest_page_row(row: dict, root: Path) -> None:
    """Finish one row of a manifest listing — the reads it needs, and only for a
    row the page returns. `bytes` the manifest did not carry (an `unextracted`
    document has no `sizes` entry) comes from one stat; a `.md` a conversion
    wrote gets its `source` and `source_state` from its head and, when the
    sidecar records one, its original's hash."""
    rel = row["path"]
    full = root / rel
    if row.get("bytes") is None:
        try:
            row["bytes"] = full.stat().st_size
        except OSError:
            pass
    if Path(rel).suffix.lower() == ".md" and row.get("status") != "failed":
        src, state = _conversion_of(full, _head(full))
        if state:
            row.update(source=src, source_state=state)


@tool(cls="atomic")
@_manifest_scoped
def silica_files(
    folder: Annotated[str, Field(description="Root-relative folder to list; empty = the whole root")] = "",
    status: Annotated[str, Field(description="Only this status: indexed, changed, excluded, failed, unconverted")] = "",
    limit: Annotated[int, Field(description="Max entries per page")] = 200,
    cursor: Annotated[str, Field(description="`next_cursor` from the previous page")] = "",
    since: Annotated[str, Field(description="Only files modified after this time, ISO 8601 (2026-11-04, 2026-11-04T09:30, with or without an offset; local time without): what changed since a session you remember. Each row then carries `modified`; deletions are not listed")] = "",
) -> dict:
    """Ask what the index could see before trusting a search or calling
    absence. Inventory of the root and what the index did with each file: `indexed`
    (in the index, matches disk), `changed` (differs from the index),
    `excluded` (an ignore rule or a type the index does not read; `reason`
    says which), `failed` (read or conversion error), `unconverted` (a PDF
    whose text layer the index could not read, or an office file with no
    extracted `.md` beside it; `reason` says which). A PDF is indexed from
    its own text layer, one section per page, unless a `.md` sits beside it —
    that sidecar is indexed instead, and both rows carry `source_state`:
    `stale` means the original changed after the conversion, `unverifiable`
    that the sidecar records no hash of it or its original is not under the
    root. `since` keeps only what was modified after a time, dated: what
    changed since a session you remember. `index.state` is `cold` when nothing was
    ever indexed: an empty listing there is not "no files"."""
    from silica_core.sources.convert import DOC_EXTS, IMG_EXTS

    root = _root()
    try:
        scope = _rel(folder) if folder.strip() else ""
    except ValueError as e:
        return _error("out_of_root", str(e))
    base = root / scope if scope else root
    if not base.is_dir():
        return _error("not_found", f"{scope or '.'} is not a folder under {root}")
    cut = None
    if since.strip():
        # "What changed after my last session": nine of nine models in the
        # 2026-11-04 survey asked for it (K12). No other reply here carries a
        # date, so without it deepseek-v4-flash-0731 gave no list in 10 of 10
        # turns, one claiming nothing changed, and with it the right list in
        # 10 of 10 in 1.6 tool calls against 20 (probe p4c). Dates ride along
        # only when asked for; `status` already says "changed since the index
        # read it" otherwise.
        import datetime as _dt
        try:
            when = _dt.datetime.fromisoformat(since.strip())
        except ValueError:
            return _error("bad_argument", f"since must be ISO 8601 (2026-11-04 or 2026-11-04T09:30), not {since!r}")
        cut = (when if when.tzinfo else when.astimezone()).timestamp()
    meta = _load_meta()
    stamps = meta.get("stamps", {})
    skipped = meta.get("no_text", {})
    failed = _failures()
    manifest = _manifest_inventory()
    if manifest is not None and _manifest_covers(manifest, cut, stamps, skipped):
        # A manifest names the documents; the rows come from it, and the reads a
        # row needs — a `.md` head, a sidecar, a source hash — wait for the page.
        serve_manifest = True
        entries, excluded_dirs = _manifest_entries(manifest, stamps, skipped, failed, scope, cut)
    else:
        serve_manifest = False
        entries, excluded_dirs = _walk_entries(base, cut, stamps, skipped, failed, DOC_EXTS, IMG_EXTS)
    counts = Counter(e["status"] for e in entries)
    # The folders the walk did not enter are a count in every listing and rows
    # only under status=excluded: thirteen hidden folders opened every listing
    # of this repository before any file (measured 2026-09-08).
    if status:
        entries = [e for e in entries if e["status"] == status]
    else:
        entries = [e for e in entries if not e["path"].endswith("/")]
    try:
        offset = int(cursor) if cursor else 0
    except ValueError:
        return _error("bad_argument", "cursor must be the next_cursor of a previous reply")
    page = entries[offset:offset + limit]
    if serve_manifest:
        for row in page:  # a manifest row's bytes and conversion verdict are read here, and only here
            _manifest_page_row(row, root)
    truncated = offset + limit < len(entries)
    return {"root": str(root), "total": len(entries), "files": page,
            "counts": {k: counts.get(k, 0) for k in ("indexed", "changed", "excluded", "failed", "unconverted")},
            "excluded_dirs": excluded_dirs,
            "truncated": truncated, "next_cursor": str(offset + limit) if truncated else None,
            "index": index_state(meta)}


def _iso(ts: float) -> str:
    """A local time with its offset, to the second: what `since` takes back."""
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 2. search
# ---------------------------------------------------------------------------

_section_cache: dict[tuple[str, float | int], tuple[str, list[tuple[int, str, Counter, int, str]]]] = {}


def _sections_of(key: str) -> tuple[str, list[tuple[int, str, Counter, int, str]]]:
    """Text and (offset, section, term counts, length, title) per section — a
    heading section in a note, a page in a PDF, a symbol in a source file —
    cached by identity stamp so a second query over the same top documents does
    not retokenize (nor re-extract), and a document replaced in place is not
    served from the cache. For a unit key (`path#offset`) the one unit,
    with the whole file's text, so a line is counted from the file's start."""
    rel, unit = _split_key(key)
    full = _root() / rel
    stamp = (rel, _identity_stamp(rel, full))
    if stamp not in _section_cache:
        if full.suffix.lower() == ".pdf":
            text, pages, _reason = _pdf_text(rel, full)
            bounds = [p["offset"] for p in pages] + [len(text)]
            parts = [(p["offset"], text[p["offset"]:bounds[i + 1]], f"p. {p['page']}")
                     for i, p in enumerate(pages)]
        elif _is_code(rel):
            text = _read(full)
            parts = _code_units(rel, text)
        else:
            text = _read(full)
            parts = [(off, part, part.splitlines()[0].lstrip("#").strip() if part.startswith("#") else "")
                     for off, part in _split(text)]
        secs = []
        # The tf is the body's. A code unit's title — the qualified symbol —
        # is a field the document stage reads (the index upserts it as the
        # unit's name) and this stage does not. Weighting it was measured on
        # 2026-09-11 over the 32 offline code questions and REFUTED: extra
        # title mass here (weights 1, 2, 3, 5, 8), at the document stage
        # (weight 3, rebuilt) and at both, lexical and hybrid, left every
        # paired interval across zero, and the section-stage arms traded file
        # recall for symbol precision without either clearing it.
        # docs/baseline/2026-09-11-title-boost.md.
        for off, part, title in parts:
            tf = Counter(_tokens(part))
            secs.append((off, part, tf, sum(tf.values()), title))
        if len(_section_cache) > 64:
            _section_cache.clear()
        _section_cache[stamp] = (text, secs)
    text, secs = _section_cache[stamp]
    return text, (secs if unit is None else [s for s in secs if s[0] == unit])


@tool(cls="atomic")
@_manifest_scoped
def silica_search(
    query: Annotated[str, Field(description="Concise question or description of the concept or behavior to find; preserve known names and identifiers")],
    folder: Annotated[str, Field(description="Only documents under this root-relative folder")] = "",
    k: Annotated[int, Field(description="Max hits")] = 5,
    per_doc: Annotated[int, Field(description="Max sections per document")] = 2,
    width: Annotated[int, Field(description="Characters in each hit's window")] = 600,
    queries: Annotated[list[str] | None, Field(description="More query groups, each ranked on its own and fused with `query` by rank: the concept in one, its identifiers in another")] = None,
    exclude: Annotated[list[str] | None, Field(description="Paths already read: their documents leave the ranking, so the hits are the next best and `unseen` counts what is left")] = None,
) -> dict:
    """For a question that names no identifier (what a behaviour is, where
    it is decided, why), call this before any grep (a Grep tool, or
    grep or rg in a shell), Glob or Read: it ranks
    passages and, in a source tree, the functions, methods, classes and
    constants themselves, by the question's words and their vectors.
    Do not grep for the words of a question; grep
    for an exact string or a symbol name you already know. Each hit carries
    its densest `width`-char window and line; a PDF's section is a page, so
    on a PDF hit widen `width` (1200-1500) rather than reading the page.
    `matched_terms` are the query terms in the hit; `coverage` is the share
    of the query's idf mass they carry (near 1 = every rare term matched,
    near 0 = only the common words); `terms_absent` are query terms found
    nowhere in the corpus, `terms_near` the corpus's closest spellings of
    them. `unseen` counts ranked documents not shown that match the query
    as well as the weakest shown; pass what you read as `exclude` for the
    next. No boolean "no answer" exists: read coverage and
    matched_terms and decide to stop or rephrase. A hit the dense leg alone
    found carries `dense` and coverage 0. In a source tree a hit's `section`
    is the symbol, `span` its lines, and `silica_read(path,
    section=<symbol>)` serves its body."""
    from silica_core import embeddings
    live, unextracted = _inventory()
    # A local-hybrid warm-up over a folder nothing has indexed yet: wait for
    # the first lexical index to land, this process's or another's, rather
    # than answer empty. Anything already on disk is served at once, even
    # while another process holds the build through its embedding phase.
    while index_state(live=live)["state"] == "cold" and embeddings.warmup_state()["state"] == "warming":
        embeddings.wait_lexical(0.5)
    with _STORE_RW:
        return _search(query, folder, k, per_doc, width, queries, live, unextracted, exclude)


def _search(query: str, folder: str, k: int, per_doc: int, width: int, queries: list[str] | None,
            live: dict[str, float], unextracted: list[str] | None = None,
            exclude: list[str] | None = None) -> dict:
    groups = [query] + [q for q in (queries or []) if q.strip()]
    # The dense leg is the user's call, made at index time, not the caller's:
    # it runs whenever the vectors are there and the query may go to the
    # endpoint; otherwise the reply says why and the search stays lexical.
    from silica_core import embeddings
    warm = embeddings.warmup_state()
    leg: dict[str, Any]
    if warm["state"] == "warming":
        leg = {"state": "warming", "hint": "the vectors are building in the background; this search is lexical"}
    elif warm["state"] == "failed":
        leg = {"state": "failed", "hint": warm["hint"]}
    elif not embeddings.enabled():
        leg = {"state": "off"}
    elif embeddings.load() is None:
        leg = {"state": "no_vectors", "hint": "run `silica-core index --embed`"}
    elif host := embeddings.consent_needed():
        leg = {"state": "consent_required", "hint": f"the query would leave this machine for {host}: "
               "run `silica-core index --embed --allow-remote` once to allow it"}
    else:
        leg = {"state": "ready"}
    hybrid = leg["state"] == "ready"
    state = index_state(live=live)
    if warm["state"] != "warming" and (
            state["state"] == "cold" or (state["state"] == "stale" and state["pending"] <= STALE_BUDGET)):
        # `wait=False`: another process mid-build is served around, not waited for
        state = build_index(rebuild=state["state"] == "cold", embed=embeddings.lazy_embed(), live=live,
                            wait=False)["index"]
    # else: served as it is; `index` in the reply says stale and how far behind
    store = get_store(INDEX)
    group_terms = [set(_tokens(q)) for q in groups]
    idf = store.idf(set().union(*group_terms))
    known = {t: v for t, v in idf.items() if v is not None}
    absent = sorted(t for t, v in idf.items() if v is None)
    # A term found nowhere weighs what BM25 gives a term at df=0, the heaviest
    # in the corpus, so a query whose rare words are all absent reads low even
    # when its one surviving word matches: measured 2026-09-08, "kubernetes
    # ingress controller nginx annotations" read coverage 1.0 on "annotations".
    idf_absent = math.log(1 + (len(store) + 0.5) / 0.5)
    mass = (sum(known.values()) + idf_absent * len(absent)) or 1.0
    scope = ""
    if folder.strip():
        try:
            scope = _rel(folder)
        except ValueError as e:
            return _error("out_of_root", str(e))
    rankings = [[d for d in store.bm25(q) if _paths.in_folder(d[0], scope)] for q in groups]
    # What the search could see: a relevant document among the unconverted or
    # the failed never ranks, and a term the corpus holds may still be absent
    # from the folder. `in_folder` with an empty folder is the whole root.
    # `unconverted` counted only the PDFs found empty until 2026-11-04, so a
    # folder of .docx contracts read `docs 0, unconverted 0`: an absence the
    # search had no right to imply (survey card C1, silica-agent's docs).
    unread = _unread_rows(scope, unextracted or [])
    visible = {
        "folder": scope,
        "docs": len({_split_key(p)[0] for p in store.paths() if _paths.in_folder(p, scope)}),
        "unconverted": sum(1 for r in unread if r["status"] == "unconverted"),
        "failed": sum(1 for r in unread if r["status"] == "failed")}
    # Not named here: through this surface deepseek-v4-flash-0731 called a
    # folder of unread .docx silent in 0 of 20 turns even before this fix,
    # with the count at 0 and no path named, because it asks silica_files
    # itself; the paths fixed nothing measurable (probe p1, 2026-11-04).
    # `silica_files(status=...)` and `unreadable()` name them.
    absent_in_scope = sorted(t for t in known
                             if not any(_paths.in_folder(p, scope) for p in store.paths_with(t)))
    dense_docs: list[tuple[str, float]] = []
    dense_secs: dict[tuple[str, int], float] = {}
    if hybrid:
        try:
            dense_docs, dense_secs = embeddings.rank(query, live)
        except Exception as e:  # the endpoint is the extension's business: the reply says so, the search stays lexical
            leg, hybrid = {"state": "failed", "hint": f"embeddings failed: {e}"}, False
        else:
            # the dense ranking at the store's granularity: a note by its best
            # section, a code file unit by unit, so the fusion ranks like keys
            best: dict[str, float] = {}
            for (r, o), c in dense_secs.items():
                dk = f"{r}#{o}" if _is_code(r) else r
                if c > best.get(dk, -2.0) and _paths.in_folder(dk, scope):
                    best[dk] = c
            dense_docs = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
            leg["docs"] = len({_split_key(dk)[0] for dk in best})
    fuse = hybrid or len(groups) > 1
    if not fuse:
        docs = rankings[0]
    else:
        # one ranking per group, plus the dense one: the fused order is by
        # rank; a document keeps its best raw score and the union of its terms
        lex: dict[str, tuple[float, set]] = {}
        for ranking in rankings:
            for p, sc, m in ranking:
                sc0, m0 = lex.get(p, (0.0, set()))
                lex[p] = (max(sc0, sc), m0 | m)
        lists: list[list[Any]] = [[p for p, _s, _m in ranking] for ranking in rankings]
        if hybrid:
            lists.append([p for p, _c in dense_docs])
        fused = _rrf(lists)
        docs = [(p, *lex.get(p, (0.0, set()))) for p in sorted(fused, key=lambda p: (-fused[p], p))]
    # What the caller already read leaves the ranking before the cut, so the
    # pool and the hits are the next best rather than the same five again;
    # a path that names nothing ranked is no error, it simply removes nothing.
    gone: set[str] = set()
    if exclude:
        drop = set()
        for x in exclude:
            try:
                drop.add(_rel(x))
            except ValueError:
                continue  # outside the root ranks nothing anyway
        gone = {_split_key(p)[0] for p, _s, _m in docs} & drop
        docs = [d for d in docs if _split_key(d[0])[0] not in gone]
    top = docs[: _pool(k)]
    dense = dict(dense_docs)  # a document's best section cosine, for `documents`
    text_all = " ".join(groups)
    weights = store.query_idf(_query_terms(text_all))
    secs: list[tuple[str, int, str, Counter, int, str]] = []
    texts: dict[str, str] = {}
    for path, _score, _matched in top:
        rel = _split_key(path)[0]
        try:
            texts[rel], parts = _sections_of(path)
        except OSError:
            continue
        secs.extend((rel, off, part, tf, dl, title) for off, part, tf, dl, title in parts)
    avg = (sum(s[4] for s in secs) / len(secs)) if secs else 1.0

    def bm25(tf: Counter, dl: int, terms: set) -> float:
        return sum(known[t] * (tf[t] * (K1 + 1)) / (tf[t] + K1 * (1 - B + B * dl / avg)) for t in terms)

    # every section of the pool that some leg can rank: a lexical match in any
    # group, or (hybrid) a vector; `scored` keeps what the reply shows
    scored: dict[tuple[str, int], tuple[float, str, set, str]] = {}
    per_group: list[list[tuple[float, tuple[str, int]]]] = [[] for _g in groups]
    for path, off, part, tf, dl, title in secs:
        matched = {t for t in known if tf.get(t)}
        key = (path, off)
        if not matched and key not in dense_secs:
            continue
        scored[key] = (bm25(tf, dl, matched), part, matched, title)
        for gi, tg in enumerate(group_terms):
            mg = {t for t in tg if t in matched}
            if mg:
                per_group[gi].append((bm25(tf, dl, mg), key))
    if not fuse:
        order = sorted(scored, key=lambda key: (-scored[key][0], key[0], key[1]))
    else:
        sec_lists: list[list[Any]] = [[key for _s, key in sorted(g, key=lambda x: (-x[0], x[1]))] for g in per_group]
        if hybrid:
            sec_lists.append([key for key, _c in sorted(((key, c) for key, c in dense_secs.items() if key in scored),
                                                        key=lambda kv: (-kv[1], kv[0]))])
        fused_secs = _rrf(sec_lists)
        order = sorted(fused_secs, key=lambda key: (-fused_secs[key], key[0], key[1]))
    hits: list[dict] = []
    seen: dict[str, int] = {}
    versions: dict[str, str] = {}  # the hash silica_read checks, from the text already in hand
    conversions: dict[str, tuple[str | None, str | None]] = {}
    for key in order:
        path, off = key
        score, part, matched, title = scored[key]
        if seen.get(path, 0) >= per_doc:
            continue
        seen[path] = seen.get(path, 0) + 1
        o, window = best_window_spans(part, text_all, width, 1, weights, snap=True)[0]
        hit = {"path": path, "section": title,
               "line": texts[path].count("\n", 0, off + o) + 1,
               "version": versions.setdefault(path, _version(texts[path])),
               "score": round(score, 2), "matched_terms": sorted(matched),
               "coverage": round(sum(known[t] for t in matched) / mass, 2),
               "window": window}
        if _is_code(path):  # the unit's lines, for a read of the whole symbol
            first = texts[path].count("\n", 0, off) + 1
            hit["span"] = [first, first + part.rstrip("\n").count("\n")]
        if key in dense_secs:
            hit["dense"] = round(dense_secs[key], 3)
        if path.lower().endswith(".md"):
            # A converted note quotes its original as it was: when the PDF was
            # replaced since, the hit says so before anything is cited, not
            # only a read of the note (survey card C2, K08: six of nine).
            if path not in conversions:
                conversions[path] = _conversion_of(_root() / path, texts[path])
            src, src_state = conversions[path]
            if src_state:
                hit.update(source=src, source_state=src_state)
        hits.append(hit)
        if len(hits) == k:
            break
    return {"root": str(_root()), "query": query, **({"queries": groups[1:]} if len(groups) > 1 else {}),
            "hits": hits,
            # the first k of the ranking plus any document a hit came from lower
            # down: enough to tell "right paper, wrong section" from "wrong
            # paper" without repaying the whole candidate list (43% of the reply
            # at 15 entries, measured 2026-09-08)
            "documents": [{"path": _split_key(p)[0], "score": round(sc, 2), "matched_terms": sorted(m),
                           "coverage": round(sum(known[t] for t in m) / mass, 2),
                           **({"line": texts[_split_key(p)[0]].count("\n", 0, _split_key(p)[1]) + 1}
                              if _split_key(p)[1] is not None and _split_key(p)[0] in texts else {}),
                           **({"dense": round(dense[p], 3)} if p in dense else {})}
                          for i, (p, sc, m) in enumerate(top) if i < k or _split_key(p)[0] in versions],
            "candidates": len(docs),
            **_unseen(docs, hits, known, mass),
            **({"excluded": len(gone)} if exclude else {}),
            "terms_absent": absent,
            **({"terms_near": near} if (near := _terms_near(store, absent)) else {}),
            **({"terms_absent_in_scope": absent_in_scope} if scope else {}),
            "scope": visible,
            "dense": leg,
            "index": state}


_NEAR_TERMS = 5  # absent terms looked up per search: ~7 ms each on an 87k-term vocabulary


def _terms_near(store, absent: list[str]) -> dict[str, list[str]]:
    """For each absent word of four letters or more, the corpus's closest
    spellings: an absent term says the folder never uses the word, not which
    word it uses instead (K04, five of nine models in the 2026-11-04 survey).
    Lexical and offered, never substituted: the caller decides whether
    `neuroscience` is what `neurosience` meant. A misspelt surname among
    200 contacts went unanswered in 14 of 20 turns without it and 0 with it
    on deepseek-v4-flash-0731 (probe p5b, 2026-11-04); a misspelt common
    word the model fixes on its own (p5)."""
    out: dict[str, list[str]] = {}
    for t in [t for t in absent if len(t) >= 4 and not t.isdigit()][:_NEAR_TERMS]:
        if close := store.near(t):
            out[t] = close
    return out


def _unseen(docs: list, hits: list[dict], known: dict[str, float], mass: float) -> dict:
    """`unseen`: the ranked documents no hit came from whose own coverage is
    at least that of the weakest document a lexical hit came from. A count
    of the whole ranking would always be large (BM25 ranks every document
    holding one query word: 21,500 candidates on silica-agent's tree), so
    the bar is the reply's own: zero means nothing left ranks as strongly
    as what was shown. Absent when no hit matched a word, the dense leg's
    alone giving no bar. Seven of nine models in the 2026-11-04 survey asked
    to know what remains unseen before they stop (K05); `exclude` pages on.
    On a 50-item list under the reference loop's 12,000-character result cap,
    deepseek-v4-flash-0731 finished it either way but spent 318,535 prompt
    tokens and 21.9 tool calls a turn without these, 122,182 and 10.0 with
    them (probe p2c recheck, 2026-11-05)."""
    cov: dict[str, float] = {}
    for p, _sc, m in docs:
        rel = _split_key(p)[0]
        c = sum(known[t] for t in m) / mass
        if c > cov.get(rel, -1.0):
            cov[rel] = c
    shown = {h["path"] for h in hits}
    bar = min((cov[r] for r in shown if cov.get(r, 0.0) > 0), default=None)
    if bar is None:
        return {}
    return {"unseen": sum(1 for r, c in cov.items() if r not in shown and c >= bar)}


# ---------------------------------------------------------------------------
# 3. read
# ---------------------------------------------------------------------------

def _served(rel: str) -> dict:
    """The text `silica_read` serves for `rel`, and what it says about it: a
    note's own text, a binary source's sidecar, a PDF's text layer, or the
    error a read returns. One place, so `version_of` cannot drift from the
    version a read checks."""
    root = _root()
    full = root / rel
    if not full.is_file():
        return _error("not_found", f"{rel} is not a file under {root}")
    pages: list[dict] = []
    source, paged, source_state = None, False, None
    if full.suffix.lower() != ".pdf" and _is_text(full):
        text = _read(full)
        if full.suffix.lower() == ".md":
            source, source_state = _conversion_of(full, text)
    else:
        sidecar = full.with_suffix(".md")
        if sidecar.is_file():
            text = _read(sidecar)
            source_state = _source_state(full, text)
            source, rel = rel, sidecar.relative_to(root).as_posix()
        elif full.suffix.lower() == ".pdf":
            text, pages, reason = _pdf_text(rel, full)
            if not text:
                return _error("unconverted", f"{reason} in {rel}; run `silica-core import {rel}`")
            source, paged = rel, True
        else:
            return _error("unconverted", f"no extracted text beside {rel}; run `silica-core import {rel}`")
    return {"path": rel, "text": text, "pages": pages, "paged": paged, "source": source,
            "source_state": source_state}


def version_of(path: str) -> str | None:
    """The `version` `silica_read` would serve for `path` now, or None when
    the read would refuse (missing, outside the root, unconverted). For a
    harness that remembers a citation and checks it later: compare with the
    version it kept instead of hashing the file its own way."""
    try:
        served = _served(_rel(path))
    except ValueError:
        return None
    return None if "error" in served else _version(served["text"])


@tool(cls="atomic")
@_manifest_scoped
def silica_read(
    path: Annotated[str, Field(description="Root-relative file path (from silica_files or a search hit)")],
    start: Annotated[int, Field(description="First line, 1-based")] = 1,
    end: Annotated[int, Field(description="Last line; 0 = to the end, under max_chars")] = 0,
    section: Annotated[str, Field(description="Serve this heading's section instead (exact title or unique prefix)")] = "",
    max_chars: Annotated[int, Field(description="Cap on the served text")] = 16000,
    expect_version: Annotated[str, Field(description="Refuse with error.changed when the file's version differs")] = "",
) -> dict:
    """Read a located slice before citing it: lines `start..end`, one
    `section` by heading, a PDF page, or the whole file when it fits
    `max_chars`, always with the outline and a `version`;
    `truncated` and `next_start` say how to continue. A PDF is served from
    its own text layer, page by page: `section="p. 8"` serves one page —
    the whole page, ~1.2k tokens (measured 2026-09-09), so ask for it to cite
    a passage a search located, and widen the search's `width` to explore —
    `pages` counts them, `page_map` gives the pages the slice covers, and
    `extract_path` is that text on disk, for grep and the harness's own
    reader (a cache, rebuilt when the PDF changes, under no `version`
    guard). A PDF or office file with a `.md` extracted beside it is served
    from that instead, and `source_state` says whether the original still
    hashes to what the conversion recorded: `stale` means the PDF changed
    and the `.md` did not, `unverifiable` that the sidecar records no hash or
    its original is gone. Reading the `.md` itself, the path a search hit
    carries, reports the same `source` and `source_state`.
    Either way `source` names the original, and a PDF
    with no readable text layer is `error.unconverted`. Carry
    `version` forward as `expect_version` to be refused instead of reading
    different bytes under an old citation; `version` is the served text's,
    never the original's."""
    root = _root()
    try:
        rel = _rel(path)
    except ValueError as e:
        return _error("out_of_root", str(e))
    served = _served(rel)
    if "error" in served:
        return served
    rel, text, pages, source, source_state = (served["path"], served["text"], served["pages"],
                                              served["source"], served["source_state"])
    paged = served["paged"]
    version = _version(text)
    if expect_version and expect_version != version:
        return _error("changed", f"{rel} is now version {version}", version=version)
    lines = text.splitlines()
    # An extracted text layer is not markdown: its pages are the structure, and
    # scanning it for `#` would read a paper's own markdown examples as headings.
    page_starts = [(p["page"], p["line"]) for p in pages] if paged else []
    if paged:
        outline = []
    elif _is_code(rel):
        # symbols, methods included while the file is small enough that the
        # table stays a fraction of the read (a 500-symbol module lists its classes)
        outline = _code_outline(rel, text)
        if len(outline) > 150:
            outline = [h for h in outline if h["level"] == 1]
    else:
        outline = [{"level": h["level"], "title": h["text"], "line": text.count("\n", 0, h["pos"]) + 1}
                   for h in parse_headings(text)]
    if section.strip() and paged:
        m = _PAGE.fullmatch(section.strip())
        if not m or not 1 <= int(m.group(1)) <= len(page_starts):
            return _error("not_found", f"no page matches {section!r}", pages=len(page_starts))
        n = int(m.group(1))
        start = page_starts[n - 1][1]
        end = page_starts[n][1] - 2 if n < len(page_starts) else len(lines)
    elif section.strip():
        want = section.strip().casefold()
        idx = [i for i, h in enumerate(outline) if h["title"].casefold() == want] or \
              [i for i, h in enumerate(outline) if h["title"].casefold().startswith(want)]
        if not idx:
            return _error("not_found", f"no heading matches {section!r}", outline=outline)
        if len(idx) > 1:
            return _error("bad_argument", f"{len(idx)} headings match {section!r}; be exact", outline=outline)
        h = outline[idx[0]]
        start = h["line"]
        end = h.get("end_line") or next((o["line"] - 1 for o in outline[idx[0] + 1:] if o["level"] <= h["level"]), len(lines))
    start = max(1, start)
    end = len(lines) if end <= 0 else min(end, len(lines))
    out: list[str] = []
    size, truncated = 0, False
    for line in lines[start - 1:end]:
        if size + len(line) + 1 > max_chars:
            truncated = True
            break
        out.append(line)
        size += len(line) + 1
    served_end = start + len(out) - 1 if out else start - 1
    return {"root": str(root), "path": rel, "version": version, "start": start, "end": served_end,
            "text": "\n".join(out), "truncated": truncated,
            "next_start": served_end + 1 if truncated else None,
            "outline": outline, "source": source, "source_state": source_state,
            "pages": len(page_starts) if paged else None,
            # the page the slice opens on, plus every page that starts inside it:
            # a 500-page book must not spend a page table on every read
            "page_map": [{"page": n, "line": ln} for n, ln in
                         [pl for pl in page_starts if pl[1] <= start][-1:]
                         + [pl for pl in page_starts if start < pl[1] <= served_end]] if paged else None,
            "extract_path": str(_extract_cache(rel)) if paged else None}


# ---------------------------------------------------------------------------
# 4. code_pack
# ---------------------------------------------------------------------------

@tool(cls="atomic")
def silica_code_pack(
    target: Annotated[str, Field(description="Root-relative source path, optionally narrowed with '#Class', '#Class.member' or '#L<line>' (the symbol whose declaration spans that line)")],
    budget_chars: Annotated[int, Field(description="Character budget for the whole pack; the target is always served")] = 24000,
    sections: Annotated[list[str] | None, Field(description="Sections besides the target: any of 'hierarchy', 'callers', 'neighborhood', 'external', 'importers'. Empty = all")] = None,
) -> dict:
    """Use before editing or explaining a source file, or when a traceback or
    a diff names `file:line`: one call serves the file, a `#Class.member`, or
    the symbol at `#L<line>`, with its supertypes, extenders, callers, the
    signatures it names, external dependencies and importers, inside a
    character budget. Static AST only, same repo state, same bytes. Check
    `truncated` before treating the target as complete; `dropped` names what
    did not fit. `languages` lists the parsers this install has."""
    from silica_core.kernel.code import codepack
    from silica_core.kernel.code.codeast import EXTENSION_MAP

    try:
        pack = codepack.code_pack(str(_root()), target, budget_chars, sections=sections or None)
    except (ValueError, OSError) as e:
        return _error("bad_argument", str(e))
    return {"root": str(_root()), **pack, "languages": sorted(set(EXTENSION_MAP.values()))}


# ---------------------------------------------------------------------------
# 5. write_note
# ---------------------------------------------------------------------------

def _name_map() -> dict[str, str]:
    """Wikilink resolution the way Obsidian does it: by note stem (case-
    insensitive), or by root-relative path with or without `.md`."""
    out: dict[str, str] = {}
    for rel in _note_paths():
        out.setdefault(Path(rel).stem.casefold(), rel)
        out[rel.casefold()] = rel
        out[rel[:-3].casefold()] = rel
    return out


def _lint(rel: str, content: str) -> list[dict]:
    from silica_core.kernel.link.ast import _balanced

    out = [{"kind": "structure", "target": v, "line": 0} for v in _balanced(content)]
    names = _name_map()
    for m in _WIKILINK.finditer(content):
        target = m.group(1).strip()
        if target.casefold() not in names:
            out.append({"kind": "unresolved_link", "target": target,
                        "line": content.count("\n", 0, m.start()) + 1})
    return out


def _unescape(body: str) -> str:
    r"""Decode a body the model escaped twice: no real newline anywhere and a
    literal \n where each one belonged. Measured through `silica-core repl` on a
    local gemma4:e4b, which escapes its tool arguments twice, so json.loads
    hands over backslash-n and the note lands as a single line. A single real
    newline means the model got it right and the body is left alone, which is
    what keeps a fenced code block containing \n intact.

    ponytail: three replaces, not codecs unicode_escape — that one also eats a
    lone backslash and every \x and \u a note is entitled to contain.
    """
    if "\n" in body or "\\n" not in body:
        return body
    return body.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


@tool(cls="atomic")
def silica_write_note(
    path: Annotated[str, Field(description="Root-relative path of the note (.md added when missing)")],
    body: Annotated[str, Field(description="Markdown body, written as given; a body escaped twice (no real newline, \\n in their place) is decoded first")],
    frontmatter: Annotated[dict[str, Any] | None, Field(description="Optional YAML frontmatter, serialised on top of the body")] = None,
    expect_version: Annotated[str, Field(description="Refuse with error.changed unless the existing file has this version")] = "",
    create_only: Annotated[bool, Field(description="Refuse when the file already exists")] = False,
) -> dict:
    """Write one note atomically, as given, then lint it: `lint` lists
    structural violations and unresolved wikilinks. Refuses to overwrite
    when `expect_version` does not match or when `create_only` is set and
    the file exists. Undo is git; the core keeps no journal."""
    root = _root()
    try:
        rel = _rel(path if path.lower().endswith(".md") else path + ".md")
    except ValueError as e:
        return _error("out_of_root", str(e))
    full = root / rel
    exists = full.is_file()
    if exists and create_only:
        return _error("bad_argument", f"{rel} exists and create_only is set")
    if exists:
        current = _version(_read(full))
        if expect_version and expect_version != current:
            return _error("changed", f"{rel} is now version {current}", version=current)
    content = body = _unescape(body)
    if frontmatter:
        content = "---\n" + yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False) + "---\n" + body
    try:
        full.parent.mkdir(parents=True, exist_ok=True)
        _paths.atomic_write_bytes(full, content.encode("utf-8"))
    except OSError as e:
        return _error("bad_argument", str(e))
    written = _read(full)
    return {"root": str(root), "path": rel, "version": _version(written), "created": not exists,
            "lint": _lint(rel, written)}


FUNCTIONS = (silica_files, silica_search, silica_read, silica_code_pack, silica_write_note)
files, search, read, code_pack, write_note = FUNCTIONS
