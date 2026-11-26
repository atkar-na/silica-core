# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""The producer side of `SILICA_INVENTORY`: a corpus listing → a manifest.

`core._walk()` traverses the whole root on every search, and on a mount that
reports a parent-directory mtime its stamps cannot tell a replaced file from an
untouched one. The reader (`core._manifest_inventory`) answers from a JSON
manifest instead when `SILICA_INVENTORY` names one, comparing stamps for
equality — so a content digest substitutes for an mtime with no schema change.
Nothing produced that manifest; this module does, from a listing a caller
already holds (a bucket listing, a sync, a database), without walking anything.

**The document set must not drift.** `stamps` and `unextracted` have to be
*exactly* the sets the walk derives, or the index set changes silently while
reporting `ready`. So this module imports upstream's own symbols — the suffix
lists, the size cap, `_is_code`, the ignore matcher — and never re-types one.
`tests/test_manifest_producer.py` pins that with a parity test against the walk.

The producer takes a root and a listing; it knows nothing about how either was
obtained. It fails closed: a candidate document row with no size or no stamp
raises, unless the caller opts in, and then the count of dropped rows is
recorded. A manifest that silently omits documents is worse than no manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import orjson

__all__ = ["build_manifest", "write_manifest", "digest_stamp", "main"]


def digest_stamp(digest: str) -> int:
    """A content digest → the stamp the index compares for equality.

    The first 12 hex digits as an int: 48 bits, exactly representable in a
    float, so the value survives JSON and the reader's `float()` unchanged.
    Take more digits and the comparison silently loses them. A missing,
    non-hex or too-short digest is a malformed row, not an incomplete one, so
    it raises here regardless of `allow_incomplete`.
    """
    if not isinstance(digest, str):
        raise ValueError(f"a digest stamp must be a hex string, not {type(digest).__name__}")
    if len(digest) < 12:
        raise ValueError(f"a digest stamp needs at least 12 hex digits, got {digest!r}")
    try:
        return int(digest[:12], 16)
    except ValueError as e:
        raise ValueError(f"a digest stamp must be hex, got {digest!r}: {e}") from e


def _excluded(rel: str, skip) -> bool:
    """Whether `_walk` would not enter or list `rel`: a hidden name at any
    level, a directory it prunes (NOISE_DIRS, `.silicaignore`), or a file its
    own matcher rejects. Mirrors `_walk`'s exclusion behaviour exactly, from
    the path alone — a listing has no directory entries to prune."""
    segs = rel.split("/")
    if any(s.startswith(".") for s in segs):
        return True
    for i in range(1, len(segs)):          # every ancestor directory
        if skip("/".join(segs[:i])):
            return True
    return skip(rel)                        # the file itself


def _stamp_value(rel: str, stamp):
    """`(value, mtime_or_None)`: a hex string is a digest, a number an mtime.
    Bool is never a stamp (it is an int in Python, and the schema forbids it)."""
    if isinstance(stamp, bool):
        raise ValueError(f"row {rel!r}: a stamp must be a number or a hex digest, not a bool")
    if isinstance(stamp, str):
        return digest_stamp(stamp), None
    if isinstance(stamp, (int, float)):
        return float(stamp), float(stamp)
    raise ValueError(f"row {rel!r}: a stamp must be a number or a hex digest, not {type(stamp).__name__}")


def build_manifest(root, rows, *, generated_at: str | None = None,
                   allow_incomplete: bool = False) -> dict:
    """A `SILICA_INVENTORY` manifest from `rows`, one per corpus object.

    Each row is `(rel, size, stamp)`: `rel` a root-relative posix path, `size`
    its byte size, `stamp` a hex digest string (content) or a float mtime.
    Returns the dict `INVENTORY.md` freezes: `version`, `root` (absolute),
    `generated_at` (ISO 8601 with offset, unless given), `count`, `stamps`
    (sorted by path), `unextracted`, `sizes`, and `mtimes` when the rows
    carried mtimes. With `allow_incomplete` a candidate document row missing a
    size or a stamp is dropped instead of raising, and the manifest records
    `skipped`; `count` still equals `len(stamps)`.
    """
    from silica_core.core import _CODE_MAX_BYTES, _PROSE_SUFFIXES, _is_code
    from silica_core.kernel.recall import paths as _paths
    from silica_core.sources.convert import DOC_EXTS, IMG_EXTS

    root_path = Path(root)
    skip = _paths.ignore_path_matcher(str(root_path))

    stamps: dict[str, float | int] = {}
    unextracted: list[str] = []
    sizes: dict[str, int] = {}
    mtimes: dict[str, float] = {}
    incomplete: list[str] = []

    def _need(rel: str, size, stamp) -> bool:
        """True when the row carries both; otherwise raise, or record and skip."""
        if size is not None and stamp is not None:
            return True
        if allow_incomplete:
            incomplete.append(rel)
            return False
        missing = "size" if size is None else "stamp"
        raise ValueError(
            f"row {rel!r} is a document the manifest must cover but has no {missing}; "
            "a manifest that silently omits documents is worse than no manifest "
            "(pass allow_incomplete=True to drop it and record it)")

    for rel, size, stamp in rows:
        rel = str(rel)
        if _excluded(rel, skip):
            continue
        suffix = Path(rel).suffix.lower()
        # `.with_suffix(".md").is_file()` is the sidecar test `_inventory`
        # makes: a converted PDF's `.md` is the note, and reading both would
        # double it. The check is on disk, exactly as the walk does it.
        no_sidecar = not (root_path / rel).with_suffix(".md").is_file()

        if suffix in _PROSE_SUFFIXES or (suffix == ".pdf" and no_sidecar):
            if not _need(rel, size, stamp):
                continue
        elif _is_code(rel):
            # A code file is a document only up to the cap; over it the index
            # reads nothing, so the row is neither stamped nor unextracted.
            if size is None:
                if not _need(rel, size, stamp):
                    continue
            elif size > _CODE_MAX_BYTES:
                continue
            elif not _need(rel, size, stamp):
                continue
        elif suffix in DOC_EXTS and suffix not in IMG_EXTS and suffix != ".pdf" and no_sidecar:
            if not _need(rel, size, stamp):
                continue
            unextracted.append(rel)
            continue
        else:
            continue                        # excluded: neither stamped nor unextracted

        # Every branch above reached `_need(rel, size, stamp)` and continued,
        # so both are present here; the assert only tells the type checker.
        assert size is not None and stamp is not None
        value, mtime = _stamp_value(rel, stamp)
        stamps[rel] = value
        sizes[rel] = size
        if mtime is not None:
            mtimes[rel] = mtime

    manifest: dict = {
        "version": 1,
        "root": str(root_path.resolve()),
        "generated_at": generated_at or datetime.now().astimezone().isoformat(),
        "count": len(stamps),
        "stamps": dict(sorted(stamps.items())),
        "unextracted": sorted(unextracted),
        "sizes": dict(sorted(sizes.items())),
    }
    if mtimes:
        manifest["mtimes"] = dict(sorted(mtimes.items()))
    if allow_incomplete:
        manifest["skipped"] = len(incomplete)
    return manifest


def write_manifest(path, manifest: dict) -> None:
    """Write `manifest` to `path` atomically: a temp file in the same
    directory, fsync, `os.replace`, so a reader never sees a half-written file
    and a failure mid-write leaves the previous one intact. UTF-8, sorted
    keys, so the bytes are stable across producers."""
    path = Path(path)
    data = orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _rows_from_listing(listing: str):
    """Each line a JSON object: `{"rel", "size", "sha1"}` or
    `{"rel", "size", "mtime"}`. A digest becomes the stamp, else the mtime;
    neither leaves the stamp unset, which `build_manifest` then decides on."""
    with open(listing, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as e:
                raise SystemExit(f"{listing}:{lineno}: not JSON: {e}")
            rel = obj.get("rel")
            if not rel:
                raise SystemExit(f"{listing}:{lineno}: a row needs a 'rel'")
            if "sha1" in obj:
                stamp = obj["sha1"]
            elif "digest" in obj:
                stamp = obj["digest"]
            elif "mtime" in obj:
                stamp = obj["mtime"]
            else:
                stamp = None
            yield (rel, obj.get("size"), stamp)


def main(argv=None) -> dict:
    """`python -m silica_core.inventory --root <root> --listing <rows.jsonl>
    --out <manifest.json>`: build the manifest from a listing and write it."""
    parser = argparse.ArgumentParser(
        prog="silica_core.inventory",
        description="Build a SILICA_INVENTORY manifest from a corpus listing.")
    parser.add_argument("--root", required=True, help="the absolute root the manifest describes")
    parser.add_argument("--listing", required=True, help="a JSONL file, one row per corpus object")
    parser.add_argument("--out", required=True, help="where to write the manifest")
    parser.add_argument("--generated-at", default=None, help="ISO 8601; defaults to now")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="drop candidate rows missing a size or a stamp, and record the count")
    args = parser.parse_args(argv)

    manifest = build_manifest(args.root, _rows_from_listing(args.listing),
                              generated_at=args.generated_at,
                              allow_incomplete=args.allow_incomplete)
    write_manifest(args.out, manifest)
    return manifest


if __name__ == "__main__":
    main()
