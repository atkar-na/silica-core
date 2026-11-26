# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""Where the index pass gets a document's bytes: the tree itself, or an
HTTP endpoint that serves the same objects by key.

The pass reads every document exactly once, and on a tree behind a slow
network mount that read is what the pass waits on: extraction and
tokenisation take milliseconds, the bytes take seconds. A plain HTTP
endpoint in front of the same objects answers `Range` requests, so the
bytes can be pulled on a bounded pool of streams and handed to the
extractor as local files. Nothing about extraction changes — it still
opens a path — only where the path came from.

`SILICA_SOURCE_URL` selects it (a base URL; a key is appended, quoted, one
segment at a time). Unset, the pass reads the tree as it always has.

    SILICA_SOURCE_URL         base URL of the byte endpoint; unset = filesystem
    SILICA_SOURCE_STREAMS     concurrent streams, default 24
    SILICA_SOURCE_CHUNK_MB    bytes per Range request, default 4
    SILICA_SOURCE_TIMEOUT     seconds per request, default 60
    SILICA_SOURCE_AUTH        one optional header, `Header-Name: value`
                              (`SILICA_SOURCE_AUTH_VALUE` supplies the value
                              alone when the name is in `SILICA_SOURCE_AUTH`)

The module holds no credentials, no hostnames and no deployment numbers: a
token arrives through the environment and never touches this file.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
from collections import deque
from collections.abc import Hashable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_STREAMS = 24
DEFAULT_CHUNK_MB = 4
DEFAULT_TIMEOUT = 60.0
# Retries per Range request: an outage that outlives this is reported, not
# waited out forever, and the backoff is capped so a long pass cannot stall
# on one document.
DEFAULT_ATTEMPTS = 4
DEFAULT_BACKOFF = 0.25
_MAX_BACKOFF = 8.0
_RETRY_STATUS = frozenset({408, 425, 429})
_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)\s*$")


class SourceError(RuntimeError):
    """A document could not be read from its byte source."""


class SourceSizeError(SourceError):
    """The source served a length that disagrees with the expected one."""


class _Retryable(Exception):
    """One request failed in a way a second attempt can still fix."""


@dataclass(frozen=True)
class SourceConfig:
    """What `SILICA_SOURCE_URL` and its companions resolve to."""

    base_url: str
    streams: int = DEFAULT_STREAMS
    chunk_bytes: int = DEFAULT_CHUNK_MB * 1024 * 1024
    timeout: float = DEFAULT_TIMEOUT
    auth: tuple[str, str] | None = None
    attempts: int = DEFAULT_ATTEMPTS
    backoff: float = DEFAULT_BACKOFF

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SourceConfig | None:
        """The configured source, or None when `SILICA_SOURCE_URL` is unset.

        None is the whole switch: every caller keeps its filesystem path when
        this returns None, so an installation that never sets the variable
        cannot tell this module exists.
        """
        env = os.environ if environ is None else environ
        base = (env.get("SILICA_SOURCE_URL") or "").strip()
        if not base:
            return None
        scheme = urllib.parse.urlsplit(base).scheme.lower()
        if scheme not in ("http", "https"):
            raise SourceError(f"SILICA_SOURCE_URL must be http or https, not {scheme or 'a relative path'!r}")
        return cls(base_url=base.rstrip("/"),
                   streams=_env_int(env, "SILICA_SOURCE_STREAMS", DEFAULT_STREAMS, 1, 256),
                   chunk_bytes=_env_int(env, "SILICA_SOURCE_CHUNK_MB", DEFAULT_CHUNK_MB, 1, 1024) * 1024 * 1024,
                   timeout=_env_float(env, "SILICA_SOURCE_TIMEOUT", DEFAULT_TIMEOUT, 1.0, 3600.0),
                   auth=_auth_from_env(env))


def _env_int(env: Mapping[str, str], key: str, default: int, low: int, high: int) -> int:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SourceError(f"{key} must be an integer, not {raw!r}") from None
    if not low <= value <= high:
        raise SourceError(f"{key} must be between {low} and {high}, not {value}")
    return value


def _env_float(env: Mapping[str, str], key: str, default: float, low: float, high: float) -> float:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise SourceError(f"{key} must be a number, not {raw!r}") from None
    if not low <= value <= high:
        raise SourceError(f"{key} must be between {low} and {high}, not {value}")
    return value


def _auth_from_env(env: Mapping[str, str]) -> tuple[str, str] | None:
    """One optional request header, `name: value`.

    Deliberately a single header and not a header table: a byte endpoint
    needs one credential, and a general mechanism invites per-host policy
    into a file that is public.
    """
    raw = (env.get("SILICA_SOURCE_AUTH") or "").strip()
    if not raw:
        return None
    value = (env.get("SILICA_SOURCE_AUTH_VALUE") or "").strip()
    if value:
        name = raw
    else:
        name, sep, value = raw.partition(":")
        if not sep:
            raise SourceError("SILICA_SOURCE_AUTH must be 'Header-Name: value' "
                              "(or name it there and put the value in SILICA_SOURCE_AUTH_VALUE)")
        name, value = name.strip(), value.strip()
    if not name or not value:
        raise SourceError("SILICA_SOURCE_AUTH needs both a header name and a value")
    if "\r" in name or "\n" in name or "\r" in value or "\n" in value:
        raise SourceError("SILICA_SOURCE_AUTH must not contain a newline")
    return name, value


class _Chunk:
    """One answered Range request: bytes written, the object's total if the
    response declared one, and the last byte the server actually sent."""

    __slots__ = ("wrote", "total", "span_end")

    def __init__(self, wrote: int, total: int | None, span_end: int | None):
        self.wrote, self.total, self.span_end = wrote, total, span_end


class Fetcher:
    """Reads an object by key over HTTP, one bounded Range request at a time.

    A shared client and a semaphore, because the caller's threads are what
    supply concurrency: N threads through this object never open more than
    `streams` requests, and one document is read by one thread in order, so a
    fetch neither reorders bytes nor races another fetch.
    """

    def __init__(self, config: SourceConfig, client: httpx.Client | None = None):
        self.config = config
        self._slots = threading.BoundedSemaphore(config.streams)
        self._client = client or httpx.Client(timeout=config.timeout, follow_redirects=True)
        self._owns_client = client is None
        self._closed = False
        self._sleep = time.sleep  # a seam: a caller that must not wait can replace it

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Abort whatever is in flight and release the connection pool.

        Closing the client first is what makes `close()` quick: a reader
        blocked in a socket wakes with a transport error instead of waiting
        out `SILICA_SOURCE_TIMEOUT`.
        """
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- addressing --------------------------------------------------------

    def url_for(self, key: str) -> str:
        """The object's URL: the base plus the key, quoted a segment at a time
        so a space or a non-ASCII name reaches the endpoint as it is spelled
        on disk rather than as a mangled path."""
        return f"{self.config.base_url}/{urllib.parse.quote(key.lstrip('/'), safe='/')}"

    # -- reading -----------------------------------------------------------

    def read(self, key: str, expected: int | None = None) -> bytes:
        """A small object in memory."""
        sink = io.BytesIO()
        self._copy(key, sink, expected)
        return sink.getvalue()

    def fetch(self, key: str, dest: str | Path, expected: int | None = None) -> int:
        """Stream one object to `dest`; return the bytes written.

        `expected` is the size the caller already knows (a listing, a stat).
        A source that disagrees — a truncated upload, the wrong object, a
        proxy serving a stale body — raises `SourceSizeError` rather than
        handing extraction a file that is not the document.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as sink:
            return self._copy(key, sink, expected)

    def _copy(self, key: str, sink: io.BufferedIOBase, expected: int | None) -> int:
        # The size is discovered from the source, not assumed from `expected`:
        # a body longer than the caller believed must not be read down to the
        # caller's number and then pass the size check. `expected` is a check
        # on what the source declares, and on the bytes actually written.
        if expected == 0:  # an empty document: no Range request is satisfiable
            return 0
        url = self.url_for(key)
        total: int | None = None
        pos = 0
        while total is None or pos < total:
            end = pos + self.config.chunk_bytes - 1
            if total is not None:
                end = min(end, total - 1)
            chunk = self._chunk(url, key, pos, end, sink)
            if chunk.wrote <= 0:
                if chunk.total == 0:  # a source that only now says "empty"
                    total = 0
                    break
                raise SourceError(f"{key}: the source returned no bytes at offset {pos}")
            pos += chunk.wrote
            if total is None:
                total = chunk.total
                if total is None and (chunk.span_end is None or chunk.span_end < end):
                    break  # a short tail and no declared length: that was the object
            elif chunk.total is not None and chunk.total != total:
                raise SourceSizeError(f"{key}: the source now reports {chunk.total} bytes, "
                                      f"not the {total} it reported a moment ago")
            if expected is not None and chunk.total is not None and chunk.total != expected:
                raise SourceSizeError(f"{key}: the source declares {chunk.total} bytes, "
                                      f"expected {expected}")
        size = sink.tell()
        if expected is not None and size != expected:
            raise SourceSizeError(f"{key}: read {size} bytes, expected {expected}")
        if total is not None and size != total:
            raise SourceSizeError(f"{key}: read {size} bytes, the source declared {total}")
        return size

    def _chunk(self, url: str, key: str, start: int, end: int, sink: io.BufferedIOBase) -> _Chunk:
        last: Exception | None = None
        for attempt in range(1, self.config.attempts + 1):
            try:
                return self._request(url, start, end, sink)
            except _Retryable as e:
                last = e
                # Discard the partial chunk before asking again: the retry
                # re-reads this range from its first byte, so a half-written
                # prelude would otherwise be duplicated.
                sink.seek(start)
                sink.truncate()
                if attempt < self.config.attempts:
                    self._sleep(min(self.config.backoff * 2 ** (attempt - 1), _MAX_BACKOFF))
        raise SourceError(f"{key}: bytes {start}-{end} unreadable after "
                          f"{self.config.attempts} attempt(s): {last}")

    def _request(self, url: str, start: int, end: int, sink: io.BufferedIOBase) -> _Chunk:
        headers = {"Range": f"bytes={start}-{end}"}
        if self.config.auth:
            headers[self.config.auth[0]] = self.config.auth[1]
        with self._slots:  # a slot per request, held until the body is read
            if self._closed:
                raise SourceError("the fetcher is closed")
            try:
                with self._client.stream("GET", url, headers=headers,
                                          timeout=self.config.timeout) as resp:
                    return self._receive(resp, start, end, sink)
            except httpx.TimeoutException as e:
                raise _Retryable(f"timeout: {e}") from e
            except httpx.TransportError as e:
                raise _Retryable(f"transport: {e}") from e

    def _receive(self, resp: httpx.Response, start: int, end: int,
                 sink: io.BufferedIOBase) -> _Chunk:
        status = resp.status_code
        # A zero-byte object has no satisfiable range, and endpoints answer
        # that differently: 200 with an empty body, 416, or a degenerate
        # `bytes 0--1/0`. Any of the three is simply a document of no bytes.
        if _declared_total(resp.headers.get("content-range", "")) == 0 \
                and resp.headers.get("content-length", "0").strip() in ("", "0"):
            return _Chunk(0, 0, None)
        if status in _RETRY_STATUS or 500 <= status < 600:
            raise _Retryable(f"HTTP {status}")
        if status == 416:
            raise SourceError(f"the source answered 416 for bytes {start}-{end} "
                              "(a shorter object than expected?)")
        if status not in (200, 206):
            raise SourceError(f"the source answered HTTP {status} for bytes {start}-{end}")
        total: int | None
        span_end: int | None = None
        if status == 200:
            # No Range support: the whole object arrived. Only a first chunk
            # can use it, and the length is the length.
            if start != 0:
                raise SourceError("the source ignored Range on a non-zero offset")
            total = _content_length(resp)
        else:
            span_start, span_end, total = _parse_content_range(resp.headers.get("content-range", ""),
                                                               start, end)
            if span_start != start:
                raise SourceError(f"asked for bytes {start}-{end}, the source sent "
                                  f"a range starting at {span_start}")
        sink.seek(start)
        wrote = 0
        for piece in resp.iter_raw():
            if piece:
                sink.write(piece)
                wrote += len(piece)
        return _Chunk(wrote, total, span_end)


def _content_length(resp: httpx.Response) -> int | None:
    raw = resp.headers.get("content-length", "").strip()
    return int(raw) if raw.isdigit() else None


def _parse_content_range(value: str, start: int, end: int) -> tuple[int, int | None, int | None]:
    m = _CONTENT_RANGE.match(value.strip())
    if not m:
        raise SourceError(f"the source sent no usable Content-Range for bytes {start}-{end}: {value!r}")
    span_start, span_end = int(m.group(1)), int(m.group(2))
    total = None if m.group(3) == "*" else int(m.group(3))
    return span_start, span_end, total


def _declared_total(value: str) -> int | None:
    """The object's total from any Content-Range form, `bytes a-b/N` or
    `bytes */N`, or None when it declares none (`/*`) or is unusable."""
    _span, sep, total = value.strip().rpartition("/")
    if not sep or not total.strip().isdigit():
        return None
    return int(total)


class Prefetcher:
    """Runs a bounded number of downloads ahead of the consumer.

    `materialize(keys)` yields `(key, path)` in the order it was given, while
    up to `window` downloads are in flight. Order in, order out: the index is
    built in the same sequence whichever download finishes first, so the store
    is the same bytes for the same documents whatever the network did.

    The window is the backpressure: at most `window` files exist that are
    downloaded and not yet consumed, so a fast source cannot outrun a slow
    extractor and fill the disk. Every file it hands out is unlinked as soon
    as the consumer moves on, and the whole work directory goes in `close()`,
    which `with` runs on an exception or a Ctrl+C as well as on the normal
    path.
    """

    def __init__(self, fetcher: Fetcher, workdir: str | Path | None = None,
                 window: int | None = None, expected: Mapping[Hashable, int] | None = None):
        self.fetcher = fetcher
        self.window = max(1, window if window is not None else fetcher.config.streams * 2)
        self.expected = expected
        self._parent = Path(workdir) if workdir is not None else None
        self._root: Path | None = None
        self._pool = ThreadPoolExecutor(max_workers=fetcher.config.streams,
                                        thread_name_prefix="silica-fetch")
        self._seq = itertools.count()
        self._closed = False

    def __enter__(self) -> Prefetcher:
        if self._parent is None:
            self._root = Path(tempfile.mkdtemp(prefix="silica-fetch-"))
        else:
            self._parent.mkdir(parents=True, exist_ok=True)
            self._root = Path(tempfile.mkdtemp(prefix="silica-fetch-", dir=self._parent))
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        """Stop the pool, drop every file, remove the work directory.

        The fetcher is closed first so a reader parked on a socket gives up
        now rather than at its timeout; the pool is then joined, so no worker
        can write into a directory this call is about to remove.
        """
        if self._closed:
            return
        self._closed = True
        self.fetcher.close()
        self._pool.shutdown(wait=True, cancel_futures=True)
        if self._root is not None:
            shutil.rmtree(self._root, ignore_errors=True)
            self._root = None

    def materialize(self, keys: Iterable[Hashable]) -> Iterator[tuple[Hashable, Path | None, Exception | None]]:
        """Yield `(key, path, error)` in input order, downloading ahead.

        A download the source could not complete is yielded as its exception
        and does not end the pipeline: one unreadable document is a document
        the pass records, not a reason to abandon the other N-1. `path` is
        None exactly when `error` is not.
        """
        if self._root is None:
            with self:
                yield from self.materialize(keys)
            return
        pending: deque[tuple[Hashable, Future, Path]] = deque()
        source = iter(keys)
        while True:
            while len(pending) < self.window:
                try:
                    key = next(source)
                except StopIteration:
                    break
                path = self._root / f"{next(self._seq):06d}-{_digest(key)}"
                pending.append((key, self._pool.submit(self._download, key, path), path))
            if not pending:
                return
            key, future, path = pending.popleft()
            error: Exception | None = None
            try:
                future.result()
            except Exception as exc:  # reported per document, never swallowed
                error = exc
            try:
                if error is not None:
                    yield key, None, error
                else:
                    yield key, path, None
            finally:
                path.unlink(missing_ok=True)

    def _download(self, key: Hashable, path: Path) -> None:
        if self._closed:
            return
        expected = self.expected.get(key) if self.expected is not None else None
        self.fetcher.fetch(str(key), path, expected)


def _digest(key: Hashable) -> str:
    return hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:16]


def prefetch(config: SourceConfig, workdir: str | Path | None = None,
             window: int | None = None, expected: Mapping[Hashable, int] | None = None
             ) -> Prefetcher:
    """A ready Prefetcher: `with prefetch(cfg, ...) as p: for key, path in p.materialize(keys)`."""
    return Prefetcher(Fetcher(config), workdir=workdir, window=window, expected=expected)