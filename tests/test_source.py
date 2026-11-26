# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""The HTTP byte source: Range reads, retries, backpressure and cleanup, and
the index pass producing the same store whichever way the bytes arrive.

Nothing here opens a socket beyond loopback: a Range-capable `http.server` in
a thread serves a temporary directory, so the module runs end to end without an
endpoint, a credential or a network.
"""
from __future__ import annotations

import contextlib
import hashlib
import http.server
import threading
import time
import urllib.parse
from types import SimpleNamespace

import orjson
import pytest

from silica_core import source
from tests.doc_factory import pdf_bytes

CHUNK = 32 * 1024


class _Plan:
    """What the server does while a test runs: which keys fail, how slowly,
    how many requests overlap, and how many requests each key received."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.status: dict[str, int] = {}      # key -> requests still answered 503
        self.requests: dict[str, int] = {}    # key -> requests seen
        self.delays: dict[str, float] = {}    # key -> seconds before the answer
        self.chunked = False
        self.ignore_range = False
        self.empty_416 = False


@contextlib.contextmanager
def endpoint(directory, plan: _Plan | None = None):
    """A loopback HTTP server over `directory` that honours Range."""
    plan = plan or _Plan()
    directory = directory.resolve()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):  # the suite's output stays readable
            pass

        def do_GET(self):
            key = urllib.parse.unquote(self.path.split("?", 1)[0].lstrip("/"))
            with plan.lock:
                plan.requests[key] = plan.requests.get(key, 0) + 1
                fail = plan.status.get(key, 0)
                if fail:
                    plan.status[key] = fail - 1
                plan.active += 1
                plan.peak = max(plan.peak, plan.active)
            try:
                delay = plan.delays.get(key, 0.0)
                if delay:
                    time.sleep(delay)
                if fail:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                path = (directory / key).resolve()
                if not path.is_relative_to(directory) or not path.is_file():
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = path.read_bytes()
                if not data:  # a real endpoint answers an empty object, not a range
                    self.send_response(416 if plan.empty_416 else 200)
                    if plan.empty_416:
                        self.send_header("Content-Range", "bytes */0")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start, end, status = 0, len(data) - 1, 200
                header = "" if plan.ignore_range else self.headers.get("Range", "")
                if header.startswith("bytes="):
                    first, _, last = header[6:].partition("-")
                    start = int(first)
                    end = min(int(last) if last else len(data) - 1, len(data) - 1)
                    status = 206
                body = data[start:end + 1]
                self.send_response(status)
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                    self.send_header("Accept-Ranges", "bytes")
                if plan.chunked and status == 206:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    middle = max(1, len(body) // 2)
                    for piece in (body[:middle], body[middle:]):
                        if piece:
                            self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            finally:
                with plan.lock:
                    plan.active -= 1

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", plan
    finally:
        server.shutdown()
        server.server_close()


def payload(size: int, seed: bytes = b"silica") -> bytes:
    """Deterministic bytes: no RNG state, so a failure is reproducible."""
    out = bytearray()
    digest = hashlib.sha256(seed).digest()
    while len(out) < size:
        digest = hashlib.sha256(digest).digest()
        out += digest
    return bytes(out[:size])


def _config(base: str, **kw) -> source.SourceConfig:
    return source.SourceConfig(base_url=base, streams=kw.pop("streams", 2),
                               chunk_bytes=kw.pop("chunk_bytes", CHUNK),
                               timeout=kw.pop("timeout", 10.0),
                               attempts=kw.pop("attempts", 1),
                               backoff=kw.pop("backoff", 0.0), **kw)


# ---------------------------------------------------------------------------
# the fetcher
# ---------------------------------------------------------------------------

def test_a_multi_chunk_download_reproduces_the_object(tmp_path):
    data = payload(300_000)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "big.bin").write_bytes(data)
    with endpoint(tmp_path) as (base, plan):
        with source.Fetcher(_config(base)) as fetcher:
            dest = tmp_path / "copy.bin"
            assert fetcher.fetch("docs/big.bin", dest, expected=len(data)) == len(data)
            assert dest.read_bytes() == data
            assert plan.requests["docs/big.bin"] > 1, "one chunk would not exercise the loop"
            assert fetcher.read("docs/big.bin") == data


def test_a_chunked_transfer_encoding_is_followed(tmp_path):
    data = payload(200_000)
    (tmp_path / "big.bin").write_bytes(data)
    with endpoint(tmp_path) as (base, plan):
        plan.chunked = True
        with source.Fetcher(_config(base)) as fetcher:
            dest = tmp_path / "copy.bin"
            assert fetcher.fetch("big.bin", dest, expected=len(data)) == len(data)
            assert dest.read_bytes() == data


def test_a_size_that_disagrees_with_the_expected_one_is_refused(tmp_path):
    (tmp_path / "big.bin").write_bytes(payload(100_000))
    with endpoint(tmp_path) as (base, _plan):
        with source.Fetcher(_config(base)) as fetcher:
            with pytest.raises(source.SourceSizeError):
                fetcher.fetch("big.bin", tmp_path / "copy.bin", expected=99_999)
            with pytest.raises(source.SourceSizeError):  # the source declares it first
                fetcher.read("big.bin", expected=100_001)


def test_a_5xx_is_retried_and_then_succeeds(tmp_path):
    data = payload(20_000)  # one chunk, so the count is the retry count
    (tmp_path / "big.bin").write_bytes(data)
    with endpoint(tmp_path) as (base, plan):
        plan.status["big.bin"] = 2
        with source.Fetcher(_config(base, attempts=4)) as fetcher:
            assert fetcher.fetch("big.bin", tmp_path / "copy.bin") == len(data)
            assert (tmp_path / "copy.bin").read_bytes() == data
        assert plan.requests["big.bin"] == 3


def test_retries_give_up_and_the_backoff_is_bounded(tmp_path):
    (tmp_path / "big.bin").write_bytes(payload(1000))
    sleeps: list[float] = []
    with endpoint(tmp_path) as (base, plan):
        plan.status["big.bin"] = 99
        with source.Fetcher(_config(base, attempts=4, backoff=5.0)) as fetcher:
            fetcher._sleep = sleeps.append  # type: ignore[attr-defined]
            with pytest.raises(source.SourceError, match="unreadable"):
                fetcher.fetch("big.bin", tmp_path / "copy.bin")
        assert plan.requests["big.bin"] == 4
    assert sleeps == [5.0, 8.0, 8.0], "exponential, then capped"


def test_a_404_is_not_retried(tmp_path):
    with endpoint(tmp_path) as (base, plan):
        with source.Fetcher(_config(base, attempts=4)) as fetcher:
            with pytest.raises(source.SourceError, match="HTTP 404"):
                fetcher.fetch("missing.bin", tmp_path / "copy.bin")
        assert plan.requests["missing.bin"] == 1


@pytest.mark.parametrize("unsatisfiable", [False, True])
def test_a_zero_byte_object_is_an_empty_document(tmp_path, unsatisfiable):
    (tmp_path / "empty.md").write_bytes(b"")
    with endpoint(tmp_path) as (base, plan):
        plan.empty_416 = unsatisfiable
        with source.Fetcher(_config(base)) as fetcher:
            assert fetcher.read("empty.md") == b""
            assert fetcher.fetch("empty.md", tmp_path / "copy.md", expected=0) == 0
            assert (tmp_path / "copy.md").read_bytes() == b""


def test_a_server_that_ignores_range_is_read_whole(tmp_path):
    data = payload(70_000)
    (tmp_path / "big.bin").write_bytes(data)
    with endpoint(tmp_path) as (base, plan):
        plan.ignore_range = True
        with source.Fetcher(_config(base)) as fetcher:
            assert fetcher.read("big.bin") == data


# ---------------------------------------------------------------------------
# the prefetch pipeline
# ---------------------------------------------------------------------------

def test_the_stream_cap_is_honoured(tmp_path):
    names = [f"f{i}.bin" for i in range(12)]
    for name in names:
        (tmp_path / name).write_bytes(payload(64 * 1024, name.encode()))
    with endpoint(tmp_path) as (base, plan):
        plan.delays.update({name: 0.05 for name in names})
        with source.prefetch(_config(base, streams=3), workdir=tmp_path / "work") as ahead:
            assert [key for key, _path, _err in ahead.materialize(names)] == names
        assert plan.peak <= 3, f"{plan.peak} requests answered at once, cap was 3"
        assert plan.peak >= 2, "the server never saw concurrency, so the cap was not the limit"


def test_materialize_yields_in_order_whatever_the_downloads_do(tmp_path):
    names = [f"f{i}.bin" for i in range(8)]
    for i, name in enumerate(names):
        (tmp_path / name).write_bytes(payload(10_000, name.encode()))
    with endpoint(tmp_path) as (base, plan):
        # Later documents arrive first: the last file is instant, the first slow.
        plan.delays.update({names[0]: 0.20, names[1]: 0.10})
        with source.prefetch(_config(base, streams=4), workdir=tmp_path / "work") as ahead:
            got = [key for key, _path, _err in ahead.materialize(names)]
        assert got == names


def test_the_window_bounds_files_waiting_for_extraction(tmp_path):
    names = [f"f{i}.bin" for i in range(10)]
    for name in names:
        (tmp_path / name).write_bytes(payload(1000, name.encode()))
    with endpoint(tmp_path) as (base, _plan):
        work = tmp_path / "work"
        with source.prefetch(_config(base, streams=2), workdir=work, window=4) as ahead:
            root = next(work.glob("silica-fetch-*"))
            seen_at_most = 0
            for key, path, error in ahead.materialize(names):
                assert error is None
                assert path.read_bytes() == payload(1000, key.encode())
                seen_at_most = max(seen_at_most, len(list(root.iterdir())))
        assert seen_at_most <= 4, f"{seen_at_most} downloaded files waiting, window was 4"


def test_the_work_directory_is_removed_after_an_exception(tmp_path):
    names = [f"f{i}.bin" for i in range(6)]
    for name in names:
        (tmp_path / name).write_bytes(payload(1000, name.encode()))
    work = tmp_path / "work"
    with endpoint(tmp_path) as (base, _plan):
        with pytest.raises(RuntimeError, match="consumer died"):
            with source.prefetch(_config(base, streams=2), workdir=work) as ahead:
                for _key, path, _error in ahead.materialize(names):
                    assert path.is_file()
                    raise RuntimeError("consumer died")
    assert not list(work.glob("silica-fetch-*"))
    assert list(work.iterdir()) == []


def test_the_work_directory_is_removed_on_interrupt(tmp_path):
    names = [f"f{i}.bin" for i in range(4)]
    for name in names:
        (tmp_path / name).write_bytes(payload(1000, name.encode()))
    work = tmp_path / "work"
    with endpoint(tmp_path) as (base, _plan):
        with pytest.raises(KeyboardInterrupt):
            with source.prefetch(_config(base, streams=2), workdir=work) as ahead:
                for _key, _path, _error in ahead.materialize(names):
                    raise KeyboardInterrupt
    assert not list(work.glob("silica-fetch-*"))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_the_variable_unset_means_no_source():
    assert source.SourceConfig.from_env({}) is None
    assert source.SourceConfig.from_env({"SILICA_SOURCE_URL": "   "}) is None


def test_the_defaults_and_the_single_auth_header():
    cfg = source.SourceConfig.from_env({"SILICA_SOURCE_URL": "http://localhost:8080/base/"})
    assert cfg is not None
    assert cfg.base_url == "http://localhost:8080/base"
    assert (cfg.streams, cfg.chunk_bytes, cfg.timeout) == (24, 4 * 1024 * 1024, 60.0)
    assert cfg.auth is None
    cfg = source.SourceConfig.from_env({"SILICA_SOURCE_URL": "https://localhost",
                                        "SILICA_SOURCE_STREAMS": "8",
                                        "SILICA_SOURCE_CHUNK_MB": "1",
                                        "SILICA_SOURCE_TIMEOUT": "5",
                                        "SILICA_SOURCE_AUTH": "Authorization: Bearer token-here"})
    assert cfg.auth == ("Authorization", "Bearer token-here")
    assert (cfg.streams, cfg.chunk_bytes, cfg.timeout) == (8, 1024 * 1024, 5.0)


@pytest.mark.parametrize("environ", [
    {"SILICA_SOURCE_URL": "ftp://localhost"},
    {"SILICA_SOURCE_URL": "http://localhost", "SILICA_SOURCE_STREAMS": "0"},
    {"SILICA_SOURCE_URL": "http://localhost", "SILICA_SOURCE_CHUNK_MB": "many"},
    {"SILICA_SOURCE_URL": "http://localhost", "SILICA_SOURCE_AUTH": "no-colon-here"},
    {"SILICA_SOURCE_URL": "http://localhost", "SILICA_SOURCE_AUTH": "X-Key: a\r\nY: b"},
])
def test_a_bad_variable_is_an_error_not_a_guess(environ):
    with pytest.raises(source.SourceError):
        source.SourceConfig.from_env(environ)


def test_the_url_quotes_the_key_a_segment_at_a_time(tmp_path):
    fetcher = source.Fetcher(_config("http://localhost:9/base"))
    assert fetcher.url_for("a b/c+d.pdf") == "http://localhost:9/base/a%20b/c%2Bd.pdf"
    fetcher.close()


# ---------------------------------------------------------------------------
# the index pass
# ---------------------------------------------------------------------------

@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A small tree with prose, a PDF and a code file, and the module under it."""
    from silica_core.config import CONFIG
    import silica_core.core as core
    import silica_core.kernel.recall.paths as paths

    root = tmp_path / "vault"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "a.md").write_text("# A\n\ncompaction merges the runs\n", encoding="utf-8")
    (root / "notes" / "b.md").write_text("# B\n\nzebra crossing on the plain\n", encoding="utf-8")
    (root / "papers").mkdir()
    (root / "papers" / "one.pdf").write_bytes(pdf_bytes(["# Results\n\ncompaction beats the baseline\n"]))
    monkeypatch.setattr(CONFIG, "vault_path", str(root))
    monkeypatch.delenv("SILICA_SOURCE_URL", raising=False)
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    holder = {"dir": tmp_path / "index-disk"}
    monkeypatch.setattr(paths, "index_dir_for", lambda _vault: holder["dir"])
    core._section_cache.clear()
    return SimpleNamespace(core=core, root=root, holder=holder, tmp=tmp_path)


def _build(vault, index_dir):
    import silica_core.kernel.recall.lexical as lex

    vault.core._section_cache.clear()
    lex._STORE_CACHE.clear()
    vault.holder["dir"] = index_dir
    out = vault.core.build_index(rebuild=True)
    store = (index_dir / "lexical.json").read_bytes()
    meta = orjson.loads((index_dir / "stamps.json").read_bytes())
    return out, store, meta


def test_the_store_is_the_same_from_disk_and_from_http(vault, monkeypatch):
    disk_out, disk_store, disk_meta = _build(vault, vault.tmp / "index-disk")
    assert disk_out["docs"] == 3 and not disk_out["failed"]

    with endpoint(vault.root) as (base, _plan):
        monkeypatch.setenv("SILICA_SOURCE_URL", base)
        http_out, http_store, http_meta = _build(vault, vault.tmp / "index-http")

    assert http_out["docs"] == disk_out["docs"]
    assert http_out["changed"] == disk_out["changed"]
    assert http_out["failed"] == disk_out["failed"]
    assert http_store == disk_store, "the store must not know where the bytes came from"
    assert http_meta["stamps"] == disk_meta["stamps"]
    assert http_meta["no_text"] == disk_meta["no_text"]
    assert http_meta["units"] == disk_meta["units"]


def test_the_store_does_not_depend_on_the_order_the_downloads_finish(vault, monkeypatch):
    _out, disk_store, _meta = _build(vault, vault.tmp / "index-disk")
    orders = [{"papers/one.pdf": 0.15}, {"notes/a.md": 0.15, "papers/one.pdf": 0.05}]
    stores = []
    for i, delays in enumerate(orders):
        with endpoint(vault.root) as (base, plan):
            plan.delays.update(delays)
            monkeypatch.setenv("SILICA_SOURCE_URL", base)
            _o, store, _m = _build(vault, vault.tmp / f"index-order-{i}")
        stores.append(store)
    assert stores == [disk_store, disk_store]


def test_the_filesystem_path_is_untouched_when_the_variable_is_unset(vault, monkeypatch):
    def never(*_args, **_kwargs):
        raise AssertionError("the prefetcher must not exist when SILICA_SOURCE_URL is unset")

    monkeypatch.setattr(source, "prefetch", never)
    monkeypatch.setattr(source, "Fetcher", never)
    out, _store, _meta = _build(vault, vault.tmp / "index-disk")
    assert out["docs"] == 3 and not out["failed"]
    assert not (vault.tmp / "index-disk" / "tmp").exists()


def test_a_download_failure_fails_the_document_and_the_pass_continues(vault, monkeypatch):
    with endpoint(vault.root) as (base, plan):
        plan.status["notes/a.md"] = 99
        monkeypatch.setenv("SILICA_SOURCE_URL", base)
        out, _store, _meta = _build(vault, vault.tmp / "index-http")
    assert set(out["failed"]) == {"notes/a.md"}
    assert out["docs"] == 2, "the other two documents were still indexed"
    assert out["index"]["state"] == "stale"