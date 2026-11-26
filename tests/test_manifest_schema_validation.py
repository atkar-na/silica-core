# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""The `SILICA_INVENTORY` reader enforces its schema strictly.

A manifest is a claim about the corpus that a caller hands in, and the reader
either accepts it whole or falls back to the walk — there is no half-trusted
middle, and a refusal is never a failure. Three faults an auditor reproduced
against the reader are pinned here:

* a JSON bool is an `int` in Python, so `version: true` and `count: true` passed
  an `isinstance(x, int)` check and were accepted as `1`;
* a `../..` key names a file outside the root, so a manifest could point a read
  at anything on the filesystem;
* a digest stamp took the first 12 characters and ignored the rest, so
  `0123456789abXYZ` was accepted where a clean hex digest is required.

The producer (`inventory.build_manifest`) and the reader must still agree: a
manifest the former produces validates through the stricter latter unchanged.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def root(tmp_path, monkeypatch):
    from silica_core.config import CONFIG
    from silica_core.kernel.recall import paths

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "lsm.md").write_text(
        "# LSM trees\n\nWrites land in a memtable.\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=tmp_path / ".idx": _d)
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    import silica_core.core as core
    return core


def _write(tmp_path, data, name="inventory.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _manifest(root, stamps=None, **overrides):
    stamps = {"docs/lsm.md": 1.0} if stamps is None else stamps
    data = {"version": 1, "root": str(root._root()), "count": len(stamps),
            "stamps": stamps}
    data.update(overrides)
    return data


def _rejected(tmp_path, monkeypatch, root, data):
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, data)))
    assert root._manifest_inventory() is None, f"the reader accepted {data!r}"
    # A refusal is never fatal: the walk still answers, and `index_state` says why.
    assert "docs/lsm.md" in root._inventory()[0]
    state = root.index_state()["inventory"]
    assert state["source"] == "walk" and state.get("reason"), \
        "a refused manifest must be visible in the state report"


def _accepted(tmp_path, monkeypatch, root, data):
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, data)))
    parsed = root._manifest_inventory()
    assert parsed is not None, "a valid manifest must be accepted"
    return parsed


# --------------------------------------------------------------------------
# version — exactly the integer 1
# --------------------------------------------------------------------------

def test_version_must_be_the_integer_one(root, tmp_path, monkeypatch):
    for bad in (True, False, 0, 2, "1", 1.0):
        _rejected(tmp_path, monkeypatch, root, _manifest(root, version=bad))
    absent = _manifest(root)
    del absent["version"]
    _rejected(tmp_path, monkeypatch, root, absent)


# --------------------------------------------------------------------------
# count — a real int equal to len(stamps)
# --------------------------------------------------------------------------

def test_count_true_is_not_a_real_count(root, tmp_path, monkeypatch):
    # `True == 1`, so a bool slipped past `count != len(stamps)`.
    _rejected(tmp_path, monkeypatch, root, _manifest(root, count=True))


def test_a_mismatched_or_negative_or_float_count_is_rejected(root, tmp_path, monkeypatch):
    _rejected(tmp_path, monkeypatch, root, _manifest(root, count=99))    # a truncated write
    _rejected(tmp_path, monkeypatch, root, _manifest(root, count=-1))
    _rejected(tmp_path, monkeypatch, root, _manifest(root, count=1.0))   # a float, not an int


def test_a_correct_count_is_accepted(root, tmp_path, monkeypatch):
    parsed = _accepted(tmp_path, monkeypatch, root, _manifest(root))
    assert parsed["stamps"] == {"docs/lsm.md": 1.0}


# --------------------------------------------------------------------------
# keys — root-relative and unable to escape the root
# --------------------------------------------------------------------------

def test_an_escaping_or_empty_key_is_rejected(root, tmp_path, monkeypatch):
    for bad in ("../../outside.md", "/abs/path.md", "a/../../b.md", ""):
        _rejected(tmp_path, monkeypatch, root, _manifest(root, stamps={bad: 1.0}))
    # The rule holds for every map, not just `stamps`.
    _rejected(tmp_path, monkeypatch, root, _manifest(root, sizes={"../x": 1}))
    _rejected(tmp_path, monkeypatch, root, _manifest(root, mtimes={"../x": 1.0}))
    _rejected(tmp_path, monkeypatch, root, _manifest(root, unextracted=["../x"]))


def test_a_bool_stamp_or_size_is_rejected(root, tmp_path, monkeypatch):
    # `bool` is an `int`; the schema forbids it as a number.
    _rejected(tmp_path, monkeypatch, root, _manifest(root, stamps={"docs/lsm.md": True}))
    _rejected(tmp_path, monkeypatch, root, _manifest(root, sizes={"docs/lsm.md": True}))


# --------------------------------------------------------------------------
# a valid manifest is accepted, maps and all
# --------------------------------------------------------------------------

def test_a_valid_manifest_is_accepted(root, tmp_path, monkeypatch):
    parsed = _accepted(tmp_path, monkeypatch, root,
                       _manifest(root, stamps={"docs/lsm.md": 1.0},
                                 sizes={"docs/lsm.md": 41},
                                 mtimes={"docs/lsm.md": 1764111240.0},
                                 unextracted=["docs/report.docx"]))
    assert parsed["stamps"] == {"docs/lsm.md": 1.0}
    assert parsed["sizes"] == {"docs/lsm.md": 41.0}
    assert parsed["mtimes"] == {"docs/lsm.md": 1764111240.0}
    assert parsed["unextracted"] == ["docs/report.docx"]


# --------------------------------------------------------------------------
# digest_stamp — a clean 40-hex digest, or ValueError
# --------------------------------------------------------------------------

def test_digest_stamp_rejects_anything_but_a_clean_40_hex_digest():
    from silica_core.inventory import digest_stamp

    for bad in ("0123456789abXYZ", "abc", "", None, True, "Z" * 40):
        with pytest.raises(ValueError):
            digest_stamp(bad)


def test_digest_stamp_accepts_a_clean_40_hex_digest():
    from silica_core.inventory import digest_stamp

    clean = "0123456789abcdef" * 2 + "01234567"       # exactly 40 hex characters
    assert len(clean) == 40
    assert digest_stamp(clean) == int(clean[:12], 16)


# --------------------------------------------------------------------------
# producer → reader round trip
# --------------------------------------------------------------------------

def _corpus(tmp_path, monkeypatch):
    from silica_core.config import CONFIG
    from silica_core.kernel.recall import paths

    corpus = tmp_path / "corpus"
    (corpus / "docs").mkdir(parents=True)
    (corpus / "docs" / "lsm.md").write_text("# LSM\n\nA memtable.\n", encoding="utf-8")
    (corpus / "docs" / "report.docx").write_bytes(b"PK\x03\x04x")
    monkeypatch.setattr(CONFIG, "vault_path", str(corpus))
    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=tmp_path / ".idx": _d)
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    import silica_core.core as core
    rows = []
    for rel, full in core._walk():
        if full is None:
            continue
        st = full.stat()
        rows.append((rel, st.st_size, st.st_mtime))
    return core, corpus, rows


def test_the_producer_manifest_validates_through_the_strict_reader(tmp_path, monkeypatch):
    from silica_core.inventory import build_manifest, write_manifest

    core, corpus, rows = _corpus(tmp_path, monkeypatch)
    manifest = build_manifest(corpus, rows)
    out = tmp_path / "inventory.json"
    write_manifest(out, manifest)

    monkeypatch.setenv("SILICA_INVENTORY", str(out))
    parsed = core._manifest_inventory()
    assert parsed is not None, "the producer's own manifest must pass the strict reader"
    assert parsed["stamps"] == manifest["stamps"]
    assert sorted(parsed["unextracted"]) == sorted(manifest["unextracted"])


def test_a_digest_manifest_round_trips_through_the_reader(tmp_path, monkeypatch):
    import hashlib

    from silica_core.inventory import build_manifest, write_manifest

    core, corpus, rows = _corpus(tmp_path, monkeypatch)
    digest_rows = []
    for rel, size, _mtime in rows:
        digest = hashlib.sha1((corpus / rel).read_bytes()).hexdigest()
        digest_rows.append((rel, size, digest))
    manifest = build_manifest(corpus, digest_rows)
    out = tmp_path / "digest-inventory.json"
    write_manifest(out, manifest)

    monkeypatch.setenv("SILICA_INVENTORY", str(out))
    parsed = core._manifest_inventory()
    assert parsed is not None
    assert parsed["stamps"] == manifest["stamps"]
