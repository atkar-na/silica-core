# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Alessandro Carosia

"""The consumers of `SILICA_INVENTORY` besides a search: the index's state
report, the listing tool, and the prune guard.

A manifest lets a deployment hand in what the corpus holds so a search does not
pay a whole traversal (the reason this exists). The same manifest answers the
listing tool and says which source the state came from. What these tests pin,
against the frozen interface in INVENTORY.md:

* `index_state()` carries an `inventory` block naming the source — `manifest` or
  `walk` — with the manifest's `generated_at`, so a stale or subset manifest is
  visible rather than silently freezing the index at `ready`.
* On a manifest pass, rows absent from the manifest are removed, but a pass that
  would remove more than `SILICA_INVENTORY_MAX_DROP` of the stored rows aborts
  before anything is written and reports the refusal: a manifest that covers a
  subset of the root cannot empty an index.
* `silica_files` serves its rows from the manifest when `sizes` covers `stamps`
  (and `mtimes` too, for `since=`), reading a `.md` head, a sidecar or a source
  hash only for the rows of the returned page.
* A manifest that is unset, for another root, short-counted or without the
  `mtimes` a `since=` needs falls back to the walk, never raising.
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
        "# LSM trees\n\n## 1 Write path\n\nWrites land in a memtable.\n\n"
        "## 2 Compaction\n\nCompaction merges sorted runs; leveled compaction bounds read amplification.\n",
        encoding="utf-8")
    (tmp_path / "docs" / "btree.md").write_text("# B-trees\n\nA B-tree keeps pages sorted.\n",
                                                encoding="utf-8")
    (tmp_path / "docs" / "ctree.md").write_text("# C-trees\n\nA C-tree bounds write amplification.\n",
                                                encoding="utf-8")
    (tmp_path / "docs" / "report.docx").write_bytes(b"PK\3\4not really a docx")
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=tmp_path.parent / (tmp_path.name + "-idx"): _d)
    import silica_core.core as core
    core._section_cache.clear()
    return core


@pytest.fixture
def plain(tmp_path, monkeypatch):
    """An empty root, for counting reads over a corpus built to size."""
    from silica_core.config import CONFIG
    from silica_core.kernel.recall import paths

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(paths, "index_dir_for", lambda vault, _d=tmp_path.parent / (tmp_path.name + "-idx"): _d)
    import silica_core.core as core
    core._section_cache.clear()
    return core


class _Walked(Exception):
    """Raised by the rigged `_walk`: seeing it proves the walk ran."""


def _boom(*_a, **_k):
    raise _Walked("the walk ran")


def _covering(root, **overrides):
    """A manifest describing the root as a real walk sees it plus the sizes and
    mtimes a listing needs, so a manifest-driven reply can be compared with the
    walk's own. `SILICA_INVENTORY` must be unset when this is called."""
    stamps, unextracted = root._inventory()
    r = root._root()
    data = {"version": 1, "root": str(r), "generated_at": "2026-11-25T22:14:00-05:00",
            "count": len(stamps), "stamps": stamps,
            "sizes": {rel: (r / rel).stat().st_size for rel in stamps},
            "mtimes": {rel: (r / rel).stat().st_mtime for rel in stamps},
            "unextracted": list(unextracted)}
    data.update(overrides)
    return data


def _write(tmp_path, data, name="inventory.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _by_path(reply):
    return {row["path"]: row for row in reply["files"]}


def _index_bytes(root):
    idx = root._paths.index_dir()
    return {str(p.relative_to(idx)): p.read_bytes() for p in sorted(idx.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------
# a. index_state names the source
# ---------------------------------------------------------------------------

def test_state_names_the_walk_when_no_manifest_is_set(root):
    root.build_index()
    st = root.index_state()
    assert st["state"] == "ready" and st["docs"] == 3 and st["pending"] == 0  # every old key kept
    assert st["inventory"] == {"source": "walk", "root": str(root._root()),
                               "generated_at": None, "count": 3}


def test_state_names_the_manifest_and_how_old_it_is(root, tmp_path, monkeypatch):
    root.build_index()
    data = _covering(root, generated_at="2026-11-25T22:14:00-05:00")
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, data)))
    st = root.index_state()
    assert st["state"] == "ready" and st["docs"] == 3  # every old key kept
    assert st["inventory"] == {"source": "manifest", "root": str(root._root()),
                               "generated_at": "2026-11-25T22:14:00-05:00", "count": 3}


def test_a_cold_index_still_names_its_inventory(root):
    st = root.index_state()
    assert st["state"] == "cold"
    assert st["inventory"]["source"] == "walk" and st["inventory"]["count"] is None


def test_a_manifest_for_another_root_or_a_short_count_falls_back_to_the_walk(root, tmp_path, monkeypatch):
    root.build_index()
    wrong_root = _covering(root)
    wrong_root["root"] = "/somewhere/else"
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, wrong_root, name="wrong_root.json")))
    assert root.index_state()["inventory"]["source"] == "walk"

    short = _covering(root)
    short["count"] = 99  # a truncated write
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, short, name="short.json")))
    assert root.index_state()["inventory"]["source"] == "walk"


# ---------------------------------------------------------------------------
# b. the prune guard
# ---------------------------------------------------------------------------

def test_prune_guard_refuses_a_pass_that_would_empty_the_index(root, tmp_path, monkeypatch):
    root.build_index()
    assert root.index_state()["docs"] == 3
    stamps = root._load_meta()["stamps"]
    subset = {"version": 1, "root": str(root._root()), "count": 1,
              "stamps": {"docs/btree.md": stamps["docs/btree.md"]}}  # 1 of 3 stored rows
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, subset)))
    before = _index_bytes(root)

    out = root.build_index()

    assert out["error"]["code"] == "inventory_prune_refused"
    assert out["removed"] == 2 and out["stored"] == 3
    assert _index_bytes(root) == before, "the store is left byte-identical"
    assert root.index_state()["docs"] == 3


def test_a_manifest_that_drops_nothing_proceeds_normally(root, tmp_path, monkeypatch):
    root.build_index()
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, _covering(root))))
    out = root.build_index()
    assert "error" not in out and out["changed"] == 0
    assert root.index_state()["docs"] == 3


def test_the_prune_threshold_is_honoured(root, tmp_path, monkeypatch):
    root.build_index()
    stamps = root._load_meta()["stamps"]
    subset = {"version": 1, "root": str(root._root()), "count": 1,
              "stamps": {"docs/btree.md": stamps["docs/btree.md"]}}

    # 2 of 3 = 0.67: over the default 0.1, but not over 0.7
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, subset)))
    monkeypatch.setenv("SILICA_INVENTORY_MAX_DROP", "0.7")
    assert "error" not in root.build_index()
    assert root.index_state()["docs"] == 1

    # back to the full corpus, then refuse at 0.5
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    monkeypatch.delenv("SILICA_INVENTORY_MAX_DROP", raising=False)
    root.build_index(rebuild=True)
    assert root.index_state()["docs"] == 3
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, subset, name="subset2.json")))
    monkeypatch.setenv("SILICA_INVENTORY_MAX_DROP", "0.5")
    assert root.build_index()["error"]["code"] == "inventory_prune_refused"
    assert root.index_state()["docs"] == 3


# ---------------------------------------------------------------------------
# c. the listing serves the manifest's rows
# ---------------------------------------------------------------------------

def test_the_listing_serves_the_manifest_rows_with_no_walk(root, tmp_path, monkeypatch):
    root.build_index()
    walked = root.silica_files()
    assert _by_path(walked)["docs/report.docx"]["status"] == "unconverted"
    assert "docs/report.docx" not in root._load_meta()["stamps"]

    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, _covering(root))))
    monkeypatch.setattr(root, "_walk", _boom)  # any traversal is now an error

    served = root.silica_files()
    assert sorted(served["files"], key=lambda r: r["path"]) == sorted(walked["files"], key=lambda r: r["path"])
    assert served["counts"] == walked["counts"]
    assert served["total"] == walked["total"] == 4
    assert served["excluded_dirs"] == walked["excluded_dirs"] == 0
    assert served["next_cursor"] is None and served["truncated"] is False


# ---------------------------------------------------------------------------
# d. since= needs mtimes or the walk
# ---------------------------------------------------------------------------

def test_since_without_mtimes_walks_instead_of_answering_undated(root, tmp_path, monkeypatch):
    root.build_index()
    data = _covering(root)
    del data["mtimes"]  # a manifest that cannot date a row
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, data)))
    monkeypatch.setattr(root, "_walk", _boom)

    with pytest.raises(_Walked):
        root.silica_files(since="2020-01-01")  # the documented fallback is the walk


def test_since_with_mtimes_is_answered_from_the_manifest(root, tmp_path, monkeypatch):
    root.build_index()
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, _covering(root))))
    monkeypatch.setattr(root, "_walk", _boom)

    reply = root.silica_files(since="2000-01-01")  # every document is newer
    assert {f["path"] for f in reply["files"]} == {"docs/lsm.md", "docs/btree.md", "docs/ctree.md"}
    assert all("modified" in f for f in reply["files"])


# ---------------------------------------------------------------------------
# e. reads happen only for the returned page
# ---------------------------------------------------------------------------

def _corpus(root, n):
    r = root._root()
    for i in range(n):
        (r / f"file{i:03d}.md").write_text(f"# Note {i}\n\ncontent {i}\n", encoding="utf-8")


def test_page_reads_do_not_scale_with_the_corpus(plain, tmp_path, monkeypatch):
    monkeypatch.delenv("SILICA_INVENTORY", raising=False)
    _corpus(plain, 5)
    small = _covering(plain)
    _corpus(plain, 60)  # the same root, a much larger corpus
    large = _covering(plain)
    assert len(small["stamps"]) == 5 and len(large["stamps"]) == 60

    calls = []
    real_head = plain._head

    def counting_head(full, limit=65536):
        calls.append(str(full))
        return real_head(full, limit)

    monkeypatch.setattr(plain, "_head", counting_head)
    for data, name in ((small, "small.json"), (large, "large.json")):
        monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, data, name=name)))
        monkeypatch.setattr(plain, "_walk", _boom)
        calls.clear()
        reply = plain.silica_files(limit=1)
        assert len(reply["files"]) == 1
        assert len(calls) == 1, "one returned row, one `.md` head, whatever the corpus size"
        assert reply["total"] == len(data["stamps"])


# ---------------------------------------------------------------------------
# f. unset: the walk, and the same rows
# ---------------------------------------------------------------------------

def test_unset_manifest_walks_and_the_rows_match_the_manifest(root, tmp_path, monkeypatch):
    root.build_index()
    walked = []
    real_walk = root._walk

    def counting(base=None):
        walked.append(base)
        yield from real_walk(base)

    monkeypatch.setattr(root, "_walk", counting)
    unset = root.silica_files()
    assert walked, "with no manifest the listing is the walk"
    assert set(unset) == {"root", "total", "files", "counts", "excluded_dirs",
                          "truncated", "next_cursor", "index"}

    data = _covering(root)
    monkeypatch.setenv("SILICA_INVENTORY", str(_write(tmp_path, data)))
    monkeypatch.setattr(root, "_walk", _boom)
    served = root.silica_files()
    assert sorted(served["files"], key=lambda r: r["path"]) == sorted(unset["files"], key=lambda r: r["path"])
    assert served["counts"] == unset["counts"] and served["total"] == unset["total"]
