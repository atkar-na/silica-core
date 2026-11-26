# `SILICA_INVENTORY` — the manifest, frozen

One interface, three consumers: the reader in `core.py`, the listing tool, and whatever produces the file.
Change this document before changing an implementation.

## Why

`silica_search` reaches `_inventory()` (`core.py:846`) → `_walk()` (`core.py:341`), a full traversal of the
root per call. On a root behind a network filesystem that is ~163 s against a 60 s call cap. Freshness is an
**equality** comparison against stored stamps (`core.py:481`, `:547`), so the stamp may be a content digest
instead of an mtime — which is also the only way to detect a replaced file on a mount that reports a
parent-directory mtime. A caller that already knows what the corpus holds hands the answer in.

## File

`SILICA_INVENTORY` is an absolute path to a JSON object. Unset, unreadable, malformed, or failing any rule
below → the walk is used, and no error is raised: an inventory source is worth an optimisation, never a
failure.

```json
{
  "version": 1,
  "root": "/absolute/path/of/the/root",
  "generated_at": "2026-11-25T22:14:00-05:00",
  "count": 7391,
  "stamps": { "docs/lsm.md": 1041063.0 },
  "unextracted": ["docs/report.docx"],
  "sizes": { "docs/lsm.md": 41231 },
  "mtimes": { "docs/lsm.md": 1764111240.0 }
}
```

| Field | Required | Meaning |
|---|---|---|
| `version` | yes | `1`. Anything else → walk. |
| `root` | yes | The absolute root the manifest describes. Must equal `str(_root())` → else walk. |
| `count` | yes | `len(stamps)`. Must match → else walk (a truncated write). |
| `stamps` | yes | `rel → number`, one row per document the index reads. Values must be int or float, never bool or string. |
| `generated_at` | no | ISO 8601. Reported, never enforced — a stale inventory is surfaced, not rejected. |
| `unextracted` | no | Documents the index holds no text for (in `silica_files` terms: `unconverted`). Absent → `[]`. |
| `sizes` | no | `rel → bytes`. Present and complete over `stamps` → the listing tool answers without walking. |
| `mtimes` | no | `rel → number`. Needed only for a listing with `since=`. |

## Stamp

Opaque, compared for equality only. Two producers:

- **Content digest** — the first 12 hex digits of the digest as an integer (48 bits, exactly representable in
  a float; take more and the comparison silently loses digits). This is the one that gives real change
  detection.
- **mtime** — `stat().st_mtime`, for a caller with no digest. Change detection is exactly as good as the
  filesystem's mtime, i.e. no better than today.

Rows are sorted by path when written; the reader sorts too.

## The document set — the rule that must not drift

`stamps` and `unextracted` must be the same set `_walk()` + `_inventory()` produce, or the index set drifts
silently while reporting `ready`. A producer replicates, in this order:

1. Skip hidden names (leading `.`) and everything `_walk` does not enter: `NOISE_DIRS`, `.silicaignore` rules.
2. `suffix in _PROSE_SUFFIXES` → `stamps`.
3. `suffix == ".pdf"` **with** no `.md` beside it → `stamps`.
4. `_is_code(rel)` and `size <= _CODE_MAX_BYTES` → `stamps`.
5. `suffix in DOC_EXTS`, not in `IMG_EXTS`, not `.pdf`, **and** no `.md` beside it → `unextracted`.
6. A row that is none of these is neither (it appears in a listing as `excluded`, and nowhere in the
   manifest).

`_PROSE_SUFFIXES`, `_CODE_MAX_BYTES`, `_is_code`, `DOC_EXTS`, `IMG_EXTS` and `NOISE_DIRS` come from the
pinned source — import them, never re-type them.

## Consumer rules

- **`index_state`** reports which source answered: `inventory: {"source": "manifest"|"walk", "root":
  ..., "generated_at": ..., "count": ...}`. A deployment must be able to see that a manifest was used and how
  old it is; otherwise a stale manifest silently freezes the index while reporting `ready`.
- **Prune guard.** On a manifest pass, rows absent from the manifest are removed — that is the point — but if
  a single pass would remove more than `SILICA_INVENTORY_MAX_DROP` (a fraction, default `0.1`) of the stored
  rows, the pass aborts, the store is left untouched, and the caller sees an error. A manifest that covers a
  subset of the root must not be able to empty an index.
- **The listing tool** (`silica_files`) serves its rows from the manifest when `sizes` covers `stamps` and,
  when `since=` is given, when `mtimes` does too. Any read that a row needs — a `.md` head, a sidecar, a
  source hash — happens only for the rows of the returned page, never for the whole corpus. Paging and the
  reply shape are unchanged.
- **The caches** (`_section_cache`, `_sha_cache`, the PDF extract cache) key on the identity stamp, not on
  `stat().st_mtime`: on a `--no-modtime` mount that mtime is constant, so `(rel, constant)` never changes and
  a replaced file is served as stale text and stale page bounds for the life of the process.
