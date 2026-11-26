# Robust index pass

Branch `feat/robust-index`, cut from the process-parallel pass and its
batching/timeout patch (both are off `atkar/main`): the crash isolation in
section 2 replaces that pass's pool handling, and the batch boundary in
section 4 is the `SILICA_INDEX_BATCH` generation that patch introduced.

A first index pass over a large corpus must survive one pathological document
and must not rewrite the whole store while it runs. This branch lands four
changes; each names the failure it prevents and the test that proves it.

## 1. The manifest is the authority for a pass, and a pass never walks

*Commit `fix(index): the manifest is the authority for a pass, and a pass never
walks`.*

**Before.** On a manifest-driven pass (`SILICA_INVENTORY`), `_guard_live` ran a
full walk and, whenever the walk's key set differed from the manifest's in any
way, returned the walk's map as the document map. The pass then stamped every
document it processed with that map's value. A mount that reports no mtimes
answers `0.0` for every file, which never equals a manifest's content digest, so
every document looked changed on the next pass: the pass re-queued the whole
corpus, settled a few documents, and did it again. The corpus never converged
and the store was rewritten continuously. Nothing in the pass was wrong except
the change key it wrote.

**After.** A manifest pass does not traverse at all. The manifest's integer
digest stamps are the change keys written into the store, and every row the
manifest names is in the pass whether or not a filesystem would list it — a
listing is the claim, and pruning a row because a mount cannot see it is the
empty-inventory defect in the other direction.

The protection the reconciliation existed for does not need a walk: a manifest
that would remove more of the *stored* rows than `SILICA_INVENTORY_MAX_DROP`
allows is a truncated or subset write, and the pass returns the existing
`inventory_prune_refused` error with the store byte-identical. The cross-walk
survives as an explicit opt-in, `SILICA_INVENTORY_VERIFY_WALK=1`, where it may
add a row the manifest omits and may never restamp one.

The walk path is untouched when `SILICA_INVENTORY` is unset.

**Proof.** `tests/test_manifest_authority.py`

* `test_a_manifest_pass_never_walks_and_stores_the_manifest_digests` — with a
  manifest set, `_walk` is rigged to raise; the build succeeds, every manifest
  row is indexed, and the stored stamps equal the manifest digests.
* `test_a_manifest_row_the_filesystem_does_not_list_is_still_in_the_pass` and
  `test_the_guard_keeps_a_manifest_row_absent_from_disk_and_from_the_walk` — a
  row the manifest names is kept with its digest, never pruned.
* `test_a_manifest_smaller_than_the_store_is_refused_and_the_store_is_untouched`
  — the store bytes are identical after the refusal.
* `test_a_second_pass_over_an_unchanged_manifest_writes_nothing` and
  `test_a_digest_stamp_does_not_drift_into_a_walk_mtime_on_the_next_pass` —
  `changed == 0`, `pending == 0`, zero store serialisations.
* `test_verification_is_the_only_way_a_manifest_pass_walks` — the opt-in walk
  adds a row the manifest omits and restamps nothing.

## 2. Per-document crash isolation, with a quarantine budget

*Commit `fix(index): isolate a crashing document ...`.*

**Before.** One worker dying raised `BrokenProcessPool` for the whole pool, and
every document in that generation that had not yet been applied was recorded as
failed with `worker process died before this document was indexed`. A single
malfunctioning PDF therefore failed a whole batch and the pass restarted the
same work next time.

**After.** `_run_generation` blames the earliest document it could not apply,
tears the pool down, and retries that document alone in a fresh interpreter
(`python -m silica_core._isolate`). The remainder of the generation is
re-submitted to a fresh pool; nothing else is failed for it. Only when the
isolated retry also dies or exceeds `SILICA_INDEX_TIMEOUT` is the document
quarantined, with a machine-readable reason in `failed`:

| reason | meaning |
| --- | --- |
| `crashed_pdfium` | the isolated retry exited non-zero or was killed |
| `quarantined` | the isolated retry outlived the timeout and was killed |
| `quarantine_budget_exhausted` | `SILICA_INDEX_MAX_QUARANTINE` (default 200, 0 = no cap) reached; no further retry is spent |

A quarantined document keeps no stamp, so the next pass re-runs it; it is never
retried again inside the same pass. A fresh *spawned* interpreter, not a fork:
PDFium and MuPDF are not thread-safe and `pypdfium2` initialises the library at
import, so a forked child inherits the state that may have crashed.

**Proof.** `tests/test_index_robustness.py` —
`test_a_dying_worker_is_retried_alone_and_the_pass_finishes` (the generation is
not failed, every document is indexed), `..._whose_retry_also_dies_is_quarantined`
(`crashed_pdfium`), `..._outlives_the_timeout_is_quarantined` (`quarantined`,
and the pass returns promptly), `test_the_quarantine_budget_stops_a_corpus_wide_crash`
(two retries for a budget of two, then `quarantine_budget_exhausted`, and the
healthy documents are still indexed). `tests/test_parallel_pass.py` carries the
same expectation through the public `build_index()` path.

## 3. Oversize / fruitless-document guard

**Before.** A very large drawing set cost minutes of extraction to yield a few
hundred characters, and a PDF whose page count is in the thousands filled the
index with nothing.

**After.** Before a PDF is read whole, its size is stat'ed. Past
`SILICA_INDEX_MAX_MB` (default 64, 0 disables) or past `SILICA_INDEX_MAX_PAGES`
(default 400, 0 disables) the read is bounded to the page cap. A sampled read
that yields fewer than 200 characters per 10 pages is recorded as
`oversize_low_text` (or `page_cap_low_text` when only the page cap triggered) in
the existing `no_text` map, with the characters and pages measured, so status
reporting shows it as unconverted rather than silently indexed. A sampled read
that does carry text is indexed from the pages it read, the pass reports it
under the new `truncated` key, and the partial page map is never written to the
extraction cache.

**Proof.** `tests/test_index_robustness.py` —
`test_an_oversize_drawing_set_is_skipped_with_its_measurement` (reason carries
`chars from 3 of 6 pages`, the document is in neither `stamps` nor the store and
`files()` reports it unconverted), `test_a_page_cap_alone_can_skip_a_fruitless_document`,
`test_a_sampled_read_with_text_is_indexed_and_reported_as_truncated`.

## 4. Write amplification

**Before.** The store was serialised and written when a five-second timer fired
during a pass, and once per applied document in the worst case. `save()`
serialises every row, so on a large index that is a full rewrite of hundreds of
megabytes every few seconds.

**After.** No timer. The store is written once at each `SILICA_INDEX_BATCH`
generation boundary, in both the sequential and the parallel pass, and only when
it holds something the file does not (`is_dirty()`), except for the first write
of an index that has never been saved.

**Proof.** `tests/test_index_robustness.py` —
`test_the_store_is_written_once_per_generation` counts `_serialize` calls for a
synthetic run and asserts `<= ceil(N/batch) + 1`;
`test_the_parallel_pass_writes_at_the_same_boundary` does the same with a pool;
`test_an_unchanged_pass_does_not_rewrite_the_store` asserts zero serialisations
for a pass with nothing to do.

## Untouched

`TOKENIZER_VERSION`, MCP tool names, `TOOLS.md` shapes, the `INVENTORY.md`
schema and `uv.lock`.

## Test status

`pytest -q` — all tests pass except the three known pre-existing upstream
failures: `test_mcp_surface.py::test_one_sigint_stops_the_server`,
`test_mcp_surface.py::test_search_and_read_are_marked_always_loaded_on_the_wire`,
`test_root_gate.py::test_over_the_wire_a_folder_without_vault_yaml_gets_not_initialised`.