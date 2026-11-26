# Atkar fork of silica-core

This is a fork of [`kiycoh/silica-core`](https://github.com/kiycoh/silica-core) (MIT), pinned at
**`46d981ab` / v0.13.0** — upstream `main` HEAD as of the fork date.

## Why this fork exists

Upstream discovers its corpus by walking the filesystem, and decides what changed by comparing
`st_mtime`. On the deployment this fork serves (a remote object store mounted read-only over a
network filesystem) both of those are wrong in ways that break search outright:

- the walk runs on the query path, so a search pays the full traversal;
- the mount reports a parent-directory mtime, so "has this file changed?" is always answered "no".

The fork addresses those two faults. Everything else stays upstream's.

## Branch model

| Branch | Meaning |
|---|---|
| `main` | **Pristine mirror of upstream.** Fast-forward only: `git fetch upstream && git checkout main && git merge --ff-only upstream/main`. |
| `atkar/main` | **Our integration branch** (the repository default). Everything we intend to run lives here or in a branch merged into it. |
| `feat/*`, `fix/*` | Work branches, off `atkar/main`, one concern each. |

An upstream sync means: update `main`, then merge or rebase `atkar/main` onto it on a branch, run the
gates below, and only then integrate.

## Planned changes

Every behavioural change is **additive and environment-gated, default off**. With the gates unset the
output must be byte-identical to upstream's own output for the same corpus — that is the acceptance
test, not an aspiration.

1. **Inventory from an explicit source.** Upstream already accepts an injected inventory
   (`build_index(..., live=...)`, `index_state(..., live=...)`); this adds a way to *load* one from a
   manifest so the query path never walks.
2. **Freshness from a content hash when one is supplied**, falling back to the existing mtime
   comparison when it is not.
3. **Optional bounded reader concurrency**, off by default, only if a measurement on the target mount
   shows it helps — a previous attempt at parallelism measured *slower* at N=8 and was reverted.
4. **A byte source over HTTP.** `SILICA_SOURCE_URL` names an endpoint that serves each document by
   key and honours `Range`; the index pass then reads every document through it, a bounded number of
   streams ahead of extraction, into a temporary local file. Unset, the pass reads the tree exactly as
   upstream does.

### The byte source (`SILICA_SOURCE_URL`)

A remote tree reaches the pass through a network mount, and on that mount the bytes are the cost:
extraction is milliseconds and the read is seconds. The same objects are usually also reachable
through a plain HTTP endpoint that answers `Range`, so the pass can pull them on a bounded pool of
streams and hand the extractor a local file — extraction itself is unchanged, and so is everything
downstream of it.

| Variable | Meaning |
|---|---|
| `SILICA_SOURCE_URL` | base URL of the endpoint; a document's key is appended, quoted per segment. Unset = filesystem, byte-for-byte the upstream path. |
| `SILICA_SOURCE_STREAMS` | concurrent streams, default `24`. Also the pool size. |
| `SILICA_SOURCE_CHUNK_MB` | bytes per Range request, default `4`. |
| `SILICA_SOURCE_TIMEOUT` | seconds per request, default `60`. |
| `SILICA_SOURCE_AUTH` | one optional request header, `Header-Name: value`; `SILICA_SOURCE_AUTH_VALUE` supplies the value alone when the name sits in the first. The token is never a file in this repository. |

Two invariants hold this together, and `tests/test_source.py` asserts both without a network (a
Range-capable `http.server` over a temporary directory stands in for an endpoint):

- **The same documents produce the same store.** The pipeline hands files over in the pass's own
  order, whatever order the downloads finish in, and each fetched copy takes the modification time
  the inventory reported so the extraction caches key on the same identity they do on disk. A pass
  over a tree and a pass over the same bytes served over HTTP write byte-identical `lexical.json` and
  the same `stamps.json`.
- **Failure is per document, cleanup is per pass.** A document the endpoint will not serve (a 5xx past
  the retry budget, a size that disagrees with the expected one) lands in `failed` like an unreadable
  file does, and the pass continues. Downloads run a bounded window ahead of extraction — at most
  `streams × 2` files exist that are downloaded and not yet extracted — and the whole work directory
  is removed when the pass leaves, by exception or by Ctrl+C as well as by finishing.

The provider lives in `silica_core/source.py` (`Fetcher`, `Prefetcher`); the pass only chooses a
source, at `core._source_documents`.

The five-tool reply contract (`silica_files`, `silica_search`, `silica_read`, `silica_code_pack`,
`silica_write_note`) and the token stream (`TOKENIZER_VERSION`) are frozen: a change to either is a
contract change, not a patch.

## Gates before anything merges into `atkar/main`

- `uv run pytest` green.
- Byte-identity check: same corpus in, same index bytes and same search replies out, with every new
  gate **off**.
- With the gates **on**, the failure modes this fork exists to remove are demonstrably gone: a search
  performs no traversal, and a replaced file is seen as changed.

## What does not belong in this repository

It is public. Credentials, bucket names, hostnames, internal paths and deployment topology stay out of
it — including commit messages and test fixtures. Anything that needs that context lives in the private
planning repository alongside the design and the audit.

## Upstream

Small, clean, generally-useful fixes discovered here go upstream as PRs, one concern each, rather than
accumulating in the fork.
