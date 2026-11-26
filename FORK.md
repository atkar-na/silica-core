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
