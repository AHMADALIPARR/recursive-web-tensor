# Global pruning: MinHash + LSH (Phase 5, real)

`src/minhash.rs` (signatures, banding, decisions) +
`engine::cmd_prune_global` (the `prune-global` subcommand). This
removes the shard-local approximation: cross-shard near-dup pairs
no longer both survive.

## The problem

Shard-local prune compares each document only against canonicals
in its own shard. A near-dup pair split across shards survives
both shards' passes, and both copies reach training. Exact
global comparison of all retained pairs is O(n^2) — the cost this
step exists to avoid.

## The mechanism

1. **Shingle sets** — the same 5-shingle sets shard-local prune
   uses (`tokenizer::shingles`), over the run's tokens (reference
   words or BPE hex tokens, either works).
2. **MinHash signatures** — 128 permutations by default
   (config `minhash_permutations`). Each shingle is hashed with
   FNV-1a 64; permutation `i` applies universal hashing
   `h_i(x) = (a_i * x + b_i) mod 2^64` (wrapping arithmetic,
   `(a_i, b_i)` derived deterministically from splitmix64 seeded
   by `i`); the signature is the per-permutation minimum.
3. **LSH banding** — 16 bands x 8 rows by default (config
   `lsh_bands`, `lsh_rows`; bands * rows must equal the
   permutation count, else the command errors). Documents sharing
   at least one full band are candidate pairs.
4. **Exact verification** — every candidate pair's true shingle
   Jaccard is computed, and only pairs at or above the configured
   `near_dup_threshold` (default 0.9, same as shard-local prune)
   are demoted. No MinHash estimate ever demotes a document.
5. **Canonical rule** — identical to shard-local prune: retained
   docs are processed in SHA order, first-seen wins, so the
   canonical is the lowest SHA; a doc pairing with an
   already-demoted doc maps directly to that doc's canonical, so
   no alias chains are produced. Demoted docs become `Compress`
   with `alias_of` set, and their aliases merge into
   `reduced.json`'s `resolved_aliases` as well as the new
   `global_aliases` / `global_pruned` sections.

## What it writes

- `state/reduced.json` — rewritten in place: retained list
  without demoted docs, token total recomputed, Merkle root
  recomputed over the kept set, plus `global_aliases`,
  `global_pruned` (sha, alias_of, exact Jaccard, disposition),
  and `global_minhash` (parameters, candidate count, verified
  pairs, demoted counts split cross-shard / same-shard,
  unverified-missing count).
- `state/prune-global.json` — a copy of the same output, for
  audit without re-reading reduced state.

Retained docs missing from disk are kept retained and counted as
unverified, never silently dropped.

## The approximation, honestly

- **No false positives from MinHash.** Decisions use exact
  Jaccard; a pair below threshold is never demoted, however its
  signature looks.
- **Recall loss is the only error.** A true near-dup pair is
  missed when it shares no LSH band. For true Jaccard `s`, rows
  `r`, bands `b`, the miss probability is `(1 - s^r)^b`. With
  the defaults (r=8, b=16): s=0.9 -> ~0.00065; s=0.95 ->
  ~0.00002; s=0.8 -> ~0.055 (but 0.8 is below threshold anyway,
  so missing it is correct behavior, not an error).
- **Equivalence on fixtures.** `lsh_decisions` and the exact
  O(n^2) `exact_decisions` agree exactly on the fixture groups
  (unit-tested); the engine test proves the shard path end to
  end (see below).
- **Empty shingle sets never demote.** Empty-vs-empty Jaccard is
  1.0 in the tokenizer, so empty sets are excluded from banding
  and from decisions. (Empty-token docs are quarantined before
  prune anyway; this is defense in depth.)

## Commands and wiring

```
ite prune-global [--num-shards 4]
ite cycle ...   # runs reduce -> prune-global -> train automatically
```

Workflow: the `prune-global` job runs after `reduce` and before
`train` / `train-gpu`; it re-ingests the corpus (deterministic),
downloads `reduced`, runs `prune-global`, and uploads
`reduced-global`, which train, train-gpu, evaluate, and publish
then download. The local cycle runs the same step, where it is
normally a no-op: local prune already compared all pairs, so
nothing new is demoted (verified: the standard e2e cycle's
retained count is unchanged by the added step).

## Proof (measured)

- Engine test `prune_global_finds_cross_shard_pairs`: 10
  near-dup family docs (Jaccard ~0.98) spread across all four
  shards + 1 distinct doc. Shard prune + reduce retain 5 (4
  family canonicals + solo); prune-global demotes exactly 3,
  all cross-shard, all to the lowest shard-canonical SHA, all at
  exact Jaccard 0.976; retained becomes 2; Merkle root recomputed
  and verified against independent recomputation.
- e2e script Phase 5 repeats the same scenario through the CLI
  (shard loop + reduce + prune-global) with the same assertions.
- Unit tests: signature determinism, MinHash estimate tracks
  true Jaccard (0.9-set estimate within 0.12), identical sets
  share bands / disjoint sets do not, LSH == exact on fixture
  groups, empty sets never candidates.

## Limits

- Same-shard demotions are reported separately
  (`same_shard_demoted`); in a healthy run that count is 0 — a
  nonzero value means shard-local prune missed something (config
  drift between prune and prune-global, or data changed between
  the two steps) and is worth investigating, not ignoring.
- The cross/same split uses the `--num-shards` value passed to
  the command (workflow: 4). It is reporting only; demotion
  itself does not depend on it.
- MinHash signatures are computed over retained docs only, not
  the full corpus: a doc rejected by shard-local prune for
  scoring (below tau) is not resurrected or re-examined here.
  Global prune only demotes; it never promotes.
- LSH parameters are fixed per run (config); there is no
  adaptive band tuning. If the corpus grows, the candidate
  volume — not the correctness — is what to watch.
