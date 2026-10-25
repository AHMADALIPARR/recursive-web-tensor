# RECURSIVE WEB TENSOR (RWT)

GitHub Actions as the **distributed control plane** for a recursively
pruned internet-token learning system — not one giant GPU job.

Every workflow execution performs one bounded cycle over persistent
state, then stops:

```
Discover -> Tokenize -> Score -> Prune -> Train -> Evaluate
    -> Commit checkpoint -> Repeat (next scheduled run)
```

Recursion happens **across epochs** (repeated runs over persistent
state), never inside an infinitely running job.

## The design decision

GitHub Actions **plus** the owner's own GPUs. GitHub-hosted runners
do crawling, parsing, tokenization, pruning, and testing. Actual GPU
training runs on a Jetson Orin NX or RTX 3080 box through a secured
self-hosted runner (see `docs/runners.md`). Schedule runs are
CPU-only; the GPU job runs only on manual dispatch with the GPU input
enabled. Without a registered self-hosted GPU runner, the GPU job
stays queued — GPU training is a slot, not a claim that it works
today.

## What exists now (honest scope)

- A **reference CLI** (`internet-token-engine`, Rust, **zero external
  crates**, so `cargo build --release --locked` works offline) that
  runs the full cycle end to end on a small corpus: ingest, prune,
  train, evaluate, checkpoint — all real, all tested.
- A **reference tokenizer**: deterministic word/byte-level tokens.
  This is **NOT BPE**. BPE training is a later milestone, not
  claimed.
- A **reference trainer**: an incremental n-gram language model
  (default order 3) merged into the prior checkpoint (adapter-style
  update, count-space replay). This is **statistical, not neural**.
  The neural fine-tune of a small existing model on GPU runners is
  the Phase-1 milestone and is **NOT in this repo yet** — but the
  train command's interface (inputs: retained shards; outputs:
  checkpoint manifest + metrics) is designed so a neural trainer
  replaces it.
- A **validation gate**: a checkpoint is promoted only if held-out
  loss does not regress beyond tolerance; otherwise the prior
  checkpoint stands and the epoch is recorded as rejected.

## What does NOT exist yet

- Internet-scale ingestion (see Phase 1 below).
- BPE tokenization, neural training, WARC record fetching.
- A registered self-hosted GPU runner (owner's hardware step).
- 100 million tokens processed. That number is a **target**, and the
  section below says exactly what exists toward it.

## Architecture

```
 GitHub Actions (control plane)
   schedule "17 */6 * * *" + manual dispatch
   process [matrix shards 0..3]  (ubuntu-latest, 45 min each)
        |  ingest --shard N, prune --shard N, upload shard manifest
   reduce                          (merge retained sets, Merkle root)
        |
   train  (CPU; re-ingests deterministically, merges n-gram counts)
        |
   train-gpu [self-hosted, gpu]   (manual dispatch + gpu input only;
        |  same CLI train command; artifact only, never checkpoints)
   evaluate (held-out loss + probe; writes evaluation, not checkpoints)
        |
   publish (contents:write; checkpoint gate, upload artifact,
        commit state to the rwt-state branch)

 Persistent state (rwt-state branch, never main):
   state/current.json, state/validation.json,
   state/reduced.json, state/evaluation.json,
   checkpoints/<version>/manifest.json (+ model.json if promoted)
```

Rules that keep this sound:

- **Shard jobs never write the model.** Only the publish job
  writes checkpoints, after the gate.
- **Deterministic re-ingest.** Train/evaluate/publish jobs
  re-ingest the corpus instead of shipping shard blobs between
  jobs. Ingest is deterministic (SHA-sorted, content-addressed),
  so the re-ingested corpus matches the sharded one.
- **Shard-local pruning is an approximation.** Each shard prunes
  only its own docs, so a near-duplicate pair split across shards
  can both survive. Exact duplicates cannot (same SHA = same
  shard). Global pruning is a later milestone, not claimed.
- **Size discipline.** The `rwt-state` branch holds manifests,
  metrics, and small state only. Never commit large model files to
  main or to the state branch; model files travel as artifacts
  (checkpoints 90 days, shard outputs 7 days, reduced set and
  candidate models 30 days).

See `docs/architecture.md` for the full job/artifact flow.

## The cycle and the state model

```
data/shards/<sha256>.json   content-addressed documents
data/vocab.json             reference-tokenizer vocabulary
state/registry.json         ingest index (or registry-shard-N)
state/prune.json            latest prune dispositions
state/shards/shard-N.json   per-shard prune outputs (CI jobs)
state/reduced.json          merged retained set + Merkle root
state/model-next.json       candidate model (train output)
state/evaluation.json       gate metrics (evaluate output)
state/current.json          promoted checkpoint pointer
state/validation.json       loss trend driving adaptive tau
checkpoints/<version>/      manifest.json (+ model.json if promoted)
```

Shard assignment is deterministic: `shard = first_sha_byte mod
num_shards`, so shards are stable across runs.

## Scoring

```
S(d) = a*Q(d) + b*N(d) + g*I(d) - d*R(d)
```

Defaults: `a=0.3, b=0.3, g=0.2, d=0.2` (configurable in
`config.json`; see `config.example.json`).

- **Q** — quality from text properties: length band, printable
  ratio, alphabetic ratio, repeated-line penalty.
- **N** — novelty: `1 - max` word-shingle Jaccard against the other
  candidate documents.
- **I** — information density: unique-token ratio + normalized
  Shannon entropy of the token distribution.
- **R** — redundancy: that same max Jaccard.
- Exact SHA-256 duplicates are deduped at ingest, before scoring.

Full definitions and a worked example: `docs/scoring.md`.

## Pruning: adaptive thresholds, never fixed percentages

Repeated fixed 50% pruning would leave ~0.1% of the corpus after ten
iterations and destroy coverage. So the threshold `tau` adapts to
measured validation performance:

- Base `tau0 = 0.30` (configurable).
- Validation improved since the last promotion: `tau += 0.02`
  (tighten). Degraded: `tau -= 0.02` (loosen). Clamped to
  `[0.15, 0.50]`.
- Top-`k` documents per category (`k = 2` by default) are always
  kept as representative samples, even below `tau`.

Dispositions: **Keep** (`S >= tau` or top-k), **Compress**
(near-duplicate: keep the lowest-SHA canonical only, record the
alias), **Quarantine** (unknown license or invalid content — never
trained on), **Reject** (explicitly denied license, or below `tau`
and not top-k). Compress alias chains resolve in `reduce`, capped
by `--max-depth` (default 8).

License policy (`sources/license-policy.md`): an allowlist
(public-domain, CC0, CC-BY, CC-BY-SA, MIT, Apache-2.0, BSD) and a
denylist (proprietary, all-rights-reserved, no-license). Unknown
licenses quarantine; denied licenses reject. License detection uses
source metadata and manifests — it is not legal advice.

## Checkpoints and the gate

`checkpoints/<version>/manifest.json` carries version, parent
version, the SHA-256 of the parent manifest (hash chain), the
Merkle root over retained shard hashes, metrics, and timestamp.
Promotion requires `new_heldout_loss <= prev_heldout_loss +
gate_tolerance` (default `0.10` nats). On rejection the manifest is
still written (status `rejected`), `current.json` is untouched, and
the prior checkpoint stands. A corrupted chain pointer (stored
parent hash not matching the parent manifest file) refuses to
promote at all.

## Safeguards (non-negotiable)

- **Fetched content is untrusted data, never executed.** The CLI
  fetches with `curl` (HTTP GET only, timeout, byte cap) and treats
  every byte as text to tokenize. No fetched script, instruction,
  or embedded command is ever run, interpreted, or passed to a
  shell.
- **Source permissions and training licenses are enforced at ingest
  and again at prune** (allowlist/denylist above). A document that
  fails either gate never reaches training.
- **No secrets in this repo.** Workflows use `GITHUB_TOKEN` only.
  Runner registration tokens are entered on the hardware, never
  committed (see `docs/runners.md`).
- **No license file in this repo.** The license is TBD by the
  owner; the CLI's own `Cargo.toml` carries a placeholder
  `LicenseRef-Owner-TBD` and nothing here grants rights.

## Scaling limits (stated, not hidden)

- GitHub documents a **6-hour limit per GitHub-hosted job** and a
  **256-job matrix maximum**. This workflow uses 45-minute shard
  jobs and a 4-job matrix — far inside both.
- Standard public-repository Actions usage is generally free
  within the applicable rules, but that is **not an unlimited
  training-compute entitlement**. The bounded-cycle design exists
  because of this limit, not in spite of it.
- The reference trainer is O(corpus) per epoch with O(n^2)
  shingle comparisons inside prune (shard-local). That is fine at
  reference scale and a known wall at internet scale; the
  production path needs shingle indexing (e.g. MinHash/LSH), which
  is a later milestone, not a hidden assumption.

## Phase 1 target: 100 MILLION TOKENS

**Target, not a claim.** What exists toward it today: a small-corpus
end-to-end reference run (8 fixture documents, 605 tokens retained,
real checkpoint — see below). What the milestone requires:

1. Ingest permitted public documents with provenance recorded.
2. Produce deterministic, content-addressed token shards.
3. Prune duplicates and low-quality content recursively.
4. Fine-tune a small existing model on the retained samples (the
   neural trainer that plugs into the `train-gpu` slot).
5. Measure held-out loss, retrieval quality, and knowledge
   retention against a baseline.

## Run it locally

```sh
cargo build --release --locked
./target/release/internet-token-engine --root /tmp/rwt-demo \
  cycle --corpus fixtures/corpus --probe-dir fixtures/probe --steps 0
```

The fixture run produces a real `checkpoints/v1/manifest.json`
with a verifiable Merkle root. Verify it independently:

```sh
sh scripts/e2e.sh            # runs the cycle + Python-hashlib checks
cargo test --release --locked
```

Subcommands: `ingest`, `prune`, `train`, `evaluate`, `reduce`,
`checkpoint`, `cycle`. Run any of them with no arguments for usage.
`--root` may appear before the command or among its flags.

## Sources

- `sources/duckduckgo.md` — discovery via the html-lite endpoint.
- `sources/common-crawl.md` — index lookup (implemented) and WARC
  fetching (later milestone).
- `sources/github-corpus.md` — GitHub corpus via URL manifests and
  local clones (code search API needs auth; not faked).
- `sources/license-policy.md` — allowlist/denylist and enforcement
  points.

## Layout

```
cli (Rust, std-only): src/{main,engine,tokenizer,scoring,sha256,json,fetch}.rs
.github/workflows/rwt-cycle.yml   Actions control plane
docs/                              architecture, scoring, runners
sources/                           discovery families + license policy
fixtures/corpus, fixtures/probe    reference corpora (tiny, real)
config.example.json                every tunable, with defaults
scripts/e2e.sh                     cycle + independent verification
```
