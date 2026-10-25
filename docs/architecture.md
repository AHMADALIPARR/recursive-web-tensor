# Architecture

## Job graph (one epoch)

```
process[0..3]  (matrix, ubuntu-latest, 45 min, contents:read)
  ingest --corpus fixtures/corpus --shard N --num-shards 4
  prune  --shard N --num-shards 4 --max-depth 8
  upload artifact shard-N (state/shards/shard-N.json, 7 days)
        |
reduce  (needs process, 30 min, contents:read)
  download shard-* (merge-multiple into state/shards/)
  reduce --max-depth 8
  upload artifact reduced (state/reduced.json, 30 days)
        |
prune-global (needs reduce, 30 min, contents:read)
  re-ingest full corpus (deterministic; provides the docs compared)
  download reduced -> state/reduced.json
  prune-global --num-shards 4
  upload artifact reduced-global (state/reduced.json, 30 days)
        |
train   (needs prune-global, 45 min, contents:read)
  restore state/ + checkpoints/ from rwt-state branch
  re-ingest full corpus (deterministic)
  download reduced-global -> state/reduced.json
  train --steps 100
  upload artifact model-next (state/model-next.json, 30 days)
        |
train-gpu (needs prune-global, [self-hosted, gpu], 360 min, contents:read)
  ONLY on workflow_dispatch with gpu input true. Schedule runs
  never start it. Runs the same CLI train command on the GPU
  runner slot and uploads model-next-gpu as an artifact ONLY.
  It never writes checkpoints and is not on the publish path.
        |
evaluate (needs train, 30 min, contents:read)
  re-ingest full corpus
  download reduced-global + model-next
  evaluate --probe-dir fixtures/probe
  upload artifact evaluation (state/evaluation.json, 90 days)
        |
publish (needs evaluate, 30 min, contents:write)
  restore state/ + checkpoints/ from rwt-state branch
  re-ingest full corpus
  download reduced-global + model-next + evaluation
  checkpoint            <- the ONLY checkpoint write in the system
  upload artifact checkpoint (checkpoints/, 90 days)
  commit state/ + checkpoints/ to the rwt-state branch
```

If any `process` job fails, `reduce` is skipped: a partial shard
set never promotes. If the gate rejects, `publish` still commits
the rejected manifest (audit trail) but `current.json` stands.

## Why restore-then-download in train/publish

`main` carries no state. The `rwt-state` branch is the only place
with `current.json`, `validation.json`, and prior checkpoint
manifests/models. So:

- `process` restores only `state/validation.json` (the adaptive
  threshold input). Without it, tau falls back to tau0 — safe, just
  less adaptive.
- `train` restores `state/` + `checkpoints/` so the prior model is
  available for count merging (replay). Then the downloaded
  `reduced.json` overwrites the branch's stale reduced set.
- `publish` restores `state/` + `checkpoints/` so the parent
  manifest exists for chain verification and `current.json` points
  at the true prior checkpoint. Then downloads overwrite
  `reduced.json`, `model-next.json`, `evaluation.json` with this
  epoch's artifacts. Checkpoint then verifies the parent hash
  before writing anything.

The commit step copies the new `state/` + `checkpoints/` aside,
checks out the `rwt-state` branch, copies them back, and pushes.
That ordering matters: checking out the branch first would
overwrite the fresh checkpoint output with the branch's old state.

## Why deterministic re-ingest instead of shipping blobs

Shard jobs could upload their `data/shards/` blobs, but that ships
corpus bytes through artifacts on every epoch and couples jobs to
artifact size limits. Ingest is deterministic (documents keyed by
SHA-256 of content, processed in SHA order, shard by first hash
byte), so a later job re-ingesting the same corpus + manifest
reproduces the same shard files. The tradeoff is repeated ingest
work; the benefit is that no job ever depends on another job's
filesystem, and the only cross-job payloads are small manifests.

## State branch discipline

- `rwt-state` holds manifests, metrics, and small state only.
- Model files in the reference implementation are small JSON count
  tables and MAY live in `checkpoints/` on the state branch.
- Neural model files (the Phase-1 milestone) must NOT be committed
  to main or to `rwt-state`: they travel as artifacts only. The
  publish job's `git add state checkpoints` would otherwise sweep
  a multi-gigabyte file into git history. When the neural trainer
  lands, the publish step must exclude model binaries explicitly.
- First run: the `rwt-state` branch does not exist. Every restore
  step tolerates that (`|| true`) and the publish step creates the
  branch. The first checkpoint is `v1` with no parent.

## Failure modes

- **No held-out docs**: evaluate writes null held-out loss and the
  gate rejects the epoch (a gate that cannot verify must not
  promote). Fix by growing the corpus, not by bypassing the gate.
- **Chain break**: stored parent hash not matching the parent
  manifest file -> checkpoint errors before writing. Fix by
  inspecting `rwt-state`, never by force-pushing state.
- **GPU runner absent**: `train-gpu` stays queued on manual
  dispatch. It is not on the publish path, so the CPU chain still
  promotes. Never mark it `required` for publishing until the
  neural trainer exists.
- **Partial shards**: `reduce` requires all `process` jobs, so a
  failed shard blocks the epoch rather than silently training on
  three quarters of the corpus.
