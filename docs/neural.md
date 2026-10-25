# Neural trainer (Phase 3, real)

`src/neural.rs`. A small neural language model trained from scratch
in std-only Rust, with hand-implemented backprop and Adam. There
are no pretrained weights anywhere in this repo, and nothing here
is a fine-tune of a named model.

## Architecture (exact)

Feedforward NNLM — embedding + one tanh MLP hidden layer over a
context window + linear output + softmax cross-entropy:

- Context `CTX = 4` tokens; embedding dimension `D = 16`; hidden
  dimension `H = 32`. These are constants in `neural.rs`, not
  config — changing them is a code change, so a checkpoint's
  layout can never silently disagree with the trainer.
- Parameters, one flat `Vec<f64>` in fixed layout: `E` (vocab x D),
  `W1` ((CTX*D) x H), `b1` (H), `W2` (H x vocab), `b2` (vocab).
  The model JSON carries `kind: "neural"`, the vocab list, dims,
  and the flat params, so `model_from_json` can verify the length
  against the layout and refuse a mismatch.
- Padding token `"<s>"` is vocab id 0 (prepended when absent).
- **NOT claimed:** no attention, no recurrence, no transformer,
  no GPU. Attention/RNN/GPU are later scale milestones.

## Optimizer and determinism

Full-batch Adam (`beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`),
default 100 epochs, default learning rate 0.01. Init is
Xavier-style uniform from a fixed-seed xorshift PRNG; doc order is
sorted SHA order; no sampling, no dropout. The same data and
flags produce bit-identical params (unit-tested).

## Continued training and replay (the adapter story)

- **Init-from-prior-checkpoint:** when a prior neural checkpoint
  exists, its `W1`/`b1` are copied wholesale and its `E` rows,
  `W2` columns, and `b2` entries are copied for tokens present
  in both vocabularies; new tokens get fresh random init. A prior
  n-gram checkpoint (or no prior) means fresh init.
- **Replay:** the prior checkpoint's training docs (its
  `doc_shas`), when still present and not held out, are included
  in the training set alongside new retained docs.
- **Held-out split unchanged:** the same `heldout(sha)` rule as
  the n-gram trainer; held-out docs are never trained on, by
  either trainer.
- **Same gating interface:** `evaluate` computes held-out
  cross-entropy and probe loss for neural models in the same
  output shape, and `checkpoint` applies the same promotion gate
  (new loss <= prior loss + tolerance). The gate is
  loss-based regardless of trainer kind, so switching trainers
  compares the new kind's loss against the prior kind's loss.
- **Cross-trainer hygiene:** the n-gram trainer only inherits a
  prior `total_tokens` denominator when the prior model carries
  n-gram counts; a neural prior's token count is not adopted
  (it counts doc tokens, not n-gram events, and would miscalibrate
  the unigram backoff). Verified: n-gram after a neural
  checkpoint reproduces fresh n-gram metrics exactly.

## Selecting the trainer

```
ite train --trainer neural|ngram [--epochs N] [--lr F]
ite cycle ... --trainer neural|ngram [--epochs N] [--lr F]
```

Flag wins; otherwise config keys `trainer` (default `"neural"`),
`neural_epochs` (default 100), `neural_lr` (default 0.01).
**Neural is the default deliberately:** it is the primary trainer
going forward. The n-gram trainer stays available via
`--trainer ngram`. An unknown trainer name is an error, never a
silent fallback.

## Probe evaluation fix (Phase 3, applies to both trainers)

Probe documents used to be tokenized with the reference tokenizer
unconditionally. They are now tokenized with the run's tokenizer
(registry field; BPE loads the merges file and fails closed
without it; reference when no registry exists). Held-out docs were
already consistent because they come from ingest shards.

## Honest numbers at reference scale (measured, fixtures/corpus)

The fixture corpus is 8 docs, 6 training docs, ~455 training
tokens, against 18,642 neural parameters. The neural model
memorizes: 100 epochs take train loss 5.83 -> 0.04, while
held-out loss goes to 9.28 (perplexity ~10,724). Fewer epochs
stay near init (5 epochs: held-out 5.78; 10: 5.78; 20: 6.29;
40: 7.42). The n-gram trainer with backoff smoothing gets
held-out 4.98 / perplexity 144.82 on the same split.

Read that plainly: **at fixture scale the n-gram trainer
generalizes better than the neural trainer, and the neural
default is NOT claimed to beat it there.** What Phase 3 proves is
that neural training is real (gradient check: finite-difference
vs analytic, max error < 1e-4; training measurably reduces
held-out loss on a learnable pattern in tests; full pipeline
works; the gate correctly rejects a worse second epoch), and
that the neural trainer needs more data than 455 tokens before
it earns its keep. That is a data-scale statement, not a
model-quality claim, and it is exactly what the later phases
(more ingestion, WARC, global pruning) are for.

## Limits

- Full-batch Adam is O(epochs x examples x params); fine at
  reference scale, a known wall before internet scale
  (minibatching is a later milestone).
- No early stopping, no weight decay, no label smoothing, no
  dropout: overfitting control at this scale is the epoch count
  and the promotion gate.
- CPU only. The GPU runner slot runs this same CPU trainer until
  GPU acceleration lands (see `docs/runners.md`).
