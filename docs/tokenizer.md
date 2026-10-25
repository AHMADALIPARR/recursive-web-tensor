# Tokenizers

Two tokenizers, one interface. Ingest records which tokenizer produced
a run in `state/registry.json` (`"tokenizer": "reference"|"bpe"`).

## Reference tokenizer (default)

`src/tokenizer.rs`. Lowercased alphanumeric runs plus one token per
non-alphanumeric, non-whitespace character. Fixed, deterministic,
independent of training order. Whitespace is dropped. This is the
fallback path and the original v1 behavior.

## BPE tokenizer (Phase 2, real)

`src/bpe.rs`. Byte-level BPE, implemented from scratch in std-only
Rust:

- **Base vocabulary:** the 256 single bytes.
- **Learning:** count adjacent symbol pairs over the corpus byte
  streams; repeatedly merge the most frequent pair, up to
  `--merges` (default 200). Ties break toward the
  lexicographically smallest pair, so training is fully
  deterministic for a given corpus and merge count. Learning stops
  early when no adjacent pair remains.
- **Encoding:** apply merges in learned rank order — at each step
  the lowest-rank adjacent pair present is merged — until no
  learned pair remains adjacent.
- **Decoding:** concatenate symbol bytes. For text that came from
  valid UTF-8, `decode(encode(text)) == text` exactly.
- **Document form:** tokens are stored as hex strings of the symbol
  bytes (e.g. `74686520` for `the `). Hex contains no spaces or
  separators, so every downstream string-based stage — shingle
  Jaccard, scoring, n-gram counts — works unchanged and
  unambiguously.
- **Persistence:** `state/bpe.json` is
  `{"merges": [[hexA, hexB], ...], "vocab": [hex, ...]}` where
  `vocab[0..256]` are the single bytes and `vocab[256+i]` is the
  i-th merge's concatenation. Load cross-checks vocab against the
  derived vocab from merges and refuses a mismatch rather than
  silently reinterpreting it.

### Commands

```
# Train on a corpus of .txt files (files read in sorted order)
ite --root <root> tokenizer train-bpe --corpus fixtures/corpus --merges 200
# -> state/bpe.json (default), prints doc count, merge count, vocab size

# Inspect
ite --root <root> tokenizer encode --merges state/bpe.json --text "The river flows"
ite --root <root> tokenizer decode --merges state/bpe.json --tokens "54686520 726976657220 ..."
```

### Selecting BPE for ingest / cycle

```
ite --root <root> cycle --corpus ... --tokenizer bpe
ite --root <root> cycle --corpus ... --tokenizer bpe --bpe-merges state/bpe.json
```

The flag wins; otherwise `config.json` fields `tokenizer` and
`bpe_merges` (defaults `"reference"` and `"state/bpe.json"`) apply.
BPE fails closed when the merges file is missing: silently falling
back to the reference tokenizer would corrupt token-id
comparability across epochs. The vocab written to
`data/vocab.json` in BPE mode is the trained vocab in id order;
in reference mode it is first-seen order as before.

### What BPE changes downstream

Token counts differ from the reference tokenizer (BPE keeps
whitespace inside symbols and splits words into subwords), so
held-out loss and perplexity are not comparable across tokenizer
modes — only within a mode. The Merkle root is unchanged by the
tokenizer: it is computed over document content hashes, not tokens.

### Limits (honest)

- Merges are learned per corpus, not a universal vocabulary; a
  merges file trained on one corpus applied to unrelated text still
  encodes correctly (byte fallback is always available) but may
  tokenize coarsely.
- No pre-tokenization (no word splitting before BPE): merges can
  span spaces and punctuation, which is standard for byte-level
  BPE but means the learned vocabulary is corpus-shaped.
- No dropout, no sampling: encoding is deterministic.
- Encoding is O(n * merges) in the naive rank-scan loop; fine at
  reference scale, a known scaling cost before internet scale.
