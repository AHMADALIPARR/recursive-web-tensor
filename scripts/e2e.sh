#!/bin/sh
# End-to-end reference run + independent verification (Python hashlib,
# not the Rust SHA-256 under test). Exits non-zero on any mismatch.
# Usage: scripts/e2e.sh [path-to-binary]
set -eu
BIN="${1:-./target/release/internet-token-engine}"
ROOT="$(pwd)"
WORK="$(mktemp -d /tmp/rwt-e2e-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
echo "work dir: $WORK"

# Default path: neural trainer (Phase 3). The verify block below
# pins structure and finiteness, not a loss value: on 8 fixture
# docs the neural model overfits (see docs/neural.md), so its
# held-out loss is worse than the n-gram's — the n-gram pins live
# in the explicit n-gram section at the end.
"$BIN" --root "$WORK" cycle --corpus "$ROOT/fixtures/corpus" --probe-dir "$ROOT/fixtures/probe" --steps 0

MANIFEST="$WORK/checkpoints/v1/manifest.json"
[ -f "$MANIFEST" ] || { echo "FAIL: no v1 manifest"; exit 1; }
[ -f "$WORK/checkpoints/v1/model.json" ] || { echo "FAIL: no v1 model"; exit 1; }
[ -f "$WORK/state/current.json" ] || { echo "FAIL: no current.json"; exit 1; }

python3 - "$WORK" <<'EOF'
import hashlib, json, sys, math
work = sys.argv[1]
man = json.load(open(f"{work}/checkpoints/v1/manifest.json"))
cur = json.load(open(f"{work}/state/current.json"))
red = json.load(open(f"{work}/state/reduced.json"))
ev = json.load(open(f"{work}/state/evaluation.json"))

assert man["status"] == "promoted", man["status"]
assert man["version"] == "v1", man["version"]
assert man["parent"] is None, "first checkpoint must have no parent"
assert cur["version"] == "v1"
# Chain pointer: current.json manifest sha must equal sha256 of the manifest file.
raw = open(f"{work}/checkpoints/v1/manifest.json", "rb").read()
assert cur["manifest_sha256"] == hashlib.sha256(raw).hexdigest(), "current manifest sha mismatch"
# Merkle root recomputed independently (pairwise SHA-256, sorted leaves, odd promoted).
leaves = sorted(r["sha256"] for r in red["retained"])
assert leaves, "no retained docs"
level = [bytes.fromhex(h) for h in leaves]
while len(level) > 1:
    nxt = []
    i = 0
    while i < len(level):
        if i + 1 < len(level):
            nxt.append(hashlib.sha256(level[i] + level[i + 1]).digest())
            i += 2
        else:
            nxt.append(level[i])
            i += 1
    level = nxt
assert level[0].hex() == red["merkle_root"] == man["merkle_root"], "merkle mismatch"
# Held-out metrics must be finite numbers (a real evaluation, not a placeholder).
assert isinstance(ev["heldout_loss"], (int, float)) and math.isfinite(ev["heldout_loss"])
assert isinstance(ev["perplexity"], (int, float)) and ev["perplexity"] > 1.0
assert man["retained_count"] == len(leaves)
print("e2e verify OK:")
print(f"  retained={len(leaves)} merkle_root={red['merkle_root']}")
print(f"  heldout_loss={ev['heldout_loss']:.4f} perplexity={ev['perplexity']:.2f} probe_loss={ev['probe_loss']}")
print(f"  manifest_sha256={cur['manifest_sha256']}")
EOF
echo "--- v1 manifest ---"
cat "$MANIFEST"

echo "=== BPE path (Phase 2) ==="
WORK2="$(mktemp -d /tmp/rwt-e2e-bpe-XXXXXX)"
"$BIN" --root "$WORK2" tokenizer train-bpe --corpus "$ROOT/fixtures/corpus" --merges 200
"$BIN" --root "$WORK2" cycle --corpus "$ROOT/fixtures/corpus" --probe-dir "$ROOT/fixtures/probe" --steps 0 --tokenizer bpe
python3 - "$WORK2" <<'EOF'
import json, sys
work = sys.argv[1]
man = json.load(open(f"{work}/checkpoints/v1/manifest.json"))
assert man["status"] == "promoted" and man["version"] == "v1", man
reg = json.load(open(f"{work}/state/registry.json"))
assert reg["tokenizer"] == "bpe", reg.get("tokenizer")
vocab = json.load(open(f"{work}/data/vocab.json"))["tokens"]
assert len(vocab) == 456, f"expected BPE vocab 456, got {len(vocab)}"
# Every retained doc's hex tokens must decode back to its stored text.
import glob, os
checked = 0
for path in glob.glob(f"{work}/data/shards/*.json"):
    doc = json.load(open(path))
    raw = b"".join(bytes.fromhex(t) for t in doc["tokens"])
    assert raw.decode("utf-8") == doc["text"], f"round-trip failed for {doc['sha256']}"
    checked += 1
assert checked > 0, "no shards checked"
print(f"BPE verify OK: {checked} docs round-trip, vocab 456, v1 promoted")
EOF
rm -rf "$WORK2"

echo "=== n-gram path (Phase 3: trainer still available) ==="
WORK3="$(mktemp -d /tmp/rwt-e2e-ngram-XXXXXX)"
"$BIN" --root "$WORK3" cycle --corpus "$ROOT/fixtures/corpus" --probe-dir "$ROOT/fixtures/probe" --steps 0 --trainer ngram
python3 - "$WORK3" <<'EOF'
import json, sys
work = sys.argv[1]
man = json.load(open(f"{work}/checkpoints/v1/manifest.json"))
assert man["status"] == "promoted" and man["version"] == "v1", man
ev = json.load(open(f"{work}/state/evaluation.json"))
# Pinned regression values for the deterministic n-gram path on fixtures.
assert abs(ev["heldout_loss"] - 4.9755) < 1e-3, ev["heldout_loss"]
assert abs(ev["perplexity"] - 144.82) < 0.1, ev["perplexity"]
assert abs(ev["probe_loss"] - 5.0683) < 1e-3, ev["probe_loss"]
model = json.load(open(f"{work}/checkpoints/v1/model.json"))
assert "counts" in model and model.get("kind") != "neural", "not an n-gram model"
print(f"n-gram verify OK: heldout_loss={ev['heldout_loss']:.4f} perplexity={ev['perplexity']:.2f} probe_loss={ev['probe_loss']:.4f}")
EOF
rm -rf "$WORK3"
