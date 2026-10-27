#!/usr/bin/env python3
"""Rebuild the pictures in docs/img/ from the repo's own data.

Run from the repo root:  python3 docs/tools/make_figures.py

Needs: matplotlib, graphviz (the `dot` program), cargo, git.

Data used:
  - rwt-state branch: state/reduced.json, state/registry.json,
    state/model-next.json, checkpoints/v1 and v2 manifests
  - a fresh fixture run of the engine (cycle over fixtures/corpus)
    for the per-document scores in state/prune.json
"""
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1f2933"
GRID = "#d9dee3"
BLUE = "#2f6fb3"
ORANGE = "#d9822b"
GREEN = "#3b8f5a"
RED = "#c0392b"
GREY = "#8a949e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
})


def sh(*cmd, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def state_file(path):
    for ref in ("origin/rwt-state", "rwt-state"):
        try:
            return json.loads(sh("git", "show", f"{ref}:{path}"))
        except subprocess.CalledProcessError:
            continue
    raise SystemExit(f"cannot read {path} from the rwt-state branch")


# ---------- 1. architecture diagram (graphviz) ----------
DOT = r"""
digraph rwt {
  rankdir=TB; nodesep=0.35; ranksep=0.42; bgcolor="white";
  graph [fontname="DejaVu Sans", fontsize=12, labeljust=l, pad=0.3];
  node  [fontname="DejaVu Sans", fontsize=11, shape=box, style="rounded,filled",
         fillcolor="#eef3f8", color="#2f6fb3", penwidth=1.3, margin="0.18,0.09"];
  edge  [fontname="DejaVu Sans", fontsize=9, color="#4a5560", arrowsize=0.8];

  trigger [label="Trigger\nschedule: every 6 hours (17 */6 * * *)\nor manual dispatch",
           fillcolor="#fff4e0", color="#d9822b"];

  subgraph cluster_a {
    label="GitHub Actions: one cycle per run"; style="rounded,dashed"; color="#8a949e";
    process [label="process x4 (matrix shards 0-3)\ningest + prune each shard"];
    reduce  [label="reduce\nmerge shards, Merkle root"];
    pglobal [label="prune-global\nMinHash / LSH near-duplicates"];
    train   [label="train\nneural trainer, 100 steps"];
    gpu     [label="train-gpu\nself-hosted runner, manual only\nartifact only", style="rounded,dashed,filled", fillcolor="white", color="#8a949e"];
    evaluate [label="evaluate\nheld-out loss + probe loss"];
    publish [label="publish\nthe only job that writes checkpoints"];
    gate    [label="gate\nheld-out loss <= previous + 0.10 ?", shape=diamond, style="filled",
             fillcolor="#fff4e0", color="#d9822b", margin="0.05,0.05"];
    process -> reduce -> pglobal -> train -> evaluate -> publish -> gate;
    pglobal -> gpu [style=dashed, color="#8a949e"];
  }

  promoted [label="promoted\ncurrent.json moves to the new version", fillcolor="#e6f4ea", color="#3b8f5a"];
  rejected [label="rejected\nmanifest kept, previous checkpoint stands", fillcolor="#fbe9e7", color="#c0392b"];
  branch [label="rwt-state branch\nstate/  checkpoints/", shape=cylinder, fillcolor="#f3f0fa", color="#6b4fa3"];
  art    [label="Actions artifacts\nmodels, shard outputs, evaluation", shape=folder, fillcolor="#f3f0fa", color="#6b4fa3"];

  trigger -> process;
  gate -> promoted [label=" yes"];
  gate -> rejected [label=" no"];
  promoted -> branch; rejected -> branch;
  publish -> art [style=dotted];
  branch -> trigger [label=" next run reads back the\n loss trend and prior model", style=dashed, constraint=false, color="#6b4fa3"];
}
"""
(OUT / "architecture.dot").write_text(DOT)
sh("dot", "-Tsvg", str(OUT / "architecture.dot"), "-o", str(OUT / "architecture.svg"))
sh("dot", "-Tpng", "-Gdpi=150", str(OUT / "architecture.dot"), "-o", str(OUT / "architecture.png"))
(OUT / "architecture.dot").unlink()

# ---------- real data ----------
reduced = state_file("state/reduced.json")
registry = state_file("state/registry.json")
model = state_file("state/model-next.json")
v1 = state_file("checkpoints/v1/manifest.json")
v2 = state_file("checkpoints/v2/manifest.json")

names = {d["sha256"]: d["source"].replace("local:", "").replace(".txt", "") for d in registry["documents"]}
trained = set(model["doc_shas"])

# ---------- 2. tokens per document ----------
docs = sorted(reduced["retained"], key=lambda d: d["token_count"])
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
labels = [names[d["sha256"]] for d in docs]
vals = [d["token_count"] for d in docs]
cols = [BLUE if d["sha256"] in trained else ORANGE for d in docs]
bars = ax.barh(labels, vals, color=cols, height=0.62)
for b, d in zip(bars, docs):
    ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
            f'{d["token_count"]}   shard {d["shards"][0]}', va="center", fontsize=9.5)
ax.set_xlim(0, max(vals) * 1.28)
ax.set_xlabel("tokens")
tr = sum(d["token_count"] for d in docs if d["sha256"] in trained)
ho = sum(d["token_count"] for d in docs if d["sha256"] not in trained)
ax.set_title(f"Tokens kept per document ({reduced['total_tokens']} total, all {len(docs)} kept)",
             loc="left", fontsize=12.5, fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label=f"trained on ({tr} tokens)"),
                   Patch(color=ORANGE, label=f"held out ({ho} tokens)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "tokens-per-doc.png")
plt.close(fig)

# ---------- 3. loss per checkpoint ----------
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
xs = [v1["version"], v2["version"]]
held = [v1["metrics"]["heldout_loss"], v2["metrics"]["heldout_loss"]]
probe = [v1["metrics"]["probe_loss"], v2["metrics"]["probe_loss"]]
gate = v1["metrics"]["heldout_loss"] + 0.10
ax.axhline(gate, color=GREY, ls="--", lw=1.4)
ax.text(1.06, gate + 0.15, f"gate limit {gate:.2f} (v1 loss + 0.10)", ha="right", color=GREY, fontsize=9.5)
ax.plot(xs, held, marker="o", color=BLUE, lw=2.2, ms=8, label="held-out loss")
ax.plot(xs, probe, marker="s", color=ORANGE, lw=2.2, ms=8, label="probe loss")
for x, h, st in zip(xs, held, (v1["status"], v2["status"])):
    ax.annotate(f"{h:.2f}\n{st}", (x, h), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=10, color=GREEN if st == "promoted" else RED, fontweight="bold")
for x, p in zip(xs, probe):
    ax.annotate(f"{p:.2f}", (x, p), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=10)
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(5.5, 14)
ax.set_ylabel("loss (nats per token)")
ax.set_title("Checkpoints on the rwt-state branch", loc="left", fontsize=12.5, fontweight="bold")
ax.legend(loc="lower right", frameon=False)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "checkpoint-loss.png")
plt.close(fig)

# ---------- 4. prune scores from a fresh fixture run ----------
with tempfile.TemporaryDirectory() as tmp:
    sh("cargo", "build", "--release", "--locked")
    sh("./target/release/internet-token-engine", "--root", tmp, "cycle",
       "--corpus", "fixtures/corpus", "--probe-dir", "fixtures/probe", "--steps", "0")
    prune = json.loads((Path(tmp) / "state" / "prune.json").read_text())
tau = prune["tau"]
dis = sorted(prune["dispositions"], key=lambda d: d["score"])
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
labels = [names.get(d["sha256"], d["sha256"][:8]) for d in dis]
ax.barh(labels, [d["score"] for d in dis], color=GREEN, height=0.62)
ax.axvline(tau, color=RED, ls="--", lw=1.6)
ax.text(tau + 0.01, -0.85, f"keep threshold {tau:.2f}", color=RED, fontsize=9.5)
for i, d in enumerate(dis):
    ax.text(d["score"] + 0.01, i, f'{d["score"]:.2f}', va="center", fontsize=9.5)
ax.set_xlim(0, 1)
ax.set_ylim(-1.1, len(dis) - 0.4)
ax.set_xlabel("score S(d)")
ax.set_title(f"Prune scores, fixture run (Keep={prune['counts']['Keep']}, Reject={prune['counts']['Reject']})",
             loc="left", fontsize=12.5, fontweight="bold")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "prune-scores.png")
plt.close(fig)
print("done")
