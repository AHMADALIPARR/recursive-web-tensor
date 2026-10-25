# Scoring

```
S(d) = alpha*Q(d) + beta*N(d) + gamma*I(d) - delta*R(d)
```

Defaults: `alpha=0.3, beta=0.3, gamma=0.2, delta=0.2`
(configurable in `config.json`; see `config.example.json`).
All four components lie in [0, 1], so `S(d)` lies in
`[-delta, alpha+beta+gamma]` = `[-0.2, 0.8]` at defaults.

## Q: quality (text properties only)

```
length    = 0                     if 0 tokens
          = (n/10)*0.5            if n < 10
          = 1.0                   if 10 <= n <= 2000
          = max(0.3, 2000/n)      if n > 2000
printable = fraction of chars that are not control characters
            (newline, tab, carriage return count as printable)
alpha     = fraction of chars that are alphabetic
line_rep  = fraction of lines that exactly duplicate an earlier line
Q         = 0.4*length + 0.3*printable + 0.2*alpha + 0.1*(1 - line_rep)
```

No model, no external signal. A 60-token clean paragraph scores
~0.96; boilerplate with repeated lines scores lower through
`line_rep`; an empty document scores exactly 0.

## N and R: novelty and redundancy (shingle Jaccard)

Tokens are shingled into windows of 5 consecutive tokens
(`shingle_n` in config). For a document `d`:

```
max_j(d) = max over other candidates c of Jaccard(shingles(d), shingles(c))
N(d)     = 1 - max_j(d)
R(d)     = max_j(d)
```

Keeping both terms explicit is deliberate: `beta` rewards novelty
while `delta` separately penalizes redundancy, so the two can be
tuned independently. Exact SHA-256 duplicates never reach scoring:
ingest dedupes them first.

## I: information density

```
unique_ratio = unique tokens / total tokens
H            = Shannon entropy of the token distribution (bits)
H_norm       = H / log2(total tokens)     (0 when total <= 1)
I            = 0.5*unique_ratio + 0.5*H_norm
```

A document repeating one token scores near 0; a document with
all-distinct tokens scores near 1.

## Worked example (real output, fixture run)

`fixtures/corpus/04-astronomy.txt` under default weights:

```
Q = 0.9665, N = 1.0000, I = 0.8659, R = 0.0000
S = 0.3*0.9665 + 0.3*1.0000 + 0.2*0.8659 - 0.2*0.0000
  = 0.28995 + 0.30000 + 0.17318 - 0
  = 0.7631
```

N = 1.0 because no 5-token shingle is shared with any other
fixture document (the fixtures are topically unrelated). On a real
web corpus, N < 1 is the common case, and that is the signal the
pruner uses.

## The adaptive threshold (why not fixed-percentage pruning)

Fixed-percentage pruning compounds: pruning 50% per epoch leaves
`0.5^10 = 0.0977%` after ten epochs — coverage destroyed even if
later epochs find nothing better. RWT instead adapts `tau` to the
measured validation trend (`state/validation.json`, written by
`checkpoint` on promotion):

```
no history:            tau = tau0                      (0.30)
validation improved:   tau = min(tau_max, tau0 + step)  (0.32)
validation degraded:   tau = max(tau_min, tau0 - step)  (0.28)
step = 0.02, range [0.15, 0.50] (all configurable)
```

Plus: top-`k` documents per category (`k = 2` by default) are
always kept, so a low-scoring category is sampled, never erased.

## Dispositions

| Disposition | Condition |
|---|---|
| Keep | `S >= tau`, or top-k in its category |
| Compress | max shingle Jaccard >= 0.9 vs an earlier (lower-SHA) canonical; the alias is recorded, only the canonical trains |
| Quarantine | license not on the allowlist (unknown), or invalid content (empty, < 50% printable); never trained on |
| Reject | license on the denylist, or `S < tau` and not top-k |

Compress alias chains resolve in `reduce` by following `alias_of`
links up to `--max-depth` (default 8); chains that exceed the depth
or point at a missing target are reported as unresolved, not
silently dropped.
