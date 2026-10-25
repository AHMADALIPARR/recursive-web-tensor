# License policy

Enforced at ingest (recorded per document) and again at prune
(gated before training). A document that fails either gate never
reaches training.

## Lists (defaults; override in `config.json`)

Allowlist (eligible for training):
`public-domain, CC0, CC-BY, CC-BY-SA, MIT, Apache-2.0, BSD,
BSD-2-Clause, BSD-3-Clause`

Denylist (rejected outright):
`proprietary, all-rights-reserved, no-license`

Anything else — including `unknown` — is **quarantined**: kept in
the registry, recorded in prune dispositions, never trained on,
awaiting a human license determination.

## What "license detection" means here

The license value comes from source metadata: the URL manifest
entry, the corpus operator's annotation, or the source's own
metadata. The CLI does not read a webpage and decide its license,
does not infer licenses from content, and does not give legal
advice. An incorrect manifest entry is an operator error the gate
cannot catch; the quarantine default for unknown values exists
precisely because of that.

## Disposition summary

| License value | Disposition |
|---|---|
| on allowlist | eligible (then scored) |
| on denylist | Reject |
| anything else / unknown | Quarantine |
| invalid content (empty, < 50% printable) | Quarantine |

Changing the lists is a config change (`config.json`), reviewed
like any other policy change — not a code edit, and never a
silent one: the lists in effect are visible in the repo.
