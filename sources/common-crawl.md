# Common Crawl ingestion

## Index lookup (implemented)

```
GET https://index.commoncrawl.org/<INDEX>?url=<pattern>&output=json
```

Example index names look like `CC-MAIN-2025-30-index`; the current
index list is at `https://index.commoncrawl.org/collinfo.json`.
The CLI fetches one index URL with `curl` (30s timeout, 1MB cap),
parses the JSON-lines records (`url`, `filename`, `offset`,
`length`, `status`), and writes `state/cc-manifest.json`.
Unparseable lines are counted and skipped, never fatal.

## WARC record fetching (implemented, Phase 4)

A record's bytes live in a WARC file at
`https://data.commoncrawl.org/<filename>` at byte `offset` for
`length` bytes (a gzip member). The fetcher is real:

```
ite ingest --cc-warc-manifest state/cc-manifest.json
```

Range request per record (`curl -r`, short body = error),
gzip-member decompression via system `gzip`, byte-level WARC
parsing (`src/warc.rs`, strict Content-Length), reference text
extraction (tag stripping, small entity set — NOT a full HTML
parser), ingest with provenance
`cc:<filename>@<offset>:<page URL>`, category `common-crawl`,
license from the manifest item or `unknown` (quarantined at prune
as usual). Full detail, caps, and limits in `docs/warc.md`.

One gap is stated, not hidden: **robots.txt is NOT checked per
record** by this fetcher. Operators running at scale must add
robots handling. The index lookup's manifest is the input this
path consumes; the two steps stay separate commands.

## Terms

Common Crawl data is provided for research; per-record licensing
still depends on the source page. Every record fetched later must
pass the same license gate as every other document
(`sources/license-policy.md`) before training.
