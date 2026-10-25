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

## WARC record fetching (NOT implemented — later milestone)

A record's bytes live in a WARC file at
`https://data.commoncrawl.org/<filename>` at byte `offset` for
`length` bytes (a gzip member). Fetching, decompressing, and
parsing WARC records is real work (range requests, gzip member
handling, WARC header parsing, robots.txt compliance) and is
deliberately not faked here: the index lookup proves the discovery
path, and the manifest it produces is the input the WARC fetcher
will consume when it lands.

## Terms

Common Crawl data is provided for research; per-record licensing
still depends on the source page. Every record fetched later must
pass the same license gate as every other document
(`sources/license-policy.md`) before training.
