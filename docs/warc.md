# Common Crawl WARC fetching (Phase 4, real)

`src/warc.rs` (parse + extract) + `src/fetch.rs` (`curl_range`,
`gunzip`) + the `ingest --cc-warc-manifest` path. This completes
the remote-source story: index lookup discovers records, and this
path turns records into documents.

## Flow

```
--cc-index <index API URL>        -> state/cc-manifest.json
--cc-warc-manifest <manifest>     -> Range-fetch each record
                                     -> gunzip .warc.gz members
                                     -> parse WARC records
                                     -> extract text (response only)
                                     -> ingest as documents
```

```
ite ingest --cc-warc-manifest state/cc-manifest.json
ite ingest --cc-warc-manifest state/cc-manifest.json --cc-max-records 20
ite ingest --cc-warc-manifest state/cc-manifest.json \
    --cc-data-base https://data.commoncrawl.org/
```

The data base defaults to `https://data.commoncrawl.org/`; tests
point it at a local Range server. `--cc-max-records 0` (default)
means all items; otherwise the first N items in manifest order.

## Range semantics

Each manifest item carries `url`, `filename`, `offset`, `length`.
The fetcher requests exactly bytes `[offset, offset+length)` from
`<base>/<filename>` with `curl -r`. A short body (server ignored
the Range, range past EOF) is an error, never a silent partial
record — a truncated gzip member must not parse as a valid record
downstream. Non-http(s) data bases are refused.

## Gzip members

Real Common Crawl WARC files are multi-member gzip: one member per
record, and the index offset points at a member boundary. The
fetcher gunzips a fetched member with the system `gzip` binary
(the same dependency class as `curl`: present on GitHub-hosted
runners). Uncompressed `.warc` files (no `.gz` suffix, no gzip
magic) parse directly. Decompression is capped (see below); a
member that fails to decompress is recorded and skipped.

## WARC parsing (strict where it matters)

`parse_warc` is byte-level: version line must be `WARC/1.0` or
`WARC/1.1`; every record needs a parseable `Content-Length`;
content is exactly that many bytes; trailing garbage after the
last record is an error. Header names match case-insensitively.
CRLF and bare-LF header endings are both accepted. Only
`WARC-Type: response` records are ingested; `warcinfo`,
`metadata`, `request`, and friends are skipped (they are crawl
metadata, not page content).

## Text extraction (reference, stated plainly)

`extract_text` is NOT a full HTML parser:

1. If the record content starts with `HTTP/`, the HTTP response
   header block is stripped (through the first blank line).
2. `<script>` and `<style>` elements (ASCII case-insensitive),
   HTML comments, and all remaining tags are removed; tag removal
   inserts a space so adjacent text does not fuse.
3. A small entity set is decoded (`&amp; &lt; &gt; &quot;
   &#39; &apos; &nbsp;`); unknown entities stay literal.
4. Whitespace runs collapse to single spaces; the result is trimmed.

Unterminated constructs drop the rest of the input (lossy
fallback, documented — extraction must not fail a whole ingest
over one page's markup). Empty extracted text is skipped and
counted, not ingested as an empty document.

## Provenance and licensing

Each ingested document gets:

- `source`: `cc:<filename>@<offset>:<page URL>` — the record's
  own `WARC-Target-URI` when present, else the manifest URL.
- `category`: `common-crawl`.
- `license`: the manifest item's `license` when given, else
  `unknown` — which the existing license policy quarantines at
  prune time, exactly like any other unknown-license document.
  Per-record licensing still depends on the source page; Common
  Crawl provides data for research, not a license grant.

The usual pipeline then applies unchanged: SHA-256 dedup,
scoring, adaptive pruning, training, gating.

## Caps (fail closed, counted, never silent)

- Compressed member length: 10,000,000 bytes (larger items are
  recorded as errors and skipped).
- Decompressed output: 10,000,000 bytes.
- Extracted text: 2,000,000 bytes.
- Per-record failures (fetch, gunzip, parse, oversize, bad
  manifest item) are recorded in the ingest error list and
  counted in the `cc-warc:` summary line; they never abort the
  whole ingest.

## Live verification (not a fixture claim)

The chain was verified against real Common Crawl during Phase 4
(sandbox network allowed it; this is a report of a run, not a
test — tests use local fixtures and a local Range server only):
index lookup for `example.com` on `CC-MAIN-2026-43` returned 156
records; the first record (`offset 204233603, length 996`) fetched
via Range, gunzipped, parsed, and ingested as
`cc:...warc.gz@204233603:https://example.com/` with the text
"Example Domain This domain is for use in documentation examples
..." — the real page content, not a fixture.

That live run also caught a real Phase 1 bug: the index API
returns `offset`/`length` as strings, but the parser only accepted
numbers and silently defaulted both to 0. The parser now accepts
both forms and skips records with unparseable values instead of
defaulting to 0 (a 0 range would fetch the wrong bytes).

## Limits (honest)

- **robots.txt is NOT checked per record.** Common Crawl's own
  crawl respects robots, but this fetcher does not re-check the
  live robots.txt for each host before fetching a record.
  Operators running this at scale must add robots handling; the
  gap is stated here, not hidden.
- **Reference extraction only.** Script/style/comment/tag
  stripping plus a small entity set is not a readability
  algorithm and not a full HTML parser; boilerplate, navigation,
  and malformed markup survive or mangle in ways a real parser
  would handle better.
- **No WARC revisit / revisit-server-dedup handling.** Revisit
  records are not `response` records and are skipped; dedup across
  revisits is the existing SHA-256 document dedup, nothing more.
- **One Range request per record.** No connection reuse beyond
  what curl does per invocation, no parallel fetching. Fine at
  reference scale; a throughput wall before internet scale.
