# DuckDuckGo discovery

Endpoint used by the reference CLI:

```
GET https://html.duckduckgo.com/html/?q=<percent-encoded query>
```

The CLI extracts `uddg=` result URLs from the returned HTML
(first-seen order, deduplicated, http(s) only) and writes them to
`state/ddg-manifest.json`. The HTML is treated as text: no
JavaScript runs, nothing is executed.

Honest limits:

- The html-lite endpoint is an HTML scraping surface, not a stable
  API. Its markup can change without notice; extraction failures
  surface as zero URLs, not as fabricated results.
- Discovery returns URLs only. Fetching those URLs is a separate,
  explicit step (URL manifest ingest), where the license of each
  fetched document is recorded from manifest metadata.
- Rate limits and blocking are the operator's problem at scale:
  the reference implementation issues one request per dispatch,
  not a crawl. A production discovery loop needs request pacing
  and robots/terms compliance per source — not implemented here.
