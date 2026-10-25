# GitHub corpus

Two honest paths, both implemented through mechanisms that already
exist in the CLI:

1. **URL manifest.** A JSON file listing raw file URLs (or any
   HTTP(S) URL) with `source`, `category`, and `license` per entry:
   `ingest --url-manifest manifest.json`. The CLI GETs each URL
   (30s timeout, 2MB cap) and records provenance. This is the path
   for curated file lists, including `raw.githubusercontent.com`
   URLs for public files.
2. **Local clones.** Clone repositories locally (outside this
   repo) and ingest a directory of `.txt` extracts:
   `ingest --corpus <dir>`. Code files are not ingested raw: the
   reference tokenizer is a text tokenizer, and source code needs
   an extraction step (comments/strings/docstrings) that is a later
   milestone. Do not point the corpus flag at a source tree and
   expect clean tokens.

## What is NOT faked

GitHub's code search API requires authentication and has rate
limits; the search API is not called by this CLI. Any claim that
the CLI "searches GitHub" would be false. Discovery over GitHub
means: a human or a separate authenticated process produces a URL
manifest (or a local extract), and this CLI ingests exactly that.

License note: a public repository is not public-domain code. The
manifest's `license` field per file is what the license gate
enforces; a missing license quarantines the file.
