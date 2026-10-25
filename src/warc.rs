//! WARC record parsing and reference text extraction.
//!
//! Format (WARC 1.0/1.1, ISO 28500): a file is a sequence of
//! records. Each record is a version line (`WARC/1.0` or
//! `WARC/1.1`), header lines (`Name: value`), a blank line, a
//! content block of exactly `Content-Length` bytes, and a
//! CRLF CRLF separator. Headers are ASCII; content is opaque
//! bytes (for `response` records, usually an HTTP response).
//!
//! The parser is byte-level and strict about lengths: a missing
//! version line, a missing or unparseable Content-Length, or a
//! truncated content block is an error, never a silent partial
//! record. Line endings CRLF and bare LF are both accepted in the
//! header section (the spec says CRLF; the tolerance costs nothing
//! and keeps slightly-off producers parseable).
//!
//! Text extraction is a REFERENCE extractor, stated plainly: for
//! `response` records it strips an HTTP response header block when
//! present, removes `<script>`/`<style>` elements and comments,
//! strips remaining tags, decodes a small set of HTML entities,
//! and collapses whitespace. It is NOT a full HTML parser: nested
//! or malformed markup, CDATA, and entities outside the small set
//! below are handled by lossy fallback (tag stripped, entity left
//! literal), never by executing anything. Extracted text is data.

/// One parsed WARC record. The version line is validated at parse
/// time (must be WARC/1.0 or WARC/1.1) but not stored: no consumer
/// branches on it, and storing an unused field would be dead weight.
#[derive(Debug, Clone)]
pub struct WarcRecord {
    /// Header (name, value) pairs in file order. Names compared
    /// case-insensitively by `header()`.
    pub headers: Vec<(String, String)>,
    pub content: Vec<u8>,
}

impl WarcRecord {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    pub fn record_type(&self) -> Option<&str> {
        self.header("WARC-Type")
    }
    pub fn target_uri(&self) -> Option<&str> {
        self.header("WARC-Target-URI")
    }
    pub fn is_response(&self) -> bool {
        self.record_type().map(|t| t.eq_ignore_ascii_case("response")).unwrap_or(false)
    }
}

/// Find the end of a line starting at `pos`: returns (line bytes
/// without terminator, index just past the terminator). Accepts
/// CRLF or LF. Returns None when no terminator remains.
fn read_line(bytes: &[u8], pos: usize) -> Option<(&[u8], usize)> {
    if pos >= bytes.len() {
        return None;
    }
    let mut i = pos;
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            let end = if i > pos && bytes[i - 1] == b'\r' { i - 1 } else { i };
            return Some((&bytes[pos..end], i + 1));
        }
        i += 1;
    }
    None
}

/// Parse all records in `bytes`. Trailing blank lines / separators
/// after the last record are ignored; anything else trailing is an
/// error (a truncated record must not pass silently).
pub fn parse_warc(bytes: &[u8]) -> Result<Vec<WarcRecord>, String> {
    let mut records = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        // Skip inter-record blank lines (the CRLF CRLF separator
        // leaves up to two empty lines before the next version).
        while pos < bytes.len() {
            match read_line(bytes, pos) {
                Some((line, next)) if line.is_empty() => pos = next,
                _ => break,
            }
        }
        if pos >= bytes.len() {
            break;
        }
        let (version_line, mut p) = read_line(bytes, pos)
            .ok_or_else(|| format!("truncated WARC version line at offset {}", pos))?;
        let version = String::from_utf8_lossy(version_line).into_owned();
        if version != "WARC/1.0" && version != "WARC/1.1" {
            return Err(format!(
                "expected WARC version line at offset {}, got {:?}",
                pos, version
            ));
        }
        let mut headers: Vec<(String, String)> = Vec::new();
        loop {
            let (line, next) = read_line(bytes, p)
                .ok_or_else(|| format!("truncated WARC header at offset {}", p))?;
            p = next;
            if line.is_empty() {
                break;
            }
            let text = String::from_utf8_lossy(line);
            match text.split_once(':') {
                Some((name, value)) => {
                    headers.push((name.trim().to_string(), value.trim().to_string()));
                }
                None => {
                    return Err(format!("malformed WARC header line at offset {}: {:?}", p, text));
                }
            }
        }
        let length: usize = headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case("Content-Length"))
            .and_then(|(_, v)| v.parse::<usize>().ok())
            .ok_or_else(|| format!("record at offset {} has no valid Content-Length", pos))?;
        if p + length > bytes.len() {
            return Err(format!(
                "record at offset {} claims Content-Length {} but only {} bytes remain",
                pos,
                length,
                bytes.len() - p
            ));
        }
        let content = bytes[p..p + length].to_vec();
        records.push(WarcRecord { headers, content });
        pos = p + length;
    }
    Ok(records)
}

/// Strip an HTTP response header block when the content starts
/// with `HTTP/`: everything through the first blank line. Otherwise
/// the content is the body already.
fn http_body(content: &[u8]) -> &[u8] {
    if !content.starts_with(b"HTTP/") {
        return content;
    }
    // Scan for \r\n\r\n or \n\n.
    let mut i = 0;
    while i + 1 < content.len() {
        if content[i] == b'\r' && content[i + 1] == b'\n' {
            if i + 3 < content.len() && content[i + 2] == b'\r' && content[i + 3] == b'\n' {
                return &content[i + 4..];
            }
        } else if content[i] == b'\n' && i + 1 < content.len() && content[i + 1] == b'\n' {
            return &content[i + 2..];
        }
        i += 1;
    }
    content
}

/// ASCII case-insensitive prefix match on bytes (no lowered copy:
/// Unicode lowercasing can change byte lengths and invalidate
/// offsets).
fn starts_with_ignore_case(hay: &[u8], needle: &str) -> bool {
    let n = needle.as_bytes();
    hay.len() >= n.len()
        && hay[..n.len()]
            .iter()
            .zip(n.iter())
            .all(|(a, b)| a.to_ascii_lowercase() == *b)
}

/// ASCII case-insensitive substring search. Returns the byte offset
/// of the first match, or None.
fn find_ignore_case(hay: &[u8], needle: &str) -> Option<usize> {
    let n = needle.as_bytes();
    if n.is_empty() || hay.len() < n.len() {
        return None;
    }
    (0..=hay.len() - n.len()).find(|&i| starts_with_ignore_case(&hay[i..], needle))
}

/// Remove `<script>...</script>` and `<style>...</style>` elements
/// (ASCII case-insensitive tags), HTML comments, and all remaining
/// tags. Tag removal inserts a space so adjacent text does not fuse
/// ("a<b>b</b>c" -> "a b c", collapsed later). A self-closed
/// `<script/>` open tag is treated as a plain tag. Unterminated
/// constructs drop the rest of the input (documented lossy
/// fallback, never an error: extraction must not fail a whole
/// ingest over one page's markup).
fn strip_tags(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    'outer: while i < bytes.len() {
        // Comments.
        if text[i..].starts_with("<!--") {
            match text[i..].find("-->") {
                Some(end) => {
                    out.push(' ');
                    i += end + 3;
                    continue 'outer;
                }
                None => break, // unterminated comment: drop the rest
            }
        }
        // Script/style elements.
        for tag in ["script", "style"] {
            let open = format!("<{}", tag);
            if starts_with_ignore_case(&bytes[i..], &open)
                && matches!(
                    bytes.get(i + 1 + tag.len()),
                    Some(b'>') | Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') | Some(b'/')
                )
            {
                // Self-closed open tag (<script/>): plain-tag handling.
                let open_end = match text[i..].find('>') {
                    Some(gt) => i + gt,
                    None => break 'outer,
                };
                if open_end > i && bytes[open_end - 1] == b'/' {
                    out.push(' ');
                    i = open_end + 1;
                    continue 'outer;
                }
                let close = format!("</{}>", tag);
                match find_ignore_case(&bytes[i..], &close) {
                    Some(end) => {
                        out.push(' ');
                        i += end + close.len();
                        continue 'outer;
                    }
                    None => break 'outer, // unterminated element: drop the rest
                }
            }
        }
        // Plain tag: skip through '>', push a space separator.
        if bytes[i] == b'<' {
            match text[i..].find('>') {
                Some(gt) => {
                    out.push(' ');
                    i += gt + 1;
                    continue 'outer;
                }
                None => break, // unterminated tag: drop the rest
            }
        }
        // Regular character: copy one UTF-8 char.
        let ch_len = utf8_len(bytes[i]);
        let end = (i + ch_len).min(bytes.len());
        out.push_str(&text[i..end]);
        i = end;
    }
    out
}

/// Length in bytes of the UTF-8 sequence starting with `b` (1 for
/// ASCII and for invalid starters, so the loop always advances).
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

/// Decode the small entity set this reference extractor knows.
/// Unknown entities are left literal (documented lossy fallback).
fn decode_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

/// Extract readable text from a WARC record: HTTP-strip, tag-strip,
/// entity-decode, whitespace-collapse, trim.
pub fn extract_text(record: &WarcRecord) -> String {
    let body = http_body(&record.content);
    let text = String::from_utf8_lossy(body);
    let stripped = strip_tags(&text);
    let decoded = decode_entities(&stripped);
    decoded
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_warc() -> Vec<u8> {
        let html = b"<html><head><title>River Valley</title><style>.x{}</style><script>var x=1;</script></head><body><!-- c --><h1>The River &amp; The Valley</h1><p>The river flows &lt;swiftly&gt; through the valley.</p></body></html>";
        let http = [b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n".as_slice(), html].concat();
        let mut out = Vec::new();
        let warcinfo = b"software: test\r\n";
        out.extend_from_slice(
            format!(
                "WARC/1.0\r\nWARC-Type: warcinfo\r\nContent-Length: {}\r\n\r\n",
                warcinfo.len()
            )
            .as_bytes(),
        );
        out.extend_from_slice(warcinfo);
        out.extend_from_slice(b"\r\n\r\n");
        out.extend_from_slice(
            format!(
                "WARC/1.0\r\nWARC-Type: response\r\nWARC-Target-URI: http://example.com/a\r\nContent-Type: application/http; msgtype=response\r\nContent-Length: {}\r\n\r\n",
                http.len()
            )
            .as_bytes(),
        );
        out.extend_from_slice(&http);
        out.extend_from_slice(b"\r\n\r\n");
        out
    }

    #[test]
    fn parse_two_records_and_extract() {
        let bytes = sample_warc();
        let recs = parse_warc(&bytes).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].record_type(), Some("warcinfo"));
        assert!(recs[1].is_response());
        assert_eq!(recs[1].target_uri(), Some("http://example.com/a"));
        let text = extract_text(&recs[1]);
        assert_eq!(
            text,
            "River Valley The River & The Valley The river flows <swiftly> through the valley."
        );
        // Script/style/comment content must not survive.
        assert!(!text.contains("var x"));
        assert!(!text.contains(".x{}"));
    }

    #[test]
    fn range_slice_parses() {
        // A byte-range slice of a WARC file parses standalone: this
        // is exactly what a Range-fetched record looks like.
        let bytes = sample_warc();
        let mut count = 0;
        let mut start = 0;
        for i in 0..bytes.len() - 8 {
            if &bytes[i..i + 8] == b"WARC/1.0" {
                count += 1;
                if count == 2 {
                    start = i;
                    break;
                }
            }
        }
        assert_eq!(count, 2, "fixture must have two records");
        let recs = parse_warc(&bytes[start..]).unwrap();
        assert_eq!(recs.len(), 1);
        assert!(recs[0].is_response());
    }

    #[test]
    fn parse_errors_are_strict() {
        // Truncated content.
        let bad = b"WARC/1.0\r\nWARC-Type: response\r\nContent-Length: 100\r\n\r\nshort";
        assert!(parse_warc(bad).is_err());
        // Missing Content-Length.
        let bad2 = b"WARC/1.0\r\nWARC-Type: response\r\n\r\nabc";
        assert!(parse_warc(bad2).is_err());
        // Not a WARC file at all.
        assert!(parse_warc(b"hello world, not warc").is_err());
        // Trailing garbage after a valid record.
        let mut bad3 = sample_warc();
        bad3.extend_from_slice(b"garbage");
        assert!(parse_warc(&bad3).is_err());
    }

    #[test]
    fn plain_text_record_extracts_verbatim() {
        let body = b"Just some plain text, no markup at all.";
        let mut bytes = format!(
            "WARC/1.0\r\nWARC-Type: response\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();
        bytes.extend_from_slice(body);
        bytes.extend_from_slice(b"\r\n\r\n");
        let recs = parse_warc(&bytes).unwrap();
        assert_eq!(extract_text(&recs[0]), "Just some plain text, no markup at all.");
    }

    #[test]
    fn fixture_file_parses() {
        let bytes = std::fs::read("fixtures/warc-sample.warc").unwrap();
        let recs = parse_warc(&bytes).unwrap();
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[1].record_type(), Some("response"));
        let text = extract_text(&recs[1]);
        assert!(text.contains("The river flows <swiftly> through the valley."));
        assert!(text.contains("Morning mist hangs over the water."));
        assert!(!text.contains("var x"));
    }
}
