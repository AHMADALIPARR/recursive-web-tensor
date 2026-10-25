//! Remote fetching. Fetched content is DATA, never executed.
//!
//! All network access goes through the `curl` subprocess (present on
//! GitHub-hosted runners): HTTP GET only, with a timeout and a byte
//! cap. Nothing fetched is ever run, interpreted as code, or passed
//! to a shell. URLs come from an explicit manifest or from a
//! documented discovery endpoint; the CLI never follows a crawl
//! frontier on its own.
//!
//! Implemented remote paths:
//! - URL manifest: GET each listed URL, cap 2,000,000 bytes.
//! - Common Crawl index lookup: GET an index API URL, cap 1,000,000
//!   bytes, parse the JSON-lines records into a manifest. WARC
//!   record fetching is NOT implemented (later milestone).
//! - DuckDuckGo discovery: GET the html-lite endpoint, extract
//!   `uddg=` result URLs. HTML is treated as text; no JS runs.

use std::process::Command;

/// Run curl for a single GET. Returns the raw response bytes.
/// Errors are returned as strings; callers record and skip.
pub fn curl_get(url: &str, max_time_secs: u64, max_bytes: u64) -> Result<Vec<u8>, String> {
    let out = Command::new("curl")
        .args([
            "-sS",
            "-L",
            "--max-time",
            &max_time_secs.to_string(),
            "--max-filesize",
            &max_bytes.to_string(),
            url,
        ])
        .output()
        .map_err(|e| format!("failed to run curl: {}", e))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("curl failed for {}: {}", url, stderr.trim()));
    }
    Ok(out.stdout)
}

/// Percent-encode a query string (RFC 3986 unreserved set kept bare).
pub fn percent_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

/// Percent-decode `%XX` sequences and `+` as space. Invalid sequences
/// are left literal rather than failing the whole extraction.
pub fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let hex = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() + 1 && i + 2 <= bytes.len() - 1 + 1 {
            if i + 2 < bytes.len() {
                if let (Some(hi), Some(lo)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                    out.push((hi << 4) | lo);
                    i += 3;
                    continue;
                }
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Extract DuckDuckGo html-lite result URLs from page text: every
/// `uddg=<encoded>` parameter, percent-decoded, in first-seen order,
/// deduplicated. Only http(s) results are kept.
pub fn extract_ddg_urls(html: &str) -> Vec<String> {
    let mut urls = Vec::new();
    let mut rest = html;
    while let Some(pos) = rest.find("uddg=") {
        let after = &rest[pos + 5..];
        let end = after
            .find(|c: char| c == '&' || c == '"' || c == '\'' || c == '<' || c.is_whitespace())
            .unwrap_or(after.len());
        let decoded = percent_decode(&after[..end]);
        if (decoded.starts_with("http://") || decoded.starts_with("https://"))
            && !urls.contains(&decoded)
        {
            urls.push(decoded);
        }
        rest = &after[end.min(after.len())..];
    }
    urls
}

/// One Common Crawl index record, as returned by the index API
/// (one JSON object per line). Only the fields this CLI uses.
#[derive(Debug, Clone, PartialEq)]
pub struct CcRecord {
    pub url: String,
    pub filename: String,
    pub offset: i64,
    pub length: i64,
}

/// Parse Common Crawl index JSON-lines text into records. Lines that
/// fail to parse are skipped and counted, never fatal: the index is
/// untrusted remote input.
pub fn parse_cc_index(text: &str) -> (Vec<CcRecord>, usize) {
    let mut records = Vec::new();
    let mut skipped = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match crate::json::parse(line) {
            Ok(v) => {
                let url = v.get("url").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let filename =
                    v.get("filename").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let offset = v.get("offset").and_then(|x| x.as_i64()).unwrap_or(0);
                let length = v.get("length").and_then(|x| x.as_i64()).unwrap_or(0);
                if url.is_empty() || filename.is_empty() {
                    skipped += 1;
                } else {
                    records.push(CcRecord { url, filename, offset, length });
                }
            }
            Err(_) => skipped += 1,
        }
    }
    (records, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddg_extraction_fixture() {
        let html = r#"<a href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=1">one</a>
        <a href='/l/?uddg=http%3A%2F%2Fexample.org%2Fb'>two</a>
        <a href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa">dup</a>
        <a href="/l/?uddg=javascript%3Aalert%281%29">evil</a>"#;
        let urls = extract_ddg_urls(html);
        assert_eq!(urls, vec!["https://example.com/a", "http://example.org/b"]);
    }

    #[test]
    fn cc_index_fixture() {
        let text = r#"{"url":"example.com/a","status":"200","filename":"crawl-data/CC-MAIN-2025-30/segments/x.warc.gz","offset":123,"length":456,"digest":"abc"}
{"url":"example.com/b","status":"200","filename":"crawl-data/CC-MAIN-2025-30/segments/y.warc.gz","offset":789,"length":10}
not json at all
{"url":"","filename":""}"#;
        let (records, skipped) = parse_cc_index(text);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].offset, 123);
        assert_eq!(records[0].length, 456);
        assert_eq!(skipped, 2);
    }

    #[test]
    fn percent_round_trip() {
        assert_eq!(percent_decode("hello%20world%21"), "hello world!");
        assert_eq!(percent_encode("hello world!"), "hello%20world%21");
        assert_eq!(percent_decode("a+b"), "a b");
    }
}
