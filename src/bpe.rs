//! Byte-level BPE tokenizer: merge learning, rank-order encoding,
//! decoding, and deterministic persistence.
//!
//! Alphabet: the 256 single bytes. Training counts adjacent symbol
//! pairs over the corpus byte streams and repeatedly merges the most
//! frequent pair, up to `num_merges`. Ties break toward the
//! lexicographically smallest pair (by byte content), so training is
//! fully deterministic for a given corpus and merge count.
//!
//! Encoding applies merges in learned rank order: at each step the
//! lowest-rank adjacent pair present in the sequence is merged, until
//! no learned pair remains adjacent. Decoding concatenates symbol
//! bytes, so for text that came from valid UTF-8,
//! decode(encode(text)) == text exactly.
//!
//! Persistence: `state/bpe.json` is
//! `{"merges": [[hexA, hexB], ...], "vocab": [hex, ...]}` where
//! vocab[0..256] are the single bytes and vocab[256+i] is the i-th
//! merge's concatenation. Token ids are vocab indices.
//!
//! Documents in BPE mode store tokens as hex strings (the symbol
//! bytes in hex). Hex contains no spaces or separators, so every
//! downstream string-based stage (shingles, scoring, n-gram counts)
//! works unchanged and unambiguously. The reference tokenizer
//! remains available as a fallback; see docs/tokenizer.md.

use crate::json::{self, Value};
use std::collections::HashMap;

/// Hex-encode bytes (lowercase, two digits per byte).
pub fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

/// Hex-decode. Returns None on odd length or non-hex input.
pub fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    if bytes.len() % 2 != 0 {
        return None;
    }
    let hex = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex(bytes[i])?;
        let lo = hex(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Some(out)
}

/// Unambiguous key for an adjacent pair: length-prefixed bytes.
fn pair_key(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut k = Vec::with_capacity(4 + a.len() + b.len());
    k.extend_from_slice(&(a.len() as u32).to_be_bytes());
    k.extend_from_slice(a);
    k.extend_from_slice(b);
    k
}

#[derive(Debug, Clone)]
pub struct Bpe {
    /// Merges in rank order: merges[i] is the (i+1)-th learned pair.
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
    /// Vocabulary in id order: 256 single bytes, then merged symbols.
    pub vocab: Vec<Vec<u8>>,
    /// Pair key -> rank (0-based index into merges).
    ranks: HashMap<Vec<u8>, usize>,
}

impl Bpe {
    /// Build from a merge list (rank order). Vocabulary is derived.
    pub fn from_merges(merges: Vec<(Vec<u8>, Vec<u8>)>) -> Bpe {
        let mut vocab: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        let mut ranks = HashMap::new();
        for (i, (a, b)) in merges.iter().enumerate() {
            let mut sym = a.clone();
            sym.extend_from_slice(b);
            vocab.push(sym);
            ranks.insert(pair_key(a, b), i);
        }
        Bpe { merges, vocab, ranks }
    }

    /// Learn merges from corpus texts. `num_merges` caps the count;
    /// learning stops early when no adjacent pair remains.
    pub fn train(texts: &[&str], num_merges: usize) -> Bpe {
        let mut seqs: Vec<Vec<Vec<u8>>> =
            texts.iter().map(|t| t.bytes().map(|b| vec![b]).collect()).collect();
        let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for _ in 0..num_merges {
            let mut counts: HashMap<(Vec<u8>, Vec<u8>), usize> = HashMap::new();
            for seq in &seqs {
                for w in seq.windows(2) {
                    *counts.entry((w[0].clone(), w[1].clone())).or_insert(0) += 1;
                }
            }
            if counts.is_empty() {
                break;
            }
            let mut best_pair: Option<(Vec<u8>, Vec<u8>)> = None;
            let mut best_count = 0usize;
            for (pair, c) in &counts {
                let better = *c > best_count
                    || (*c == best_count
                        && best_pair.as_ref().map_or(true, |bp| pair < bp));
                if better {
                    best_count = *c;
                    best_pair = Some(pair.clone());
                }
            }
            let (a, b) = best_pair.expect("non-empty counts have a best pair");
            for seq in seqs.iter_mut() {
                let mut out: Vec<Vec<u8>> = Vec::with_capacity(seq.len());
                let mut i = 0;
                while i < seq.len() {
                    if i + 1 < seq.len() && seq[i] == a && seq[i + 1] == b {
                        let mut merged = a.clone();
                        merged.extend_from_slice(&b);
                        out.push(merged);
                        i += 2;
                    } else {
                        out.push(seq[i].clone());
                        i += 1;
                    }
                }
                *seq = out;
            }
            merges.push((a, b));
        }
        Bpe::from_merges(merges)
    }

    /// Encode text into symbol byte strings, applying merges in rank
    /// order until no learned pair remains adjacent.
    pub fn encode(&self, text: &str) -> Vec<Vec<u8>> {
        let mut syms: Vec<Vec<u8>> = text.bytes().map(|b| vec![b]).collect();
        loop {
            if syms.len() < 2 {
                break;
            }
            let mut best: Option<(usize, usize)> = None; // (rank, position)
            for i in 0..syms.len() - 1 {
                if let Some(r) = self.ranks.get(&pair_key(&syms[i], &syms[i + 1])) {
                    if best.map_or(true, |(br, _)| *r < br) {
                        best = Some((*r, i));
                    }
                }
            }
            match best {
                None => break,
                Some((_, i)) => {
                    let mut merged = syms[i].clone();
                    merged.extend_from_slice(&syms[i + 1]);
                    syms.splice(i..i + 2, [merged]);
                }
            }
        }
        syms
    }

    /// Encode to hex-string tokens (the on-disk document form).
    pub fn encode_hex(&self, text: &str) -> Vec<String> {
        self.encode(text).iter().map(|s| hex_encode(s)).collect()
    }

    /// Decode symbol byte strings back to text.
    pub fn decode(&self, syms: &[Vec<u8>]) -> String {
        let mut bytes = Vec::new();
        for s in syms {
            bytes.extend_from_slice(s);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode hex-string tokens back to text. Tokens that fail to
    /// hex-decode are skipped (defensive; well-formed shards never
    /// produce them).
    pub fn decode_hex(&self, tokens: &[String]) -> String {
        let syms: Vec<Vec<u8>> = tokens.iter().filter_map(|t| hex_decode(t)).collect();
        self.decode(&syms)
    }

    /// Vocabulary as hex strings, in id order.
    pub fn vocab_hex(&self) -> Vec<String> {
        self.vocab.iter().map(|s| hex_encode(s)).collect()
    }

    /// Serialize to the state/bpe.json Value.
    pub fn to_json(&self) -> Value {
        let mut o = Value::obj();
        let merges: Vec<Value> = self
            .merges
            .iter()
            .map(|(a, b)| Value::Arr(vec![Value::s(&hex_encode(a)), Value::s(&hex_encode(b))]))
            .collect();
        o.put("merges", Value::Arr(merges));
        o.put(
            "vocab",
            Value::Arr(self.vocab_hex().iter().map(|h| Value::s(h)).collect()),
        );
        o
    }

    /// Load from the state/bpe.json Value. The vocab field is
    /// cross-checked against the derived vocab; a mismatch is an
    /// error rather than a silent reinterpretation.
    pub fn from_json(v: &Value) -> Result<Bpe, String> {
        let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        if let Some(arr) = v.get("merges").and_then(|x| x.as_arr()) {
            for pair in arr {
                let parts = pair.as_arr().ok_or("bpe merges entry is not a pair")?;
                if parts.len() != 2 {
                    return Err("bpe merges entry does not have 2 parts".to_string());
                }
                let a = parts[0]
                    .as_str()
                    .and_then(hex_decode)
                    .ok_or("bpe merge part is not hex")?;
                let b = parts[1]
                    .as_str()
                    .and_then(hex_decode)
                    .ok_or("bpe merge part is not hex")?;
                merges.push((a, b));
            }
        }
        let bpe = Bpe::from_merges(merges);
        if let Some(varr) = v.get("vocab").and_then(|x| x.as_arr()) {
            let stored: Vec<String> =
                varr.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect();
            if stored != bpe.vocab_hex() {
                return Err("bpe vocab does not match merges".to_string());
            }
        }
        Ok(bpe)
    }

    /// Load from a state/bpe.json file.
    pub fn load_file(path: &std::path::Path) -> Result<Bpe, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
        let v = json::parse(&text).map_err(|e| format!("bad bpe json: {}", e))?;
        Bpe::from_json(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip() {
        assert_eq!(hex_encode(b"the"), "746865");
        assert_eq!(hex_decode("746865"), Some(b"the".to_vec()));
        assert_eq!(hex_decode("zz"), None);
        assert_eq!(hex_decode("abc"), None);
    }

    #[test]
    fn train_known_merges() {
        // "abab": (a,b) occurs twice, (b,a) once -> merge ab first;
        // then [ab, ab] has one pair (ab,ab) -> merge abab.
        let bpe = Bpe::train(&["abab"], 2);
        assert_eq!(bpe.merges.len(), 2);
        assert_eq!(bpe.merges[0], (b"a".to_vec(), b"b".to_vec()));
        assert_eq!(bpe.merges[1], (b"ab".to_vec(), b"ab".to_vec()));
        // "aaa": (a,a) occurs twice -> merge aa; then [aa, a] has
        // one pair (aa,a) -> merge aaa with a second merge.
        let bpe2 = Bpe::train(&["aaa"], 2);
        assert_eq!(bpe2.merges[0], (b"a".to_vec(), b"a".to_vec()));
        assert_eq!(bpe2.merges[1], (b"aa".to_vec(), b"a".to_vec()));
    }

    #[test]
    fn train_tie_break_deterministic() {
        // "abac": pairs (a,b), (b,a), (a,c) each occur once. The
        // lexicographically smallest pair (a,b) must win.
        let bpe = Bpe::train(&["abac"], 1);
        assert_eq!(bpe.merges[0], (b"a".to_vec(), b"b".to_vec()));
        let again = Bpe::train(&["abac"], 1);
        assert_eq!(bpe.merges, again.merges);
    }

    #[test]
    fn encode_rank_order() {
        // Hand-built merges: (a,b) rank 0, (b,c) rank 1. Encoding
        // "abc" must merge (a,b) first, leaving [ab, c].
        let bpe = Bpe::from_merges(vec![
            (b"a".to_vec(), b"b".to_vec()),
            (b"b".to_vec(), b"c".to_vec()),
        ]);
        let syms = bpe.encode("abc");
        assert_eq!(syms, vec![b"ab".to_vec(), b"c".to_vec()]);
        // Determinism: encoding twice gives the same tokens.
        assert_eq!(bpe.encode("abc"), bpe.encode("abc"));
    }

    #[test]
    fn decode_round_trip() {
        let texts = [
            "The river flows through the valley!",
            "Café naïve — punctuation, numbers 123, and\nnewlines.",
            "",
            "a",
            "   spaces   everywhere   ",
        ];
        let bpe = Bpe::train(&texts, 64);
        for t in texts {
            let syms = bpe.encode(t);
            assert_eq!(bpe.decode(&syms), t, "round-trip failed for {:?}", t);
            let hex = bpe.encode_hex(t);
            assert_eq!(bpe.decode_hex(&hex), t, "hex round-trip failed for {:?}", t);
        }
    }

    #[test]
    fn persistence_reload() {
        let bpe = Bpe::train(&["the cat sat on the mat, the cat slept."], 32);
        let json = bpe.to_json();
        let text = json::to_string(&json);
        let parsed = json::parse(&text).unwrap();
        let reloaded = Bpe::from_json(&parsed).unwrap();
        assert_eq!(reloaded.merges, bpe.merges);
        assert_eq!(reloaded.vocab, bpe.vocab);
        assert_eq!(
            reloaded.encode("the cat slept."),
            bpe.encode("the cat slept.")
        );
        // Corrupt vocab must be refused, not silently reinterpreted.
        let mut bad = json.clone();
        if let Some(v) = bad.get("vocab").and_then(|x| x.as_arr()) {
            let mut v2 = v.clone();
            v2.push(Value::s("ff"));
            // Rebuild the object with the tampered vocab.
            let mut o = Value::obj();
            o.put("merges", json.get("merges").unwrap().clone());
            o.put("vocab", Value::Arr(v2));
            bad = o;
        }
        assert!(Bpe::from_json(&bad).is_err());
    }

    #[test]
    fn encode_empty_and_single() {
        let bpe = Bpe::train(&["hello world"], 8);
        assert!(bpe.encode("").is_empty());
        assert_eq!(bpe.encode("x"), vec![b"x".to_vec()]);
    }
}
