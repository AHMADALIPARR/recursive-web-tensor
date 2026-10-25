//! Deterministic reference tokenizer (word/byte level).
//!
//! This is NOT BPE. Tokens are lowercased alphanumeric runs plus one
//! token per non-alphanumeric, non-whitespace character. The mapping
//! is fixed, deterministic, and independent of training order, so the
//! same document always tokenizes the same way. BPE training is a
//! later milestone, not claimed anywhere in this repo.
//!
//! The vocabulary is built by scanning documents in ascending SHA-256
//! order and assigning ids in first-seen order, then persisted as
//! `data/vocab.json` so later runs reuse identical ids.

/// Tokenize `text` into a deterministic token stream.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                current.push(lc);
            }
        } else {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            if !ch.is_whitespace() {
                tokens.push(ch.to_string());
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Build a vocabulary from documents given in ascending SHA-256 order.
/// Returns (vocab list in first-seen order, token -> id map as pairs).
pub fn build_vocab(docs_in_sha_order: &[Vec<String>]) -> Vec<String> {
    let mut vocab: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for tokens in docs_in_sha_order {
        for tok in tokens {
            if seen.insert(tok.clone()) {
                vocab.push(tok.clone());
            }
        }
    }
    vocab
}

/// Word shingles of `tokens`: windows of `n` consecutive tokens joined
/// by a single space. A document shorter than `n` tokens yields one
/// shingle: the whole token stream. Used for novelty/redundancy.
pub fn shingles(tokens: &[String], n: usize) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    if tokens.is_empty() {
        return set;
    }
    if tokens.len() < n {
        set.insert(tokens.join(" "));
        return set;
    }
    for window in tokens.windows(n) {
        set.insert(window.join(" "));
    }
    set
}

/// Jaccard similarity between two shingle sets. Empty-vs-empty is 1.0
/// (identical emptiness); empty-vs-nonempty is 0.0.
pub fn jaccard(a: &std::collections::HashSet<String>, b: &std::collections::HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    inter / union
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_deterministic() {
        let a = tokenize("Hello, World! Hello world.");
        let b = tokenize("Hello, World! Hello world.");
        assert_eq!(a, b);
        assert_eq!(a, vec!["hello", ",", "world", "!", "hello", "world", "."]);
    }

    #[test]
    fn tokenize_unicode_lowercases() {
        assert_eq!(tokenize("Café NAÏVE"), vec!["café", "naïve"]);
    }

    #[test]
    fn vocab_first_seen_order() {
        let docs = vec![
            vec!["b".to_string(), "a".to_string()],
            vec!["a".to_string(), "c".to_string()],
        ];
        assert_eq!(build_vocab(&docs), vec!["b", "a", "c"]);
    }

    #[test]
    fn jaccard_basics() {
        let a: std::collections::HashSet<String> =
            ["x".to_string(), "y".to_string()].into_iter().collect();
        let b: std::collections::HashSet<String> =
            ["y".to_string(), "z".to_string()].into_iter().collect();
        let j = jaccard(&a, &b);
        assert!((j - 1.0 / 3.0).abs() < 1e-12);
    }
}
