//! Retention scoring: S(d) = a*Q(d) + b*N(d) + g*I(d) - d*R(d).
//! All four components are deterministic functions of the document
//! text and the corpus, in [0, 1]. See docs/scoring.md for the exact
//! definitions, the rationale, and the worked example.

use crate::tokenizer::jaccard;
use std::collections::HashSet;

/// Scoring weights. Defaults: alpha 0.3, beta 0.3, gamma 0.2, delta 0.2.
#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub delta: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Weights { alpha: 0.3, beta: 0.3, gamma: 0.2, delta: 0.2 }
    }
}

/// Quality from text properties only. No model, no external signal.
///   length:  n tokens; 1.0 inside [10, 2000], ramps below 10,
///            decays as 2000/n above 2000, floored at 0.3.
///   printable: fraction of chars that are not control characters
///            (newline, tab, carriage return count as printable).
///   alpha:   fraction of chars that are alphabetic.
///   line_rep: fraction of lines that are exact duplicates of an
///            earlier line (boilerplate penalty).
/// Q = 0.4*length + 0.3*printable + 0.2*alpha + 0.1*(1 - line_rep).
pub fn quality(text: &str, tokens: &[String]) -> f64 {
    if tokens.is_empty() || text.is_empty() {
        return 0.0;
    }
    let n = tokens.len() as f64;
    let length = if n == 0.0 {
        0.0
    } else if n < 10.0 {
        (n / 10.0) * 0.5
    } else if n <= 2000.0 {
        1.0
    } else {
        (2000.0 / n).max(0.3)
    };
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len().max(1) as f64;
    let printable = chars
        .iter()
        .filter(|c| !c.is_control() || **c == '\n' || **c == '\t' || **c == '\r')
        .count() as f64
        / total;
    let alpha = chars.iter().filter(|c| c.is_alphabetic()).count() as f64 / total;
    let lines: Vec<&str> = text.lines().collect();
    let line_rep = if lines.len() <= 1 {
        0.0
    } else {
        let mut seen = HashSet::new();
        let mut dups = 0usize;
        for line in &lines {
            if !seen.insert(*line) {
                dups += 1;
            }
        }
        dups as f64 / lines.len() as f64
    };
    0.4 * length + 0.3 * printable + 0.2 * alpha + 0.1 * (1.0 - line_rep)
}

/// Information density: half unique-token ratio, half normalized
/// Shannon entropy of the token distribution. H is normalized by
/// log2(token count) so short and long documents are comparable.
pub fn information_density(tokens: &[String]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<&String, usize> = std::collections::HashMap::new();
    for tok in tokens {
        *counts.entry(tok).or_insert(0) += 1;
    }
    let total = tokens.len() as f64;
    let unique_ratio = counts.len() as f64 / total;
    let mut entropy = 0.0f64;
    for count in counts.values() {
        let p = *count as f64 / total;
        entropy -= p * p.log2();
    }
    let norm = if total > 1.0 { entropy / total.log2() } else { 0.0 };
    0.5 * unique_ratio + 0.5 * norm
}

/// Max Jaccard similarity of `shingles` against every other set.
/// Returns 0.0 when there are no other documents.
pub fn max_jaccard(shingles: &HashSet<String>, others: &[HashSet<String>]) -> f64 {
    let mut best = 0.0f64;
    for other in others {
        let j = jaccard(shingles, other);
        if j > best {
            best = j;
        }
    }
    best
}

/// Full score breakdown for one document.
#[derive(Debug, Clone, Copy)]
pub struct Score {
    pub quality: f64,
    pub novelty: f64,
    pub density: f64,
    pub redundancy: f64,
    pub total: f64,
}

/// Score one document. `max_j` is its max shingle Jaccard against all
/// other candidate documents (computed by the caller, which owns the
/// corpus-wide shingle sets). Novelty is 1 - max_j; redundancy is
/// max_j. Keeping both terms explicit is deliberate: beta rewards
/// novelty while delta separately penalizes redundancy, so the two
/// weights can be tuned independently.
pub fn score(text: &str, tokens: &[String], max_j: f64, weights: Weights) -> Score {
    let q = quality(text, tokens);
    let i = information_density(tokens);
    let n = 1.0 - max_j;
    let r = max_j;
    let total = weights.alpha * q + weights.beta * n + weights.gamma * i - weights.delta * r;
    Score { quality: q, novelty: n, density: i, redundancy: r, total }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    #[test]
    fn quality_bands() {
        let empty = quality("", &[]);
        assert_eq!(empty, 0.0);
        let good_text = "The quick brown fox jumps over the lazy dog and runs away fast. ".repeat(3);
        let good = quality(&good_text, &tokenize(&good_text));
        assert!(good > 0.7, "good text quality {}", good);
        // Repetition penalty: identical lines repeated.
        let boiler = "click here now\n".repeat(20);
        let rep = quality(&boiler, &tokenize(&boiler));
        assert!(rep < good, "boilerplate {} should score below good {}", rep, good);
    }

    #[test]
    fn density_unique_vs_repeated() {
        let varied = tokenize("alpha beta gamma delta epsilon zeta eta theta");
        let repeated = tokenize("same same same same same same same same");
        assert!(information_density(&varied) > information_density(&repeated));
        assert_eq!(information_density(&[]), 0.0);
    }

    #[test]
    fn score_formula_exact() {
        let w = Weights { alpha: 0.3, beta: 0.3, gamma: 0.2, delta: 0.2 };
        let text = "one two three four five six seven eight nine ten eleven twelve";
        let tokens = tokenize(text);
        let s = score(text, &tokens, 0.25, w);
        let expect = 0.3 * s.quality + 0.3 * (1.0 - 0.25) + 0.2 * s.density - 0.2 * 0.25;
        assert!((s.total - expect).abs() < 1e-12);
        assert_eq!(s.novelty, 0.75);
        assert_eq!(s.redundancy, 0.25);
    }
}
