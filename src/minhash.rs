//! MinHash signatures + LSH banding for cross-shard near-dup
//! candidate generation.
//!
//! Why this exists: shard-local pruning compares each document only
//! against canonicals in its own shard (O(n^2) per shard, bounded).
//! A near-dup pair split across shards survives both shards' prune
//! passes. The global step must find those pairs WITHOUT comparing
//! every retained pair (O(n^2) globally). MinHash + LSH does that:
//! documents that share a full LSH band are candidates; candidates
//! are then verified with EXACT Jaccard on shingle sets.
//!
//! Consequences of that design (stated plainly):
//! - Final Compress decisions are EXACT: a pair is demoted only
//!   when its true shingle Jaccard >= threshold. MinHash never
//!   demotes a pair by itself, so there are no MinHash-caused
//!   false positives.
//! - The approximation lives in RECALL only: a true near-dup pair
//!   can be missed when it shares no LSH band. For true Jaccard s,
//!   band size r, band count b, the miss probability is
//!   (1 - s^r)^b. With the defaults (r=8, b=16) a pair at s=0.9 is
//!   missed with probability (1 - 0.9^8)^16 ≈ 0.00065; at s=0.95
//!   it is ≈ 0.00002. Pairs below threshold are not "missed" —
//!   they are correctly not demoted.
//! - The MinHash estimate itself (fraction of equal signature
//!   positions) is NOT used for decisions; it is reported in tests
//!   only, to show the signatures track true Jaccard.
//!
//! Hash family (deterministic, no randomness at runtime): each
//! shingle string is hashed with FNV-1a 64, then permutation i
//! applies universal hashing h_i(x) = (a_i * x + b_i) mod 2^64
//! (wrapping arithmetic) with (a_i, b_i) derived from splitmix64
//! seeded by i. The signature is the per-permutation minimum.
//! Empty shingle sets produce no bands (they cannot be near-dups
//! of anything meaningful; note the tokenizer's empty-vs-empty
//! Jaccard is 1.0, which must NOT leak into global decisions).

use std::collections::{HashMap, HashSet};

/// FNV-1a 64-bit hash of a string's bytes.
pub fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// splitmix64: deterministic PRNG step for deriving hash params.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// (a_i, b_i) pairs for `perms` permutations. a_i is forced odd
/// (standard universal-hashing practice; the min operation does not
/// require invertibility, but odd multipliers avoid degenerate
/// even-only images).
pub fn hash_params(perms: usize) -> Vec<(u64, u64)> {
    let mut state: u64 = 0x2545F4914F6CDD1D;
    (0..perms)
        .map(|_| {
            let a = splitmix64(&mut state) | 1;
            let b = splitmix64(&mut state);
            (a, b)
        })
        .collect()
}

/// MinHash signature of a shingle set: per-permutation minimum of
/// h_i(fnv1a(shingle)). Empty sets yield all-u64::MAX (a sentinel
/// that callers must exclude from banding — see `candidate_pairs`).
pub fn signature(shingles: &HashSet<String>, params: &[(u64, u64)]) -> Vec<u64> {
    let mut sig = vec![u64::MAX; params.len()];
    for s in shingles {
        let x = fnv1a(s);
        for (i, (a, b)) in params.iter().enumerate() {
            let h = a.wrapping_mul(x).wrapping_add(*b);
            if h < sig[i] {
                sig[i] = h;
            }
        }
    }
    sig
}

/// Fraction of equal signature positions: the MinHash estimate of
/// Jaccard. Test oracle only (never used for decisions); hence
/// cfg(test).
#[cfg(test)]
pub fn estimated_jaccard(a: &[u64], b: &[u64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let eq = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    eq as f64 / a.len() as f64
}

/// Candidate pairs from LSH banding: documents sharing at least one
/// full band (bands consecutive slices of `rows` signature
/// positions). Inputs are (sha, signature) pairs; empty-signature
/// docs (all MAX, i.e. empty shingle sets) are excluded. Output
/// pairs are (lower-sha, higher-sha), deduplicated, sorted.
pub fn candidate_pairs(docs: &[(String, Vec<u64>)], bands: usize, rows: usize) -> Vec<(String, String)> {
    let mut buckets: HashMap<Vec<u64>, Vec<String>> = HashMap::new();
    for (sha, sig) in docs {
        if sig.iter().all(|v| *v == u64::MAX) {
            continue; // empty shingle set: no bands
        }
        for band in 0..bands {
            let start = band * rows;
            let end = (start + rows).min(sig.len());
            if start >= sig.len() {
                break;
            }
            buckets.entry(sig[start..end].to_vec()).or_default().push(sha.clone());
        }
    }
    let mut pairs: HashSet<(String, String)> = HashSet::new();
    for members in buckets.values() {
        if members.len() < 2 {
            continue;
        }
        let mut sorted = members.clone();
        sorted.sort();
        for i in 0..sorted.len() {
            for j in i + 1..sorted.len() {
                pairs.insert((sorted[i].clone(), sorted[j].clone()));
            }
        }
    }
    let mut out: Vec<(String, String)> = pairs.into_iter().collect();
    out.sort();
    out
}

/// Exact O(n^2) global decisions: the test oracle for
/// `lsh_decisions` (production never runs it — running it would
/// defeat the point of LSH). Input docs must be SHA-sorted; the
/// canonical of a group is its lowest SHA (first-seen wins in SHA
/// order, and a doc that pairs with an already-demoted doc maps
/// directly to that doc's canonical, so no alias chains are
/// produced).
/// Returns (demoted_sha, canonical_sha, jaccard) triples.
#[cfg(test)]
pub fn exact_decisions(
    docs: &[(String, HashSet<String>)],
    threshold: f64,
) -> Vec<(String, String, f64)> {
    let mut canonical_of: HashMap<String, String> = HashMap::new();
    let mut out = Vec::new();
    for (i, (sha, set)) in docs.iter().enumerate() {
        if set.is_empty() {
            canonical_of.insert(sha.clone(), sha.clone());
            continue;
        }
        let mut best: Option<(String, f64)> = None;
        for (j, (other, oset)) in docs.iter().enumerate() {
            if j >= i || oset.is_empty() {
                continue;
            }
            let jacc = crate::tokenizer::jaccard(set, oset);
            if jacc >= threshold {
                let canon = canonical_of.get(other).cloned().unwrap_or_else(|| other.clone());
                if best.as_ref().map_or(true, |(c, _)| canon < *c) {
                    best = Some((canon, jacc));
                }
            }
        }
        match best {
            Some((canon, jacc)) => {
                canonical_of.insert(sha.clone(), canon.clone());
                out.push((sha.clone(), canon, jacc));
            }
            None => {
                canonical_of.insert(sha.clone(), sha.clone());
            }
        }
    }
    out
}

/// LSH global decisions: same contract as `exact_decisions`, but
/// pairs are proposed by MinHash/LSH banding and verified with
/// exact Jaccard. Decisions therefore match the exact path except
/// for pairs LSH fails to propose (recall loss, see module docs).
/// Inputs must be SHA-sorted (same requirement as exact_decisions).
/// Returns (demoted_sha, canonical_sha, jaccard, candidates_seen).
pub fn lsh_decisions(
    docs: &[(String, HashSet<String>)],
    threshold: f64,
    perms: usize,
    bands: usize,
    rows: usize,
) -> (Vec<(String, String, f64)>, usize) {
    let params = hash_params(perms);
    let sigs: Vec<(String, Vec<u64>)> = docs
        .iter()
        .map(|(sha, set)| (sha.clone(), signature(set, &params)))
        .collect();
    let candidates = candidate_pairs(&sigs, bands, rows);
    let by_sha: HashMap<String, &HashSet<String>> =
        docs.iter().map(|(sha, set)| (sha.clone(), set)).collect();
    // Verified pairs (sha, canonical-candidate, jaccard), keyed by pair.
    let mut verified: HashMap<(String, String), f64> = HashMap::new();
    for (a, b) in &candidates {
        let (sa, sb) = (by_sha[a], by_sha[b]);
        if sa.is_empty() || sb.is_empty() {
            continue;
        }
        let jacc = crate::tokenizer::jaccard(sa, sb);
        if jacc >= threshold {
            verified.insert((a.clone(), b.clone()), jacc);
        }
    }
    // Same greedy SHA-order canonical assignment as exact_decisions,
    // restricted to verified pairs.
    let mut canonical_of: HashMap<String, String> = HashMap::new();
    let mut out = Vec::new();
    for (sha, set) in docs {
        if set.is_empty() {
            canonical_of.insert(sha.clone(), sha.clone());
            continue;
        }
        let mut best: Option<(String, f64)> = None;
        for (other, oset) in docs {
            if other >= sha || oset.is_empty() {
                continue;
            }
            // Candidate pairs are stored (lower-sha, higher-sha).
            let key = (other.clone(), sha.clone());
            if let Some(jacc) = verified.get(&key) {
                let canon = canonical_of.get(other).cloned().unwrap_or_else(|| other.clone());
                if best.as_ref().map_or(true, |(c, _)| canon < *c) {
                    best = Some((canon, *jacc));
                }
            }
        }
        match best {
            Some((canon, jacc)) => {
                canonical_of.insert(sha.clone(), canon.clone());
                out.push((sha.clone(), canon, jacc));
            }
            None => {
                canonical_of.insert(sha.clone(), sha.clone());
            }
        }
    }
    (out, candidates.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(words: &[&str]) -> HashSet<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn signature_deterministic_and_estimate_tracks() {
        let params = hash_params(128);
        // Two sets with true Jaccard 0.9: 90 shared + 10 unique each.
        let mut a: HashSet<String> = (0..90).map(|i| format!("s{}", i)).collect();
        let mut b = a.clone();
        for i in 90..100 {
            a.insert(format!("a{}", i));
            b.insert(format!("b{}", i));
        }
        let sa = signature(&a, &params);
        let sb = signature(&b, &params);
        assert_eq!(sa, signature(&a, &params), "signatures must be deterministic");
        let est = estimated_jaccard(&sa, &sb);
        let truth = 90.0 / 110.0;
        assert!(
            (est - truth).abs() < 0.12,
            "estimate {} vs true {}",
            est,
            truth
        );
        // Disjoint sets: estimate near 0.
        let c: HashSet<String> = (0..100).map(|i| format!("c{}", i)).collect();
        let sc = signature(&c, &params);
        assert!(estimated_jaccard(&sa, &sc) < 0.12);
    }

    #[test]
    fn lsh_banding_identical_and_disjoint() {
        let params = hash_params(128);
        let a = set(&["x", "y", "z"]);
        let docs = vec![
            ("aaa".to_string(), signature(&a, &params)),
            ("bbb".to_string(), signature(&a, &params)),
            ("ccc".to_string(), signature(&set(&["p", "q", "r"]), &params)),
        ];
        let pairs = candidate_pairs(&docs, 16, 8);
        assert!(pairs.contains(&("aaa".to_string(), "bbb".to_string())), "identical sets must share bands: {:?}", pairs);
        assert!(!pairs.iter().any(|(x, y)| x == "ccc" || y == "ccc"), "disjoint sets must not be candidates: {:?}", pairs);
    }

    #[test]
    fn lsh_matches_exact_on_fixture_groups() {
        // Two near-dup groups (Jaccard ~0.95 within group) + distinct docs.
        let mut docs: Vec<(String, HashSet<String>)> = Vec::new();
        for g in 0..2 {
            for v in 0..3 {
                let mut s: HashSet<String> = (0..40).map(|i| format!("g{}-{}", g, i)).collect();
                s.insert(format!("g{}-unique-{}", g, v));
                docs.push((format!("doc-{}-{}", g, v), s));
            }
        }
        docs.push(("doc-outlier".to_string(), set(&["o1", "o2", "o3", "o4", "o5"])));
        docs.sort_by(|a, b| a.0.cmp(&b.0));
        let exact = exact_decisions(&docs, 0.9);
        let (lsh, candidates) = lsh_decisions(&docs, 0.9, 128, 16, 8);
        assert!(candidates > 0, "no candidates proposed");
        assert_eq!(
            lsh, exact,
            "LSH decisions must match exact decisions on fixtures\nlsh: {:?}\nexact: {:?}",
            lsh, exact
        );
        // Each group demotes 2 of 3 to the lowest SHA in the group.
        assert_eq!(exact.len(), 4);
    }

    #[test]
    fn empty_sets_never_candidates() {
        let params = hash_params(128);
        let docs = vec![
            ("aaa".to_string(), signature(&HashSet::new(), &params)),
            ("bbb".to_string(), signature(&HashSet::new(), &params)),
        ];
        assert!(candidate_pairs(&docs, 16, 8).is_empty());
        let (decisions, _) = lsh_decisions(
            &[
                ("aaa".to_string(), HashSet::new()),
                ("bbb".to_string(), HashSet::new()),
            ],
            0.9,
            128,
            16,
            8,
        );
        assert!(decisions.is_empty(), "empty docs must not demote each other");
    }
}
