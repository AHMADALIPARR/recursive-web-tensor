//! RWT engine: ingest, prune, train, evaluate, reduce, checkpoint.
//!
//! State layout under the working root (`--root`, default `.`):
//!   data/shards/<sha256>.json   content-addressed documents
//!   data/vocab.json             reference-tokenizer vocabulary
//!   state/registry.json         ingest index (or registry-shard-N)
//!   state/prune.json            latest prune dispositions
//!   state/shards/shard-N.json   per-shard prune outputs (CI jobs)
//!   state/reduced.json          merged retained set + Merkle root
//!   state/model-next.json       candidate model (train output)
//!   state/evaluation.json       gate metrics (evaluate output)
//!   state/current.json          promoted checkpoint pointer
//!   state/validation.json       loss trend driving adaptive tau
//!   checkpoints/<version>/      manifest.json (+ model.json if promoted)
//!
//! Only the publish chain writes checkpoints. Shard jobs write only
//! their own shard outputs; the reduce step merges them.

use crate::bpe::Bpe;
use crate::fetch;
use crate::neural;
use crate::json::{self, Value};
use crate::scoring::{self, Weights};
use crate::sha256;
use crate::tokenizer;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------- time

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------- config

#[derive(Debug, Clone)]
pub struct Config {
    pub weights: Weights,
    pub tau0: f64,
    pub tau_min: f64,
    pub tau_max: f64,
    pub tau_step: f64,
    pub top_k_per_category: usize,
    pub gate_tolerance: f64,
    pub allowed_licenses: Vec<String>,
    pub denied_licenses: Vec<String>,
    pub ngram_order: usize,
    pub near_dup_threshold: f64,
    pub shingle_n: usize,
    pub tokenizer: String,
    pub bpe_merges: String,
    pub trainer: String,
    pub neural_epochs: usize,
    pub neural_lr: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            weights: Weights::default(),
            tau0: 0.30,
            tau_min: 0.15,
            tau_max: 0.50,
            tau_step: 0.02,
            top_k_per_category: 2,
            gate_tolerance: 0.10,
            allowed_licenses: vec![
                "public-domain".into(),
                "CC0".into(),
                "CC-BY".into(),
                "CC-BY-SA".into(),
                "MIT".into(),
                "Apache-2.0".into(),
                "BSD".into(),
                "BSD-2-Clause".into(),
                "BSD-3-Clause".into(),
            ],
            denied_licenses: vec!["proprietary".into(), "all-rights-reserved".into(), "no-license".into()],
            ngram_order: 3,
            near_dup_threshold: 0.9,
            shingle_n: 5,
            tokenizer: "reference".to_string(),
            bpe_merges: "state/bpe.json".to_string(),
            trainer: "neural".to_string(),
            neural_epochs: 100,
            neural_lr: 0.01,
        }
    }
}

fn get_f64(v: &Value, key: &str, fallback: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(fallback)
}

fn get_usize(v: &Value, key: &str, fallback: usize) -> usize {
    v.get(key).and_then(|x| x.as_i64()).map(|n| n.max(0) as usize).unwrap_or(fallback)
}

fn get_str_list(v: &Value, key: &str, fallback: Vec<String>) -> Vec<String> {
    match v.get(key).and_then(|x| x.as_arr()) {
        Some(arr) => arr.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(),
        None => fallback,
    }
}

pub fn load_config(root: &Path) -> Result<Config, String> {
    let path = root.join("config.json");
    if !path.exists() {
        return Ok(Config::default());
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    let v = json::parse(&text).map_err(|e| format!("bad config.json: {}", e))?;
    let d = Config::default();
    Ok(Config {
        weights: Weights {
            alpha: get_f64(&v, "alpha", d.weights.alpha),
            beta: get_f64(&v, "beta", d.weights.beta),
            gamma: get_f64(&v, "gamma", d.weights.gamma),
            delta: get_f64(&v, "delta", d.weights.delta),
        },
        tau0: get_f64(&v, "tau0", d.tau0),
        tau_min: get_f64(&v, "tau_min", d.tau_min),
        tau_max: get_f64(&v, "tau_max", d.tau_max),
        tau_step: get_f64(&v, "tau_step", d.tau_step),
        top_k_per_category: get_usize(&v, "top_k_per_category", d.top_k_per_category),
        gate_tolerance: get_f64(&v, "gate_tolerance", d.gate_tolerance),
        allowed_licenses: get_str_list(&v, "allowed_licenses", d.allowed_licenses),
        denied_licenses: get_str_list(&v, "denied_licenses", d.denied_licenses),
        ngram_order: get_usize(&v, "ngram_order", d.ngram_order).max(1),
        near_dup_threshold: get_f64(&v, "near_dup_threshold", d.near_dup_threshold),
        shingle_n: get_usize(&v, "shingle_n", d.shingle_n).max(1),
        tokenizer: v
            .get("tokenizer")
            .and_then(|x| x.as_str())
            .unwrap_or(&d.tokenizer)
            .to_string(),
        bpe_merges: v
            .get("bpe_merges")
            .and_then(|x| x.as_str())
            .unwrap_or(&d.bpe_merges)
            .to_string(),
        trainer: v
            .get("trainer")
            .and_then(|x| x.as_str())
            .unwrap_or(&d.trainer)
            .to_string(),
        neural_epochs: get_usize(&v, "neural_epochs", d.neural_epochs).max(1),
        neural_lr: get_f64(&v, "neural_lr", d.neural_lr),
    })
}

// ---------------------------------------------------------------- docs

#[derive(Debug, Clone)]
pub struct Doc {
    pub sha: String,
    pub source: String,
    pub category: String,
    pub license: String,
    pub fetched_at: u64,
    pub text: String,
    pub tokens: Vec<String>,
}

pub fn shard_of(sha: &str, num_shards: usize) -> usize {
    if num_shards == 0 {
        return 0;
    }
    let first = sha.get(0..2).and_then(|h| u8::from_str_radix(h, 16).ok()).unwrap_or(0);
    (first as usize) % num_shards
}

fn ensure_dir(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|e| format!("cannot create {}: {}", path.display(), e))
}

fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        ensure_dir(parent)?;
    }
    let mut text = json::to_string(v);
    text.push('\n');
    std::fs::write(path, text).map_err(|e| format!("cannot write {}: {}", path.display(), e))
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    json::parse(&text).map_err(|e| format!("bad JSON in {}: {}", path.display(), e))
}

pub fn doc_to_json(d: &Doc) -> Value {
    let mut o = Value::obj();
    o.put("sha256", Value::s(&d.sha));
    o.put("source", Value::s(&d.source));
    o.put("category", Value::s(&d.category));
    o.put("license", Value::s(&d.license));
    o.put("fetched_at", Value::n(d.fetched_at as i64));
    o.put("text", Value::s(&d.text));
    let toks: Vec<Value> = d.tokens.iter().map(|t| Value::s(t)).collect();
    o.put("tokens", Value::Arr(toks));
    o
}

pub fn doc_from_json(v: &Value) -> Result<Doc, String> {
    let req = |k: &str| -> Result<String, String> {
        v.get(k)
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| format!("doc missing string field '{}'", k))
    };
    let tokens = v
        .get("tokens")
        .and_then(|x| x.as_arr())
        .map(|arr| arr.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_else(Vec::new);
    Ok(Doc {
        sha: req("sha256")?,
        source: req("source")?,
        category: v.get("category").and_then(|x| x.as_str()).unwrap_or("unknown").to_string(),
        license: req("license")?,
        fetched_at: v.get("fetched_at").and_then(|x| x.as_i64()).unwrap_or(0) as u64,
        text: req("text")?,
        tokens,
    })
}

/// Load every content-addressed document under data/shards, sorted
/// by SHA-256 ascending (the deterministic processing order).
pub fn load_docs(root: &Path) -> Result<Vec<Doc>, String> {
    let dir = root.join("data/shards");
    let mut docs = Vec::new();
    if !dir.exists() {
        return Ok(docs);
    }
    let mut names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(|e| format!("cannot list {}: {}", dir.display(), e))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".json") {
            names.push(name);
        }
    }
    names.sort();
    for name in names {
        let v = read_json(&dir.join(&name))?;
        docs.push(doc_from_json(&v)?);
    }
    docs.sort_by(|a, b| a.sha.cmp(&b.sha));
    Ok(docs)
}

/// Tokenizer mode for ingest. Reference tokenizes to plain word
/// strings; BPE tokenizes to hex strings of the learned symbols
/// (unambiguous for every downstream string-based stage).
pub enum TokMode {
    Reference,
    Bpe(Bpe),
}

impl TokMode {
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        match self {
            TokMode::Reference => tokenizer::tokenize(text),
            TokMode::Bpe(b) => b.encode_hex(text),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            TokMode::Reference => "reference",
            TokMode::Bpe(_) => "bpe",
        }
    }
    /// BPE vocabulary in id order, if this mode is BPE.
    pub fn vocab_hex(&self) -> Option<Vec<String>> {
        match self {
            TokMode::Bpe(b) => Some(b.vocab_hex()),
            TokMode::Reference => None,
        }
    }
}

/// Resolve the effective tokenizer: CLI flag wins, then config.
/// BPE fails closed when the merges file is missing — silently
/// falling back to the reference tokenizer would corrupt token-id
/// comparability across epochs.
pub fn resolve_tok_mode(
    root: &Path,
    config: &Config,
    flag: Option<String>,
    merges_flag: Option<String>,
) -> Result<TokMode, String> {
    let name = flag.unwrap_or_else(|| config.tokenizer.clone());
    match name.as_str() {
        "reference" => Ok(TokMode::Reference),
        "bpe" => {
            let path = match merges_flag {
                Some(p) => PathBuf::from(p),
                None => root.join(&config.bpe_merges),
            };
            let bpe = Bpe::load_file(&path)?;
            Ok(TokMode::Bpe(bpe))
        }
        other => Err(format!(
            "unknown tokenizer '{}' (want 'reference' or 'bpe')",
            other
        )),
    }
}

fn make_doc(source: &str, category: &str, license: &str, text: &str, mode: &TokMode) -> Doc {
    let sha = sha256::hex(text.as_bytes());
    let tokens = mode.tokenize(text);
    Doc {
        sha,
        source: source.to_string(),
        category: category.to_string(),
        license: license.to_string(),
        fetched_at: now_secs(),
        text: text.to_string(),
        tokens,
    }
}

// ---------------------------------------------------------------- ingest

pub struct IngestArgs {
    pub corpus: Option<String>,
    pub url_manifest: Option<String>,
    pub cc_index: Option<String>,
    pub ddg_query: Option<String>,
    pub shard: Option<usize>,
    pub num_shards: usize,
    pub tokenizer: Option<String>,
    pub bpe_merges: Option<String>,
    pub config: Config,
}

/// Ingest documents from a local corpus directory, a URL manifest,
/// and optionally one Common Crawl index lookup and one DuckDuckGo
/// discovery query. Returns (ingested_new, duplicates, errors).
pub fn cmd_ingest(root: &Path, a: &IngestArgs) -> Result<(usize, usize, usize), String> {
    let mode = resolve_tok_mode(root, &a.config, a.tokenizer.clone(), a.bpe_merges.clone())?;
    let mut docs: Vec<Doc> = Vec::new();
    let mut errors: Vec<Value> = Vec::new();

    if let Some(corpus) = &a.corpus {
        let dir = Path::new(corpus);
        if !dir.exists() {
            eprintln!("warning: corpus dir {} missing, ingesting zero local docs", corpus);
        } else {
            let mut files: Vec<PathBuf> = Vec::new();
            for entry in std::fs::read_dir(dir)
                .map_err(|e| format!("cannot list corpus {}: {}", corpus, e))?
            {
                let entry = entry.map_err(|e| e.to_string())?;
                let p = entry.path();
                if p.extension().map(|e| e == "txt").unwrap_or(false) {
                    files.push(p);
                }
            }
            files.sort();
            for p in files {
                let text = std::fs::read_to_string(&p)
                    .map_err(|e| format!("cannot read {}: {}", p.display(), e))?;
                let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                docs.push(make_doc(&format!("local:{}", name), "local-corpus", "public-domain", &text, &mode));
            }
        }
    }

    if let Some(manifest) = &a.url_manifest {
        let text = std::fs::read_to_string(manifest)
            .map_err(|e| format!("cannot read manifest {}: {}", manifest, e))?;
        let v = json::parse(&text).map_err(|e| format!("bad manifest JSON: {}", e))?;
        let items: Vec<&Value> = match &v {
            Value::Arr(arr) => arr.iter().collect(),
            Value::Obj(_) => v.get("documents").and_then(|x| x.as_arr()).map(|arr| arr.iter().collect()).unwrap_or_default(),
            _ => Vec::new(),
        };
        for item in items {
            let url = match item.get("url").and_then(|x| x.as_str()) {
                Some(u) => u.to_string(),
                None => continue,
            };
            let source = item.get("source").and_then(|x| x.as_str()).unwrap_or(&url).to_string();
            let category = item.get("category").and_then(|x| x.as_str()).unwrap_or("url-manifest").to_string();
            let license = item.get("license").and_then(|x| x.as_str()).unwrap_or("unknown").to_string();
            match fetch::curl_get(&url, 30, 2_000_000) {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    docs.push(make_doc(&source, &category, &license, &text, &mode));
                }
                Err(e) => {
                    let mut err = Value::obj();
                    err.put("url", Value::s(&url));
                    err.put("error", Value::s(&e));
                    errors.push(err);
                }
            }
        }
    }

    if let Some(index_url) = &a.cc_index {
        // Index lookup only: fetch the index page, parse records,
        // persist the manifest. WARC fetching is a later milestone.
        let bytes = fetch::curl_get(index_url, 30, 1_000_000)?;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let (records, skipped) = fetch::parse_cc_index(&text);
        let mut m = Value::obj();
        m.put("index_url", Value::s(index_url));
        m.put("records", Value::n(records.len() as i64));
        m.put("skipped_lines", Value::n(skipped as i64));
        let recs: Vec<Value> = records
            .iter()
            .map(|r| {
                let mut o = Value::obj();
                o.put("url", Value::s(&r.url));
                o.put("filename", Value::s(&r.filename));
                o.put("offset", Value::n(r.offset));
                o.put("length", Value::n(r.length));
                o
            })
            .collect();
        m.put("items", Value::Arr(recs));
        write_json(&root.join("state/cc-manifest.json"), &m)?;
        println!("cc-index: {} records, {} skipped lines -> state/cc-manifest.json", records.len(), skipped);
    }

    if let Some(query) = &a.ddg_query {
        let url = format!("https://html.duckduckgo.com/html/?q={}", fetch::percent_encode(query));
        let bytes = fetch::curl_get(&url, 30, 1_000_000)?;
        let html = String::from_utf8_lossy(&bytes).into_owned();
        let urls = fetch::extract_ddg_urls(&html);
        let mut m = Value::obj();
        m.put("query", Value::s(query));
        let list: Vec<Value> = urls.iter().map(|u| Value::s(u)).collect();
        m.put("urls", Value::Arr(list));
        write_json(&root.join("state/ddg-manifest.json"), &m)?;
        println!("ddg-discovery: {} result urls -> state/ddg-manifest.json", urls.len());
    }

    // Dedup by SHA-256 (exact duplicates collapse to one shard file).
    let mut seen: HashMap<String, Doc> = HashMap::new();
    let mut duplicates = 0usize;
    for d in docs {
        if seen.contains_key(&d.sha) {
            duplicates += 1;
        } else {
            seen.insert(d.sha.clone(), d);
        }
    }
    let mut ordered: Vec<Doc> = seen.into_values().collect();
    ordered.sort_by(|a, b| a.sha.cmp(&b.sha));

    // Vocabulary: in BPE mode the trained vocab in id order; in
    // reference mode, first-seen order over SHA-sorted docs.
    let vocab: Vec<String> = match mode.vocab_hex() {
        Some(v) => v,
        None => {
            let token_lists: Vec<Vec<String>> =
                ordered.iter().map(|d| d.tokens.clone()).collect();
            tokenizer::build_vocab(&token_lists)
        }
    };
    let mut vocab_json = Value::obj();
    vocab_json.put("tokens", Value::Arr(vocab.iter().map(|t| Value::s(t)).collect()));
    write_json(&root.join("data/vocab.json"), &vocab_json)?;

    let shard_dir = root.join("data/shards");
    ensure_dir(&shard_dir)?;
    let mut registry_docs: Vec<Value> = Vec::new();
    let mut written = 0usize;
    for d in &ordered {
        let shard = shard_of(&d.sha, a.num_shards);
        if let Some(want) = a.shard {
            if shard != want {
                continue;
            }
        }
        write_json(&shard_dir.join(format!("{}.json", d.sha)), &doc_to_json(d))?;
        let mut entry = Value::obj();
        entry.put("sha256", Value::s(&d.sha));
        entry.put("source", Value::s(&d.source));
        entry.put("category", Value::s(&d.category));
        entry.put("license", Value::s(&d.license));
        entry.put("shard", Value::n(shard as i64));
        entry.put("token_count", Value::n(d.tokens.len() as i64));
        registry_docs.push(entry);
        written += 1;
    }
    let mut registry = Value::obj();
    registry.put("tokenizer", Value::s(mode.name()));
    registry.put("documents", Value::Arr(registry_docs));
    registry.put("duplicates", Value::n(duplicates as i64));
    registry.put("total_unique", Value::n(ordered.len() as i64));
    let registry_name = match a.shard {
        Some(n) => format!("registry-shard-{}.json", n),
        None => "registry.json".to_string(),
    };
    write_json(&root.join("state").join(registry_name), &registry)?;
    let err_count = errors.len();
    if !errors.is_empty() {
        let mut e = Value::obj();
        e.put("errors", Value::Arr(errors));
        write_json(&root.join("state/ingest-errors.json"), &e)?;
    }
    println!(
        "ingest: {} unique docs, {} written (shard filter {:?}), {} exact duplicates, {} fetch errors",
        ordered.len(),
        written,
        a.shard,
        duplicates,
        err_count
    );
    Ok((written, duplicates, err_count))
}

// ---------------------------------------------------------------- prune

#[derive(Debug, Clone, PartialEq)]
pub enum Disposition {
    Keep,
    Compress,
    Quarantine,
    Reject,
}

impl Disposition {
    pub fn as_str(&self) -> &'static str {
        match self {
            Disposition::Keep => "Keep",
            Disposition::Compress => "Compress",
            Disposition::Quarantine => "Quarantine",
            Disposition::Reject => "Reject",
        }
    }
}

fn printable_ratio(text: &str) -> f64 {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return 0.0;
    }
    let ok = chars.iter().filter(|c| !c.is_control() || **c == '\n' || **c == '\t' || **c == '\r').count();
    ok as f64 / chars.len() as f64
}

/// Adaptive threshold from the validation trend. Tightens by one step
/// when validation improved, loosens by one step when it degraded,
/// clamped to [tau_min, tau_max]. No validation file: tau0.
pub fn adaptive_tau(config: &Config, validation: Option<&Value>) -> (f64, String) {
    match validation {
        Some(v) => {
            let loss = v.get("loss").and_then(|x| x.as_f64());
            let prev = v.get("prev_loss").and_then(|x| x.as_f64());
            match (loss, prev) {
                (Some(l), Some(p)) if l < p => {
                    ((config.tau0 + config.tau_step).min(config.tau_max), "tightened: validation improved".to_string())
                }
                (Some(l), Some(p)) if l > p => {
                    ((config.tau0 - config.tau_step).max(config.tau_min), "loosened: validation degraded".to_string())
                }
                _ => (config.tau0, "unchanged: no comparable trend".to_string()),
            }
        }
        None => (config.tau0, "base: no validation history".to_string()),
    }
}

pub struct PruneArgs {
    pub shard: Option<usize>,
    pub num_shards: usize,
    pub max_depth: usize,
    pub tau_override: Option<f64>,
}

pub struct PruneOutput {
    pub json: Value,
    pub retained_shas: Vec<String>,
}

/// Prune the loaded corpus. Writes state/prune.json (or, with a shard
/// filter, state/shards/shard-N.json) and returns the output.
pub fn cmd_prune(root: &Path, config: &Config, a: &PruneArgs) -> Result<PruneOutput, String> {
    let all_docs = load_docs(root)?;
    let docs: Vec<Doc> = match a.shard {
        Some(n) => all_docs.into_iter().filter(|d| shard_of(&d.sha, a.num_shards) == n).collect(),
        None => all_docs,
    };
    let validation = read_json(&root.join("state/validation.json")).ok();
    let (mut tau, adjustment) = adaptive_tau(config, validation.as_ref());
    if let Some(t) = a.tau_override {
        tau = t;
    }
    let _ = adjustment;

    // License + validity gate first: denied -> Reject, unknown license
    // or invalid content -> Quarantine (never trained on).
    let mut dispositions: Vec<Value> = Vec::new();
    let mut counts = [0usize; 4]; // Keep, Compress, Quarantine, Reject
    let mut candidates: Vec<Doc> = Vec::new();
    for d in &docs {
        let disp = if config.denied_licenses.iter().any(|x| x == &d.license) {
            Disposition::Reject
        } else if !config.allowed_licenses.iter().any(|x| x == &d.license) {
            Disposition::Quarantine
        } else if d.tokens.is_empty() || printable_ratio(&d.text) < 0.5 {
            Disposition::Quarantine
        } else {
            candidates.push(d.clone());
            continue;
        };
        let mut o = Value::obj();
        o.put("sha256", Value::s(&d.sha));
        o.put("disposition", Value::s(disp.as_str()));
        o.put("category", Value::s(&d.category));
        dispositions.push(o);
        match disp {
            Disposition::Quarantine => counts[2] += 1,
            Disposition::Reject => counts[3] += 1,
            _ => {}
        }
    }

    // Shingle sets for candidates, then near-dup compression:
    // candidates are SHA-sorted, so the canonical of a near-dup group
    // is deterministically its lowest SHA.
    let mut shingle_sets: HashMap<String, HashSet<String>> = HashMap::new();
    for d in &candidates {
        shingle_sets.insert(d.sha.clone(), tokenizer::shingles(&d.tokens, config.shingle_n));
    }
    let mut canonicals: Vec<Doc> = Vec::new();
    let mut canonical_sets: Vec<HashSet<String>> = Vec::new();
    for d in &candidates {
        let set = &shingle_sets[&d.sha];
        let mut alias: Option<String> = None;
        for (i, cset) in canonical_sets.iter().enumerate() {
            if tokenizer::jaccard(set, cset) >= config.near_dup_threshold {
                alias = Some(canonicals[i].sha.clone());
                break;
            }
        }
        match alias {
            Some(target) => {
                let mut o = Value::obj();
                o.put("sha256", Value::s(&d.sha));
                o.put("disposition", Value::s(Disposition::Compress.as_str()));
                o.put("alias_of", Value::s(&target));
                o.put("category", Value::s(&d.category));
                dispositions.push(o);
                counts[1] += 1;
            }
            None => {
                canonicals.push(d.clone());
                canonical_sets.push(set.clone());
            }
        }
    }

    // Score canonicals. max_j is against the other canonicals.
    let mut scored: Vec<(Doc, scoring::Score)> = Vec::new();
    for (i, d) in canonicals.iter().enumerate() {
        let others: Vec<HashSet<String>> = canonical_sets
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, s)| s.clone())
            .collect();
        let max_j = scoring::max_jaccard(&shingle_sets[&d.sha], &others);
        scored.push((d.clone(), scoring::score(&d.text, &d.tokens, max_j, config.weights)));
    }

    // Top-k per category are always kept (representative samples),
    // then the adaptive threshold decides the rest.
    let mut keep: HashSet<String> = HashSet::new();
    let mut by_category: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    for (d, s) in &scored {
        by_category.entry(d.category.clone()).or_default().push((d.sha.clone(), s.total));
    }
    for (_, mut items) in by_category {
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        for (sha, _) in items.into_iter().take(config.top_k_per_category) {
            keep.insert(sha);
        }
    }
    let mut retained: Vec<Value> = Vec::new();
    for (d, s) in &scored {
        let kept = keep.contains(&d.sha) || s.total >= tau;
        let mut o = Value::obj();
        o.put("sha256", Value::s(&d.sha));
        o.put("disposition", Value::s(if kept { Disposition::Keep.as_str() } else { Disposition::Reject.as_str() }));
        o.put("score", Value::f(s.total));
        o.put("quality", Value::f(s.quality));
        o.put("novelty", Value::f(s.novelty));
        o.put("density", Value::f(s.density));
        o.put("redundancy", Value::f(s.redundancy));
        o.put("category", Value::s(&d.category));
        dispositions.push(o);
        if kept {
            counts[0] += 1;
            let mut r = Value::obj();
            r.put("sha256", Value::s(&d.sha));
            r.put("token_count", Value::n(d.tokens.len() as i64));
            retained.push(r);
        } else {
            counts[3] += 1;
        }
    }

    let mut out = Value::obj();
    match a.shard {
        Some(n) => out.put("shard", Value::n(n as i64)),
        None => out.put("shard", Value::Null),
    }
    out.put("tau", Value::f(tau));
    out.put("max_depth", Value::n(a.max_depth as i64));
    let mut c = Value::obj();
    c.put("Keep", Value::n(counts[0] as i64));
    c.put("Compress", Value::n(counts[1] as i64));
    c.put("Quarantine", Value::n(counts[2] as i64));
    c.put("Reject", Value::n(counts[3] as i64));
    out.put("counts", c);
    out.put("dispositions", Value::Arr(dispositions));
    out.put("retained", Value::Arr(retained.clone()));
    let path = match a.shard {
        Some(n) => root.join("state/shards").join(format!("shard-{}.json", n)),
        None => root.join("state/prune.json"),
    };
    write_json(&path, &out)?;
    let retained_shas: Vec<String> = retained
        .iter()
        .filter_map(|r| r.get("sha256").and_then(|x| x.as_str()).map(|s| s.to_string()))
        .collect();
    println!(
        "prune: tau={:.3} Keep={} Compress={} Quarantine={} Reject={} -> {}",
        tau,
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        path.display()
    );
    Ok(PruneOutput { json: out, retained_shas })
}

// ---------------------------------------------------------------- reduce

/// Follow Compress alias chains through the merged disposition map,
/// up to `max_depth` hops. Returns (resolved, unresolved).
pub fn resolve_aliases(
    alias_of: &HashMap<String, String>,
    max_depth: usize,
) -> (HashMap<String, String>, Vec<String>) {
    let mut resolved = HashMap::new();
    let mut unresolved = Vec::new();
    for sha in alias_of.keys() {
        let mut current = sha.clone();
        let mut hops = 0usize;
        let mut target: Option<String> = None;
        while hops < max_depth {
            match alias_of.get(&current) {
                Some(next) => {
                    current = next.clone();
                    hops += 1;
                    if !alias_of.contains_key(&current) {
                        target = Some(current.clone());
                        break;
                    }
                }
                None => break,
            }
        }
        match target {
            Some(t) => {
                resolved.insert(sha.clone(), t);
            }
            None => unresolved.push(sha.clone()),
        }
    }
    (resolved, unresolved)
}

pub struct ReduceArgs {
    pub max_depth: usize,
}

/// Merge per-shard prune outputs (or the single prune output) into
/// one retained set with a recomputed Merkle root.
pub fn cmd_reduce(root: &Path, a: &ReduceArgs) -> Result<Value, String> {
    let shard_dir = root.join("state/shards");
    let mut shard_files: Vec<PathBuf> = Vec::new();
    if shard_dir.exists() {
        for entry in std::fs::read_dir(&shard_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let p = entry.path();
            if p.extension().map(|e| e == "json").unwrap_or(false) {
                shard_files.push(p);
            }
        }
    }
    shard_files.sort();
    let mut retained_map: HashMap<String, (i64, Vec<i64>)> = HashMap::new();
    let mut alias_of: HashMap<String, String> = HashMap::new();
    let mut shard_counts: BTreeMap<String, i64> = BTreeMap::new();
    if shard_files.is_empty() {
        let prune = read_json(&root.join("state/prune.json"))?;
        let retained = prune.get("retained").and_then(|x| x.as_arr()).cloned().unwrap_or_default();
        for r in &retained {
            if let Some(sha) = r.get("sha256").and_then(|x| x.as_str()) {
                let tc = r.get("token_count").and_then(|x| x.as_i64()).unwrap_or(0);
                retained_map.insert(sha.to_string(), (tc, vec![]));
            }
        }
        if let Some(disps) = prune.get("dispositions").and_then(|x| x.as_arr()) {
            for d in disps {
                if d.get("disposition").and_then(|x| x.as_str()) == Some("Compress") {
                    if let (Some(sha), Some(target)) = (
                        d.get("sha256").and_then(|x| x.as_str()),
                        d.get("alias_of").and_then(|x| x.as_str()),
                    ) {
                        alias_of.insert(sha.to_string(), target.to_string());
                    }
                }
            }
        }
    } else {
        for p in &shard_files {
            let v = read_json(p)?;
            let shard_id = v.get("shard").and_then(|x| x.as_i64()).unwrap_or(-1);
            let retained = v.get("retained").and_then(|x| x.as_arr()).cloned().unwrap_or_default();
            shard_counts.insert(format!("{}", shard_id), retained.len() as i64);
            for r in &retained {
                if let Some(sha) = r.get("sha256").and_then(|x| x.as_str()) {
                    let tc = r.get("token_count").and_then(|x| x.as_i64()).unwrap_or(0);
                    retained_map
                        .entry(sha.to_string())
                        .and_modify(|e| {
                            if !e.1.contains(&shard_id) {
                                e.1.push(shard_id);
                            }
                        })
                        .or_insert((tc, vec![shard_id]));
                }
            }
            if let Some(disps) = v.get("dispositions").and_then(|x| x.as_arr()) {
                for d in disps {
                    if d.get("disposition").and_then(|x| x.as_str()) == Some("Compress") {
                        if let (Some(sha), Some(target)) = (
                            d.get("sha256").and_then(|x| x.as_str()),
                            d.get("alias_of").and_then(|x| x.as_str()),
                        ) {
                            alias_of.insert(sha.to_string(), target.to_string());
                        }
                    }
                }
            }
        }
    }
    let mut retained_shas: Vec<String> = retained_map.keys().cloned().collect();
    retained_shas.sort();
    let merkle = sha256::merkle_root(retained_shas.clone());
    let (resolved, unresolved) = resolve_aliases(&alias_of, a.max_depth);
    let mut retained: Vec<Value> = Vec::new();
    let mut total_tokens = 0i64;
    for sha in &retained_shas {
        let (tc, shards) = &retained_map[sha];
        total_tokens += *tc;
        let mut r = Value::obj();
        r.put("sha256", Value::s(sha));
        r.put("token_count", Value::n(*tc));
        r.put("shards", Value::Arr(shards.iter().map(|s| Value::n(*s)).collect()));
        retained.push(r);
    }
    let mut out = Value::obj();
    out.put("retained", Value::Arr(retained));
    out.put("merkle_root", Value::s(&merkle));
    out.put("total_tokens", Value::n(total_tokens));
    out.put("max_depth", Value::n(a.max_depth as i64));
    let mut sc = Value::obj();
    for (k, v) in &shard_counts {
        sc.put(k, Value::n(*v));
    }
    out.put("shard_counts", sc);
    let mut ra = Value::obj();
    for (k, v) in &resolved {
        ra.put(k, Value::s(v));
    }
    out.put("resolved_aliases", ra);
    out.put(
        "unresolved_aliases",
        Value::Arr(unresolved.iter().map(|s| Value::s(s)).collect()),
    );
    write_json(&root.join("state/reduced.json"), &out)?;
    println!(
        "reduce: {} retained docs, {} tokens, merkle_root={} -> state/reduced.json",
        retained_shas.len(),
        total_tokens,
        merkle
    );
    Ok(out)
}

// ---------------------------------------------------------------- train

pub fn heldout(sha: &str) -> bool {
    let first = sha.get(0..2).and_then(|h| u8::from_str_radix(h, 16).ok()).unwrap_or(0);
    first % 5 == 0
}

pub struct TrainArgs {
    pub order: usize,
    pub steps: usize, // 0 = unlimited; otherwise max retained docs consumed
    pub trainer: Option<String>,
    pub epochs: usize, // 0 = config default (neural only)
    pub lr: f64,       // 0.0 = config default (neural only)
}

/// Train dispatch: neural (default) or n-gram. Both write
/// state/model-next.json through the same checkpoint/gating
/// interface. An unknown trainer name is an error, never a silent
/// fallback to the other trainer.
pub fn cmd_train(root: &Path, config: &Config, a: &TrainArgs) -> Result<Value, String> {
    let trainer = a.trainer.clone().unwrap_or_else(|| config.trainer.clone());
    match trainer.as_str() {
        "neural" => cmd_train_neural(root, config, a),
        "ngram" => cmd_train_ngram(root, config, a),
        other => Err(format!("unknown trainer '{}' (want 'neural' or 'ngram')", other)),
    }
}

/// Train the reference n-gram model: merge prior checkpoint counts
/// (count-space replay: old knowledge persists) with new counts from
/// retained, non-held-out docs. Writes state/model-next.json.
pub fn cmd_train_ngram(root: &Path, config: &Config, a: &TrainArgs) -> Result<Value, String> {
    let order = if a.order == 0 { config.ngram_order } else { a.order };
    let reduced = read_json(&root.join("state/reduced.json"))
        .or_else(|_| read_json(&root.join("state/prune.json")))?;
    let retained = reduced.get("retained").and_then(|x| x.as_arr()).cloned().unwrap_or_default();
    let mut shas: Vec<String> = retained
        .iter()
        .filter_map(|r| r.get("sha256").and_then(|x| x.as_str()).map(|s| s.to_string()))
        .collect();
    shas.sort();
    let docs = load_docs(root)?;
    let by_sha: HashMap<String, Doc> = docs.into_iter().map(|d| (d.sha.clone(), d)).collect();

    // Prior model (replay): counts from the promoted checkpoint.
    let mut counts: HashMap<String, HashMap<String, i64>> = HashMap::new();
    let mut unigram: HashMap<String, i64> = HashMap::new();
    let mut total_tokens: i64 = 0;
    let mut prior_docs: Vec<String> = Vec::new();
    if let Ok(current) = read_json(&root.join("state/current.json")) {
        if let Some(ver) = current.get("version").and_then(|x| x.as_str()) {
            let model_path = root.join("checkpoints").join(ver).join("model.json");
            if let Ok(model) = read_json(&model_path) {
                if let Some(c) = model.get("counts").and_then(|x| x.as_arr()) {
                    for entry in c {
                        let ctx = entry.get("context").and_then(|x| x.as_str()).unwrap_or("").to_string();
                        let mut m = HashMap::new();
                        if let Some(next) = entry.get("next").and_then(|x| x.as_arr()) {
                            for pair in next {
                                if let (Some(tok), Some(n)) = (
                                    pair.get("token").and_then(|x| x.as_str()),
                                    pair.get("count").and_then(|x| x.as_i64()),
                                ) {
                                    m.insert(tok.to_string(), n);
                                }
                            }
                        }
                        counts.insert(ctx, m);
                    }
                }
                if let Some(u) = model.get("unigram").and_then(|x| x.as_arr()) {
                    for pair in u {
                        if let (Some(tok), Some(n)) = (
                            pair.get("token").and_then(|x| x.as_str()),
                            pair.get("count").and_then(|x| x.as_i64()),
                        ) {
                            unigram.insert(tok.to_string(), n);
                        }
                    }
                }
                // Only inherit the token denominator when the prior
                // model actually carries n-gram counts. A neural
                // prior's total_tokens counts doc tokens, not
                // n-gram events; adopting it would halve every
                // unigram probability (counts sum to the new epoch
                // only, denominator to both epochs).
                if model.get("counts").and_then(|x| x.as_arr()).is_some() {
                    total_tokens = model.get("total_tokens").and_then(|x| x.as_i64()).unwrap_or(0);
                }
                if let Some(ds) = model.get("doc_shas").and_then(|x| x.as_arr()) {
                    prior_docs = ds.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect();
                }
            }
        }
    }

    let mut trained_docs: Vec<String> = Vec::new();
    let mut new_tokens: i64 = 0;
    let limit = if a.steps == 0 { usize::MAX } else { a.steps };
    for sha in &shas {
        if trained_docs.len() >= limit {
            break;
        }
        if heldout(sha) {
            continue; // held-out split is never trained on
        }
        let doc = match by_sha.get(sha) {
            Some(d) => d,
            None => continue,
        };
        let mut padded: Vec<String> = vec!["<s>".to_string(); order - 1];
        padded.extend(doc.tokens.iter().cloned());
        for window in padded.windows(order) {
            let (ctx, next) = window.split_at(order - 1);
            let key = ctx.join("\u{1f}");
            let next_tok = next[0].clone();
            *counts.entry(key).or_default().entry(next_tok.clone()).or_insert(0) += 1;
            *unigram.entry(next_tok).or_insert(0) += 1;
            total_tokens += 1;
            new_tokens += 1;
        }
        trained_docs.push(sha.clone());
    }
    let mut all_docs = prior_docs.clone();
    for sha in &trained_docs {
        if !all_docs.contains(sha) {
            all_docs.push(sha.clone());
        }
    }
    all_docs.sort();

    let vocab_size = match read_json(&root.join("data/vocab.json")) {
        Ok(v) => v.get("tokens").and_then(|x| x.as_arr()).map(|a| a.len() as i64).unwrap_or(unigram.len() as i64),
        Err(_) => unigram.len() as i64,
    }
    .max(unigram.len() as i64);

    let mut model = Value::obj();
    model.put("order", Value::n(order as i64));
    let mut count_arr: Vec<Value> = Vec::new();
    let mut ctx_keys: Vec<String> = counts.keys().cloned().collect();
    ctx_keys.sort();
    for ctx in ctx_keys {
        let mut entry = Value::obj();
        entry.put("context", Value::s(&ctx));
        let mut next_arr: Vec<Value> = Vec::new();
        let mut toks: Vec<String> = counts[&ctx].keys().cloned().collect();
        toks.sort();
        for tok in toks {
            let mut pair = Value::obj();
            pair.put("token", Value::s(&tok));
            pair.put("count", Value::n(counts[&ctx][&tok]));
            next_arr.push(pair);
        }
        entry.put("next", Value::Arr(next_arr));
        count_arr.push(entry);
    }
    model.put("counts", Value::Arr(count_arr));
    let mut uni_arr: Vec<Value> = Vec::new();
    let mut utoks: Vec<String> = unigram.keys().cloned().collect();
    utoks.sort();
    for tok in utoks {
        let mut pair = Value::obj();
        pair.put("token", Value::s(&tok));
        pair.put("count", Value::n(unigram[&tok]));
        uni_arr.push(pair);
    }
    model.put("unigram", Value::Arr(uni_arr));
    model.put("total_tokens", Value::n(total_tokens));
    model.put("vocab_size", Value::n(vocab_size));
    model.put("trained_docs", Value::n(trained_docs.len() as i64));
    model.put("doc_shas", Value::Arr(all_docs.iter().map(|s| Value::s(s)).collect()));
    write_json(&root.join("state/model-next.json"), &model)?;
    println!(
        "train: order={} new_docs={} new_tokens={} total_tokens={} -> state/model-next.json",
        order,
        trained_docs.len(),
        new_tokens,
        total_tokens
    );
    Ok(model)
}

/// Train the neural model (see neural.rs for the exact
/// architecture): from scratch, or init-from-prior-checkpoint when
/// a prior neural checkpoint exists, plus replay of the prior
/// checkpoint's training docs. Held-out docs are never trained on.
/// Writes state/model-next.json with kind "neural".
pub fn cmd_train_neural(root: &Path, config: &Config, a: &TrainArgs) -> Result<Value, String> {
    let epochs = if a.epochs == 0 { config.neural_epochs } else { a.epochs };
    let lr = if a.lr == 0.0 { config.neural_lr } else { a.lr };
    let reduced = read_json(&root.join("state/reduced.json"))
        .or_else(|_| read_json(&root.join("state/prune.json")))?;
    let retained = reduced.get("retained").and_then(|x| x.as_arr()).cloned().unwrap_or_default();
    let mut shas: Vec<String> = retained
        .iter()
        .filter_map(|r| r.get("sha256").and_then(|x| x.as_str()).map(|s| s.to_string()))
        .collect();
    shas.sort();
    let docs = load_docs(root)?;
    let by_sha: HashMap<String, Doc> = docs.into_iter().map(|d| (d.sha.clone(), d)).collect();

    // Prior neural checkpoint, if any (weights + replay doc list).
    let mut prior_vocab: Option<Vec<String>> = None;
    let mut prior_params: Option<Vec<f64>> = None;
    let mut prior_docs: Vec<String> = Vec::new();
    if let Ok(current) = read_json(&root.join("state/current.json")) {
        if let Some(ver) = current.get("version").and_then(|x| x.as_str()) {
            let model_path = root.join("checkpoints").join(ver).join("model.json");
            if let Ok(model) = read_json(&model_path) {
                if let Ok((pv, pp)) = neural::model_from_json(&model) {
                    prior_vocab = Some(pv);
                    prior_params = Some(pp);
                }
                if let Some(ds) = model.get("doc_shas").and_then(|x| x.as_arr()) {
                    prior_docs = ds.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect();
                }
            }
        }
    }

    // New docs: retained, non-held-out, steps-limited.
    let limit = if a.steps == 0 { usize::MAX } else { a.steps };
    let mut new_docs: Vec<String> = Vec::new();
    for sha in &shas {
        if new_docs.len() >= limit {
            break;
        }
        if heldout(sha) {
            continue;
        }
        if by_sha.contains_key(sha) {
            new_docs.push(sha.clone());
        }
    }
    // Replay docs: prior training docs still present, non-held-out.
    let mut train_shas = new_docs.clone();
    for sha in &prior_docs {
        if !heldout(sha) && by_sha.contains_key(sha) && !train_shas.contains(sha) {
            train_shas.push(sha.clone());
        }
    }
    train_shas.sort();

    // Model vocabulary: "<s>" first, then data/vocab.json order,
    // then any training-doc token missing from it (first-seen over
    // sorted shas). Deterministic; guarantees no unknown tokens in
    // the training set.
    let mut vocab: Vec<String> = vec!["<s>".to_string()];
    if let Ok(v) = read_json(&root.join("data/vocab.json")) {
        if let Some(toks) = v.get("tokens").and_then(|x| x.as_arr()) {
            for t in toks {
                if let Some(s) = t.as_str() {
                    if !vocab.contains(&s.to_string()) {
                        vocab.push(s.to_string());
                    }
                }
            }
        }
    }
    for sha in &train_shas {
        for tok in &by_sha[sha].tokens {
            if !vocab.contains(tok) {
                vocab.push(tok.clone());
            }
        }
    }
    let index: HashMap<String, usize> =
        vocab.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
    let mut seqs: Vec<Vec<usize>> = Vec::new();
    let mut total_tokens: i64 = 0;
    for sha in &train_shas {
        let doc = &by_sha[sha];
        total_tokens += doc.tokens.len() as i64;
        let (ids, _) = neural::tokens_to_ids(&index, &doc.tokens);
        seqs.push(ids);
    }
    let examples = neural::examples_from_ids(&seqs);
    let layout = neural::Layout::new(vocab.len());
    let params0 = match (&prior_vocab, &prior_params) {
        (Some(pv), Some(pp)) => neural::init_from_prior(&vocab, pv, pp, neural::SEED),
        _ => neural::init_params(vocab.len(), neural::SEED),
    };
    let init_loss = neural::mean_loss(&params0, &layout, &examples);
    let params = neural::train_adam(params0, &layout, &examples, epochs, lr, 0.9, 0.999, 1e-8);
    let final_loss = neural::mean_loss(&params, &layout, &examples);
    let model = neural::model_to_json(&vocab, &params, train_shas.len(), &train_shas, total_tokens);
    write_json(&root.join("state/model-next.json"), &model)?;
    println!(
        "train: trainer=neural epochs={} lr={} new_docs={} replay_docs={} params={} train_loss={:.4} (init {:.4}) -> state/model-next.json",
        epochs,
        lr,
        new_docs.len(),
        train_shas.len() - new_docs.len(),
        layout.total,
        final_loss,
        init_loss
    );
    Ok(model)
}

// ---------------------------------------------------------------- evaluate

pub struct EvaluateArgs {
    pub model_path: Option<String>,
    pub probe_dir: Option<String>,
}

struct EvalModel {
    order: usize,
    counts: HashMap<String, HashMap<String, i64>>,
    unigram: HashMap<String, i64>,
    total_tokens: i64,
    vocab_size: i64,
}

fn load_eval_model(v: &Value) -> EvalModel {
    let order = v.get("order").and_then(|x| x.as_i64()).unwrap_or(3).max(1) as usize;
    let mut counts = HashMap::new();
    if let Some(c) = v.get("counts").and_then(|x| x.as_arr()) {
        for entry in c {
            let ctx = entry.get("context").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let mut m = HashMap::new();
            if let Some(next) = entry.get("next").and_then(|x| x.as_arr()) {
                for pair in next {
                    if let (Some(tok), Some(n)) = (
                        pair.get("token").and_then(|x| x.as_str()),
                        pair.get("count").and_then(|x| x.as_i64()),
                    ) {
                        m.insert(tok.to_string(), n);
                    }
                }
            }
            counts.insert(ctx, m);
        }
    }
    let mut unigram = HashMap::new();
    if let Some(u) = v.get("unigram").and_then(|x| x.as_arr()) {
        for pair in u {
            if let (Some(tok), Some(n)) = (
                pair.get("token").and_then(|x| x.as_str()),
                pair.get("count").and_then(|x| x.as_i64()),
            ) {
                unigram.insert(tok.to_string(), n);
            }
        }
    }
    EvalModel {
        order,
        counts,
        unigram,
        total_tokens: v.get("total_tokens").and_then(|x| x.as_i64()).unwrap_or(0),
        vocab_size: v.get("vocab_size").and_then(|x| x.as_i64()).unwrap_or(1).max(1),
    }
}

/// Mean per-token cross-entropy (nats) with backoff smoothing:
/// context counts, else unigram, else uniform over the vocabulary.
fn cross_entropy(model: &EvalModel, docs_tokens: &[Vec<String>]) -> (f64, i64) {
    let alpha = 0.1f64;
    let v = model.vocab_size as f64;
    let mut loss_sum = 0.0f64;
    let mut ntokens = 0i64;
    for tokens in docs_tokens {
        let mut padded: Vec<String> = vec!["<s>".to_string(); model.order - 1];
        padded.extend(tokens.iter().cloned());
        for window in padded.windows(model.order) {
            let (ctx, next) = window.split_at(model.order - 1);
            let key = ctx.join("\u{1f}");
            let tok = &next[0];
            let p = match model.counts.get(&key) {
                Some(nexts) => {
                    let total: i64 = nexts.values().sum();
                    let c = nexts.get(tok).copied().unwrap_or(0) as f64;
                    (c + alpha) / (total as f64 + alpha * v)
                }
                None => match model.unigram.get(tok) {
                    Some(c) => (*c as f64 + alpha) / (model.total_tokens as f64 + alpha * v),
                    None => 1.0 / v,
                },
            };
            loss_sum += -p.ln();
            ntokens += 1;
        }
    }
    if ntokens == 0 {
        (0.0, 0)
    } else {
        (loss_sum / ntokens as f64, ntokens)
    }
}

/// Tokenize probe text with the run's tokenizer: the registry's
/// "tokenizer" field when present (BPE needs the merges file and
/// fails closed without it), reference when no registry exists
/// (e.g. shard-job contexts that never ran ingest locally).
fn probe_tokenize(root: &Path, config: &Config, text: &str) -> Result<Vec<String>, String> {
    let name = read_json(&root.join("state/registry.json"))
        .ok()
        .and_then(|r| r.get("tokenizer").and_then(|x| x.as_str()).map(|s| s.to_string()))
        .unwrap_or_else(|| "reference".to_string());
    match name.as_str() {
        "bpe" => {
            let bpe = Bpe::load_file(&root.join(&config.bpe_merges))?;
            Ok(bpe.encode_hex(text))
        }
        _ => Ok(tokenizer::tokenize(text)),
    }
}

/// Evaluate a neural candidate model on the held-out split and the
/// probe set. Same output shape as the n-gram path, so the
/// checkpoint gate is unchanged across trainers.
fn cmd_evaluate_neural(
    root: &Path,
    config: &Config,
    a: &EvaluateArgs,
    model_json: &Value,
) -> Result<Value, String> {
    let (vocab, params) = neural::model_from_json(model_json)?;
    let layout = neural::Layout::new(vocab.len());
    let index: HashMap<String, usize> =
        vocab.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
    let reduced = read_json(&root.join("state/reduced.json"))
        .or_else(|_| read_json(&root.join("state/prune.json")))?;
    let retained = reduced.get("retained").and_then(|x| x.as_arr()).cloned().unwrap_or_default();
    let shas: Vec<String> = retained
        .iter()
        .filter_map(|r| r.get("sha256").and_then(|x| x.as_str()).map(|s| s.to_string()))
        .collect();
    let docs = load_docs(root)?;
    let by_sha: HashMap<String, Doc> = docs.into_iter().map(|d| (d.sha.clone(), d)).collect();
    let mut held_seqs: Vec<Vec<usize>> = Vec::new();
    let mut held_unknown = 0usize;
    for s in shas.iter().filter(|s| heldout(s)) {
        if let Some(d) = by_sha.get(s) {
            let (ids, unk) = neural::tokens_to_ids(&index, &d.tokens);
            held_seqs.push(ids);
            held_unknown += unk;
        }
    }
    let (heldout_loss, heldout_n) = neural::neural_cross_entropy(&params, &layout, &held_seqs, held_unknown);
    let (probe_loss, probe_n) = eval_probe_neural(root, config, a, &index, &params, &layout)?;
    let mut out = Value::obj();
    if heldout_n > 0 {
        out.put("heldout_loss", Value::f(heldout_loss));
        out.put("perplexity", Value::f(heldout_loss.exp()));
    } else {
        out.put("heldout_loss", Value::Null);
        out.put("perplexity", Value::Null);
    }
    out.put("heldout_tokens", Value::n(heldout_n));
    out.put("probe_loss", probe_loss);
    out.put("probe_tokens", Value::n(probe_n));
    write_json(&root.join("state/evaluation.json"), &out)?;
    println!(
        "evaluate: trainer=neural heldout_loss={} perplexity={} probe_loss={} -> state/evaluation.json",
        match out.get("heldout_loss").and_then(|x| x.as_f64()) {
            Some(l) => format!("{:.4}", l),
            None => "null (no held-out docs)".to_string(),
        },
        match out.get("perplexity").and_then(|x| x.as_f64()) {
            Some(p) => format!("{:.2}", p),
            None => "null".to_string(),
        },
        match out.get("probe_loss").and_then(|x| x.as_f64()) {
            Some(l) => format!("{:.4}", l),
            None => "null".to_string(),
        }
    );
    Ok(out)
}

fn eval_probe_neural(
    root: &Path,
    config: &Config,
    a: &EvaluateArgs,
    index: &HashMap<String, usize>,
    params: &[f64],
    layout: &neural::Layout,
) -> Result<(Value, i64), String> {
    let probe_dir = match &a.probe_dir {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            let p = root.join("probe");
            if p.exists() {
                Some(p)
            } else {
                None
            }
        }
    };
    let mut probe_loss = Value::Null;
    let mut probe_n = 0i64;
    if let Some(dir) = probe_dir {
        let mut files: Vec<PathBuf> = Vec::new();
        if dir.exists() {
            for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                let p = entry.path();
                if p.extension().map(|e| e == "txt").unwrap_or(false) {
                    files.push(p);
                }
            }
        }
        files.sort();
        let mut seqs: Vec<Vec<usize>> = Vec::new();
        let mut unknown = 0usize;
        for p in files {
            let text = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
            let toks = probe_tokenize(root, config, &text)?;
            let (ids, unk) = neural::tokens_to_ids(index, &toks);
            seqs.push(ids);
            unknown += unk;
        }
        let (loss, n) = neural::neural_cross_entropy(params, layout, &seqs, unknown);
        probe_loss = Value::f(loss);
        probe_n = n;
    }
    Ok((probe_loss, probe_n))
}

/// Evaluate the candidate model on the held-out split and the probe
/// set. Writes state/evaluation.json. Dispatches on model kind:
/// neural models take the neural path, anything else the n-gram
/// path (the original format carries no kind field).
pub fn cmd_evaluate(root: &Path, config: &Config, a: &EvaluateArgs) -> Result<Value, String> {
    let model_path = match &a.model_path {
        Some(p) => PathBuf::from(p),
        None => root.join("state/model-next.json"),
    };
    let model_json = read_json(&model_path)?;
    if model_json.get("kind").and_then(|x| x.as_str()) == Some("neural") {
        return cmd_evaluate_neural(root, config, a, &model_json);
    }
    let model = load_eval_model(&model_json);
    let reduced = read_json(&root.join("state/reduced.json"))
        .or_else(|_| read_json(&root.join("state/prune.json")))?;
    let retained = reduced.get("retained").and_then(|x| x.as_arr()).cloned().unwrap_or_default();
    let shas: Vec<String> = retained
        .iter()
        .filter_map(|r| r.get("sha256").and_then(|x| x.as_str()).map(|s| s.to_string()))
        .collect();
    let docs = load_docs(root)?;
    let by_sha: HashMap<String, Doc> = docs.into_iter().map(|d| (d.sha.clone(), d)).collect();
    let heldout_tokens: Vec<Vec<String>> = shas
        .iter()
        .filter(|s| heldout(s))
        .filter_map(|s| by_sha.get(s).map(|d| d.tokens.clone()))
        .collect();
    let (heldout_loss, heldout_n) = cross_entropy(&model, &heldout_tokens);

    // Probe set: explicit dir, else <root>/probe if it exists.
    let probe_dir = match &a.probe_dir {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            let p = root.join("probe");
            if p.exists() {
                Some(p)
            } else {
                None
            }
        }
    };
    let mut probe_loss = Value::Null;
    let mut probe_n = 0i64;
    if let Some(dir) = probe_dir {
        let mut files: Vec<PathBuf> = Vec::new();
        if dir.exists() {
            for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                let p = entry.path();
                if p.extension().map(|e| e == "txt").unwrap_or(false) {
                    files.push(p);
                }
            }
        }
        files.sort();
        let mut probe_docs: Vec<Vec<String>> = Vec::new();
        for p in files {
            let text = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
            probe_docs.push(probe_tokenize(root, config, &text)?);
        }
        let (loss, n) = cross_entropy(&model, &probe_docs);
        probe_loss = Value::f(loss);
        probe_n = n;
    }

    let mut out = Value::obj();
    if heldout_n > 0 {
        out.put("heldout_loss", Value::f(heldout_loss));
        out.put("perplexity", Value::f(heldout_loss.exp()));
    } else {
        out.put("heldout_loss", Value::Null);
        out.put("perplexity", Value::Null);
    }
    out.put("heldout_tokens", Value::n(heldout_n));
    out.put("probe_loss", probe_loss);
    out.put("probe_tokens", Value::n(probe_n));
    write_json(&root.join("state/evaluation.json"), &out)?;
    println!(
        "evaluate: heldout_loss={} perplexity={} probe_loss={} -> state/evaluation.json",
        match out.get("heldout_loss").and_then(|x| x.as_f64()) {
            Some(l) => format!("{:.4}", l),
            None => "null (no held-out docs)".to_string(),
        },
        match out.get("perplexity").and_then(|x| x.as_f64()) {
            Some(p) => format!("{:.2}", p),
            None => "null".to_string(),
        },
        match out.get("probe_loss").and_then(|x| x.as_f64()) {
            Some(l) => format!("{:.4}", l),
            None => "null".to_string(),
        }
    );
    Ok(out)
}

// ---------------------------------------------------------------- checkpoint

/// Write the checkpoint for the candidate model. Promotion is gated:
/// the new held-out loss must not exceed the promoted loss plus the
/// configured tolerance. A rejected epoch is recorded (manifest with
/// status "rejected" + state/rejected.json) and current.json stands.
pub fn cmd_checkpoint(root: &Path, config: &Config) -> Result<Value, String> {
    let model_next = root.join("state/model-next.json");
    if !model_next.exists() {
        return Err("no candidate model: run train first (state/model-next.json missing)".to_string());
    }
    let evaluation = read_json(&root.join("state/evaluation.json"))?;
    let reduced = read_json(&root.join("state/reduced.json"))
        .or_else(|_| read_json(&root.join("state/prune.json")))?;
    let retained = reduced.get("retained").and_then(|x| x.as_arr()).cloned().unwrap_or_default();
    let retained_shas: Vec<String> = retained
        .iter()
        .filter_map(|r| r.get("sha256").and_then(|x| x.as_str()).map(|s| s.to_string()))
        .collect();
    let recomputed = sha256::merkle_root(retained_shas.clone());
    if let Some(stored) = reduced.get("merkle_root").and_then(|x| x.as_str()) {
        if stored != recomputed {
            return Err(format!(
                "merkle mismatch: reduced.json says {} but recomputation gives {}",
                stored, recomputed
            ));
        }
    }
    let total_tokens = reduced.get("total_tokens").and_then(|x| x.as_i64()).unwrap_or(0);

    let current = read_json(&root.join("state/current.json")).ok();
    let (version, parent, parent_sha) = match &current {
        Some(c) => {
            let ver = c.get("version").and_then(|x| x.as_str()).unwrap_or("v0").to_string();
            let num: i64 = ver.trim_start_matches('v').parse().unwrap_or(0);
            let parent_manifest = root.join("checkpoints").join(&ver).join("manifest.json");
            let parent_bytes = std::fs::read(&parent_manifest)
                .map_err(|e| format!("cannot read parent manifest {}: {}", parent_manifest.display(), e))?;
            let recomputed_parent = sha256::hex(&parent_bytes);
            let stored_parent = c.get("manifest_sha256").and_then(|x| x.as_str()).unwrap_or("");
            if stored_parent != recomputed_parent {
                return Err(format!(
                    "checkpoint chain broken: current.json records {} but {} hashes to {}",
                    stored_parent,
                    parent_manifest.display(),
                    recomputed_parent
                ));
            }
            (format!("v{}", num + 1), Some(ver), Some(recomputed_parent))
        }
        None => ("v1".to_string(), None, None),
    };

    let new_loss = evaluation.get("heldout_loss").and_then(|x| x.as_f64());
    let prev_loss = current
        .as_ref()
        .and_then(|c| c.get("heldout_loss"))
        .and_then(|x| x.as_f64());
    let (promoted, reason) = match (new_loss, prev_loss) {
        (Some(n), Some(p)) => {
            if n <= p + config.gate_tolerance {
                (true, "held-out loss within tolerance".to_string())
            } else {
                (false, format!("held-out loss {:.4} exceeds {:.4} + tolerance {:.4}", n, p, config.gate_tolerance))
            }
        }
        (Some(_), None) => (true, "first checkpoint (no prior loss)".to_string()),
        (None, _) => (false, "no held-out docs: gate cannot verify, epoch rejected".to_string()),
    };

    let dir = root.join("checkpoints").join(&version);
    ensure_dir(&dir)?;
    let mut manifest = Value::obj();
    manifest.put("version", Value::s(&version));
    manifest.put("status", Value::s(if promoted { "promoted" } else { "rejected" }));
    match &parent {
        Some(p) => manifest.put("parent", Value::s(p)),
        None => manifest.put("parent", Value::Null),
    }
    match &parent_sha {
        Some(s) => manifest.put("parent_manifest_sha256", Value::s(s)),
        None => manifest.put("parent_manifest_sha256", Value::Null),
    }
    manifest.put("merkle_root", Value::s(&recomputed));
    manifest.put("retained_count", Value::n(retained_shas.len() as i64));
    manifest.put("total_tokens", Value::n(total_tokens));
    manifest.put("timestamp", Value::n(now_secs() as i64));
    manifest.put("metrics", evaluation.clone());
    manifest.put("gate_reason", Value::s(&reason));
    write_json(&dir.join("manifest.json"), &manifest)?;

    if promoted {
        std::fs::copy(&model_next, dir.join("model.json"))
            .map_err(|e| format!("cannot copy model: {}", e))?;
        let manifest_sha = sha256::hex(&std::fs::read(dir.join("manifest.json")).map_err(|e| e.to_string())?);
        let mut cur = Value::obj();
        cur.put("version", Value::s(&version));
        cur.put("manifest_sha256", Value::s(&manifest_sha));
        if let Some(l) = new_loss {
            cur.put("heldout_loss", Value::f(l));
        } else {
            cur.put("heldout_loss", Value::Null);
        }
        cur.put("merkle_root", Value::s(&recomputed));
        write_json(&root.join("state/current.json"), &cur)?;
        let mut validation = Value::obj();
        match prev_loss {
            Some(p) => validation.put("prev_loss", Value::f(p)),
            None => validation.put("prev_loss", Value::Null),
        }
        match new_loss {
            Some(l) => validation.put("loss", Value::f(l)),
            None => validation.put("loss", Value::Null),
        }
        write_json(&root.join("state/validation.json"), &validation)?;
        println!("checkpoint: {} PROMOTED ({}) -> checkpoints/{}/", version, reason, version);
    } else {
        let mut rej = Value::obj();
        rej.put("version", Value::s(&version));
        rej.put("reason", Value::s(&reason));
        rej.put("metrics", evaluation.clone());
        write_json(&root.join("state/rejected.json"), &rej)?;
        println!("checkpoint: {} REJECTED ({}) -- prior checkpoint stands", version, reason);
    }
    Ok(manifest)
}

// ---------------------------------------------------------------- cycle

pub struct CycleArgs {
    pub corpus: Option<String>,
    pub url_manifest: Option<String>,
    pub probe_dir: Option<String>,
    pub order: usize,
    pub max_depth: usize,
    pub steps: usize,
    pub tokenizer: Option<String>,
    pub bpe_merges: Option<String>,
    pub trainer: Option<String>,
    pub epochs: usize,
    pub lr: f64,
}

/// Run the full bounded cycle locally: ingest -> prune -> reduce ->
/// train -> evaluate -> checkpoint. One epoch, over persistent state.
/// Recursion happens across epochs (repeated runs), never inside an
/// infinitely running job.
pub fn cmd_cycle(root: &Path, config: &Config, a: &CycleArgs) -> Result<Value, String> {
    let (_, _, _) = cmd_ingest(
        root,
        &IngestArgs {
            corpus: a.corpus.clone(),
            url_manifest: a.url_manifest.clone(),
            cc_index: None,
            ddg_query: None,
            shard: None,
            num_shards: 4,
            tokenizer: a.tokenizer.clone(),
            bpe_merges: a.bpe_merges.clone(),
            config: config.clone(),
        },
    )?;
    let prune_out = cmd_prune(
        root,
        config,
        &PruneArgs { shard: None, num_shards: 4, max_depth: a.max_depth, tau_override: None },
    )?;
    let tau = prune_out.json.get("tau").and_then(|x| x.as_f64()).unwrap_or(0.0);
    println!("cycle: prune retained {} docs at tau {:.3}", prune_out.retained_shas.len(), tau);
    cmd_reduce(root, &ReduceArgs { max_depth: a.max_depth })?;
    cmd_train(
        root,
        config,
        &TrainArgs { order: a.order, steps: a.steps, trainer: a.trainer.clone(), epochs: a.epochs, lr: a.lr },
    )?;
    cmd_evaluate(
        root,
        config,
        &EvaluateArgs { model_path: None, probe_dir: a.probe_dir.clone() },
    )?;
    let manifest = cmd_checkpoint(root, config)?;
    let mut summary = Value::obj();
    summary.put("checkpoint", manifest);
    println!("cycle: epoch complete");
    Ok(summary)
}

// ---------------------------------------------------------------- tokenizer

/// Train BPE merges from a corpus directory of .txt files and write
/// the merges+vocab file (default <root>/state/bpe.json).
pub fn cmd_tokenizer_train(
    root: &Path,
    corpus: &str,
    num_merges: usize,
    out: Option<String>,
) -> Result<Bpe, String> {
    let dir = Path::new(corpus);
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("cannot list corpus {}: {}", corpus, e))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let p = entry.path();
        if p.extension().map(|e| e == "txt").unwrap_or(false) {
            files.push(p);
        }
    }
    files.sort();
    let mut texts: Vec<String> = Vec::new();
    for p in &files {
        texts.push(
            std::fs::read_to_string(p).map_err(|e| format!("cannot read {}: {}", p.display(), e))?,
        );
    }
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let bpe = Bpe::train(&refs, num_merges);
    let out_path = match out {
        Some(p) => PathBuf::from(p),
        None => root.join("state/bpe.json"),
    };
    write_json(&out_path, &bpe.to_json())?;
    println!(
        "tokenizer train-bpe: {} docs, {} merges, vocab {} -> {}",
        texts.len(),
        bpe.merges.len(),
        bpe.vocab.len(),
        out_path.display()
    );
    Ok(bpe)
}

/// Encode one text with a trained merges file; prints hex tokens.
pub fn cmd_tokenizer_encode(merges_path: &str, text: &str) -> Result<Vec<String>, String> {
    let bpe = Bpe::load_file(Path::new(merges_path))?;
    let tokens = bpe.encode_hex(text);
    println!("{} tokens: {}", tokens.len(), tokens.join(" "));
    Ok(tokens)
}

/// Decode space-separated hex tokens with a trained merges file;
/// prints the reconstructed text.
pub fn cmd_tokenizer_decode(merges_path: &str, tokens: &str) -> Result<String, String> {
    let bpe = Bpe::load_file(Path::new(merges_path))?;
    let toks: Vec<String> = tokens.split_whitespace().map(|s| s.to_string()).collect();
    let text = bpe.decode_hex(&toks);
    println!("{}", text);
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_root(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("rwt-test-{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_corpus(root: &Path, docs: &[(&str, &str)]) {
        let dir = root.join("corpus");
        fs::create_dir_all(&dir).unwrap();
        for (name, text) in docs {
            fs::write(dir.join(name), text).unwrap();
        }
    }

    #[test]
    fn adaptive_tau_rules() {
        let c = Config::default();
        assert_eq!(adaptive_tau(&c, None).0, 0.30);
        let mut v = Value::obj();
        v.put("loss", Value::f(1.0));
        v.put("prev_loss", Value::f(1.5));
        assert!((adaptive_tau(&c, Some(&v)).0 - 0.32).abs() < 1e-12);
        v.put("loss", Value::f(2.0));
        assert!((adaptive_tau(&c, Some(&v)).0 - 0.28).abs() < 1e-12);
    }

    #[test]
    fn prune_dispositions_all_four() {
        let root = tmp_root("prune");
        // Good doc, near-dup of it, unknown-license doc, denied doc.
        let good = [
            "The river flows through the valley while morning mist hangs over the water.",
            "Mountains rise beyond the trees and snow covers the highest peaks in winter.",
            "Farmers plant wheat in dark soil and wait for rain through the dry season.",
            "Old sailors watch the tides and mend their nets before the evening storm arrives.",
        ]
        .join(" ");
        write_corpus(
            &root,
            &[
                ("a.txt", good.as_str()),
                // Near-dup of a.txt (one extra token): shingle Jaccard ~0.97, above the 0.9 bar,
                // but not an exact duplicate, so ingest does not collapse it.
                ("b.txt", &format!("{} Fin.", good)),
                ("c.txt", "Quantum entanglement links particles across vast distances in strange ways."),
            ],
        );
        cmd_ingest(
            &root,
            &IngestArgs { corpus: Some(root.join("corpus").to_string_lossy().into_owned()), url_manifest: None, cc_index: None, ddg_query: None, shard: None, num_shards: 4, tokenizer: None, bpe_merges: None, config: Config::default() },
        )
        .unwrap();
        // Force licenses: rewrite shard files for c (unknown) and add a denied doc.
        let docs = load_docs(&root).unwrap();
        for d in &docs {
            if d.source.contains("c.txt") {
                let mut dd = d.clone();
                dd.license = "unknown".to_string();
                write_json(&root.join("data/shards").join(format!("{}.json", dd.sha)), &doc_to_json(&dd)).unwrap();
            }
        }
        let denied_text = "A completely different document about cooking rice and beans with garlic and oil.";
        let mut denied = make_doc("local:denied.txt", "local-corpus", "proprietary", denied_text, &TokMode::Reference);
        denied.tokens = tokenizer::tokenize(denied_text);
        write_json(&root.join("data/shards").join(format!("{}.json", denied.sha)), &doc_to_json(&denied)).unwrap();

        let cfg = Config::default();
        let out = cmd_prune(&root, &cfg, &PruneArgs { shard: None, num_shards: 4, max_depth: 8, tau_override: None }).unwrap();
        let disps = out.json.get("dispositions").and_then(|x| x.as_arr()).unwrap();
        let kinds: HashSet<String> = disps
            .iter()
            .filter_map(|d| d.get("disposition").and_then(|x| x.as_str()).map(|s| s.to_string()))
            .collect();
        // Keep (good doc + top-k), Compress (near-dup), Quarantine (unknown license), Reject (denied).
        assert!(kinds.contains("Keep"), "kinds: {:?}", kinds);
        assert!(kinds.contains("Compress"), "kinds: {:?}", kinds);
        assert!(kinds.contains("Quarantine"), "kinds: {:?}", kinds);
        assert!(kinds.contains("Reject"), "kinds: {:?}", kinds);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn alias_resolution_depth() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), "b".to_string());
        m.insert("b".to_string(), "c".to_string());
        m.insert("c".to_string(), "d".to_string());
        let (resolved, unresolved) = resolve_aliases(&m, 8);
        assert_eq!(resolved.get("a"), Some(&"d".to_string()));
        let (_, unresolved2) = resolve_aliases(&m, 2);
        assert!(unresolved2.contains(&"a".to_string()));
        assert!(unresolved.is_empty());
    }

    #[test]
    fn gate_rejects_regression_and_chain_verifies() {
        let root = tmp_root("gate");
        write_corpus(
            &root,
            &[
                ("a.txt", "The river flows through the valley while morning mist hangs over the water and birds sing."),
                ("b.txt", "Mountains rise beyond the trees and snow covers the highest peaks through the winter months."),
                ("c.txt", "To cook rice well, rinse the grains until the water runs clear, then soak them before cooking."),
                ("d.txt", "Astronomers study the light from distant stars to learn what they are made of and how far away."),
                ("e.txt", "The ancient city kept records on clay tablets, pressing wedge-shaped marks into wet clay."),
                ("f.txt", "Ocean currents carry warm water from the equator toward the poles and cold water back again."),
            ],
        );
        let cfg = Config::default();
        cmd_cycle(
            &root,
            &cfg,
            &CycleArgs {
                corpus: Some(root.join("corpus").to_string_lossy().into_owned()),
                url_manifest: None,
                probe_dir: None,
                order: 0,
                max_depth: 8,
                steps: 0,
                tokenizer: None,
                bpe_merges: None,
                trainer: None,
                epochs: 0,
                lr: 0.0,
            },
        )
        .unwrap();
        let cur = read_json(&root.join("state/current.json")).unwrap();
        assert_eq!(cur.get("version").and_then(|x| x.as_str()), Some("v1"));
        let v1_loss = cur.get("heldout_loss").and_then(|x| x.as_f64()).unwrap();

        // Forged terrible evaluation: gate must reject, v1 must stand.
        let mut ev = Value::obj();
        ev.put("heldout_loss", Value::f(v1_loss + 5.0));
        ev.put("perplexity", Value::f(999.0));
        ev.put("heldout_tokens", Value::n(10));
        ev.put("probe_loss", Value::Null);
        ev.put("probe_tokens", Value::n(0));
        write_json(&root.join("state/evaluation.json"), &ev).unwrap();
        let m = cmd_checkpoint(&root, &cfg).unwrap();
        assert_eq!(m.get("status").and_then(|x| x.as_str()), Some("rejected"));
        let cur2 = read_json(&root.join("state/current.json")).unwrap();
        assert_eq!(cur2.get("version").and_then(|x| x.as_str()), Some("v1"));

        // Good evaluation: promotes to v2 with a verified parent hash.
        let mut ev2 = Value::obj();
        ev2.put("heldout_loss", Value::f(v1_loss));
        ev2.put("perplexity", Value::f(50.0));
        ev2.put("heldout_tokens", Value::n(10));
        ev2.put("probe_loss", Value::Null);
        ev2.put("probe_tokens", Value::n(0));
        write_json(&root.join("state/evaluation.json"), &ev2).unwrap();
        let m2 = cmd_checkpoint(&root, &cfg).unwrap();
        assert_eq!(m2.get("status").and_then(|x| x.as_str()), Some("promoted"));
        assert_eq!(m2.get("version").and_then(|x| x.as_str()), Some("v2"));
        let v1_bytes = fs::read(root.join("checkpoints/v1/manifest.json")).unwrap();
        assert_eq!(
            m2.get("parent_manifest_sha256").and_then(|x| x.as_str()),
            Some(sha256::hex(&v1_bytes).as_str())
        );

        // Corrupted chain pointer: checkpoint must refuse, not promote.
        let mut cur3 = read_json(&root.join("state/current.json")).unwrap();
        cur3.put("manifest_sha256", Value::s("0000000000000000000000000000000000000000000000000000000000000000"));
        write_json(&root.join("state/current.json"), &cur3).unwrap();
        assert!(cmd_checkpoint(&root, &cfg).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn license_quarantine_and_deny() {
        let cfg = Config::default();
        assert!(cfg.allowed_licenses.contains(&"MIT".to_string()));
        assert!(cfg.denied_licenses.contains(&"proprietary".to_string()));
        assert!(!cfg.allowed_licenses.contains(&"unknown".to_string()));
    }
}
