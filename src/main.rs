//! internet-token-engine: reference CLI for RECURSIVE WEB TENSOR.
//!
//! Subcommands: ingest, prune, train, evaluate, reduce, checkpoint,
//! cycle. Global flag: --root <dir> (default ".").
//!
//! This is a reference implementation: deterministic tokenization
//! (reference or BPE), heuristic scoring, and a small neural trainer
//! (from scratch, CPU, std-only) with the n-gram trainer kept as an
//! alternative. GPU acceleration is a later milestone, NOT in this
//! binary. Nothing fetched from the network is ever executed.

mod bpe;
mod engine;
mod fetch;
mod minhash;
mod neural;
mod warc;
mod json;
mod scoring;
mod sha256;
mod tokenizer;

use std::collections::HashMap;
use std::path::PathBuf;

struct Args {
    map: HashMap<String, String>,
}

impl Args {
    fn parse(pairs: &[String]) -> Args {
        let mut map = HashMap::new();
        let mut i = 0;
        while i < pairs.len() {
            let arg = &pairs[i];
            if let Some(key) = arg.strip_prefix("--") {
                if i + 1 < pairs.len() && !pairs[i + 1].starts_with("--") {
                    map.insert(key.to_string(), pairs[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        Args { map }
    }
    fn get(&self, key: &str) -> Option<String> {
        self.map.get(key).cloned()
    }
    fn get_usize(&self, key: &str, default: usize) -> usize {
        self.map.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
    }
    fn get_f64(&self, key: &str) -> Option<f64> {
        self.map.get(key).and_then(|v| v.parse().ok())
    }
}

fn usage() -> String {
    "internet-token-engine <command> [flags]\n\
     Global: --root <dir> (default .)\n\
     Commands:\n\
     \x20 ingest    --corpus <dir> [--url-manifest <file>] [--cc-index <url>]\n\
     \x20             [--cc-warc-manifest <file>] [--cc-data-base <url>]\n\
     \x20             [--cc-max-records <n>] [--ddg-query <q>]\n\
     \x20             [--shard <n>] [--num-shards <n>]\n\
     \x20             [--tokenizer reference|bpe] [--bpe-merges <file>]\n\
     \x20 prune     [--shard <n>] [--num-shards <n>] [--max-depth <n>] [--tau <f>]\n\
     \x20 reduce    [--max-depth <n>]\n\
     \x20 prune-global [--num-shards <n>]\n\
     \x20 train     [--order <n>] [--steps <n>]  (steps 0 = unlimited)\n\
     \x20             [--trainer neural|ngram] [--epochs <n>] [--lr <f>]\n\
     \x20 evaluate  [--model <path>] [--probe-dir <dir>]\n\
     \x20 checkpoint\n\
     \x20 cycle     --corpus <dir> [--probe-dir <dir>] [--order <n>]\n\
     \x20             [--cc-warc-manifest <file>] [--cc-data-base <url>]\n\
     \x20             [--cc-max-records <n>] [--max-depth <n>] [--steps <n>]\n\
     \x20             [--tokenizer reference|bpe] [--bpe-merges <file>]\n\
     \x20             [--trainer neural|ngram] [--epochs <n>] [--lr <f>]\n\
     \x20 tokenizer train-bpe --corpus <dir> [--merges <n>] [--out <file>]\n\
     \x20 tokenizer encode --merges <file> --text <text>\n\
     \x20 tokenizer decode --merges <file> --tokens <hex...>\n"
        .to_string()
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().collect();
    // --root is global: it may appear before the command
    // (`--root DIR cycle ...`) or among the command flags.
    let mut tokens: Vec<String> = argv[1..].to_vec();
    let mut root = PathBuf::from(".");
    if let Some(pos) = tokens.iter().position(|t| t == "--root") {
        if pos + 1 < tokens.len() {
            root = PathBuf::from(tokens[pos + 1].clone());
            tokens.drain(pos..pos + 2);
        }
    }
    if tokens.is_empty() {
        return Err(usage());
    }
    let command = tokens[0].clone();
    let command = command.as_str();
    let args = Args::parse(&tokens[1..]);
    if root == PathBuf::from(".") {
        if let Some(r) = args.get("root") {
            root = PathBuf::from(r);
        }
    }
    let config = engine::load_config(&root)?;
    match command {
        "ingest" => {
            let shard_raw = args.get_usize("shard", usize::MAX);
            let shard = if shard_raw == usize::MAX { None } else { Some(shard_raw) };
            let (written, dups, errs) = engine::cmd_ingest(
                &root,
                &engine::IngestArgs {
                    corpus: args.get("corpus"),
                    url_manifest: args.get("url-manifest"),
                    cc_index: args.get("cc-index"),
                    cc_warc_manifest: args.get("cc-warc-manifest"),
                    cc_data_base: args.get("cc-data-base"),
                    cc_max_records: args.get_usize("cc-max-records", 0),
                    ddg_query: args.get("ddg-query"),
                    shard,
                    num_shards: args.get_usize("num-shards", 4),
                    tokenizer: args.get("tokenizer"),
                    bpe_merges: args.get("bpe-merges"),
                    config: config.clone(),
                },
            )?;
            println!("ingest done: written={} duplicates={} errors={}", written, dups, errs);
            Ok(())
        }
        "tokenizer" => {
            let sub = tokens.get(1).cloned().unwrap_or_default();
            let sub_args = Args::parse(&tokens[2..]);
            match sub.as_str() {
                "train-bpe" => {
                    let corpus = sub_args.get("corpus").ok_or("tokenizer train-bpe needs --corpus <dir>")?;
                    engine::cmd_tokenizer_train(
                        &root,
                        &corpus,
                        sub_args.get_usize("merges", 200),
                        sub_args.get("out"),
                    )?;
                    Ok(())
                }
                "encode" => {
                    let merges = sub_args.get("merges").ok_or("tokenizer encode needs --merges <file>")?;
                    let text = sub_args.get("text").ok_or("tokenizer encode needs --text <text>")?;
                    engine::cmd_tokenizer_encode(&merges, &text)?;
                    Ok(())
                }
                "decode" => {
                    let merges = sub_args.get("merges").ok_or("tokenizer decode needs --merges <file>")?;
                    let tokens = sub_args.get("tokens").ok_or("tokenizer decode needs --tokens <hex...>")?;
                    engine::cmd_tokenizer_decode(&merges, &tokens)?;
                    Ok(())
                }
                _ => Err(format!("unknown tokenizer subcommand '{}'\n{}", sub, usage())),
            }
        }
        "prune" => {
            let shard_raw = args.get_usize("shard", usize::MAX);
            let shard = if shard_raw == usize::MAX { None } else { Some(shard_raw) };
            engine::cmd_prune(
                &root,
                &config,
                &engine::PruneArgs {
                    shard,
                    num_shards: args.get_usize("num-shards", 4),
                    max_depth: args.get_usize("max-depth", 8),
                    tau_override: args.get_f64("tau"),
                },
            )?;
            Ok(())
        }
        "reduce" => {
            engine::cmd_reduce(&root, &engine::ReduceArgs { max_depth: args.get_usize("max-depth", 8) })?;
            Ok(())
        }
        "prune-global" => {
            engine::cmd_prune_global(&root, &config, args.get_usize("num-shards", 4))?;
            Ok(())
        }
        "train" => {
            engine::cmd_train(
                &root,
                &config,
                &engine::TrainArgs {
                    order: args.get_usize("order", 0),
                    steps: args.get_usize("steps", 100),
                    trainer: args.get("trainer"),
                    epochs: args.get_usize("epochs", 0),
                    lr: args.get_f64("lr").unwrap_or(0.0),
                },
            )?;
            Ok(())
        }
        "evaluate" => {
            engine::cmd_evaluate(
                &root,
                &config,
                &engine::EvaluateArgs { model_path: args.get("model"), probe_dir: args.get("probe-dir") },
            )?;
            Ok(())
        }
        "checkpoint" => {
            engine::cmd_checkpoint(&root, &config)?;
            Ok(())
        }
        "cycle" => {
            engine::cmd_cycle(
                &root,
                &config,
                &engine::CycleArgs {
                    corpus: args.get("corpus"),
                    url_manifest: args.get("url-manifest"),
                    cc_warc_manifest: args.get("cc-warc-manifest"),
                    cc_data_base: args.get("cc-data-base"),
                    cc_max_records: args.get_usize("cc-max-records", 0),
                    probe_dir: args.get("probe-dir"),
                    order: args.get_usize("order", 0),
                    max_depth: args.get_usize("max-depth", 8),
                    steps: args.get_usize("steps", 0),
                    tokenizer: args.get("tokenizer"),
                    bpe_merges: args.get("bpe-merges"),
                    trainer: args.get("trainer"),
                    epochs: args.get_usize("epochs", 0),
                    lr: args.get_f64("lr").unwrap_or(0.0),
                },
            )?;
            Ok(())
        }
        other => Err(format!("unknown command '{}'\n{}", other, usage())),
    }
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }
}
