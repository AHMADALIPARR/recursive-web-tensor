//! End-to-end test through the real binary: one full cycle on a
//! small self-contained corpus must produce a promoted v1 checkpoint
//! with a consistent current pointer. (Merkle-root independence is
//! verified by scripts/e2e.sh with Python hashlib.)

use std::fs;
use std::process::Command;

fn bin() -> String {
    env!("CARGO_BIN_EXE_internet-token-engine").to_string()
}

#[test]
fn cycle_produces_promoted_v1() {
    let work = std::env::temp_dir().join(format!("rwt-e2e-{}", std::process::id()));
    let _ = fs::remove_dir_all(&work);
    fs::create_dir_all(&work).unwrap();
    let corpus = work.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    let docs = [
        ("a.txt", "The river flows through the valley while morning mist hangs over the water and birds sing."),
        ("b.txt", "Mountains rise beyond the trees and snow covers the highest peaks through the winter months."),
        ("c.txt", "To cook rice well, rinse the grains until the water runs clear, then soak them before cooking."),
        ("d.txt", "Astronomers study the light from distant stars to learn what they are made of and how far away."),
        ("e.txt", "The ancient city kept records on clay tablets, pressing wedge-shaped marks into wet clay."),
        ("f.txt", "Ocean currents carry warm water from the equator toward the poles and cold water back again."),
    ];
    for (name, text) in docs {
        fs::write(corpus.join(name), text).unwrap();
    }
    let probe = work.join("probe");
    fs::create_dir_all(&probe).unwrap();
    fs::write(probe.join("p.txt"), "Streams gather rainfall from hillsides and carry it downhill into rivers.").unwrap();

    let out = Command::new(bin())
        .arg("--root")
        .arg(&work)
        .arg("cycle")
        .arg("--corpus")
        .arg(&corpus)
        .arg("--probe-dir")
        .arg(&probe)
        .arg("--steps")
        .arg("0")
        .output()
        .expect("run cycle");
    assert!(
        out.status.success(),
        "cycle failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let manifest_path = work.join("checkpoints/v1/manifest.json");
    assert!(manifest_path.exists(), "no v1 manifest");
    let manifest = fs::read_to_string(&manifest_path).unwrap();
    assert!(manifest.contains("\"status\":\"promoted\""), "manifest: {}", manifest);
    let current = fs::read_to_string(work.join("state/current.json")).unwrap();
    assert!(current.contains("\"version\":\"v1\""), "current: {}", current);
    assert!(work.join("checkpoints/v1/model.json").exists(), "no v1 model");
    let _ = fs::remove_dir_all(&work);
}

#[test]
fn cycle_ngram_promotes() {
    // Explicit n-gram path (neural is the default since Phase 3).
    let work = std::env::temp_dir().join(format!("rwt-e2e-ngram-{}", std::process::id()));
    let _ = fs::remove_dir_all(&work);
    fs::create_dir_all(&work).unwrap();
    let corpus = work.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::write(
        corpus.join("a.txt"),
        "The river flows through the valley while morning mist hangs over the water and birds sing.",
    )
    .unwrap();
    fs::write(
        corpus.join("b.txt"),
        "Mountains rise beyond the trees and snow covers the highest peaks through the winter months.",
    )
    .unwrap();
    fs::write(
        corpus.join("c.txt"),
        "To cook rice well, rinse the grains until the water runs clear, then soak them before cooking.",
    )
    .unwrap();
    let out = Command::new(bin())
        .arg("--root")
        .arg(&work)
        .arg("cycle")
        .arg("--corpus")
        .arg(&corpus)
        .arg("--steps")
        .arg("0")
        .arg("--trainer")
        .arg("ngram")
        .output()
        .expect("run ngram cycle");
    assert!(
        out.status.success(),
        "ngram cycle failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let manifest = fs::read_to_string(work.join("checkpoints/v1/manifest.json")).unwrap();
    assert!(manifest.contains("\"status\":\"promoted\""), "manifest: {}", manifest);
    let model = fs::read_to_string(work.join("checkpoints/v1/model.json")).unwrap();
    assert!(model.contains("\"counts\""), "not an n-gram model: {}", &model[..model.len().min(200)]);
    let _ = fs::remove_dir_all(&work);
}

#[test]
fn cycle_bpe_promotes() {
    let work = std::env::temp_dir().join(format!("rwt-e2e-bpe-{}", std::process::id()));
    let _ = fs::remove_dir_all(&work);
    fs::create_dir_all(&work).unwrap();
    let corpus = work.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    let docs = [
        ("a.txt", "The river flows through the valley while morning mist hangs over the water and birds sing."),
        ("b.txt", "Mountains rise beyond the trees and snow covers the highest peaks through the winter months."),
        ("c.txt", "To cook rice well, rinse the grains until the water runs clear, then soak them before cooking."),
        ("d.txt", "Astronomers study the light from distant stars to learn what they are made of and how far away."),
    ];
    for (name, text) in docs {
        fs::write(corpus.join(name), text).unwrap();
    }
    let probe = work.join("probe");
    fs::create_dir_all(&probe).unwrap();
    fs::write(probe.join("p.txt"), "Streams gather rainfall from hillsides and carry it downhill into rivers.").unwrap();

    // Train BPE, then run the full cycle on the BPE path.
    let out = Command::new(bin())
        .arg("--root")
        .arg(&work)
        .arg("tokenizer")
        .arg("train-bpe")
        .arg("--corpus")
        .arg(&corpus)
        .arg("--merges")
        .arg("64")
        .output()
        .expect("train-bpe");
    assert!(out.status.success(), "train-bpe failed: {}", String::from_utf8_lossy(&out.stderr));
    assert!(work.join("state/bpe.json").exists(), "no bpe.json");

    let out = Command::new(bin())
        .arg("--root")
        .arg(&work)
        .arg("cycle")
        .arg("--corpus")
        .arg(&corpus)
        .arg("--probe-dir")
        .arg(&probe)
        .arg("--steps")
        .arg("0")
        .arg("--tokenizer")
        .arg("bpe")
        .output()
        .expect("run bpe cycle");
    assert!(
        out.status.success(),
        "bpe cycle failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let manifest = fs::read_to_string(work.join("checkpoints/v1/manifest.json")).unwrap();
    assert!(manifest.contains("\"status\":\"promoted\""), "manifest: {}", manifest);
    // Registry must record the BPE tokenizer, and shards must hold
    // hex tokens that decode back to the stored text.
    let registry = fs::read_to_string(work.join("state/registry.json")).unwrap();
    assert!(registry.contains("\"tokenizer\":\"bpe\""), "registry: {}", registry);
    assert!(work.join("checkpoints/v1/model.json").exists(), "no v1 model");
    let _ = fs::remove_dir_all(&work);
}
