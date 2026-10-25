//! Small neural language model: feedforward NNLM, hand-implemented
//! backprop + Adam, std-only.
//!
//! Architecture (exact, no attention, no recurrence — those are NOT
//! claimed): token embedding (dimension D) + one tanh MLP hidden
//! layer (dimension H) over the concatenation of the last CTX token
//! embeddings, + linear output layer + softmax cross-entropy.
//! Default dims: CTX=4, D=16, H=32. This is training FROM SCRATCH
//! on retained documents; there are no pretrained weights anywhere
//! in this repo, and nothing here is a fine-tune of a named model.
//!
//! Parameters are one flat Vec<f64> with a fixed layout:
//!   E  : vocab * D        (embedding rows, token id order)
//!   W1 : (CTX*D) * H      (input -> hidden, row-major: input-major)
//!   b1 : H
//!   W2 : H * vocab        (hidden -> logits, row-major: hidden-major)
//!   b2 : vocab
//! All math is index arithmetic over that layout, so the gradient
//! check (finite differences vs analytic gradients) covers every
//! parameter with no special cases.
//!
//! Training is full-batch Adam over the retained, non-held-out
//! documents (plus replay of prior-checkpoint docs, see engine.rs):
//! deterministic — fixed-seed xorshift init, sorted doc order,
//! no sampling, no dropout. Padding uses a "<s>" token prepended
//! to the vocabulary (id 0) when it is not already present.
//!
//! Init-from-prior-checkpoint: when a prior neural checkpoint has
//! the same CTX/D/H, its E rows, W2 columns, and b2 entries are
//! copied for tokens present in both vocabularies; W1 and b1 are
//! copied wholesale; tokens new to this epoch get fresh random
//! init. A prior n-gram checkpoint (or no prior) means fresh init.

use crate::json::Value;

pub const CTX: usize = 4;
pub const DIM: usize = 16;
pub const HIDDEN: usize = 32;

/// Fixed seed: training is deterministic across runs on the same
/// data. This is a training seed, not a security claim.
pub const SEED: u64 = 0x9E3779B97F4A7C15;

pub struct Layout {
    pub vocab: usize,
    pub e_off: usize,
    pub w1_off: usize,
    pub b1_off: usize,
    pub w2_off: usize,
    pub b2_off: usize,
    pub total: usize,
}

impl Layout {
    pub fn new(vocab: usize) -> Layout {
        let e_off = 0;
        let w1_off = e_off + vocab * DIM;
        let b1_off = w1_off + CTX * DIM * HIDDEN;
        let w2_off = b1_off + HIDDEN;
        let b2_off = w2_off + HIDDEN * vocab;
        let total = b2_off + vocab;
        Layout { vocab, e_off, w1_off, b1_off, w2_off, b2_off, total }
    }
}

/// xorshift64* — deterministic PRNG for init only.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in [-r, r].
    pub fn uniform(&mut self, r: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (u * 2.0 - 1.0) * r
    }
}

/// Fresh init: Xavier-style uniform bounds per matrix fan-in.
pub fn init_params(vocab: usize, seed: u64) -> Vec<f64> {
    let layout = Layout::new(vocab);
    let mut rng = Rng(seed);
    let mut p = vec![0.0f64; layout.total];
    let e_r = 1.0 / (DIM as f64).sqrt();
    for i in 0..vocab * DIM {
        p[layout.e_off + i] = rng.uniform(e_r);
    }
    let w1_r = 1.0 / ((CTX * DIM) as f64).sqrt();
    for i in 0..CTX * DIM * HIDDEN {
        p[layout.w1_off + i] = rng.uniform(w1_r);
    }
    let w2_r = 1.0 / (HIDDEN as f64).sqrt();
    for i in 0..HIDDEN * vocab {
        p[layout.w2_off + i] = rng.uniform(w2_r);
    }
    // b1, b2 start at zero.
    p
}

/// Init from a prior neural checkpoint's params. `prior_vocab` and
/// `vocab` are the token lists (id order, "<s>" first). Returns
/// fresh init when the prior layout is incompatible (different
/// CTX/D/H is impossible here since dims are constants, but a
/// length mismatch means the file is not a neural model of this
/// shape, so fail safe to fresh init rather than misaligning).
pub fn init_from_prior(
    vocab: &[String],
    prior_vocab: &[String],
    prior_params: &[f64],
    seed: u64,
) -> Vec<f64> {
    let layout = Layout::new(vocab.len());
    let prior_layout = Layout::new(prior_vocab.len());
    if prior_params.len() != prior_layout.total {
        return init_params(vocab.len(), seed);
    }
    let mut p = init_params(vocab.len(), seed);
    // W1, b1: shape-independent of vocab, copy wholesale.
    p[layout.w1_off..layout.b1_off]
        .copy_from_slice(&prior_params[prior_layout.w1_off..prior_layout.b1_off]);
    p[layout.b1_off..layout.w2_off]
        .copy_from_slice(&prior_params[prior_layout.b1_off..prior_layout.w2_off]);
    // Per-token rows/columns/entries for shared tokens.
    for (new_id, tok) in vocab.iter().enumerate() {
        if let Some(old_id) = prior_vocab.iter().position(|t| t == tok) {
            for d in 0..DIM {
                p[layout.e_off + new_id * DIM + d] =
                    prior_params[prior_layout.e_off + old_id * DIM + d];
            }
            for h in 0..HIDDEN {
                p[layout.w2_off + h * vocab.len() + new_id] =
                    prior_params[prior_layout.w2_off + h * prior_vocab.len() + old_id];
            }
            p[layout.b2_off + new_id] = prior_params[prior_layout.b2_off + old_id];
        }
    }
    p
}

/// One training example: CTX context token ids + target token id.
pub type Example = ([usize; CTX], usize);

/// Build examples from token-id sequences: pad with <s> (id 0) at
/// the start, then every position predicts the next token.
pub fn examples_from_ids(seqs: &[Vec<usize>]) -> Vec<Example> {
    let mut out = Vec::new();
    for seq in seqs {
        // For each target position t in the padded stream, the
        // context is the CTX tokens before it.
        let mut stream: Vec<usize> = vec![0; CTX];
        stream.extend(seq.iter().copied());
        for t in CTX..stream.len() {
            let mut ctx = [0usize; CTX];
            ctx.copy_from_slice(&stream[t - CTX..t]);
            out.push((ctx, stream[t]));
        }
    }
    out
}

/// Forward pass for one example. Returns (loss, logits, hidden
/// activations) — the activations are needed by backprop.
fn forward(params: &[f64], layout: &Layout, ex: &Example) -> (f64, Vec<f64>, Vec<f64>) {
    let (ctx, target) = ex;
    // x = concat of context embeddings.
    let mut x = vec![0.0f64; CTX * DIM];
    for (i, tok) in ctx.iter().enumerate() {
        for d in 0..DIM {
            x[i * DIM + d] = params[layout.e_off + tok * DIM + d];
        }
    }
    // Hidden: z1 = W1^T x + b1, a = tanh(z1). W1 is row-major with
    // input index major: W1[in * HIDDEN + h].
    let mut a = vec![0.0f64; HIDDEN];
    for h in 0..HIDDEN {
        let mut z = params[layout.b1_off + h];
        for i in 0..CTX * DIM {
            z += x[i] * params[layout.w1_off + i * HIDDEN + h];
        }
        a[h] = z.tanh();
    }
    // Logits: W2 row-major hidden-major: W2[h * vocab + v].
    let mut logits = vec![0.0f64; layout.vocab];
    for v in 0..layout.vocab {
        let mut z = params[layout.b2_off + v];
        for h in 0..HIDDEN {
            z += a[h] * params[layout.w2_off + h * layout.vocab + v];
        }
        logits[v] = z;
    }
    // Log-softmax cross-entropy (max-shifted for stability).
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0f64;
    for l in &logits {
        sum += (l - max).exp();
    }
    let log_z = max + sum.ln();
    let loss = log_z - logits[*target];
    (loss, logits, a)
}

/// Mean loss over examples (targets must be valid ids).
pub fn mean_loss(params: &[f64], layout: &Layout, examples: &[Example]) -> f64 {
    if examples.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for ex in examples {
        sum += forward(params, layout, ex).0;
    }
    sum / examples.len() as f64
}

/// Analytic gradients of the mean loss, same flat layout as params.
pub fn gradients(params: &[f64], layout: &Layout, examples: &[Example]) -> Vec<f64> {
    let mut grads = vec![0.0f64; layout.total];
    if examples.is_empty() {
        return grads;
    }
    let n = examples.len() as f64;
    for ex in examples {
        let (ctx, target) = ex;
        let (_, logits, a) = forward(params, layout, ex);
        // Softmax probabilities.
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs = vec![0.0f64; layout.vocab];
        let mut sum = 0.0f64;
        for (i, l) in logits.iter().enumerate() {
            probs[i] = (l - max).exp();
            sum += probs[i];
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        // dL/dlogits = p - onehot(target), scaled by 1/n.
        let mut dlogits = probs;
        dlogits[*target] -= 1.0;
        for d in dlogits.iter_mut() {
            *d /= n;
        }
        // dL/db2, dL/dW2.
        for v in 0..layout.vocab {
            grads[layout.b2_off + v] += dlogits[v];
            for h in 0..HIDDEN {
                grads[layout.w2_off + h * layout.vocab + v] += a[h] * dlogits[v];
            }
        }
        // dL/da = W2 * dlogits; dL/dz1 = dL/da * (1 - a^2).
        let mut da = vec![0.0f64; HIDDEN];
        for h in 0..HIDDEN {
            let mut s = 0.0f64;
            for v in 0..layout.vocab {
                s += params[layout.w2_off + h * layout.vocab + v] * dlogits[v];
            }
            da[h] = s * (1.0 - a[h] * a[h]);
        }
        // dL/db1, dL/dW1, dL/dx (x recomputed from the context
        // embeddings below).
        let mut dx = vec![0.0f64; CTX * DIM];
        for h in 0..HIDDEN {
            grads[layout.b1_off + h] += da[h];
        }
        let mut x = vec![0.0f64; CTX * DIM];
        for (i, tok) in ctx.iter().enumerate() {
            for d in 0..DIM {
                x[i * DIM + d] = params[layout.e_off + tok * DIM + d];
            }
        }
        for h in 0..HIDDEN {
            for i in 0..CTX * DIM {
                grads[layout.w1_off + i * HIDDEN + h] += x[i] * da[h];
                dx[i] += params[layout.w1_off + i * HIDDEN + h] * da[h];
            }
        }
        // Embedding grads: scatter-add dx into context rows.
        for (i, tok) in ctx.iter().enumerate() {
            for d in 0..DIM {
                grads[layout.e_off + tok * DIM + d] += dx[i * DIM + d];
            }
        }
    }
    grads
}

/// Full-batch Adam. Returns the trained params. `lr`, `beta1`,
/// `beta2`, `eps` are the standard Adam hyperparameters.
pub fn train_adam(
    mut params: Vec<f64>,
    layout: &Layout,
    examples: &[Example],
    epochs: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
) -> Vec<f64> {
    let mut m = vec![0.0f64; layout.total];
    let mut v = vec![0.0f64; layout.total];
    for t in 1..=epochs {
        let grads = gradients(&params, layout, examples);
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);
        for i in 0..layout.total {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
    params
}

/// Serialize a neural model to the checkpoint JSON Value.
pub fn model_to_json(
    vocab: &[String],
    params: &[f64],
    trained_docs: usize,
    doc_shas: &[String],
    total_tokens: i64,
) -> Value {
    let mut model = Value::obj();
    model.put("kind", Value::s("neural"));
    model.put("ctx", Value::n(CTX as i64));
    model.put("d", Value::n(DIM as i64));
    model.put("h", Value::n(HIDDEN as i64));
    model.put("vocab", Value::Arr(vocab.iter().map(|t| Value::s(t)).collect()));
    model.put("params", Value::Arr(params.iter().map(|p| Value::f(*p)).collect()));
    model.put("trained_docs", Value::n(trained_docs as i64));
    model.put("doc_shas", Value::Arr(doc_shas.iter().map(|s| Value::s(s)).collect()));
    model.put("total_tokens", Value::n(total_tokens));
    model
}

/// Parse a neural model JSON Value. Returns (vocab, params).
pub fn model_from_json(v: &Value) -> Result<(Vec<String>, Vec<f64>), String> {
    if v.get("kind").and_then(|x| x.as_str()) != Some("neural") {
        return Err("model is not a neural model".to_string());
    }
    let vocab: Vec<String> = v
        .get("vocab")
        .and_then(|x| x.as_arr())
        .ok_or("neural model has no vocab")?
        .iter()
        .filter_map(|x| x.as_str().map(|s| s.to_string()))
        .collect();
    let params: Vec<f64> = v
        .get("params")
        .and_then(|x| x.as_arr())
        .ok_or("neural model has no params")?
        .iter()
        .filter_map(|x| x.as_f64())
        .collect();
    let layout = Layout::new(vocab.len());
    if params.len() != layout.total {
        return Err(format!(
            "neural params length {} does not match vocab {} layout {}",
            params.len(),
            vocab.len(),
            layout.total
        ));
    }
    Ok((vocab, params))
}

/// Cross-entropy of a neural model on token-id sequences. Unknown
/// target ids (outside the model vocab) contribute uniform loss
/// ln(vocab) — the same honest fallback spirit as the n-gram
/// backoff — and are counted, never silently skipped.
pub fn neural_cross_entropy(
    params: &[f64],
    layout: &Layout,
    seqs_ids: &[Vec<usize>],
    unknown_targets: usize,
) -> (f64, i64) {
    let examples = examples_from_ids(seqs_ids);
    let mut loss_sum = 0.0f64;
    let mut n = 0i64;
    for ex in &examples {
        loss_sum += forward(params, layout, ex).0;
        n += 1;
    }
    if unknown_targets > 0 {
        loss_sum += unknown_targets as f64 * (layout.vocab as f64).ln();
        n += unknown_targets as i64;
    }
    if n == 0 {
        (0.0, 0)
    } else {
        (loss_sum / n as f64, n)
    }
}

/// Map token strings to ids with a vocab list. Returns (ids,
/// unknown_count). Unknown tokens map to nothing — the caller
/// decides how to count them.
pub fn tokens_to_ids(vocab_index: &std::collections::HashMap<String, usize>, tokens: &[String]) -> (Vec<usize>, usize) {
    let mut ids = Vec::new();
    let mut unknown = 0usize;
    for t in tokens {
        match vocab_index.get(t) {
            Some(id) => ids.push(*id),
            None => unknown += 1,
        }
    }
    (ids, unknown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_setup() -> (Layout, Vec<f64>) {
        // Vocab: <s>, a, b, c (4 tokens).
        let layout = Layout::new(4);
        let params = init_params(4, 12345);
        (layout, params)
    }

    #[test]
    fn gradient_check_finite_difference() {
        let (layout, params) = tiny_setup();
        let examples = vec![([0, 0, 0, 1], 2), ([0, 0, 1, 2], 3), ([0, 0, 0, 0], 1)];
        let analytic = gradients(&params, &layout, &examples);
        let eps = 1e-5;
        let mut max_err = 0.0f64;
        for i in 0..layout.total {
            let mut plus = params.clone();
            plus[i] += eps;
            let mut minus = params.clone();
            minus[i] -= eps;
            let fd = (mean_loss(&plus, &layout, &examples)
                - mean_loss(&minus, &layout, &examples))
                / (2.0 * eps);
            let err = (fd - analytic[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 1e-4,
            "gradient check failed: max abs error {}",
            max_err
        );
    }

    #[test]
    fn training_reduces_loss() {
        // Part A — generalizable pattern: alternating "a b a b ..."
        // sequences. The rule (after a comes b, after b comes a) is
        // fully determined by context, so a held-out alternating
        // sequence must get LOWER loss after training, not just the
        // training sequences.
        let vocab = vec!["<s>".to_string(), "a".to_string(), "b".to_string()];
        let index: HashMap<String, usize> =
            vocab.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
        let seq = |tokens: &[&str]| -> Vec<usize> {
            tokens.iter().map(|t| index[*t]).collect()
        };
        let train_seqs = vec![
            seq(&["a", "b", "a", "b", "a", "b", "a", "b"]),
            seq(&["b", "a", "b", "a", "b", "a", "b", "a"]),
            seq(&["a", "b", "a", "b", "a", "b"]),
        ];
        let held_seqs = vec![seq(&["b", "a", "b", "a", "b", "a"])];
        let layout = Layout::new(vocab.len());
        let params0 = init_params(vocab.len(), SEED);
        let train_ex = examples_from_ids(&train_seqs);
        let held_ex = examples_from_ids(&held_seqs);
        let init_train = mean_loss(&params0, &layout, &train_ex);
        let init_held = mean_loss(&params0, &layout, &held_ex);
        let trained = train_adam(params0, &layout, &train_ex, 120, 0.01, 0.9, 0.999, 1e-8);
        let final_train = mean_loss(&trained, &layout, &train_ex);
        let final_held = mean_loss(&trained, &layout, &held_ex);
        assert!(
            final_train < init_train,
            "train loss did not fall: {} -> {}",
            init_train,
            final_train
        );
        assert!(
            final_held < init_held,
            "held-out loss did not fall: {} -> {}",
            init_held,
            final_held
        );

        // Part B — sentence-shaped data: training loss must fall
        // (capacity check). No held-out claim here: with three
        // training sentences the held-out combination is genuinely
        // surprising under the learned distribution, and asserting
        // otherwise would be fitting the test to the model.
        let vocab2 = vec![
            "<s>".to_string(),
            "the".to_string(),
            "cat".to_string(),
            "sat".to_string(),
            "dog".to_string(),
            "ran".to_string(),
        ];
        let index2: HashMap<String, usize> =
            vocab2.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
        let sentences = [
            vec!["the", "cat", "sat"],
            vec!["the", "dog", "ran"],
            vec!["the", "cat", "ran"],
        ];
        let seqs2: Vec<Vec<usize>> = sentences
            .iter()
            .map(|s| s.iter().map(|t| index2[*t]).collect())
            .collect();
        let layout2 = Layout::new(vocab2.len());
        let p0 = init_params(vocab2.len(), SEED);
        let ex2 = examples_from_ids(&seqs2);
        let before = mean_loss(&p0, &layout2, &ex2);
        let after = train_adam(p0, &layout2, &ex2, 120, 0.01, 0.9, 0.999, 1e-8);
        let after_loss = mean_loss(&after, &layout2, &ex2);
        assert!(after_loss < before, "sentence train loss did not fall: {} -> {}", before, after_loss);
    }

    #[test]
    fn init_from_prior_preserves_shared_tokens() {
        let prior_vocab = vec!["<s>".to_string(), "a".to_string(), "b".to_string()];
        let prior = init_params(prior_vocab.len(), 777);
        let vocab = vec![
            "<s>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ];
        let p = init_from_prior(&vocab, &prior_vocab, &prior, 999);
        let layout = Layout::new(vocab.len());
        let prior_layout = Layout::new(prior_vocab.len());
        // Shared token "a" (id 1 in both): embedding row identical.
        assert_eq!(
            p[layout.e_off + 1 * DIM..layout.e_off + 2 * DIM],
            prior[prior_layout.e_off + 1 * DIM..prior_layout.e_off + 2 * DIM]
        );
        // W1/b1 copied wholesale.
        assert_eq!(
            p[layout.w1_off..layout.b1_off],
            prior[prior_layout.w1_off..prior_layout.b1_off]
        );
        // New token "c" differs from fresh-seed init only by seed;
        // it must at least be finite and non-zero somewhere.
        let row = &p[layout.e_off + 3 * DIM..layout.e_off + 4 * DIM];
        assert!(row.iter().all(|x| x.is_finite()));
        assert!(row.iter().any(|x| *x != 0.0));
    }

    #[test]
    fn training_is_deterministic() {
        let vocab = vec!["<s>".to_string(), "a".to_string(), "b".to_string()];
        let layout = Layout::new(vocab.len());
        let ex = vec![([0, 0, 0, 1], 2), ([0, 0, 1, 2], 1), ([0, 0, 0, 0], 2)];
        let a = train_adam(init_params(3, SEED), &layout, &ex, 10, 0.01, 0.9, 0.999, 1e-8);
        let b = train_adam(init_params(3, SEED), &layout, &ex, 10, 0.01, 0.9, 0.999, 1e-8);
        assert_eq!(a, b);
    }

    #[test]
    fn model_json_round_trip() {
        let vocab = vec!["<s>".to_string(), "a".to_string(), "b".to_string()];
        let params = init_params(vocab.len(), 42);
        let json = model_to_json(&vocab, &params, 2, &["abc".to_string()], 10);
        let text = crate::json::to_string(&json);
        let parsed = crate::json::parse(&text).unwrap();
        let (v2, p2) = model_from_json(&parsed).unwrap();
        assert_eq!(v2, vocab);
        assert_eq!(p2, params);
    }

    #[test]
    fn examples_cover_positions() {
        // Sequence [a, b] with CTX=4: stream = [<s>x4, a, b];
        // targets: a (context all <s>), b (context <s>,<s>,<s>,a).
        let seqs = vec![vec![1usize, 2]];
        let ex = examples_from_ids(&seqs);
        assert_eq!(ex.len(), 2);
        assert_eq!(ex[0], ([0, 0, 0, 0], 1));
        assert_eq!(ex[1], ([0, 0, 0, 1], 2));
    }
}
