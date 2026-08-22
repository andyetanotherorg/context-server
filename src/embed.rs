//! Sentence embeddings via fastembed (ONNX Runtime, BGE-small-en-v1.5).

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;

/// Stored in the DB so we refuse to search against an incompatible index.
pub const MODEL_ID: &str = "BGESmallENV15";
pub const DIM: usize = 384;

/// BGE retrieval instruction: prefix queries only; passages are embedded as-is.
/// See https://huggingface.co/BAAI/bge-small-en-v1.5
const QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";
/// Persisted provenance for every behavior that affects vector compatibility.
pub const EMBEDDING_FINGERPRINT: &str =
    "v1|fastembed:BGESmallENV15|dim:384|pool:model-default|l2:true|query:bge-v1.5";

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        let cache_dir = model_cache_dir()?;
        std::fs::create_dir_all(&cache_dir)
            .with_context(|| format!("create model cache dir {}", cache_dir.display()))?;
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(true),
        )
        .context("load embedding model (BGESmallENV15)")?;
        Ok(Self { model })
    }

    /// Embed a search query (applies the BGE query instruction).
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let instructed = format!("{QUERY_INSTRUCTION}{text}");
        let mut out = self.model.embed(vec![instructed], None).context("embed")?;
        let mut v = out.pop().context("empty embedding")?;
        l2_normalize(&mut v);
        Ok(v)
    }

    /// Embed document passages (no query instruction).
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let mut out = self.model.embed(texts, None).context("embed batch")?;
        for v in &mut out {
            l2_normalize(v);
        }
        Ok(out)
    }
}

/// Prefer explicit env overrides, otherwise use the XDG cache (not the process cwd).
/// fastembed's default is `.fastembed_cache` in PWD, which pollutes project trees.
pub fn model_cache_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("FASTEMBED_CACHE_DIR") {
        return Ok(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(p));
    }
    Ok(dirs::cache_dir()
        .context("no cache directory (set XDG_CACHE_HOME or HOME)")?
        .join("context-server")
        .join("fastembed"))
}

pub fn model_is_cached() -> Result<bool> {
    let dir = model_cache_dir()?;
    if !dir.is_dir() {
        return Ok(false);
    }
    Ok(walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .any(|entry| entry.file_type().is_file() && entry.file_name() == "model.onnx"))
}

pub fn estimated_tokens(text: &str) -> usize {
    text.split_whitespace()
        .map(|word| word.chars().count().max(1).div_ceil(4))
        .sum::<usize>()
        .max(1)
}

fn l2_normalize(v: &mut [f32]) {
    let mut sum = 0.0f32;
    for x in v.iter() {
        sum += x * x;
    }
    let norm = sum.sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity for L2-normalized vectors (dot product).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_cache_dir_defaults_under_xdg_cache() {
        if std::env::var_os("FASTEMBED_CACHE_DIR").is_some()
            || std::env::var_os("HF_HOME").is_some()
        {
            return;
        }
        let dir = model_cache_dir().expect("cache dir");
        assert!(dir.is_absolute(), "cache dir should be absolute: {dir:?}");
        assert_eq!(dir.file_name().and_then(|s| s.to_str()), Some("fastembed"));
        assert!(
            dir.components().any(|c| c.as_os_str() == "context-server"),
            "unexpected cache dir: {dir:?}"
        );
    }
}
