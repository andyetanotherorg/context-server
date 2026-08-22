//! Retrieval evaluation metrics and golden-case loading.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct EvalCase {
    pub query: String,
    pub relevant: Vec<String>,
}

#[derive(Debug, Default, PartialEq)]
pub struct Metrics {
    pub cases: usize,
    pub recall_at_k: f64,
    pub mrr: f64,
}

pub fn load_cases(path: &Path) -> Result<Vec<EvalCase>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read evaluation cases {}", path.display()))?;
    serde_json::from_str(&text).context("parse evaluation cases JSON")
}

pub fn score_case(ranked: &[String], relevant: &[String], k: usize) -> (f64, f64) {
    if relevant.is_empty() {
        return (if ranked.is_empty() { 1.0 } else { 0.0 }, 0.0);
    }
    let found = ranked
        .iter()
        .take(k)
        .filter(|citation| relevant.contains(citation))
        .count();
    let recall = found as f64 / relevant.len() as f64;
    let reciprocal_rank = ranked
        .iter()
        .position(|citation| relevant.contains(citation))
        .map(|rank| 1.0 / (rank + 1) as f64)
        .unwrap_or(0.0);
    (recall, reciprocal_rank)
}

pub fn aggregate(values: &[(f64, f64)]) -> Metrics {
    if values.is_empty() {
        return Metrics::default();
    }
    Metrics {
        cases: values.len(),
        recall_at_k: values.iter().map(|v| v.0).sum::<f64>() / values.len() as f64,
        mrr: values.iter().map(|v| v.1).sum::<f64>() / values.len() as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_recall_and_reciprocal_rank() {
        let ranked = vec!["a#0".into(), "b#0".into(), "c#0".into()];
        let relevant = vec!["b#0".into(), "c#0".into()];
        assert_eq!(score_case(&ranked, &relevant, 2), (0.5, 0.5));
    }

    #[test]
    fn no_answer_case_rewards_empty_results() {
        assert_eq!(score_case(&[], &[], 5), (1.0, 0.0));
        assert_eq!(score_case(&["a#0".into()], &[], 5), (0.0, 0.0));
    }
}
