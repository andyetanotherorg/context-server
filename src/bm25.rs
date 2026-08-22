//! Simple BM25 over in-memory tokenized documents.

use std::collections::HashMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;

#[derive(Debug, Clone)]
struct DocStats {
    /// term -> raw tf
    tfs: HashMap<String, u32>,
    len: usize,
}

/// Corpus-level BM25 index.
pub struct Bm25 {
    docs: Vec<DocStats>,
    /// term -> number of docs containing it
    df: HashMap<String, u32>,
    avgdl: f32,
    n_docs: usize,
}

impl Bm25 {
    pub fn build(texts: &[String]) -> Self {
        let mut docs = Vec::with_capacity(texts.len());
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut total_len = 0usize;

        for text in texts {
            let tokens = tokenize(text);
            let len = tokens.len();
            total_len += len;
            let mut tfs: HashMap<String, u32> = HashMap::new();
            for t in tokens {
                *tfs.entry(t).or_insert(0) += 1;
            }
            for term in tfs.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            docs.push(DocStats { tfs, len });
        }

        let n_docs = docs.len();
        let avgdl = if n_docs == 0 {
            0.0
        } else {
            total_len as f32 / n_docs as f32
        };

        Self {
            docs,
            df,
            avgdl,
            n_docs,
        }
    }

    /// BM25 scores for every document (same order as build input).
    pub fn scores(&self, query: &str) -> Vec<f32> {
        let q_terms = tokenize(query);
        if self.n_docs == 0 || q_terms.is_empty() {
            return vec![0.0; self.docs.len()];
        }

        let mut out = vec![0.0f32; self.docs.len()];
        for term in unique_terms(&q_terms) {
            let df = *self.df.get(term).unwrap_or(&0) as f32;
            // Lucene-style IDF (always non-negative)
            let idf = ((self.n_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            for (i, doc) in self.docs.iter().enumerate() {
                let tf = *doc.tfs.get(term).unwrap_or(&0) as f32;
                if tf == 0.0 {
                    continue;
                }
                let dl = doc.len as f32;
                let denom = tf + K1 * (1.0 - B + B * dl / self.avgdl.max(1.0));
                out[i] += idf * (tf * (K1 + 1.0)) / denom;
            }
        }
        out
    }
}

/// Lowercase alphanumeric tokens; keeps identifiers like `alice`, `cdn`.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' || ch == '-' {
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            push_token_and_components(&mut tokens, std::mem::take(&mut cur));
        }
    }
    push_token_and_components(&mut tokens, cur);
    tokens
}

fn push_token_and_components(tokens: &mut Vec<String>, token: String) {
    if token.chars().count() < 2 {
        return;
    }
    tokens.push(token.clone());
    if token.contains(['-', '_']) {
        tokens.extend(
            token
                .split(['-', '_'])
                .filter(|part| part.chars().count() >= 2)
                .map(str::to_string),
        );
    }
}

fn unique_terms(terms: &[String]) -> Vec<&String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for t in terms {
        if seen.insert(t.as_str()) {
            out.push(t);
        }
    }
    out
}

/// Reciprocal Rank Fusion over one or more ranked id lists (best-first).
pub fn reciprocal_rank_fusion(rankings: &[Vec<usize>], k: f32) -> Vec<(usize, f32)> {
    let mut scores: HashMap<usize, f32> = HashMap::new();
    for ranking in rankings {
        for (rank, &doc_id) in ranking.iter().enumerate() {
            *scores.entry(doc_id).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut scored: Vec<(usize, f32)> = scores.into_iter().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_exact_token_match() {
        let texts = vec![
            "storage team owns ceph csi".into(),
            "networking team owns ovn".into(),
            "jsmith is the manager for platform virtualization".into(),
        ];
        let bm = Bm25::build(&texts);
        let scores = bm.scores("jsmith virtualization");
        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(best, 2);
    }

    #[test]
    fn rrf_merges_rankings() {
        let dense = vec![0usize, 1, 2];
        let lex = vec![2usize, 0, 1];
        let fused = reciprocal_rank_fusion(&[dense, lex], 60.0);
        assert_eq!(fused[0].0, 0); // appears high in both
    }

    #[test]
    fn tokenizes_unicode_and_identifier_components() {
        let tokens = tokenize("Café foo-bar storage_team 深い");
        for expected in [
            "café",
            "foo-bar",
            "foo",
            "bar",
            "storage_team",
            "storage",
            "team",
            "深い",
        ] {
            assert!(
                tokens.contains(&expected.to_string()),
                "missing {expected}: {tokens:?}"
            );
        }
    }
}
