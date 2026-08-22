//! Hybrid dense + BM25 search over loaded documents.

use crate::bm25::{reciprocal_rank_fusion, Bm25};
use crate::embed::{self, Embedder};
use crate::store::{Db, Document};
use anyhow::{Context, Result};

const RRF_K: f32 = 60.0;
/// How many candidates to take from each ranker before fusion.
const CANDIDATE_POOL: usize = 50;
const MIN_DENSE_SCORE: f32 = 0.30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Hybrid,
    Dense,
    Lexical,
}

#[derive(Clone, Copy)]
enum ScoreKind {
    Dense,
    Lexical,
}

impl SearchMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "hybrid" | "both" => Some(Self::Hybrid),
            "dense" | "semantic" | "vector" => Some(Self::Dense),
            "lexical" | "bm25" | "text" => Some(Self::Lexical),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResultHit {
    pub score: f32,
    pub dense_score: f32,
    pub lexical_score: f32,
    pub source_path: String,
    pub chunk_index: usize,
    pub headings: Vec<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub text: String,
}

/// Optional constraints applied after scoring (non-matching docs are dropped).
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    /// Keep chunks whose `source_path` starts with this prefix (e.g. `teams/`).
    pub path_prefix: Option<String>,
    /// Keep chunks where any heading contains this substring (case-insensitive).
    pub heading: Option<String>,
    /// Keep chunks whose metadata `tags` array contains this value (case-insensitive),
    /// or whose string metadata values equal it.
    pub tag: Option<String>,
}

impl SearchFilter {
    pub fn is_empty(&self) -> bool {
        self.path_prefix.as_ref().is_none_or(|s| s.is_empty())
            && self.heading.as_ref().is_none_or(|s| s.is_empty())
            && self.tag.as_ref().is_none_or(|s| s.is_empty())
    }

    pub fn matches(&self, doc: &Document) -> bool {
        if let Some(prefix) = self.path_prefix.as_ref().filter(|s| !s.is_empty()) {
            if !doc.source_path.starts_with(prefix.as_str()) {
                return false;
            }
        }
        if let Some(heading) = self.heading.as_ref().filter(|s| !s.is_empty()) {
            let needle = heading.to_ascii_lowercase();
            let ok = doc
                .headings
                .iter()
                .any(|h| h.to_ascii_lowercase().contains(&needle));
            if !ok {
                return false;
            }
        }
        if let Some(tag) = self.tag.as_ref().filter(|s| !s.is_empty()) {
            if !metadata_has_tag(&doc.metadata, tag) {
                return false;
            }
        }
        true
    }
}

fn metadata_has_tag(metadata: &serde_json::Map<String, serde_json::Value>, tag: &str) -> bool {
    let needle = tag.to_ascii_lowercase();
    if let Some(tags) = metadata.get("tags") {
        match tags {
            serde_json::Value::Array(arr) => {
                if arr.iter().any(|v| match v {
                    serde_json::Value::String(s) => s.to_ascii_lowercase() == needle,
                    _ => v.to_string().trim_matches('"').to_ascii_lowercase() == needle,
                }) {
                    return true;
                }
            }
            serde_json::Value::String(s)
                if s.split(',')
                    .map(str::trim)
                    .any(|t| t.to_ascii_lowercase() == needle) =>
            {
                return true;
            }
            _ => {}
        }
    }
    metadata.values().any(|v| match v {
        serde_json::Value::String(s) => s.to_ascii_lowercase() == needle,
        _ => false,
    })
}

pub struct Index {
    docs: Vec<Document>,
    bm25: Bm25,
}

impl Index {
    pub fn load(db: &Db) -> Result<Self> {
        let docs = db.load_all().context("load documents")?;
        let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let bm25 = Bm25::build(&texts);
        Ok(Self { docs, bm25 })
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Fetch a single chunk by stable citation key (`source_path` + `chunk_index`).
    pub fn get(&self, source_path: &str, chunk_index: usize) -> Option<&Document> {
        self.docs
            .iter()
            .find(|d| d.source_path == source_path && d.chunk_index == chunk_index)
    }

    pub fn get_by_chunk_id(&self, chunk_id: &str) -> Option<&Document> {
        self.docs.iter().find(|document| {
            document
                .metadata
                .get("_chunk_id")
                .and_then(|value| value.as_str())
                == Some(chunk_id)
        })
    }

    /// All chunks for a source path, ordered by `chunk_index`.
    pub fn get_by_path(&self, source_path: &str) -> Vec<&Document> {
        let mut docs: Vec<&Document> = self
            .docs
            .iter()
            .filter(|d| d.source_path == source_path)
            .collect();
        docs.sort_by_key(|d| d.chunk_index);
        docs
    }

    /// Unfiltered search (all indexed chunks).
    #[allow(dead_code)]
    pub fn query(
        &self,
        emb: &mut Embedder,
        query: &str,
        limit: usize,
        mode: SearchMode,
    ) -> Result<Vec<ResultHit>> {
        self.query_filtered(emb, query, limit, mode, &SearchFilter::default())
    }

    pub fn query_filtered(
        &self,
        emb: &mut Embedder,
        query: &str,
        limit: usize,
        mode: SearchMode,
        filter: &SearchFilter,
    ) -> Result<Vec<ResultHit>> {
        let limit = if limit == 0 { 5 } else { limit };
        if self.docs.is_empty() {
            return Ok(vec![]);
        }

        let mask: Vec<bool> = if filter.is_empty() {
            vec![true; self.docs.len()]
        } else {
            self.docs.iter().map(|d| filter.matches(d)).collect()
        };
        if !mask.iter().any(|&m| m) {
            return Ok(vec![]);
        }

        let dense_scores: Vec<f32> = match mode {
            SearchMode::Lexical => vec![0.0; self.docs.len()],
            SearchMode::Dense | SearchMode::Hybrid => {
                let qv = emb.embed(query)?;
                self.docs
                    .iter()
                    .enumerate()
                    .map(|(i, d)| {
                        if mask[i] {
                            embed::cosine(&qv, &d.vector)
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }
        };
        let mut lexical_scores: Vec<f32> = match mode {
            SearchMode::Dense => vec![0.0; self.docs.len()],
            SearchMode::Lexical | SearchMode::Hybrid => self.bm25.scores(query),
        };
        for (i, m) in mask.iter().enumerate() {
            if !m {
                lexical_scores[i] = 0.0;
            }
        }

        let hits = match mode {
            SearchMode::Dense => rank_by_scores(
                &self.docs,
                &dense_scores,
                &lexical_scores,
                limit,
                ScoreKind::Dense,
            ),
            SearchMode::Lexical => rank_by_scores(
                &self.docs,
                &lexical_scores,
                &dense_scores,
                limit,
                ScoreKind::Lexical,
            ),
            SearchMode::Hybrid => hybrid_rrf(&self.docs, &dense_scores, &lexical_scores, limit),
        };
        // Drop zero-score hits that slipped through when everything was filtered out of ranking ties
        Ok(hits
            .into_iter()
            .filter(|h| {
                filter.is_empty()
                    || self
                        .docs
                        .iter()
                        .find(|d| d.source_path == h.source_path && d.chunk_index == h.chunk_index)
                        .is_some_and(|d| filter.matches(d))
            })
            .collect())
    }

    #[allow(dead_code)] // used by unit tests; kept for dense-only callers
    pub fn query_vector(&self, qv: &[f32], limit: usize) -> Vec<ResultHit> {
        let dense: Vec<f32> = self
            .docs
            .iter()
            .map(|d| embed::cosine(qv, &d.vector))
            .collect();
        let lexical = vec![0.0; self.docs.len()];
        rank_by_scores(&self.docs, &dense, &lexical, limit, ScoreKind::Dense)
    }
}

fn top_ids(scores: &[f32], pool: usize) -> Vec<usize> {
    let mut idxs: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    idxs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idxs.into_iter()
        .filter(|(_, s)| *s > 0.0)
        .take(pool)
        .map(|(i, _)| i)
        .collect()
}

fn hybrid_rrf(docs: &[Document], dense: &[f32], lexical: &[f32], limit: usize) -> Vec<ResultHit> {
    let pool = CANDIDATE_POOL.max(limit);
    let dense_rank = top_ids(dense, pool);
    let lex_rank = top_ids(lexical, pool);
    // If one side is all zeros, fall back to the other.
    let fused = if dense_rank.is_empty() {
        lex_rank.into_iter().map(|i| (i, lexical[i])).collect()
    } else if lex_rank.is_empty() {
        dense_rank.into_iter().map(|i| (i, dense[i])).collect()
    } else {
        reciprocal_rank_fusion(&[dense_rank, lex_rank], RRF_K)
    };
    let mut per_source = std::collections::HashMap::<&str, usize>::new();
    fused
        .into_iter()
        .filter(|(i, _)| lexical[*i] > 0.0 || dense[*i] >= MIN_DENSE_SCORE)
        .filter(|(i, _)| {
            let count = per_source.entry(docs[*i].source_path.as_str()).or_default();
            if *count >= 2 {
                false
            } else {
                *count += 1;
                true
            }
        })
        .take(limit)
        .map(|(i, score)| hit_from(docs, i, score, dense[i], lexical[i]))
        .collect()
}

fn rank_by_scores(
    docs: &[Document],
    primary: &[f32],
    secondary: &[f32],
    limit: usize,
    kind: ScoreKind,
) -> Vec<ResultHit> {
    let mut idxs: Vec<usize> = (0..docs.len()).filter(|&i| primary[i] > 0.0).collect();
    idxs.sort_by(|a, b| {
        primary[*b]
            .partial_cmp(&primary[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| docs[*a].source_path.cmp(&docs[*b].source_path))
            .then_with(|| docs[*a].chunk_index.cmp(&docs[*b].chunk_index))
    });
    idxs.truncate(limit.min(docs.len()));
    idxs.into_iter()
        .map(|i| match kind {
            ScoreKind::Dense => hit_from(docs, i, primary[i], primary[i], secondary[i]),
            ScoreKind::Lexical => hit_from(docs, i, primary[i], secondary[i], primary[i]),
        })
        .collect()
}

fn hit_from(
    docs: &[Document],
    i: usize,
    score: f32,
    dense_score: f32,
    lexical_score: f32,
) -> ResultHit {
    let d = &docs[i];
    ResultHit {
        score,
        dense_score,
        lexical_score,
        source_path: d.source_path.clone(),
        chunk_index: d.chunk_index,
        headings: d.headings.clone(),
        metadata: d.metadata.clone(),
        text: d.text.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Document;

    fn doc(id: i64, text: &str, vector: Vec<f32>) -> Document {
        Document {
            id,
            source_path: format!("{id}.md"),
            chunk_index: 0,
            text: text.into(),
            headings: vec![],
            metadata: Default::default(),
            vector,
        }
    }

    #[test]
    fn ranks_by_cosine() {
        let docs = vec![
            doc(1, "dogs", vec![1.0, 0.0, 0.0]),
            doc(2, "cats", vec![0.0, 1.0, 0.0]),
        ];
        let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let idx = Index {
            bm25: Bm25::build(&texts),
            docs,
        };
        let hits = idx.query_vector(&[0.9, 0.1, 0.0], 2);
        assert_eq!(hits[0].text, "dogs");
    }

    #[test]
    fn hybrid_surfaces_exact_id() {
        // Dense would prefer the storage vector; lexical should pull the username doc up.
        let docs = vec![
            doc(
                1,
                "storage team owns persistent volumes and ceph",
                vec![1.0, 0.0, 0.0],
            ),
            doc(
                2,
                "jsmith manages platform virtualization",
                vec![0.0, 1.0, 0.0],
            ),
        ];
        let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let idx = Index {
            bm25: Bm25::build(&texts),
            docs,
        };
        let dense = vec![0.95f32, 0.10]; // favors doc 0
        let lexical = idx.bm25.scores("jsmith");
        let hits = hybrid_rrf(&idx.docs, &dense, &lexical, 2);
        assert!(hits[0].text.contains("jsmith"), "{hits:?}");
    }

    fn doc_at(
        id: i64,
        path: &str,
        chunk_index: usize,
        text: &str,
        headings: &[&str],
        tags: &[&str],
        vector: Vec<f32>,
    ) -> Document {
        let mut metadata = serde_json::Map::new();
        if !tags.is_empty() {
            metadata.insert(
                "tags".into(),
                serde_json::Value::Array(
                    tags.iter()
                        .map(|t| serde_json::Value::String((*t).into()))
                        .collect(),
                ),
            );
        }
        Document {
            id,
            source_path: path.into(),
            chunk_index,
            text: text.into(),
            headings: headings.iter().map(|s| (*s).into()).collect(),
            metadata,
            vector,
        }
    }

    #[test]
    fn lexical_result_reports_lexical_not_dense_score() {
        let docs = vec![doc(1, "exact identifier", vec![1.0, 0.0])];
        let dense = vec![0.0];
        let lexical = vec![4.2];
        let hits = rank_by_scores(&docs, &lexical, &dense, 1, ScoreKind::Lexical);
        assert_eq!(hits[0].dense_score, 0.0);
        assert_eq!(hits[0].lexical_score, 4.2);
    }

    #[test]
    fn equal_scores_use_stable_citation_order() {
        let docs = vec![
            doc_at(2, "z.md", 0, "same", &[], &[], vec![1.0]),
            doc_at(1, "a.md", 0, "same", &[], &[], vec![1.0]),
        ];
        let hits = rank_by_scores(&docs, &[1.0, 1.0], &[0.0, 0.0], 2, ScoreKind::Dense);
        assert_eq!(hits[0].source_path, "a.md");
        assert_eq!(hits[1].source_path, "z.md");
    }

    #[test]
    fn path_prefix_filter() {
        let filter = SearchFilter {
            path_prefix: Some("teams/".into()),
            ..Default::default()
        };
        let team = doc_at(
            1,
            "teams/storage.md",
            0,
            "storage",
            &[],
            &[],
            vec![1.0, 0.0],
        );
        let guide = doc_at(
            2,
            "guides/onboarding.md",
            0,
            "onboard",
            &[],
            &[],
            vec![0.0, 1.0],
        );
        assert!(filter.matches(&team));
        assert!(!filter.matches(&guide));
    }

    #[test]
    fn heading_and_tag_filters() {
        let d = doc_at(
            1,
            "org/process.md",
            0,
            "body",
            &["Release", "Backports"],
            &["ga", "storage"],
            vec![1.0],
        );
        assert!(SearchFilter {
            heading: Some("back".into()),
            ..Default::default()
        }
        .matches(&d));
        assert!(!SearchFilter {
            heading: Some("network".into()),
            ..Default::default()
        }
        .matches(&d));
        assert!(SearchFilter {
            tag: Some("Storage".into()),
            ..Default::default()
        }
        .matches(&d));
        assert!(!SearchFilter {
            tag: Some("network".into()),
            ..Default::default()
        }
        .matches(&d));
    }

    /// End-to-end parity test: the `--tag` CLI flag and the MCP `tag` argument
    /// both build a `SearchFilter { tag }` and rely on the same
    /// `metadata_has_tag` matching. This test drives that identical path with
    /// real front-matter tag parsing (`index::split_markdown`), so it proves
    /// markdown `tags:` front-matter flows into `metadata.tags` and can be
    /// filtered by tag exactly the same way a CLI or MCP caller would.
    #[test]
    fn cli_and_mcp_tag_filter_are_equivalent_over_front_matter() {
        let tagged_md = "---\ntags:\n  - Storage\n  - kubernetes\n---\n\n# Storage Team\nOwners of the storage cluster.\n";
        let untagged_md =
            "---\ntags:\n  - networking\n---\n\n# Networking\nModels the network paths.\n";

        let tagged_chunk = &crate::index::split_markdown("teams/storage.md", tagged_md)[0];
        let untagged_chunk = &crate::index::split_markdown("teams/net.md", untagged_md)[0];

        fn as_doc(c: &crate::index::Chunk) -> crate::store::Document {
            crate::store::Document {
                id: 0,
                source_path: c.source_path.clone(),
                chunk_index: c.chunk_index,
                text: c.text.clone(),
                headings: c.headings.clone(),
                metadata: c.metadata.clone(),
                vector: vec![],
            }
        }

        let tagged_doc = as_doc(tagged_chunk);
        let untagged_doc = as_doc(untagged_chunk);

        // Front-matter tags were parsed into metadata.
        let tags = tagged_doc.metadata.get("tags").expect("tags present");
        let arr = tags.as_array().expect("tags is an array");
        let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(names, ["Storage", "kubernetes"]);

        // This is the exact filter both `search --tag <t>` (CLI) and
        // `semantic_search(tag=<t>)` (MCP) build.
        let filter = SearchFilter {
            tag: Some("storage".into()),
            ..Default::default()
        };
        assert!(
            filter.matches(&tagged_doc),
            "case-insensitive tag match on the tagged doc (CLI == MCP)"
        );
        assert!(
            !filter.matches(&untagged_doc),
            "doc with a different tag must not match"
        );

        // A second tag on the same doc also matches.
        assert!(SearchFilter {
            tag: Some("KUBERNETES".into()),
            ..Default::default()
        }
        .matches(&tagged_doc));
    }

    #[test]
    fn get_chunk_by_citation() {
        let docs = vec![
            doc_at(1, "a.md", 0, "first", &[], &[], vec![1.0, 0.0]),
            doc_at(2, "a.md", 1, "second", &[], &[], vec![0.0, 1.0]),
            doc_at(3, "b.md", 0, "other", &[], &[], vec![0.5, 0.5]),
        ];
        let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let idx = Index {
            bm25: Bm25::build(&texts),
            docs,
        };
        assert_eq!(idx.get("a.md", 1).map(|d| d.text.as_str()), Some("second"));
        assert!(idx.get("a.md", 9).is_none());
        let all_a = idx.get_by_path("a.md");
        assert_eq!(all_a.len(), 2);
        assert_eq!(all_a[0].chunk_index, 0);
        assert_eq!(all_a[1].chunk_index, 1);
    }

    #[test]
    fn rank_by_scores_filters_zeros() {
        let docs = vec![
            doc(1, "hit", vec![1.0, 0.0]),
            doc(2, "zero", vec![0.0, 1.0]),
        ];
        let primary = vec![0.8, 0.0];
        let secondary = vec![0.0, 0.0];
        let ranked = rank_by_scores(&docs, &primary, &secondary, 5, ScoreKind::Dense);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].source_path, "1.md");
    }
}
