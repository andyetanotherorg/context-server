//! MCP stdio server tools.

use crate::embed::Embedder;
use crate::search::{Index, SearchFilter, SearchMode};
use crate::store::Db;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, ServerHandler,
};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const MAX_SEARCH_RESULTS: usize = 20;
const MAX_LIST_RESULTS: usize = 200;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    #[schemars(
        description = "Natural-language search query. Include names, teams, acronyms, or topic keywords (e.g. 'who manages the storage team', 'backport process')."
    )]
    pub query: String,
    #[schemars(description = "Max passages to return (default 5, maximum 20)")]
    pub limit: Option<usize>,
    #[schemars(
        description = "Only search chunks whose source_path starts with this prefix (e.g. 'teams/' or 'guides/')."
    )]
    pub path_prefix: Option<String>,
    #[schemars(
        description = "Only search chunks where a heading contains this substring (case-insensitive)."
    )]
    pub heading: Option<String>,
    #[schemars(
        description = "Only search chunks tagged with this value in metadata.tags (case-insensitive)."
    )]
    pub tag: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ListRequest {
    #[schemars(description = "Max chunks to list (default 50, maximum 200)")]
    pub limit: Option<usize>,
    #[schemars(description = "Only list chunks whose source_path starts with this prefix.")]
    pub path_prefix: Option<String>,
    #[schemars(description = "Zero-based chunk offset for pagination")]
    pub offset: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetDocumentRequest {
    #[schemars(
        description = "Indexed source path as returned in search hits (e.g. 'teams/storage.md')."
    )]
    pub source_path: String,
    #[schemars(
        description = "Chunk index within that file (the number after '#' in citations like path#3). Omit to return all chunks for the file."
    )]
    pub chunk_index: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetChunkByIdRequest {
    #[schemars(description = "Stable chunk ID returned in chunk metadata")]
    pub chunk_id: String,
}

pub struct ContextService {
    db: Mutex<Db>,
    index: Mutex<Index>,
    db_path: PathBuf,
    db_stamp: Mutex<Option<DbStamp>>,
    embedder: Mutex<Embedder>,
    instructions: Mutex<String>,
    tool_router: ToolRouter<Self>,
}

/// Detects when the on-disk DB (and its WAL) has been rewritten by a separate
/// `index` process, so `serve` can hot-reload the search index in place instead
/// of requiring a restart.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DbStamp {
    mtime: std::time::SystemTime,
    size: u64,
    wal_mtime: Option<std::time::SystemTime>,
    wal_size: u64,
    generation: Option<String>,
    file_id: u64,
}

fn generation_for(path: &Path) -> Option<String> {
    let conn =
        rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .ok()?;
    conn.query_row(
        "SELECT value FROM meta WHERE key = 'generation'",
        [],
        |row| row.get(0),
    )
    .ok()
}

fn db_stamp_for(path: &Path) -> Option<DbStamp> {
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(path).ok()?;
    let wal = std::fs::metadata(format!("{}-wal", path.display())).ok();
    Some(DbStamp {
        mtime: meta.modified().ok()?,
        size: meta.len(),
        wal_mtime: wal.as_ref().and_then(|m| m.modified().ok()),
        wal_size: wal.as_ref().map(|m| m.len()).unwrap_or(0),
        generation: generation_for(path),
        #[cfg(unix)]
        file_id: meta.ino(),
        #[cfg(not(unix))]
        file_id: 0,
    })
}

/// Default MCP instructions shown to the agent when the corpus was indexed
/// without an explicit `--instructions`. Kept measured so the agent only
/// searches when the question plausibly touches the knowledge base, instead of
/// over-firing on unrelated questions. Users who want the strongly-worded
/// "always search, never guess" guidance can set it explicitly via
/// `index --instructions` (stored in DB meta).
const DEFAULT_INSTRUCTIONS: &str =
    "This server searches an indexed markdown knowledge base (teams, people, \
ownership, processes, guides). If a question may be answerable from this \
corpus, call semantic_search rather than relying only on general knowledge. \
Use list_documents to browse what is indexed, and cite passages as \
source_path#chunk_index (fetch full text via get_document). For unrelated \
general questions, answer normally without forcing a search.";

impl ContextService {
    pub fn new(db: Db, index: Index, embedder: Embedder, db_path: PathBuf) -> Self {
        let instructions = db
            .get_meta("instructions")
            .ok()
            .flatten()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_INSTRUCTIONS.to_string());
        let db_stamp = db_stamp_for(&db_path);
        Self {
            db: Mutex::new(db),
            index: Mutex::new(index),
            db_path,
            db_stamp: Mutex::new(db_stamp),
            embedder: Mutex::new(embedder),
            instructions: Mutex::new(instructions),
            tool_router: Self::tool_router(),
        }
    }

    /// If a separate `index` process rewrote the on-disk DB since we last
    /// loaded, reload the search index (and any MCP `instructions` meta) in
    /// place. Called at the top of every read tool so long-lived MCP sessions
    /// stay fresh after a re-index. Errors reloading leave the previous state
    /// intact (e.g. the DB is mid-write) and are retried next call.
    fn refresh(&self) {
        let Some((new_db, new_index, instructions)) =
            try_reload(&self.db_path, &mut self.db_stamp.lock().unwrap())
        else {
            return;
        };
        *self.db.lock().unwrap() = new_db;
        *self.index.lock().unwrap() = new_index;
        *self.instructions.lock().unwrap() = instructions;
        eprintln!(
            "context-server: reloaded index from {}",
            self.db_path.display()
        );
    }
}

/// Re-open and reload the search index from `path`, but only if the on-disk DB
/// changed since `last` (which is updated on success). Returns `None` when the
/// DB is unchanged or cannot be safely read yet (mid-write).
fn try_reload(path: &Path, last: &mut Option<DbStamp>) -> Option<(Db, Index, String)> {
    let current = db_stamp_for(path)?;
    if last.as_ref().is_some_and(|previous| {
        previous == &current
            || previous.file_id == current.file_id
                && previous.generation.is_some()
                && previous.generation == current.generation
    }) {
        return None;
    }
    let new_db = Db::open_read_only(path).ok()?;
    let new_index = Index::load(&new_db).ok()?;
    let instructions = new_db
        .get_meta("instructions")
        .ok()
        .flatten()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_INSTRUCTIONS.to_string());
    let after = db_stamp_for(path)?;
    if after.file_id != current.file_id || after.generation != current.generation {
        // The committed database changed while the snapshot was loading. Keep
        // serving the prior snapshot and retry on the next tool call. WAL
        // timestamps may move during a read, so identity is inode+generation.
        return None;
    }
    *last = Some(after);
    Some((new_db, new_index, instructions))
}

fn filter_from(
    path_prefix: Option<String>,
    heading: Option<String>,
    tag: Option<String>,
) -> SearchFilter {
    SearchFilter {
        path_prefix,
        heading,
        tag,
    }
}

#[tool_router]
impl ContextService {
    #[tool(
        description = "REQUIRED for org/knowledge questions: search the indexed markdown knowledge base (people, teams, ownership, processes, guides). Call this instead of guessing whenever the user asks who owns something, how a process works, or anything that may be in team/org docs. Returns ranked passages with scores and citations (source_path#chunk_index). Optional path_prefix/heading/tag narrow the corpus."
    )]
    fn semantic_search(
        &self,
        Parameters(SearchRequest {
            query,
            limit,
            path_prefix,
            heading,
            tag,
        }): Parameters<SearchRequest>,
    ) -> String {
        let limit = limit.unwrap_or(5).clamp(1, MAX_SEARCH_RESULTS);
        if query.trim().is_empty() {
            return "error: query is required".into();
        }
        self.refresh();
        let filter = filter_from(path_prefix, heading, tag);
        let mut emb = match self.embedder.lock() {
            Ok(e) => e,
            Err(_) => return "error: embedder lock poisoned; restart the server".into(),
        };
        let index = match self.index.lock() {
            Ok(i) => i,
            Err(_) => return "error: index lock poisoned; restart the server".into(),
        };
        match index.query_filtered(&mut emb, &query, limit, SearchMode::Hybrid, &filter) {
            Ok(hits) => format_hits(&query, &hits),
            Err(e) => format!("error: {e:#}"),
        }
    }

    #[tool(
        description = "List what is indexed in the knowledge base (paths, headings, previews). Use when the user asks what docs are available or to browse the corpus. Optional path_prefix scopes the listing."
    )]
    fn list_documents(
        &self,
        Parameters(ListRequest {
            limit,
            path_prefix,
            offset,
        }): Parameters<ListRequest>,
    ) -> String {
        let limit = limit.unwrap_or(50).clamp(1, MAX_LIST_RESULTS);
        self.refresh();
        let db = match self.db.lock() {
            Ok(d) => d,
            Err(_) => return "error: database lock poisoned; restart the server".into(),
        };
        match db.list_page(limit, offset.unwrap_or(0), path_prefix.as_deref()) {
            Ok(filtered) => {
                let mut out = format!("Showing {} document chunks:\n", filtered.len());
                for d in filtered {
                    let preview = crate::index::truncate_preview(&d.text, 157);
                    let heading = if d.headings.is_empty() {
                        "(root)".into()
                    } else {
                        d.headings.join(" > ")
                    };
                    out.push_str(&format!(
                        "- {}#{} [{}] {}\n",
                        d.source_path, d.chunk_index, heading, preview
                    ));
                }
                out
            }
            Err(e) => format!("error: {e:#}"),
        }
    }

    #[tool(description = "Fetch a full chunk by stable chunk ID.")]
    fn get_chunk_by_id(
        &self,
        Parameters(GetChunkByIdRequest { chunk_id }): Parameters<GetChunkByIdRequest>,
    ) -> String {
        self.refresh();
        let index = match self.index.lock() {
            Ok(index) => index,
            Err(_) => return "error: index lock poisoned; restart the server".into(),
        };
        match index.get_by_chunk_id(chunk_id.trim()) {
            Some(document) => format_document(document),
            None => format!("error: no chunk with id {}", chunk_id.trim()),
        }
    }

    #[tool(
        description = "Fetch a full indexed chunk by citation for quoting. Pass source_path and chunk_index from a search hit (path#N). Omit chunk_index to return every chunk in that file."
    )]
    fn get_document(
        &self,
        Parameters(GetDocumentRequest {
            source_path,
            chunk_index,
        }): Parameters<GetDocumentRequest>,
    ) -> String {
        let path = source_path.trim();
        if path.is_empty() {
            return "error: source_path is required".into();
        }
        self.refresh();
        let index = self.index.lock().unwrap();
        match chunk_index {
            Some(idx) => match index.get(path, idx) {
                Some(d) => format_document(d),
                None => format!("error: no chunk {path}#{idx}"),
            },
            None => {
                let docs = index.get_by_path(path);
                if docs.is_empty() {
                    return format!("error: no chunks for {path}");
                }
                let mut out = format!("{} chunk(s) in {path}:\n", docs.len());
                for d in docs {
                    out.push('\n');
                    out.push_str(&format_document(d));
                    out.push('\n');
                }
                out
            }
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ContextService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(self.instructions.lock().unwrap().clone())
    }
}

fn format_document(d: &crate::store::Document) -> String {
    let heading = if d.headings.is_empty() {
        String::new()
    } else {
        format!("Headings: {}\n", d.headings.join(" > "))
    };
    let provenance = d
        .metadata
        .get("_chunk_id")
        .and_then(|value| value.as_str())
        .map(|id| format!("Chunk ID: {id}\n"))
        .unwrap_or_default();
    format!(
        "Citation: {}#{}\n{}{}---\n{}\n",
        d.source_path, d.chunk_index, provenance, heading, d.text
    )
}

fn format_hits(query: &str, hits: &[crate::search::ResultHit]) -> String {
    let mut out = format!("Results for {query:?} (hybrid):\n");
    if hits.is_empty() {
        out.push_str("(no hits)\n");
        return out;
    }
    for (i, h) in hits.iter().enumerate() {
        let chunk_id = h
            .metadata
            .get("_chunk_id")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        let heading = if h.headings.is_empty() {
            String::new()
        } else {
            format!(" [{}]", h.headings.join(" > "))
        };
        out.push_str(&format!(
            "\n{}. score={:.4} (dense={:.4} lexical={:.4})  {}#{}{} id={}\n{}\n",
            i + 1,
            h.score,
            h.dense_score,
            h.lexical_score,
            h.source_path,
            h.chunk_index,
            heading,
            chunk_id,
            h.text
        ));
    }
    out.push_str("\nUse get_document with source_path and chunk_index to fetch a full citation.\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed;
    use crate::index::Chunk;
    use tempfile::tempdir;

    fn chunk(path: &str, text: &str) -> Chunk {
        Chunk {
            source_path: path.into(),
            chunk_index: 0,
            text: text.into(),
            headings: vec![],
            metadata: serde_json::Map::new(),
        }
    }

    /// Write the given docs into `path` via a fresh connection (simulates a
    /// separate `context-server index` process rewriting the on-disk DB).
    fn rewrite_db(path: &std::path::Path, docs: &[(&str, &str)]) {
        let mut db = Db::open(path).unwrap();
        let chunks: Vec<Chunk> = docs.iter().map(|(p, t)| chunk(p, t)).collect();
        let vectors: Vec<Vec<f32>> = chunks.iter().map(|_| vec![1.0f32; embed::DIM]).collect();
        db.replace_all(&chunks, &vectors, None).unwrap();
    }

    #[test]
    fn db_stamp_changes_when_db_rewritten() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.db");
        rewrite_db(&path, &[("a.md", "hello")]);

        let before = db_stamp_for(&path).expect("stamp after first write");
        rewrite_db(&path, &[("a.md", "hello"), ("b.md", "world")]);
        let after = db_stamp_for(&path).expect("stamp after rewrite");

        assert!(
            before != after,
            "db stamp must change when the DB is rewritten"
        );
    }

    #[test]
    fn try_reload_returns_none_when_unchanged_and_some_after_reindex() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.db");

        rewrite_db(&path, &[("a.md", "hello world")]);
        let mut last = db_stamp_for(&path);

        // Unchanged -> no reload.
        assert!(
            try_reload(&path, &mut last).is_none(),
            "no reload when the DB is unchanged"
        );

        // Re-index: add a second doc; the stamp now differs -> reload.
        rewrite_db(&path, &[("a.md", "hello world"), ("b.md", "second doc")]);
        let (new_db, new_index, _instructions) =
            try_reload(&path, &mut last).expect("reload after re-index");
        assert_eq!(new_index.len(), 2, "reloaded index has both docs");
        assert_eq!(
            new_index.get_by_path("b.md").len(),
            1,
            "new doc present after reload"
        );
        drop(new_db);

        // After a successful reload the stamp is recorded -> next call is None.
        assert!(
            try_reload(&path, &mut last).is_none(),
            "subsequent reload skipped once stamp is recorded"
        );
    }
}
