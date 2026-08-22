//! MCP stdio server tools.

use crate::embed::Embedder;
use crate::search::{Index, SearchFilter, SearchMode};
use crate::store::Db;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, ServerHandler,
};
use std::sync::Mutex;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    #[schemars(
        description = "Natural-language search query. Include names, teams, acronyms, or topic keywords (e.g. 'who manages the storage team', 'backport process')."
    )]
    pub query: String,
    #[schemars(description = "Max passages to return (default 5)")]
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
    #[schemars(description = "Max chunks to list (default 50)")]
    pub limit: Option<usize>,
    #[schemars(description = "Only list chunks whose source_path starts with this prefix.")]
    pub path_prefix: Option<String>,
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

pub struct ContextService {
    pub db: Mutex<Db>,
    pub index: Index,
    pub embedder: Mutex<Embedder>,
    instructions: String,
    tool_router: ToolRouter<Self>,
}

const DEFAULT_INSTRUCTIONS: &str =
    "Organizational markdown knowledge base (teams, people, ownership, processes, guides). \
ALWAYS call semantic_search before answering questions about \
who owns what, team structure, managers, acronyms, backports, or internal process — \
do not guess from general knowledge. Use list_documents to browse the corpus. \
Cite passages as source_path#chunk_index and call get_document to fetch a full chunk by that citation. \
Use path_prefix/heading/tag filters to scope search (e.g. path_prefix='teams/').";

impl ContextService {
    pub fn new(db: Db, index: Index, embedder: Embedder) -> Self {
        let instructions = db
            .get_meta("instructions")
            .ok()
            .flatten()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_INSTRUCTIONS.to_string());
        Self {
            db: Mutex::new(db),
            index,
            embedder: Mutex::new(embedder),
            instructions,
            tool_router: Self::tool_router(),
        }
    }
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
        let limit = limit.unwrap_or(5);
        if query.trim().is_empty() {
            return "error: query is required".into();
        }
        let filter = filter_from(path_prefix, heading, tag);
        let mut emb = self.embedder.lock().unwrap();
        match self
            .index
            .query_filtered(&mut emb, &query, limit, SearchMode::Hybrid, &filter)
        {
            Ok(hits) => format_hits(&query, &hits),
            Err(e) => format!("error: {e:#}"),
        }
    }

    #[tool(
        description = "List what is indexed in the knowledge base (paths, headings, previews). Use when the user asks what docs are available or to browse the corpus. Optional path_prefix scopes the listing."
    )]
    fn list_documents(
        &self,
        Parameters(ListRequest { limit, path_prefix }): Parameters<ListRequest>,
    ) -> String {
        let limit = limit.unwrap_or(50);
        let db = self.db.lock().unwrap();
        match db.list(limit.saturating_mul(4).max(limit)) {
            Ok(docs) => {
                let filtered: Vec<_> = docs
                    .into_iter()
                    .filter(|d| match &path_prefix {
                        Some(p) if !p.is_empty() => d.source_path.starts_with(p.as_str()),
                        _ => true,
                    })
                    .take(limit)
                    .collect();
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
        match chunk_index {
            Some(idx) => match self.index.get(path, idx) {
                Some(d) => format_document(d),
                None => format!("error: no chunk {path}#{idx}"),
            },
            None => {
                let docs = self.index.get_by_path(path);
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
            .with_instructions(self.instructions.clone())
    }
}

fn format_document(d: &crate::store::Document) -> String {
    let heading = if d.headings.is_empty() {
        String::new()
    } else {
        format!("Headings: {}\n", d.headings.join(" > "))
    };
    format!(
        "Citation: {}#{}\n{}---\n{}\n",
        d.source_path, d.chunk_index, heading, d.text
    )
}

fn format_hits(query: &str, hits: &[crate::search::ResultHit]) -> String {
    let mut out = format!("Results for {query:?} (hybrid):\n");
    if hits.is_empty() {
        out.push_str("(no hits)\n");
        return out;
    }
    for (i, h) in hits.iter().enumerate() {
        let heading = if h.headings.is_empty() {
            String::new()
        } else {
            format!(" [{}]", h.headings.join(" > "))
        };
        out.push_str(&format!(
            "\n{}. score={:.4} (dense={:.4} lexical={:.4})  {}#{}{}\n{}\n",
            i + 1,
            h.score,
            h.dense_score,
            h.lexical_score,
            h.source_path,
            h.chunk_index,
            heading,
            h.text
        ));
    }
    out.push_str("\nUse get_document with source_path and chunk_index to fetch a full citation.\n");
    out
}
