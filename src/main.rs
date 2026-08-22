mod bm25;
mod embed;
mod eval;
mod index;
mod mcp;
mod remote;
mod search;
mod store;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use search::SearchMode;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "context-server",
    about = "Semantic search MCP server for markdown knowledge bases"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Index markdown into the SQLite database
    Index {
        /// Markdown file or directory
        #[arg(long)]
        input: PathBuf,
        /// Local SQLite database path
        #[arg(long, default_value = "context.db")]
        db: PathBuf,
        /// Chunk and print without embedding
        #[arg(long)]
        dry_run: bool,
        /// Embedding batch size
        #[arg(long, default_value_t = 16)]
        batch: usize,
        /// Re-embed every collected file even if content hashes match
        #[arg(long)]
        full: bool,
        /// Delete indexed paths missing from --input; empty input deletes all documents
        #[arg(long)]
        sync: bool,
        /// MCP server instructions stored in DB meta (when to call this corpus)
        #[arg(long)]
        instructions: Option<String>,
        /// Read MCP instructions from a UTF-8 text file
        #[arg(long)]
        instructions_file: Option<PathBuf>,
    },
    /// Start the MCP server (stdio)
    Serve {
        /// Local path, or `gs://bucket/object` /
        /// `gs://projects/PROJECT/buckets/BUCKET/objects/OBJECT`
        #[arg(long, default_value = "context.db")]
        db: String,
    },
    /// Search the database (CLI)
    Search {
        /// Local path, or `gs://bucket/object` /
        /// `gs://projects/PROJECT/buckets/BUCKET/objects/OBJECT`
        #[arg(long, default_value = "context.db")]
        db: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        /// Search mode: hybrid (default), dense, or lexical
        #[arg(long, default_value = "hybrid")]
        mode: String,
        /// Only search source_path values with this prefix
        #[arg(long)]
        path_prefix: Option<String>,
        /// Only search chunks whose heading contains this substring
        #[arg(long)]
        heading: Option<String>,
        /// Only search chunks with this metadata tag
        #[arg(long)]
        tag: Option<String>,
        /// Query text
        query: Vec<String>,
    },
    /// Fetch a chunk by citation (source_path + chunk index)
    Get {
        #[arg(long, default_value = "context.db")]
        db: String,
        /// Indexed source path (e.g. teams/storage.md)
        #[arg(long)]
        path: String,
        /// Chunk index; omit to print all chunks for the path
        #[arg(long)]
        chunk: Option<usize>,
    },
    /// Embed a search query (smoke test; applies BGE query instruction)
    Embed { text: Vec<String> },
    /// Evaluate hybrid retrieval against golden citations
    Eval {
        #[arg(long, default_value = "context.db")]
        db: String,
        #[arg(long, default_value = "eval/cases.json")]
        cases: PathBuf,
        #[arg(long, default_value_t = 5)]
        limit: usize,
    },
    /// Show corpus and index metadata
    Status {
        #[arg(long, default_value = "context.db")]
        db: String,
        #[arg(long)]
        json: bool,
    },
    /// Validate database integrity and model compatibility
    Validate {
        #[arg(long, default_value = "context.db")]
        db: String,
    },
    /// Show embedding model cache status
    ModelStatus,
    /// Download and initialize the embedding model cache
    ModelDownload,
}

fn main() -> Result<()> {
    // google-cloud-storage / reqwest enable both rustls aws-lc-rs and ring features;
    // pick an explicit process default before any TLS.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let cli = Cli::parse();
    match cli.command {
        Commands::Index {
            input,
            db,
            dry_run,
            batch,
            full,
            sync,
            instructions,
            instructions_file,
        } => run_index(IndexRun {
            input,
            db_path: db,
            dry_run,
            batch,
            full,
            sync,
            instructions,
            instructions_file,
        }),
        Commands::Serve { db } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(run_serve(db))
        }
        Commands::Search {
            db,
            limit,
            mode,
            path_prefix,
            heading,
            tag,
            query,
        } => run_search(db, limit, mode, path_prefix, heading, tag, query),
        Commands::Get { db, path, chunk } => run_get(db, path, chunk),
        Commands::Embed { text } => run_embed(text),
        Commands::Eval { db, cases, limit } => run_eval(db, cases, limit),
        Commands::Status { db, json } => run_status(db, json),
        Commands::Validate { db } => run_validate(db),
        Commands::ModelStatus => run_model_status(),
        Commands::ModelDownload => run_model_download(),
    }
}

struct IndexRun {
    input: PathBuf,
    db_path: PathBuf,
    dry_run: bool,
    batch: usize,
    full: bool,
    sync: bool,
    instructions: Option<String>,
    instructions_file: Option<PathBuf>,
}

struct PreparedFile {
    path: String,
    hash: String,
    chunks: Vec<index::Chunk>,
    vectors: Vec<Vec<f32>>,
}

fn run_index(
    IndexRun {
        input,
        db_path,
        dry_run,
        batch,
        full,
        sync,
        instructions,
        instructions_file,
    }: IndexRun,
) -> Result<()> {
    let chunks = index::collect(&input)?;
    validate_collected_chunks(chunks.len(), sync)
        .with_context(|| format!("index input {}", input.display()))?;
    if dry_run {
        println!("chunked {} pieces from {}", chunks.len(), input.display());
        for c in &chunks {
            println!("  {}: {}", c.source_path, index::format_chunk_debug(c));
        }
        return Ok(());
    }

    let instructions = match (instructions, instructions_file) {
        (Some(_), Some(_)) => {
            bail!("pass only one of --instructions or --instructions-file");
        }
        (Some(text), None) => Some(text),
        (None, Some(path)) => {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("read instructions file {}", path.display()))?;
            Some(text.trim().to_string())
        }
        (None, None) => None,
    };
    if let Some(ref text) = instructions {
        if text.is_empty() {
            bail!("instructions text is empty");
        }
    }

    let grouped = index::group_chunks_by_path(chunks);
    let files: Vec<(String, String, Vec<index::Chunk>)> = grouped
        .into_iter()
        .map(|(path, group)| {
            let hash = index::hash_chunks(group.iter());
            (path, hash, group)
        })
        .collect();

    let mut db = store::Db::open(&db_path)?;
    let migration_required = db.needs_full_reembed()?;
    validate_index_mode(sync, migration_required)?;
    let force_full = full || migration_required;
    let mut existing = db.file_hashes()?;
    for path in db.source_paths()? {
        existing.entry(path).or_default();
    }
    let incoming_hashes: Vec<(String, String)> = files
        .iter()
        .map(|(p, h, _)| (p.clone(), h.clone()))
        .collect();
    let plan = index::plan_index(&incoming_hashes, &existing, sync, force_full);

    println!(
        "indexing {} files from {} -> {} ({}){}",
        files.len(),
        input.display(),
        db_path.display(),
        embed::MODEL_ID,
        if force_full && !full {
            " [full re-embed: model or chunker changed]"
        } else if full {
            " [--full]"
        } else {
            ""
        }
    );
    if !plan.skip.is_empty() {
        eprintln!("  skip {} unchanged", plan.skip.len());
    }
    if !plan.embed.is_empty() {
        eprintln!(
            "  embed {} file(s): {}",
            plan.embed.len(),
            plan.embed.join(", ")
        );
    }
    if !plan.prune.is_empty() {
        eprintln!(
            "  prune {} removed: {}",
            plan.prune.len(),
            plan.prune.join(", ")
        );
    }
    if !sync {
        eprintln!("  upsert only; pass --sync to prune paths missing from input");
    }

    let embed_set: std::collections::HashSet<&str> =
        plan.embed.iter().map(|s| s.as_str()).collect();
    let mut updates: Vec<PreparedFile> = Vec::new();
    if plan.embed.is_empty() {
        let _ = files;
        if plan.prune.is_empty() && instructions.is_none() {
            println!("nothing to do; {}", db.summary()?);
            return Ok(());
        }
    } else {
        let mut emb = embed::Embedder::new()?;
        let batch = batch.max(1);
        for (path, hash, group) in files {
            if !embed_set.contains(path.as_str()) {
                continue;
            }
            let mut vectors = Vec::with_capacity(group.len());
            for i in (0..group.len()).step_by(batch) {
                let end = (i + batch).min(group.len());
                eprintln!("  embedding {} {}-{}/{}", path, i + 1, end, group.len());
                let texts: Vec<String> = group[i..end].iter().map(|c| c.text.clone()).collect();
                let batch_vecs = emb.embed_batch(&texts)?;
                vectors.extend(batch_vecs);
            }
            updates.push(PreparedFile {
                path,
                hash,
                chunks: group,
                vectors,
            });
        }
    }

    let file_updates: Vec<store::FileUpdate<'_>> = updates
        .iter()
        .map(|f| store::FileUpdate {
            source_path: &f.path,
            content_hash: &f.hash,
            chunks: &f.chunks,
            vectors: &f.vectors,
        })
        .collect();
    db.apply_index(&file_updates, &plan.prune, instructions.as_deref())?;
    if let Some(text) = db.get_meta("instructions")? {
        eprintln!(
            "MCP instructions ({} chars): {}",
            text.len(),
            text.chars().take(80).collect::<String>()
        );
    }
    println!("wrote {}", db.summary()?);
    Ok(())
}

fn validate_collected_chunks(chunk_count: usize, sync: bool) -> Result<()> {
    if chunk_count == 0 && !sync {
        bail!("no markdown chunks found; pass --sync to reconcile an empty corpus");
    }
    Ok(())
}

fn validate_index_mode(sync: bool, migration_required: bool) -> Result<()> {
    if !sync && migration_required {
        bail!(
            "the database requires a full model/chunker migration; \
             re-index the complete corpus with --sync"
        );
    }
    Ok(())
}

async fn run_serve(db_spec: String) -> Result<()> {
    use rmcp::ServiceExt;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let db_path = remote::resolve_db(&db_spec).await?;
    let db = store::Db::open_read_only(&db_path)?;
    let n = db.count()?;
    if n == 0 {
        bail!(
            "database {} has no documents; run index first",
            db_path.display()
        );
    }
    let index = search::Index::load(&db)?;
    let embedder = embed::Embedder::new()?;
    let service = mcp::ContextService::new(db, index, embedder, db_path.clone());

    eprintln!(
        "context-server: serving MCP stdio ({} chunks from {}, hybrid search, {})",
        n,
        db_path.display(),
        embed::MODEL_ID
    );
    let server = service.serve(rmcp::transport::stdio()).await?;
    server.waiting().await?;
    Ok(())
}

fn run_search(
    db_spec: String,
    limit: usize,
    mode: String,
    path_prefix: Option<String>,
    heading: Option<String>,
    tag: Option<String>,
    query: Vec<String>,
) -> Result<()> {
    let q = query.join(" ").trim().to_string();
    if q.is_empty() {
        bail!("usage: context-server search --db context.db <query>");
    }
    let mode = SearchMode::parse(&mode).ok_or_else(|| {
        anyhow::anyhow!("unknown --mode {mode:?} (expected hybrid, dense, or lexical)")
    })?;
    let filter = search::SearchFilter {
        path_prefix,
        heading,
        tag,
    };
    let db_path = remote::resolve_db_blocking(&db_spec)?;
    let db = store::Db::open_read_only(&db_path)?;
    let idx = search::Index::load(&db)?;
    if idx.is_empty() {
        bail!("database {} has no documents", db_path.display());
    }
    let mut emb = embed::Embedder::new()?;
    let hits = idx.query_filtered(&mut emb, &q, limit, mode, &filter)?;
    println!("query={q:?} mode={mode:?} ({} indexed chunks)", idx.len());
    for (i, h) in hits.iter().enumerate() {
        let preview = index::truncate_preview(&h.text, 237);
        println!(
            "\n{}. score={:.4} (dense={:.4} lexical={:.4})  {}#{}\n   {}",
            i + 1,
            h.score,
            h.dense_score,
            h.lexical_score,
            h.source_path,
            h.chunk_index,
            preview
        );
    }
    Ok(())
}

fn run_get(db_spec: String, path: String, chunk: Option<usize>) -> Result<()> {
    let db_path = remote::resolve_db_blocking(&db_spec)?;
    let db = store::Db::open_read_only(&db_path)?;
    let idx = search::Index::load(&db)?;
    match chunk {
        Some(i) => {
            let Some(d) = idx.get(&path, i) else {
                bail!("no chunk {path}#{i}");
            };
            print_chunk(d);
        }
        None => {
            let docs = idx.get_by_path(&path);
            if docs.is_empty() {
                bail!("no chunks for {path}");
            }
            for d in docs {
                print_chunk(d);
                println!();
            }
        }
    }
    Ok(())
}

fn print_chunk(d: &store::Document) {
    if !d.headings.is_empty() {
        println!(
            "{}#{} [{}]",
            d.source_path,
            d.chunk_index,
            d.headings.join(" > ")
        );
    } else {
        println!("{}#{}", d.source_path, d.chunk_index);
    }
    println!("{}", d.text);
}

fn run_embed(text: Vec<String>) -> Result<()> {
    let t = text.join(" ").trim().to_string();
    if t.is_empty() {
        bail!("usage: context-server embed <query>");
    }
    let mut emb = embed::Embedder::new().context("init embedder")?;
    // Query-style embedding (same path as search); passages use embed_batch.
    let vec = emb.embed(&t)?;
    print!("dim={} query={t:?}\nfirst8=[", vec.len());
    for (i, v) in vec.iter().take(8).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{v:.6}");
    }
    println!("]");

    let base = "The dog is running in the park";
    let similar = "A canine is running through the park";
    let other = "I love eating pizza for dinner";
    let vecs = emb.embed_batch(&[base.into(), similar.into(), other.into()])?;
    println!(
        "cosine({base:?}, {similar:?}) = {:.4}",
        embed::cosine(&vecs[0], &vecs[1])
    );
    println!(
        "cosine({base:?}, {other:?}) = {:.4}",
        embed::cosine(&vecs[0], &vecs[2])
    );
    Ok(())
}

fn run_eval(db_spec: String, cases_path: PathBuf, limit: usize) -> Result<()> {
    let db_path = remote::resolve_db_blocking(&db_spec)?;
    let db = store::Db::open_read_only(&db_path)?;
    let idx = search::Index::load(&db)?;
    let cases = eval::load_cases(&cases_path)?;
    let mut emb = embed::Embedder::new()?;
    let mut values = Vec::with_capacity(cases.len());
    for case in &cases {
        let hits = idx.query(&mut emb, &case.query, limit, SearchMode::Hybrid)?;
        let ranked: Vec<String> = hits
            .iter()
            .map(|hit| format!("{}#{}", hit.source_path, hit.chunk_index))
            .collect();
        values.push(eval::score_case(&ranked, &case.relevant, limit));
    }
    let metrics = eval::aggregate(&values);
    println!(
        "cases={} recall@{}={:.4} mrr={:.4}",
        metrics.cases, limit, metrics.recall_at_k, metrics.mrr
    );
    Ok(())
}

fn run_status(db_spec: String, json: bool) -> Result<()> {
    let path = remote::resolve_db_blocking(&db_spec)?;
    let db = store::Db::open_read_only(&path)?;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "path": path, "chunks": db.count()?, "files": db.source_paths()?.len(),
                "model_id": db.get_meta("model_id")?, "chunker_version": db.get_meta("chunker_version")?,
                "embedding_fingerprint": db.get_meta("embedding_fingerprint")?,
            })
        );
    } else {
        println!("{}", db.summary()?);
    }
    Ok(())
}

fn run_validate(db_spec: String) -> Result<()> {
    let path = remote::resolve_db_blocking(&db_spec)?;
    let db = store::Db::open_read_only(&path)?;
    db.ensure_model_compatible()?;
    let result: String = db
        .conn
        .query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
    if result != "ok" {
        bail!("SQLite integrity_check failed: {result}");
    }
    println!("ok: {}", db.summary()?);
    Ok(())
}

fn run_model_status() -> Result<()> {
    let path = embed::model_cache_dir()?;
    println!(
        "model={} cached={} path={}",
        embed::MODEL_ID,
        embed::model_is_cached()?,
        path.display()
    );
    Ok(())
}

fn run_model_download() -> Result<()> {
    let _ = embed::Embedder::new()?;
    println!(
        "model={} ready at {}",
        embed::MODEL_ID,
        embed::model_cache_dir()?.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_is_rejected_when_index_migration_is_required() {
        let err = validate_index_mode(false, true).expect_err("partial migration must fail");
        assert!(
            err.to_string().contains("--sync"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn upsert_is_allowed_when_index_is_compatible() {
        validate_index_mode(false, false).expect("compatible upsert");
    }

    #[test]
    fn sync_is_allowed_when_index_migration_is_required() {
        validate_index_mode(true, true).expect("complete migration");
    }

    #[test]
    fn compatible_sync_is_allowed() {
        validate_index_mode(true, false).expect("compatible sync");
    }

    #[test]
    fn index_defaults_to_safe_upsert() {
        let cli = Cli::try_parse_from(["context-server", "index", "--input", "docs"])
            .expect("parse index command");
        let Commands::Index { sync, .. } = cli.command else {
            panic!("expected index command");
        };
        assert!(!sync);
    }

    #[test]
    fn index_sync_requires_explicit_flag() {
        let cli = Cli::try_parse_from(["context-server", "index", "--input", "docs", "--sync"])
            .expect("parse index command");
        let Commands::Index { sync, .. } = cli.command else {
            panic!("expected index command");
        };
        assert!(sync);
    }

    #[test]
    fn removed_update_flag_is_rejected() {
        let err = Cli::try_parse_from(["context-server", "index", "--input", "docs", "--update"])
            .expect_err("removed flag must be rejected");
        assert!(err.to_string().contains("--update"));
    }

    #[test]
    fn empty_input_requires_sync() {
        assert!(validate_collected_chunks(0, false).is_err());
        validate_collected_chunks(0, true).expect("explicit empty sync");
        validate_collected_chunks(1, false).expect("non-empty upsert");
    }
}
