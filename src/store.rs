//! SQLite storage for chunks and embeddings.

use crate::embed::{self, MODEL_ID};
use crate::index::{Chunk, CHUNKER_VERSION};
use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OpenFlags, Transaction};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Document {
    #[allow(dead_code)]
    pub id: i64,
    pub source_path: String,
    pub chunk_index: usize,
    pub text: String,
    pub headings: Vec<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub vector: Vec<f32>,
}

/// One file to upsert during incremental index.
pub struct FileUpdate<'a> {
    pub source_path: &'a str,
    pub content_hash: &'a str,
    pub chunks: &'a [Chunk],
    pub vectors: &'a [Vec<f32>],
}

pub struct Db {
    pub(crate) conn: Connection,
}

const SCHEMA_VERSION: &str = "1";

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        let existed = path.exists();
        let conn = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
        // `serve` and `index` may share the same WAL DB. Without a busy timeout,
        // a reader hitting a brief writer lock fails immediately with
        // SQLITE_BUSY. A short retry window absorbs those transient write
        // commits (fastembed indexing writes can overlap a read).
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch(
            r#"
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY,
  source_path TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  headings TEXT NOT NULL DEFAULT '[]',
  metadata TEXT NOT NULL DEFAULT '{}',
  UNIQUE(source_path, chunk_index)
);

CREATE TABLE IF NOT EXISTS embeddings (
  id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
  source_path TEXT PRIMARY KEY,
  content_hash TEXT NOT NULL
);
"#,
        )?;
        let db = Self { conn };
        match db.get_meta("schema_version")? {
            Some(version) if version == SCHEMA_VERSION => {}
            Some(version) => bail!("unsupported schema_version {version:?}"),
            None if existed && db.has_compatible_legacy_schema()? => {
                db.set_meta("schema_version", SCHEMA_VERSION)?;
                // Force the next index run to rebuild all chunks with current
                // stable-ID/line-range semantics before readers accept it.
                db.conn.execute("DELETE FROM meta WHERE key IN ('chunker_version', 'embedding_fingerprint')", [])?;
            }
            None if existed => bail!(
                "database has no schema_version and its schema is incompatible; move it aside and re-run index"
            ),
            None => db.set_meta("schema_version", SCHEMA_VERSION)?,
        }
        Ok(db)
    }

    pub fn open_read_only(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("open {} read-only", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let db = Self { conn };
        let version = db
            .get_meta("schema_version")?
            .context("database has no schema_version; re-run index")?;
        if version != SCHEMA_VERSION {
            bail!("unsupported schema_version {version:?}");
        }
        Ok(db)
    }

    fn has_compatible_legacy_schema(&self) -> Result<bool> {
        for (table, required) in [
            (
                "documents",
                &[
                    "id",
                    "source_path",
                    "chunk_index",
                    "text",
                    "headings",
                    "metadata",
                ][..],
            ),
            ("embeddings", &["id", "dim", "vector"][..]),
            ("meta", &["key", "value"][..]),
            ("files", &["source_path", "content_hash"][..]),
        ] {
            let mut statement = self.conn.prepare(&format!("PRAGMA table_info({table})"))?;
            let columns: HashSet<String> = statement
                .query_map([], |row| row.get::<_, String>(1))?
                .collect::<std::result::Result<_, _>>()?;
            if !required.iter().all(|column| columns.contains(*column)) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    #[allow(dead_code)]
    pub fn clear(&self) -> Result<()> {
        self.conn.execute_batch(
            "DELETE FROM embeddings; DELETE FROM documents; DELETE FROM files; DELETE FROM meta;",
        )?;
        Ok(())
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
        let mut rows = stmt.query(params![key])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }

    #[allow(dead_code)] // used by unit tests
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Full rebuild used by tests (and as a reference for the table layout).
    #[cfg(test)]
    pub fn replace_all(
        &mut self,
        chunks: &[Chunk],
        vectors: &[Vec<f32>],
        instructions: Option<&str>,
    ) -> Result<()> {
        if chunks.len() != vectors.len() {
            bail!(
                "chunks ({}) and vectors ({}) length mismatch",
                chunks.len(),
                vectors.len()
            );
        }
        let tx = self.conn.transaction()?;
        // Preserve meta.instructions unless the caller supplies a new value.
        tx.execute_batch("DELETE FROM embeddings; DELETE FROM documents; DELETE FROM files;")?;
        insert_chunks(&tx, chunks, vectors)?;
        write_file_hashes(&tx, chunks)?;
        write_index_meta(&tx, instructions)?;
        tx.commit()?;
        Ok(())
    }

    /// Hashes from the `files` table (empty on DBs that predate the table).
    pub fn file_hashes(&self) -> Result<HashMap<String, String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT source_path, content_hash FROM files")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut out = HashMap::new();
        for row in rows {
            let (path, hash) = row?;
            out.insert(path, hash);
        }
        Ok(out)
    }

    /// Distinct `source_path` values in `documents`.
    pub fn source_paths(&self) -> Result<HashSet<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT source_path FROM documents")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = HashSet::new();
        for row in rows {
            out.insert(row?);
        }
        Ok(out)
    }

    /// True when hashes cannot be trusted and every collected file must be
    /// re-embedded (missing/legacy meta, model change, or chunker change).
    pub fn needs_full_reembed(&self) -> Result<bool> {
        if self.count()? == 0 {
            return Ok(false);
        }
        match self.get_meta("model_id")? {
            Some(m) if m == MODEL_ID => {}
            _ => return Ok(true),
        }
        match self.get_meta("dim")? {
            Some(d) if d == embed::DIM.to_string() => {}
            _ => return Ok(true),
        }
        match self.get_meta("chunker_version")? {
            Some(v) if v == CHUNKER_VERSION => {}
            _ => return Ok(true),
        }
        match self.get_meta("embedding_fingerprint")? {
            Some(v) if v == embed::EMBEDDING_FINGERPRINT => {}
            _ => return Ok(true),
        }
        if self.file_hashes()?.is_empty() {
            return Ok(true);
        }
        Ok(false)
    }

    /// Apply per-file upserts and optional deletes in one transaction.
    pub fn apply_index(
        &mut self,
        updates: &[FileUpdate<'_>],
        prune: &[String],
        instructions: Option<&str>,
    ) -> Result<()> {
        for u in updates {
            if u.chunks.len() != u.vectors.len() {
                bail!(
                    "chunks ({}) and vectors ({}) length mismatch for {}",
                    u.chunks.len(),
                    u.vectors.len(),
                    u.source_path
                );
            }
        }
        let tx = self.conn.transaction()?;
        for path in prune {
            delete_source_path(&tx, path)?;
        }
        for u in updates {
            delete_source_path(&tx, u.source_path)?;
            insert_chunks(&tx, u.chunks, u.vectors)?;
            tx.execute(
                "INSERT INTO files (source_path, content_hash) VALUES (?1, ?2)
                 ON CONFLICT(source_path) DO UPDATE SET content_hash = excluded.content_hash",
                params![u.source_path, u.content_hash],
            )?;
        }
        write_index_meta(&tx, instructions)?;
        tx.commit()?;
        Ok(())
    }

    /// Ensure DB embeddings were built with the current model.
    pub fn ensure_model_compatible(&self) -> Result<()> {
        let fingerprint = self
            .get_meta("embedding_fingerprint")?
            .context("database has no embedding_fingerprint; re-run index")?;
        if fingerprint != embed::EMBEDDING_FINGERPRINT {
            bail!("database embedding_fingerprint is incompatible; re-run index");
        }
        let model = self
            .get_meta("model_id")?
            .context("database has no model_id; re-run index")?;
        if model != MODEL_ID {
            bail!("database model {model:?} != current {MODEL_ID:?}; re-run index");
        }
        let dim = self
            .get_meta("dim")?
            .context("database has no dim; re-run index")?;
        let dim: usize = dim.parse().context("parse meta.dim")?;
        if dim != embed::DIM {
            bail!(
                "database dim {dim} != {MODEL_ID} dim {}; re-run index",
                embed::DIM
            );
        }
        Ok(())
    }

    pub fn count(&self) -> Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    pub fn list_page(
        &self,
        limit: usize,
        offset: usize,
        path_prefix: Option<&str>,
    ) -> Result<Vec<Document>> {
        let prefix = path_prefix.unwrap_or("").trim();
        let rows = if prefix.is_empty() {
            let mut stmt = self.conn.prepare(
                "SELECT id, source_path, chunk_index, text, headings, metadata FROM documents ORDER BY source_path, chunk_index LIMIT ?1 OFFSET ?2",
            )?;
            let mapped = stmt.query_map(params![limit as i64, offset as i64], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                ))
            })?;
            mapped.collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            let escaped = format!(
                "{}%",
                prefix
                    .replace('\\', "\\\\")
                    .replace('%', "\\%")
                    .replace('_', "\\_")
            );
            let mut stmt = self.conn.prepare(
                "SELECT id, source_path, chunk_index, text, headings, metadata FROM documents WHERE source_path LIKE ?1 ESCAPE '\\' ORDER BY source_path, chunk_index LIMIT ?2 OFFSET ?3",
            )?;
            let mapped = stmt.query_map(params![escaped, limit as i64, offset as i64], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                ))
            })?;
            mapped.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let mut out = Vec::new();
        for (id, source_path, chunk_index, text, headings_json, meta_json) in rows {
            out.push(Document {
                id,
                source_path,
                chunk_index: chunk_index as usize,
                text,
                headings: serde_json::from_str(&headings_json).unwrap_or_default(),
                metadata: serde_json::from_str(&meta_json).unwrap_or_default(),
                vector: vec![],
            });
        }
        Ok(out)
    }

    pub fn load_all(&self) -> Result<Vec<Document>> {
        self.ensure_model_compatible()?;
        let mut stmt = self.conn.prepare(
            r#"
SELECT d.id, d.source_path, d.chunk_index, d.text, d.headings, d.metadata, e.dim, e.vector
FROM documents d
JOIN embeddings e ON e.id = d.id
ORDER BY d.id
"#,
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, Vec<u8>>(7)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, source_path, chunk_index, text, headings_json, meta_json, dim, blob) = row?;
            let vector = bytes_to_float32(&blob, dim as usize)?;
            out.push(Document {
                id,
                source_path,
                chunk_index: chunk_index as usize,
                text,
                headings: serde_json::from_str(&headings_json).unwrap_or_default(),
                metadata: serde_json::from_str(&meta_json).unwrap_or_default(),
                vector,
            });
        }
        Ok(out)
    }

    pub fn summary(&self) -> Result<String> {
        let n = self.count()?;
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT source_path FROM documents ORDER BY source_path")?;
        let sources: Vec<String> = stmt
            .query_map([], |r| r.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        let shown: Vec<&str> = sources.iter().take(5).map(|s| s.as_str()).collect();
        let extra = if sources.len() > 5 { ", …" } else { "" };
        Ok(format!(
            "{} chunks across {} files ({}{}) [{MODEL_ID}/{}d]",
            n,
            sources.len(),
            shown.join(", "),
            extra,
            embed::DIM
        ))
    }
}

fn delete_source_path(tx: &Transaction<'_>, path: &str) -> Result<()> {
    tx.execute(
        "DELETE FROM documents WHERE source_path = ?1",
        params![path],
    )?;
    tx.execute("DELETE FROM files WHERE source_path = ?1", params![path])?;
    Ok(())
}

fn insert_chunks(tx: &Transaction<'_>, chunks: &[Chunk], vectors: &[Vec<f32>]) -> Result<()> {
    let mut doc_stmt = tx.prepare(
        "INSERT INTO documents (source_path, chunk_index, text, headings, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;
    let mut emb_stmt =
        tx.prepare("INSERT INTO embeddings (id, dim, vector) VALUES (?1, ?2, ?3)")?;

    for (c, vec) in chunks.iter().zip(vectors.iter()) {
        if vec.is_empty() {
            bail!("empty vector for {}[{}]", c.source_path, c.chunk_index);
        }
        if vec.len() != embed::DIM {
            bail!(
                "vector dim {} != expected {} for {}[{}]",
                vec.len(),
                embed::DIM,
                c.source_path,
                c.chunk_index
            );
        }
        let headings = serde_json::to_string(&c.headings)?;
        let metadata = serde_json::to_string(&c.metadata)?;
        doc_stmt.execute(params![
            c.source_path,
            c.chunk_index as i64,
            c.text,
            headings,
            metadata
        ])?;
        let id = tx.last_insert_rowid();
        emb_stmt.execute(params![id, vec.len() as i64, float32_to_bytes(vec)])?;
    }
    Ok(())
}

#[cfg(test)]
fn write_file_hashes(tx: &Transaction<'_>, chunks: &[Chunk]) -> Result<()> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<&str, Vec<&Chunk>> = BTreeMap::new();
    for c in chunks {
        groups.entry(c.source_path.as_str()).or_default().push(c);
    }
    let mut stmt = tx.prepare(
        "INSERT INTO files (source_path, content_hash) VALUES (?1, ?2)
         ON CONFLICT(source_path) DO UPDATE SET content_hash = excluded.content_hash",
    )?;
    for (path, mut group) in groups {
        group.sort_by_key(|c| c.chunk_index);
        let hash = crate::index::hash_chunks(group);
        stmt.execute(params![path, hash])?;
    }
    Ok(())
}

fn upsert_meta(tx: &Transaction<'_>, key: &str, value: &str) -> Result<()> {
    tx.execute(
        "INSERT INTO meta(key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

fn write_index_meta(tx: &Transaction<'_>, instructions: Option<&str>) -> Result<()> {
    upsert_meta(tx, "model_id", MODEL_ID)?;
    upsert_meta(tx, "dim", &embed::DIM.to_string())?;
    upsert_meta(tx, "chunker_version", CHUNKER_VERSION)?;
    let generation: i64 = tx
        .query_row(
            "SELECT value FROM meta WHERE key = 'generation'",
            [],
            |row| row.get::<_, String>(0),
        )
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
        + 1;
    upsert_meta(tx, "generation", &generation.to_string())?;
    upsert_meta(tx, "embedding_fingerprint", embed::EMBEDDING_FINGERPRINT)?;
    if let Some(text) = instructions {
        upsert_meta(tx, "instructions", text)?;
    }
    Ok(())
}

fn float32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for f in v {
        b.extend_from_slice(&f.to_le_bytes());
    }
    b
}

fn bytes_to_float32(b: &[u8], dim: usize) -> Result<Vec<f32>> {
    if b.len() != dim * 4 {
        bail!("blob length {} != dim*4 ({})", b.len(), dim * 4);
    }
    let (chunks, _) = b.as_chunks::<4>();
    let mut out = Vec::with_capacity(dim);
    for chunk in chunks {
        out.push(f32::from_le_bytes(*chunk));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Chunk;
    use tempfile::tempdir;

    #[test]
    fn replace_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.db");
        let mut db = Db::open(&path).unwrap();
        let chunks = vec![Chunk {
            source_path: "a.md".into(),
            chunk_index: 0,
            text: "hello".into(),
            headings: vec!["H".into()],
            metadata: serde_json::Map::new(),
        }];
        let vectors = vec![vec![1.0f32; embed::DIM]];
        db.replace_all(&chunks, &vectors, None).unwrap();
        assert_eq!(db.count().unwrap(), 1);
        db.ensure_model_compatible().unwrap();
        let docs = db.load_all().unwrap();
        assert_eq!(docs[0].text, "hello");
        assert_eq!(docs[0].vector.len(), embed::DIM);
    }

    #[test]
    fn instructions_meta_set_and_preserved() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.db");
        let mut db = Db::open(&path).unwrap();
        let chunks = vec![Chunk {
            source_path: "a.md".into(),
            chunk_index: 0,
            text: "hello".into(),
            headings: vec![],
            metadata: serde_json::Map::new(),
        }];
        let vectors = vec![vec![1.0f32; embed::DIM]];
        db.replace_all(&chunks, &vectors, Some("use for org docs"))
            .unwrap();
        assert_eq!(
            db.get_meta("instructions").unwrap().as_deref(),
            Some("use for org docs")
        );
        // Re-index without instructions flag keeps prior value.
        db.replace_all(&chunks, &vectors, None).unwrap();
        assert_eq!(
            db.get_meta("instructions").unwrap().as_deref(),
            Some("use for org docs")
        );
        db.replace_all(&chunks, &vectors, Some("updated")).unwrap();
        assert_eq!(
            db.get_meta("instructions").unwrap().as_deref(),
            Some("updated")
        );
        db.set_meta("instructions", "via set_meta").unwrap();
        assert_eq!(
            db.get_meta("instructions").unwrap().as_deref(),
            Some("via set_meta")
        );
    }

    fn chunk(path: &str, text: &str) -> Chunk {
        Chunk {
            source_path: path.into(),
            chunk_index: 0,
            text: text.into(),
            headings: vec![],
            metadata: serde_json::Map::new(),
        }
    }

    fn dummy_vec() -> Vec<f32> {
        vec![1.0f32; embed::DIM]
    }

    #[test]
    fn replace_all_records_file_hashes_and_chunker() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.db");
        let mut db = Db::open(&path).unwrap();
        let chunks = vec![chunk("a.md", "hello"), chunk("b.md", "world")];
        let vectors = vec![dummy_vec(), dummy_vec()];
        db.replace_all(&chunks, &vectors, None).unwrap();
        assert!(!db.needs_full_reembed().unwrap());
        let hashes = db.file_hashes().unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(
            hashes.get("a.md").unwrap(),
            &crate::index::hash_chunks(std::iter::once(&chunks[0]))
        );
        assert_eq!(
            db.get_meta("chunker_version").unwrap().as_deref(),
            Some(CHUNKER_VERSION)
        );
    }

    #[test]
    fn apply_index_upserts_and_prunes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.db");
        let mut db = Db::open(&path).unwrap();
        let a = chunk("a.md", "aaa");
        let b = chunk("b.md", "bbb");
        db.replace_all(&[a.clone(), b.clone()], &[dummy_vec(), dummy_vec()], None)
            .unwrap();

        let a2 = chunk("a.md", "aaa-edited");
        let hash = crate::index::hash_chunks(std::iter::once(&a2));
        let vecs = [dummy_vec()];
        db.apply_index(
            &[FileUpdate {
                source_path: "a.md",
                content_hash: &hash,
                chunks: std::slice::from_ref(&a2),
                vectors: &vecs,
            }],
            &["b.md".into()],
            None,
        )
        .unwrap();

        assert_eq!(db.count().unwrap(), 1);
        let docs = db.load_all().unwrap();
        assert_eq!(docs[0].source_path, "a.md");
        assert_eq!(docs[0].text, "aaa-edited");
        let hashes = db.file_hashes().unwrap();
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes.get("a.md").unwrap(), &hash);
        assert!(!hashes.contains_key("b.md"));
    }

    #[test]
    fn missing_embedding_fingerprint_requires_reembed_and_blocks_search() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.db");
        let mut db = Db::open(&path).unwrap();
        let chunks = vec![chunk("a.md", "hello")];
        db.replace_all(&chunks, &[dummy_vec()], None).unwrap();
        db.conn
            .execute("DELETE FROM meta WHERE key = 'embedding_fingerprint'", [])
            .unwrap();

        assert!(db.needs_full_reembed().unwrap());
        let err = db
            .ensure_model_compatible()
            .expect_err("ambiguous provenance must block search");
        assert!(err.to_string().contains("embedding_fingerprint"), "{err:#}");
    }

    #[test]
    fn mismatched_embedding_fingerprint_requires_reembed_and_blocks_search() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.db");
        let mut db = Db::open(&path).unwrap();
        let chunks = vec![chunk("a.md", "hello")];
        db.replace_all(&chunks, &[dummy_vec()], None).unwrap();
        db.set_meta("embedding_fingerprint", "different-model-config")
            .unwrap();

        assert!(db.needs_full_reembed().unwrap());
        assert!(db.ensure_model_compatible().is_err());
    }

    #[test]
    fn legacy_db_without_file_hashes_needs_full_reembed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.db");
        let mut db = Db::open(&path).unwrap();
        let chunks = vec![chunk("a.md", "hello")];
        db.replace_all(&chunks, &[dummy_vec()], None).unwrap();
        db.conn.execute("DELETE FROM files", []).unwrap();
        assert!(db.needs_full_reembed().unwrap());
    }

    #[test]
    fn list_with_and_without_prefix() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.db");
        let mut db = Db::open(&path).unwrap();
        let chunks = vec![
            chunk("guides/onboard.md", "guide 1"),
            chunk("teams/eng.md", "eng team"),
            chunk("teams/storage.md", "storage team"),
        ];
        let vectors = vec![dummy_vec(), dummy_vec(), dummy_vec()];
        db.replace_all(&chunks, &vectors, None).unwrap();

        let all = db.list_page(10, 0, None).unwrap();
        assert_eq!(all.len(), 3);

        let teams = db.list_page(10, 0, Some("teams/")).unwrap();
        assert_eq!(teams.len(), 2);
        assert_eq!(teams[0].source_path, "teams/eng.md");
        assert_eq!(teams[1].source_path, "teams/storage.md");

        let limited = db.list_page(1, 0, Some("teams/")).unwrap();
        assert_eq!(limited.len(), 1);
        assert_eq!(limited[0].source_path, "teams/eng.md");
    }
}
