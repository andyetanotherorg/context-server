//! Resolve `--db` to a local SQLite path, fetching from GCS when needed.
//!
//! Remote fetches look for a sibling `{object}.sha256` (sha256sum format, as
//! published next to the DB). When the local cache matches that checksum, the
//! DB download is skipped.

use anyhow::{bail, Context, Result};
use google_cloud_storage::client::Storage;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Parsed remote GCS object identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcsRef {
    /// GCP project id when known; `None` means use `projects/_/buckets/...`.
    pub project: Option<String>,
    pub bucket: String,
    pub object: String,
}

impl GcsRef {
    /// Bucket resource name for the Storage object API.
    ///
    /// Object reads require the globally-unique form `projects/_/buckets/{id}`.
    /// A project id in the URI is kept only for cache identity / display; the
    /// Storage object API does not accept `projects/{project}/buckets/...`.
    pub fn bucket_resource(&self) -> String {
        format!("projects/_/buckets/{}", self.bucket)
    }

    fn checksum_object(&self) -> String {
        format!("{}.sha256", self.object)
    }

    fn cache_key(&self) -> String {
        let identity = format!(
            "{}|{}|{}",
            self.project.as_deref().unwrap_or("_"),
            self.bucket,
            self.object
        );
        hex::encode(Sha256::digest(identity.as_bytes()))
    }
}

/// What `--db` referred to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbSpec {
    Local(PathBuf),
    Gcs(GcsRef),
}

/// Parse a `--db` value into a local path or GCS reference.
///
/// Remote DBs must use a `gs://` URL. Bare `projects/...` paths are treated as
/// local filesystem paths so we never guess remote intent from a relative path.
pub fn parse_db_spec(spec: &str) -> Result<DbSpec> {
    let spec = spec.trim();
    if spec.is_empty() {
        bail!("--db must not be empty");
    }

    if let Some(rest) = spec.strip_prefix("gs://") {
        return Ok(DbSpec::Gcs(parse_gs_url(rest)?));
    }

    Ok(DbSpec::Local(PathBuf::from(spec)))
}

fn parse_gs_url(rest: &str) -> Result<GcsRef> {
    // gs://projects/PROJECT/buckets/BUCKET/objects/OBJECT...
    // Strip gs://; Storage API calls use projects/_/buckets/{bucket} only.
    if rest.starts_with("projects/") {
        return parse_resource_name(rest);
    }

    let (bucket, object) = rest
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("gs:// URL missing object path: gs://{rest}"))?;
    if bucket.is_empty() {
        bail!("gs:// URL missing bucket name");
    }
    if object.is_empty() {
        bail!("gs:// URL missing object path: gs://{bucket}/");
    }
    Ok(GcsRef {
        project: None,
        bucket: bucket.to_string(),
        object: object.to_string(),
    })
}

fn parse_resource_name(name: &str) -> Result<GcsRef> {
    // projects/{project}/buckets/{bucket}/objects/{object...}
    let parts: Vec<&str> = name.splitn(6, '/').collect();
    if parts.len() < 6 || parts[0] != "projects" || parts[2] != "buckets" || parts[4] != "objects" {
        bail!("expected projects/PROJECT/buckets/BUCKET/objects/OBJECT..., got {name:?}");
    }
    let project = parts[1];
    let bucket = parts[3];
    let object = parts[5];
    if project.is_empty() || bucket.is_empty() || object.is_empty() {
        bail!("incomplete GCS resource name: {name:?}");
    }
    let project = if project == "_" {
        None
    } else {
        Some(project.to_string())
    };
    Ok(GcsRef {
        project,
        bucket: bucket.to_string(),
        object: object.to_string(),
    })
}

/// Parse `sha256sum` output (`<hex>  <filename>`) or a bare 64-char hex digest.
pub fn parse_sha256_text(text: &str) -> Result<String> {
    let first = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .ok_or_else(|| anyhow::anyhow!("empty sha256 file"))?;
    let hex = first.split_whitespace().next().unwrap_or("");
    if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("invalid sha256 digest in {first:?}");
    }
    Ok(hex.to_ascii_lowercase())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 64];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn write_sha256_file(path: &Path, digest: &str, labeled_name: &str) -> Result<()> {
    let body = format!("{digest}  {labeled_name}\n");
    fs::write(path, body).with_context(|| format!("write {}", path.display()))
}

fn read_local_sha256(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .and_then(|t| parse_sha256_text(&t).ok())
}

/// Verify the bytes that will actually be served, never just a cached sidecar.
fn cached_database_matches(db_path: &Path, expected: &str) -> bool {
    db_path.is_file() && sha256_file(db_path).ok().as_deref() == Some(expected)
}

/// Resolve `--db` to a local filesystem path, downloading from GCS if needed.
pub async fn resolve_db(spec: &str) -> Result<PathBuf> {
    match parse_db_spec(spec)? {
        DbSpec::Local(path) => Ok(path),
        DbSpec::Gcs(gcs) => fetch_gcs_db(&gcs).await,
    }
}

/// Sync wrapper for callers without an existing runtime.
pub fn resolve_db_blocking(spec: &str) -> Result<PathBuf> {
    match parse_db_spec(spec)? {
        DbSpec::Local(path) => Ok(path),
        DbSpec::Gcs(_) => {
            let rt = tokio::runtime::Runtime::new().context("tokio runtime for GCS download")?;
            rt.block_on(resolve_db(spec))
        }
    }
}

async fn read_object_bytes(client: &Storage, gcs: &GcsRef, object: &str) -> Result<Vec<u8>> {
    let bucket = gcs.bucket_resource();
    // Object API requires projects/_/buckets/{id}. A project in the URI is
    // kept for cache identity only; do not set quota project here (that needs
    // serviceusage.services.use on the project).
    let mut resp = client
        .read_object(&bucket, object)
        .send()
        .await
        .with_context(|| format!("read GCS object {bucket}/{object}"))?;
    let mut out = Vec::new();
    while let Some(chunk) = resp.next().await {
        out.extend_from_slice(&chunk.context("read GCS object chunk")?);
    }
    Ok(out)
}

async fn fetch_remote_sha256(client: &Storage, gcs: &GcsRef) -> Result<Option<String>> {
    let checksum_obj = gcs.checksum_object();
    match read_object_bytes(client, gcs, &checksum_obj).await {
        Ok(bytes) => {
            let text = String::from_utf8_lossy(&bytes);
            Ok(Some(parse_sha256_text(&text).with_context(|| {
                format!("parse gs://{}/{}", gcs.bucket, checksum_obj)
            })?))
        }
        Err(err) => {
            eprintln!(
                "context-server: no remote checksum at gs://{}/{} ({err}); will always re-fetch",
                gcs.bucket, checksum_obj
            );
            Ok(None)
        }
    }
}

async fn fetch_gcs_db(gcs: &GcsRef) -> Result<PathBuf> {
    let cache_dir = cache_dir_for(gcs)?;
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("create cache dir {}", cache_dir.display()))?;
    let dest = cache_dir.join("context.db");
    let checksum_path = cache_dir.join("context.db.sha256");

    let client = Storage::builder()
        .build()
        .await
        .context("build Google Cloud Storage client (check Application Default Credentials)")?;

    let remote_sum = fetch_remote_sha256(&client, gcs).await?;

    if dest.is_file() {
        if let Some(ref remote) = remote_sum {
            if cached_database_matches(&dest, remote) {
                // Refresh a missing or stale sidecar after validating the DB bytes.
                if read_local_sha256(&checksum_path).as_deref() != Some(remote) {
                    let _ = write_sha256_file(&checksum_path, remote, "context.db");
                }
                eprintln!(
                    "context-server: cache hit for gs://{}/{} (sha256 {})",
                    gcs.bucket,
                    gcs.object,
                    &remote[..12]
                );
                return Ok(dest);
            }
            eprintln!(
                "context-server: remote sha256 changed for gs://{}/{}; re-fetching",
                gcs.bucket, gcs.object
            );
        }
    }

    eprintln!(
        "context-server: fetching gs://{}/{} -> {}",
        gcs.bucket,
        gcs.object,
        dest.display()
    );

    // Unique staging file: concurrent `resolve_db` calls for the same object
    // must not trample each other's partial download. `fs::rename` below is
    // atomic, so whichever finishes first wins and the other's rename simply
    // replaces the (already-verified) destination.
    let tmp = unique_partial_path(&cache_dir, &dest)?;

    let bucket = gcs.bucket_resource();
    let mut resp = client
        .read_object(&bucket, &gcs.object)
        .send()
        .await
        .with_context(|| format!("read GCS object {bucket}/{}", gcs.object))?;

    let mut file = fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
    let mut hasher = Sha256::new();
    let mut total = 0usize;
    while let Some(chunk) = resp.next().await {
        let chunk = chunk.context("read GCS object chunk")?;
        total += chunk.len();
        hasher.update(&chunk);
        file.write_all(&chunk)
            .with_context(|| format!("write {}", tmp.display()))?;
    }
    file.sync_all().ok();
    drop(file);

    let digest = hex::encode(hasher.finalize());
    if let Some(ref remote) = remote_sum {
        if digest != *remote {
            let _ = fs::remove_file(&tmp);
            bail!(
                "downloaded sha256 {digest} does not match remote checksum {remote} for gs://{}/{}",
                gcs.bucket,
                gcs.object
            );
        }
    }

    fs::rename(&tmp, &dest)
        .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))?;
    write_sha256_file(&checksum_path, &digest, "context.db")?;
    eprintln!(
        "context-server: downloaded {total} bytes (sha256 {})",
        &digest[..12]
    );
    Ok(dest)
}

/// A per-process-unique staging path under `cache_dir`, so concurrent fetches
/// of the same remote object never overwrite each other's partial download.
///
/// Uniqueness comes from PID + a process-wide monotonic counter, not a racy
/// `exists()` probe (two callers could otherwise pick the same free name).
fn unique_partial_path(cache_dir: &Path, dest: &Path) -> Result<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let name = dest
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("context");
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(cache_dir.join(format!(
        "{}.partial.{}.{}",
        name,
        std::process::id(),
        counter
    )))
}

fn cache_dir_for(gcs: &GcsRef) -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .context("no cache directory (set XDG_CACHE_HOME or HOME)")?
        .join("context-server")
        .join("dbs")
        .join(gcs.cache_key());
    Ok(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parse_local_path() {
        assert_eq!(
            parse_db_spec("/tmp/context.db").unwrap(),
            DbSpec::Local(PathBuf::from("/tmp/context.db"))
        );
        assert_eq!(
            parse_db_spec("context.db").unwrap(),
            DbSpec::Local(PathBuf::from("context.db"))
        );
    }

    #[test]
    fn parse_short_gs() {
        let s = parse_db_spec("gs://my-context-bucket/latest/context.db").unwrap();
        assert_eq!(
            s,
            DbSpec::Gcs(GcsRef {
                project: None,
                bucket: "my-context-bucket".into(),
                object: "latest/context.db".into(),
            })
        );
        if let DbSpec::Gcs(g) = s {
            assert_eq!(g.bucket_resource(), "projects/_/buckets/my-context-bucket");
            assert_eq!(g.checksum_object(), "latest/context.db.sha256");
        }
    }

    #[test]
    fn bare_resource_name_is_local_path() {
        let path = "projects/my-gcp-project/buckets/my-context-bucket/objects/latest/context.db";
        assert_eq!(
            parse_db_spec(path).unwrap(),
            DbSpec::Local(PathBuf::from(path))
        );
    }

    #[test]
    fn parse_gs_wrapped_resource_name() {
        let s = parse_db_spec(
            "gs://projects/my-gcp-project/buckets/my-context-bucket/objects/latest/context.db",
        )
        .unwrap();
        assert_eq!(
            s,
            DbSpec::Gcs(GcsRef {
                project: Some("my-gcp-project".into()),
                bucket: "my-context-bucket".into(),
                object: "latest/context.db".into(),
            })
        );
        if let DbSpec::Gcs(g) = s {
            assert_eq!(g.bucket_resource(), "projects/_/buckets/my-context-bucket");
        }
    }

    #[test]
    fn parse_gs_underscore_project() {
        let s = parse_db_spec("gs://projects/_/buckets/my-bucket/objects/obj.db").unwrap();
        assert_eq!(
            s,
            DbSpec::Gcs(GcsRef {
                project: None,
                bucket: "my-bucket".into(),
                object: "obj.db".into(),
            })
        );
    }

    #[test]
    fn reject_gs_without_object() {
        assert!(parse_db_spec("gs://bucket-only").is_err());
        assert!(parse_db_spec("gs://bucket/").is_err());
    }

    #[test]
    fn cache_key_stable() {
        let a = GcsRef {
            project: Some("p".into()),
            bucket: "b".into(),
            object: "o/x".into(),
        };
        let b = a.clone();
        assert_eq!(a.cache_key(), b.cache_key());
        assert_eq!(a.cache_key().len(), 64);
    }

    #[test]
    fn parse_sha256sum_lines() {
        assert_eq!(
            parse_sha256_text(
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  context.db\n"
            )
            .unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            parse_sha256_text("E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855")
                .unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert!(parse_sha256_text("not-a-hash").is_err());
        assert!(parse_sha256_text("").is_err());
    }

    #[test]
    fn unique_partial_path_returns_distinct_staging_files() {
        let dir = tempdir().unwrap();
        let dest = dir.path().join("context.db");
        let a = unique_partial_path(dir.path(), &dest).unwrap();
        let b = unique_partial_path(dir.path(), &dest).unwrap();
        // Same process, so the counter bumps to avoid collision.
        assert!(a != b, "concurrent fetches must not share a staging file");
        let prefix = format!("context.db.partial.{}", std::process::id());
        assert!(a
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with(&prefix));
    }

    #[test]
    fn cached_database_is_validated_even_when_sidecar_matches() {
        let dir = tempdir().unwrap();
        let db = dir.path().join("context.db");
        let sidecar = dir.path().join("context.db.sha256");
        fs::write(&db, b"corrupted database").unwrap();
        write_sha256_file(&sidecar, &"a".repeat(64), "context.db").unwrap();

        assert!(!cached_database_matches(&db, &"a".repeat(64)));
    }

    #[test]
    fn cached_database_matches_actual_remote_digest_without_sidecar() {
        let dir = tempdir().unwrap();
        let db = dir.path().join("context.db");
        fs::write(&db, b"valid database bytes").unwrap();
        let digest = sha256_file(&db).unwrap();

        assert!(cached_database_matches(&db, &digest));
    }
}
