//! Markdown chunking and directory collection.

use anyhow::{bail, Context, Result};
use regex::Regex;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Soft cap on embedded text length. BGE-small truncates around ~512 tokens;
/// ~4 chars/token ⇒ keep chunks under this so the tail is not dropped.
pub const MAX_CHUNK_CHARS: usize = 1800;
/// Overlap between consecutive splits of an oversized section.
pub const CHUNK_OVERLAP_CHARS: usize = 200;
/// Stored in DB meta; bump when chunking rules change so incremental index
/// re-embeds even if file bytes look the same.
pub const CHUNKER_VERSION: &str = "1";

#[derive(Debug, Clone)]
pub struct Chunk {
    pub source_path: String,
    pub chunk_index: usize,
    pub text: String,
    pub headings: Vec<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// SHA-256 of a file's chunks (index, text, headings, metadata). Used to skip
/// re-embedding unchanged files. Hash post-chunk text so a chunker change
/// invalidates without relying only on [`CHUNKER_VERSION`].
pub fn hash_chunks<'a, I>(chunks: I) -> String
where
    I: IntoIterator<Item = &'a Chunk>,
{
    let mut hasher = Sha256::new();
    hasher.update(b"context-server-file-v1\0");
    for c in chunks {
        hasher.update((c.chunk_index as u64).to_le_bytes());
        hasher.update((c.text.len() as u64).to_le_bytes());
        hasher.update(c.text.as_bytes());
        let headings = serde_json::to_string(&c.headings).unwrap_or_else(|_| "[]".into());
        hasher.update((headings.len() as u64).to_le_bytes());
        hasher.update(headings.as_bytes());
        let metadata = serde_json::to_string(&c.metadata).unwrap_or_else(|_| "{}".into());
        hasher.update((metadata.len() as u64).to_le_bytes());
        hasher.update(metadata.as_bytes());
    }
    hex::encode(hasher.finalize())
}

/// Group chunks by `source_path`, sorted by path then `chunk_index`.
pub fn group_chunks_by_path(mut chunks: Vec<Chunk>) -> Vec<(String, Vec<Chunk>)> {
    chunks.sort_by(|a, b| {
        a.source_path
            .cmp(&b.source_path)
            .then(a.chunk_index.cmp(&b.chunk_index))
    });
    let mut out: Vec<(String, Vec<Chunk>)> = Vec::new();
    for c in chunks {
        if let Some((path, group)) = out.last_mut() {
            if *path == c.source_path {
                group.push(c);
                continue;
            }
        }
        let path = c.source_path.clone();
        out.push((path, vec![c]));
    }
    out
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexPlan {
    /// Unchanged files; leave existing rows alone.
    pub skip: Vec<String>,
    /// New or changed files (or everything when `full`).
    pub embed: Vec<String>,
    /// Paths in the DB that are no longer in the input (only when pruning).
    pub prune: Vec<String>,
}

/// Decide which files to re-embed, skip, or delete.
///
/// `incoming` is `(source_path, content_hash)` for files in this index run.
/// `existing` is hashes currently stored (and any document paths with an empty
/// hash if the `files` table is missing a row).
///
/// When `prune` is true, paths in `existing` but not in `incoming` are removed.
/// When `full` is true, every incoming file is re-embedded even if the hash matches.
pub fn plan_index(
    incoming: &[(String, String)],
    existing: &HashMap<String, String>,
    prune: bool,
    full: bool,
) -> IndexPlan {
    let incoming_map: HashMap<&str, &str> = incoming
        .iter()
        .map(|(p, h)| (p.as_str(), h.as_str()))
        .collect();

    let mut plan = IndexPlan::default();
    for (path, hash) in incoming {
        if !full && existing.get(path).is_some_and(|old| old == hash) {
            plan.skip.push(path.clone());
        } else {
            plan.embed.push(path.clone());
        }
    }
    if prune {
        for path in existing.keys() {
            if !incoming_map.contains_key(path.as_str()) {
                plan.prune.push(path.clone());
            }
        }
    }
    plan.skip.sort();
    plan.embed.sort();
    plan.prune.sort();
    plan
}

/// Split markdown into chunks on ## and ### boundaries.
pub fn split_markdown(source_path: &str, content: &str) -> Vec<Chunk> {
    let (metadata, content) = parse_front_matter(content);
    let heading_re = Regex::new(r"^(#{1,6})\s+(.+?)\s*$").unwrap();

    let mut doc_title = String::new();
    let mut h2 = String::new();
    let mut h3 = String::new();
    let mut body: Vec<String> = Vec::new();
    let mut chunks: Vec<Chunk> = Vec::new();

    let emit = |body: &mut Vec<String>,
                doc_title: &str,
                h2: &str,
                h3: &str,
                chunks: &mut Vec<Chunk>,
                source_path: &str,
                metadata: &serde_json::Map<String, serde_json::Value>| {
        let text = body.join("\n").trim().to_string();
        body.clear();
        if text.is_empty() {
            return;
        }
        let mut headings = Vec::new();
        if !doc_title.is_empty() {
            headings.push(doc_title.to_string());
        }
        if !h2.is_empty() {
            headings.push(h2.to_string());
        }
        if !h3.is_empty() {
            headings.push(h3.to_string());
        }
        let prefixed = if headings.is_empty() {
            text.clone()
        } else {
            format!("{}\n\n{}", headings.join(" > "), text)
        };
        let idx = chunks.len();
        chunks.push(Chunk {
            source_path: source_path.to_string(),
            chunk_index: idx,
            text: prefixed,
            headings,
            metadata: metadata.clone(),
        });
    };

    for line in content.lines() {
        if let Some(caps) = heading_re.captures(line) {
            let level = caps[1].len();
            let title = caps[2].trim().to_string();
            match level {
                1 => {
                    emit(&mut body, &doc_title, &h2, &h3, &mut chunks, source_path, &metadata);
                    doc_title = title;
                    h2.clear();
                    h3.clear();
                }
                2 => {
                    emit(&mut body, &doc_title, &h2, &h3, &mut chunks, source_path, &metadata);
                    h2 = title;
                    h3.clear();
                }
                3 => {
                    emit(&mut body, &doc_title, &h2, &h3, &mut chunks, source_path, &metadata);
                    h3 = title;
                }
                _ => body.push(line.to_string()),
            }
            continue;
        }
        body.push(line.to_string());
    }
    emit(&mut body, &doc_title, &h2, &h3, &mut chunks, source_path, &metadata);
    split_oversized(chunks)
}

/// Split any chunk whose embedded text exceeds [`MAX_CHUNK_CHARS`], keeping the
/// heading prefix on each piece and overlapping body windows.
fn split_oversized(chunks: Vec<Chunk>) -> Vec<Chunk> {
    let mut out = Vec::new();
    for chunk in chunks {
        if chunk.text.chars().count() <= MAX_CHUNK_CHARS {
            out.push(chunk);
            continue;
        }
        let prefix = if chunk.headings.is_empty() {
            String::new()
        } else {
            format!("{}\n\n", chunk.headings.join(" > "))
        };
        let body = chunk
            .text
            .strip_prefix(&prefix)
            .unwrap_or(chunk.text.as_str());
        let prefix_len = prefix.chars().count();
        let body_budget = MAX_CHUNK_CHARS.saturating_sub(prefix_len).max(200);
        let overlap = CHUNK_OVERLAP_CHARS.min(body_budget / 3);

        let body_chars: Vec<char> = body.chars().collect();
        if body_chars.is_empty() {
            out.push(chunk);
            continue;
        }

        let mut start = 0usize;
        while start < body_chars.len() {
            let mut end = (start + body_budget).min(body_chars.len());
            // Prefer breaking on whitespace when not at the end.
            if end < body_chars.len() {
                if let Some(rel) = body_chars[start..end]
                    .iter()
                    .rposition(|c| c.is_whitespace())
                {
                    if rel > body_budget / 4 {
                        end = start + rel;
                    }
                }
            }
            let piece: String = body_chars[start..end].iter().collect();
            let piece = piece.trim();
            if !piece.is_empty() {
                let text = if prefix.is_empty() {
                    piece.to_string()
                } else {
                    format!("{prefix}{piece}")
                };
                out.push(Chunk {
                    source_path: chunk.source_path.clone(),
                    chunk_index: 0, // renumbered below
                    text,
                    headings: chunk.headings.clone(),
                    metadata: chunk.metadata.clone(),
                });
            }
            if end >= body_chars.len() {
                break;
            }
            let next = end.saturating_sub(overlap);
            start = if next <= start { end } else { next };
        }
    }
    for (i, c) in out.iter_mut().enumerate() {
        c.chunk_index = i;
    }
    out
}
fn parse_front_matter(content: &str) -> (serde_json::Map<String, serde_json::Value>, String) {
    if !content.starts_with("---") {
        return (serde_json::Map::new(), content.to_string());
    }
    let re = Regex::new(r"(?s)^---\r?\n(.*?)\r?\n---\r?\n?").unwrap();
    if let Some(caps) = re.captures(content) {
        let raw_yaml = &caps[1];
        let remaining = &content[caps[0].len()..];
        let mut meta = serde_json::Map::new();

        let mut current_list_key: Option<String> = None;
        let mut list_items: Vec<serde_json::Value> = Vec::new();

        for line in raw_yaml.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if trimmed.starts_with("- ") {
                if let Some(ref _key) = current_list_key {
                    let item = trimmed[2..].trim().trim_matches('"').trim_matches('\'');
                    list_items.push(serde_json::Value::String(item.to_string()));
                    continue;
                }
            }

            if let Some(key) = current_list_key.take() {
                meta.insert(key, serde_json::Value::Array(std::mem::take(&mut list_items)));
            }

            if let Some((k, v)) = line.split_once(':') {
                let k = k.trim().to_string();
                let v = v.trim();
                if v.is_empty() {
                    current_list_key = Some(k);
                    list_items.clear();
                } else if (v.starts_with('[') && v.ends_with(']')) || v.contains(',') {
                    let inner = v.trim_matches(|c| c == '[' || c == ']');
                    let items: Vec<serde_json::Value> = inner
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(|s| serde_json::Value::String(s.trim_matches('"').trim_matches('\'').to_string()))
                        .collect();
                    meta.insert(k, serde_json::Value::Array(items));
                } else {
                    let scalar = v.trim_matches('"').trim_matches('\'');
                    meta.insert(k, serde_json::Value::String(scalar.to_string()));
                }
            }
        }

        if let Some(key) = current_list_key.take() {
            meta.insert(key, serde_json::Value::Array(list_items));
        }

        (meta, remaining.to_string())
    } else {
        (serde_json::Map::new(), content.to_string())
    }
}

pub fn heading_path(c: &Chunk) -> String {
    if c.headings.is_empty() {
        "(root)".into()
    } else {
        c.headings.join(" > ")
    }
}

/// Truncate for display without panicking on multi-byte UTF-8 (emoji, etc.).
pub fn truncate_preview(text: &str, max_chars: usize) -> String {
    let mut iter = text.chars();
    let mut out: String = iter.by_ref().take(max_chars).collect();
    if iter.next().is_some() {
        out.push_str("...");
    }
    out.replace('\n', " ")
}

pub fn format_chunk_debug(c: &Chunk) -> String {
    let preview = truncate_preview(&c.text, 117);
    format!("[{}] {} | {}", c.chunk_index, heading_path(c), preview)
}

/// Walk root and return chunks for every .md file.
pub fn collect(root: &Path) -> Result<Vec<Chunk>> {
    let meta = fs::metadata(root).with_context(|| format!("stat {}", root.display()))?;
    let mut chunks = Vec::new();

    let mut add_file = |path: &Path, rel: &str| -> Result<()> {
        let data = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        chunks.extend(split_markdown(rel, &data));
        Ok(())
    };

    if meta.is_file() {
        if !is_markdown(root) {
            bail!("{}: only .md files are supported", root.display());
        }
        let name = root
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| root.display().to_string());
        add_file(root, &name)?;
        return Ok(chunks);
    }

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_dir() {
            let name = entry.file_name().to_string_lossy();
            if name == ".git" || name == "node_modules" || name == "vendor" || name == "target" {
                // WalkDir doesn't skip easily mid-walk without filter_entry; fine to skip files only
            }
            continue;
        }
        let path = entry.path();
        if !is_markdown(path) {
            continue;
        }
        // Skip under ignored dirs
        if path.components().any(|c| {
            matches!(
                c.as_os_str().to_str(),
                Some(".git" | "node_modules" | "vendor" | "target")
            )
        }) {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        add_file(path, &rel)?;
    }
    Ok(chunks)
}

fn is_markdown(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()),
        Some(ref e) if e == "md" || e == "markdown"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headings_and_hierarchy() {
        let md = r#"---
name: example
---

# Backport Process

Intro paragraph about backports.

## Overview

When a bug fix targets the current release.

## Requirements by Bug Status

### NEW

No PR requirements.

### ASSIGNED

Required:
- Fix version set

## Branch Naming

Upstream repos use stable branches.
"#;
        let chunks = split_markdown("backport-process.md", md);
        assert_eq!(
            chunks.len(),
            5,
            "{:?}",
            chunks.iter().map(format_chunk_debug).collect::<Vec<_>>()
        );
        assert_eq!(chunks[0].headings, ["Backport Process"]);
        assert!(chunks[0].text.contains("Intro paragraph"));
        assert_eq!(
            chunks[2].headings,
            ["Backport Process", "Requirements by Bug Status", "NEW"]
        );
    }

    #[test]
    fn front_matter_tags_extracted_to_metadata() {
        let md = r#"---
title: Test Document
tags:
  - backend
  - "storage"
category: guides
---

# Guide

Body content here.
"#;
        let chunks = split_markdown("test.md", md);
        assert_eq!(chunks.len(), 1);
        let tags = chunks[0].metadata.get("tags").expect("tags present");
        assert_eq!(
            tags,
            &serde_json::json!(["backend", "storage"])
        );
        assert_eq!(
            chunks[0].metadata.get("category"),
            Some(&serde_json::json!("guides"))
        );
    }

    #[test]
    fn front_matter_inline_array_tags() {
        let md = r#"---
tags: [devops, infra, kubevirt]
---

# Title
Text.
"#;
        let chunks = split_markdown("x.md", md);
        assert_eq!(chunks.len(), 1);
        let tags = chunks[0].metadata.get("tags").unwrap();
        assert_eq!(tags, &serde_json::json!(["devops", "infra", "kubevirt"]));
    }

    #[test]
    fn empty_sections_skipped() {
        let md = "# Title\n\n## Empty\n\n## Has Content\n\nHello.\n";
        let chunks = split_markdown("x.md", md);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].headings, ["Title", "Has Content"]);
    }

    #[test]
    fn truncate_preview_handles_multibyte_at_cut() {
        // ✅ is 3 bytes; cutting at a byte index inside it used to panic.
        let s = format!("{}{}", "a".repeat(155), "✅ more text after emoji");
        let preview = truncate_preview(&s, 157);
        assert!(preview.ends_with("..."));
        assert!(!preview.contains('\u{FFFD}'));
        assert!(preview.chars().count() <= 160);
    }

    #[test]
    fn oversized_section_is_split() {
        let long = "word ".repeat(400); // well over MAX_CHUNK_CHARS
        let md = format!("# Doc\n\n## Big\n\n{long}");
        let chunks = split_markdown("big.md", &md);
        assert!(chunks.len() > 1, "expected split, got {}", chunks.len());
        for c in &chunks {
            assert!(
                c.text.chars().count() <= MAX_CHUNK_CHARS + 50,
                "chunk too long: {}",
                c.text.chars().count()
            );
            assert!(c.text.contains("Doc > Big") || c.headings.contains(&"Big".into()));
        }
    }

    fn chunk(path: &str, index: usize, text: &str) -> Chunk {
        Chunk {
            source_path: path.into(),
            chunk_index: index,
            text: text.into(),
            headings: vec![],
            metadata: serde_json::Map::new(),
        }
    }

    #[test]
    fn content_hash_is_stable_and_changes_with_text() {
        let a = [chunk("a.md", 0, "hello")];
        let b = [chunk("a.md", 0, "hello")];
        let c = [chunk("a.md", 0, "hello!")];
        assert_eq!(hash_chunks(a.iter()), hash_chunks(b.iter()));
        assert_ne!(hash_chunks(a.iter()), hash_chunks(c.iter()));
    }

    #[test]
    fn group_chunks_by_path_sorts() {
        let grouped = group_chunks_by_path(vec![
            chunk("b.md", 0, "b"),
            chunk("a.md", 1, "a1"),
            chunk("a.md", 0, "a0"),
        ]);
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].0, "a.md");
        assert_eq!(grouped[0].1[0].chunk_index, 0);
        assert_eq!(grouped[0].1[1].chunk_index, 1);
        assert_eq!(grouped[1].0, "b.md");
    }

    #[test]
    fn plan_skips_unchanged_embeds_changed_prunes_missing() {
        let incoming = vec![
            ("keep.md".into(), "hash-keep".into()),
            ("edit.md".into(), "hash-new".into()),
            ("new.md".into(), "hash-new-file".into()),
        ];
        let existing = HashMap::from([
            ("keep.md".into(), "hash-keep".into()),
            ("edit.md".into(), "hash-old".into()),
            ("gone.md".into(), "hash-gone".into()),
        ]);
        let plan = plan_index(&incoming, &existing, true, false);
        assert_eq!(plan.skip, ["keep.md"]);
        assert_eq!(plan.embed, ["edit.md", "new.md"]);
        assert_eq!(plan.prune, ["gone.md"]);
    }

    #[test]
    fn plan_update_does_not_prune() {
        let incoming = vec![("a.md".into(), "h".into())];
        let existing = HashMap::from([("a.md".into(), "h".into()), ("b.md".into(), "x".into())]);
        let plan = plan_index(&incoming, &existing, false, false);
        assert_eq!(plan.skip, ["a.md"]);
        assert!(plan.embed.is_empty());
        assert!(plan.prune.is_empty());
    }

    #[test]
    fn plan_full_reembeds_even_when_hash_matches() {
        let incoming = vec![("a.md".into(), "h".into())];
        let existing = HashMap::from([("a.md".into(), "h".into())]);
        let plan = plan_index(&incoming, &existing, true, true);
        assert!(plan.skip.is_empty());
        assert_eq!(plan.embed, ["a.md"]);
    }
}
