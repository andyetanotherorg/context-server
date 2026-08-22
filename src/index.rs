//! Markdown chunking and directory collection.

use anyhow::{bail, Context, Result};
use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use regex::Regex;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Soft cap on embedded text length. BGE-small truncates around ~512 tokens;
/// ~4 chars/token ⇒ keep chunks under this so the tail is not dropped.
pub const MAX_CHUNK_CHARS: usize = 1800;
pub const MAX_CHUNK_TOKENS: usize = 450;
/// Overlap between consecutive splits of an oversized section.
pub const CHUNK_OVERLAP_CHARS: usize = 200;
/// Stored in DB meta; bump when chunking rules change so incremental index
/// re-embeds even if file bytes look the same.
pub const CHUNKER_VERSION: &str = "4";

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

/// Split Markdown at structural heading boundaries. The CommonMark event parser
/// ensures heading-looking lines inside fenced code blocks remain body content.
pub fn split_markdown(source_path: &str, content: &str) -> Vec<Chunk> {
    let (metadata, content) = parse_front_matter(content);
    let mut headings_found: Vec<(usize, usize, usize, String)> = Vec::new();
    let mut active: Option<(usize, usize, String)> = None;

    for (event, range) in Parser::new_ext(&content, Options::all()).into_offset_iter() {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                active = Some((heading_level(level), range.start, String::new()));
            }
            Event::Text(text) | Event::Code(text) if active.is_some() => {
                if let Some((_, _, title)) = active.as_mut() {
                    title.push_str(&text);
                }
            }
            Event::End(TagEnd::Heading(_)) => {
                if let Some((level, start, title)) = active.take() {
                    headings_found.push((level, start, range.end, title.trim().to_string()));
                }
            }
            _ => {}
        }
    }

    let mut chunks = Vec::new();
    let mut hierarchy: Vec<String> = Vec::new();
    let mut body_start = 0usize;
    let mut section_start = 0usize;
    for (level, heading_start, heading_end, title) in headings_found {
        emit_section(
            &content,
            &content[body_start..heading_start],
            section_start,
            heading_start,
            &hierarchy,
            source_path,
            &metadata,
            &mut chunks,
        );
        hierarchy.truncate(level.saturating_sub(1));
        while hierarchy.len() < level.saturating_sub(1) {
            hierarchy.push(String::new());
        }
        hierarchy.push(title);
        body_start = heading_end;
        section_start = heading_start;
    }
    emit_section(
        &content,
        &content[body_start..],
        section_start,
        content.len(),
        &hierarchy,
        source_path,
        &metadata,
        &mut chunks,
    );
    let mut seen = std::collections::HashMap::<String, usize>::new();
    for chunk in &mut chunks {
        let base = chunk
            .metadata
            .get("_chunk_id")
            .and_then(|value| value.as_str())
            .unwrap_or("chunk")
            .to_string();
        let occurrence = seen.entry(base.clone()).or_default();
        if *occurrence > 0 {
            chunk.metadata.insert(
                "_chunk_id".into(),
                serde_json::Value::String(format!("{base}-occ{occurrence}")),
            );
        }
        *occurrence += 1;
    }
    split_oversized(chunks)
}

fn heading_level(level: HeadingLevel) -> usize {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_section(
    full_content: &str,
    body: &str,
    section_start: usize,
    section_end: usize,
    hierarchy: &[String],
    source_path: &str,
    metadata: &serde_json::Map<String, serde_json::Value>,
    chunks: &mut Vec<Chunk>,
) {
    let body = body.trim();
    if body.is_empty() {
        return;
    }
    let headings: Vec<String> = hierarchy
        .iter()
        .filter(|h| !h.is_empty())
        .cloned()
        .collect();
    let text = if headings.is_empty() {
        body.to_string()
    } else {
        format!("{}\n\n{}", headings.join(" > "), body)
    };
    let mut metadata = metadata.clone();
    metadata.insert(
        "_chunk_id".into(),
        serde_json::Value::String(stable_chunk_id(source_path, &headings, body)),
    );
    metadata.insert(
        "_start_line".into(),
        serde_json::json!(line_at(full_content, section_start)),
    );
    metadata.insert(
        "_end_line".into(),
        serde_json::json!(line_at(
            full_content,
            section_start + full_content[section_start..section_end].trim_end().len()
        )),
    );
    chunks.push(Chunk {
        source_path: source_path.to_string(),
        chunk_index: chunks.len(),
        text,
        headings,
        metadata,
    });
}

fn stable_chunk_id(source_path: &str, headings: &[String], body: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"context-server-chunk-v1\0");
    hasher.update(source_path.as_bytes());
    hasher.update(b"\0");
    hasher.update(headings.join("\0").as_bytes());
    hasher.update(b"\0");
    hasher.update(body.trim().as_bytes());
    hex::encode(hasher.finalize())[..16].to_string()
}

fn line_at(content: &str, byte_offset: usize) -> usize {
    content.as_bytes()[..byte_offset.min(content.len())]
        .iter()
        .filter(|&&byte| byte == b'\n')
        .count()
        + 1
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
        let mut split_index = 0usize;
        while start < body_chars.len() {
            let mut end = (start + body_budget).min(body_chars.len());
            while end > start + 1 {
                let candidate: String = body_chars[start..end].iter().collect();
                if crate::embed::estimated_tokens(&candidate) <= MAX_CHUNK_TOKENS {
                    break;
                }
                end = start + ((end - start) * 9 / 10).max(1);
            }
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
                let mut metadata = chunk.metadata.clone();
                let parent_id = metadata
                    .get("_chunk_id")
                    .and_then(|value| value.as_str())
                    .unwrap_or("chunk")
                    .to_string();
                metadata.insert(
                    "_chunk_id".into(),
                    serde_json::Value::String(format!("{parent_id}-{split_index}")),
                );
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
                    metadata,
                });
                split_index += 1;
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
const MAX_FRONT_MATTER_BYTES: usize = 64 * 1024;

fn parse_front_matter(content: &str) -> (serde_json::Map<String, serde_json::Value>, String) {
    if !content.starts_with("---\n") && !content.starts_with("---\r\n") {
        return (serde_json::Map::new(), content.to_string());
    }
    let re = Regex::new(r"(?s)^---\r?\n(.*?)\r?\n---\r?\n?").unwrap();
    let Some(caps) = re.captures(content) else {
        return (serde_json::Map::new(), content.to_string());
    };
    let raw_yaml = &caps[1];
    if raw_yaml.len() > MAX_FRONT_MATTER_BYTES {
        return (serde_json::Map::new(), content.to_string());
    }
    let Ok(value) = serde_yaml_ng::from_str::<serde_json::Value>(raw_yaml) else {
        return (serde_json::Map::new(), content.to_string());
    };
    let Some(metadata) = value.as_object().cloned() else {
        return (serde_json::Map::new(), content.to_string());
    };
    let remaining = &content[caps[0].len()..];
    (metadata, remaining.to_string())
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

    let walker = WalkDir::new(root).into_iter().filter_entry(|entry| {
        if entry.file_type().is_dir() {
            let name = entry.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                ".git" | "node_modules" | "vendor" | "target" | ".target" | ".venv"
            )
        } else {
            true
        }
    });

    for entry in walker {
        let entry = entry.with_context(|| format!("walk directory {}", root.display()))?;
        if entry.file_type().is_dir() {
            continue;
        }
        let path = entry.path();
        if !is_markdown(path) {
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
        assert_eq!(tags, &serde_json::json!(["backend", "storage"]));
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
    fn headings_inside_fenced_code_are_not_structural() {
        let md = "# Real Title\n\nBefore.\n\n```bash\n# shell comment\necho hi\n## another comment\n```\n\nAfter.\n";
        let chunks = split_markdown("fenced.md", md);
        assert_eq!(chunks.len(), 1, "{chunks:#?}");
        assert_eq!(chunks[0].headings, ["Real Title"]);
        assert!(chunks[0].text.contains("# shell comment"));
        assert!(chunks[0].text.contains("After."));
    }

    #[test]
    fn front_matter_preserves_comma_scalars() {
        let md = "---\ntitle: \"Hello, world\"\ndescription: A doc about a, b, and c\ntags: [one, two]\n---\n# Doc\nBody.\n";
        let chunks = split_markdown("meta.md", md);
        assert_eq!(
            chunks[0].metadata["title"],
            serde_json::json!("Hello, world")
        );
        assert_eq!(
            chunks[0].metadata["description"],
            serde_json::json!("A doc about a, b, and c")
        );
        assert_eq!(
            chunks[0].metadata["tags"],
            serde_json::json!(["one", "two"])
        );
    }

    #[test]
    fn malformed_front_matter_is_preserved_as_body() {
        let md = "---\ntags: [unterminated\n---\n# Doc\nBody.\n";
        let chunks = split_markdown("bad.md", md);
        assert!(!chunks[0].metadata.contains_key("tags"));
        assert!(chunks[0].text.contains("tags: [unterminated"));
    }

    #[test]
    fn oversized_front_matter_is_preserved_as_body() {
        let raw = "x".repeat(MAX_FRONT_MATTER_BYTES + 1);
        let md = format!("---\ndescription: {raw}\n---\n# Doc\nBody.\n");
        let chunks = split_markdown("large.md", &md);
        assert!(!chunks[0].metadata.contains_key("description"));
        assert!(chunks
            .iter()
            .any(|chunk| chunk.text.contains("description:")));
    }

    #[test]
    fn chunk_identity_and_line_ranges_are_stable_when_earlier_sections_change() {
        let original = "# Doc\n\n## First\n\nAlpha.\n\n## Target\n\nStable body.\n";
        let edited =
            "# Doc\n\nIntro added.\n\n## First\n\nAlpha changed.\n\n## Target\n\nStable body.\n";
        let before = split_markdown("doc.md", original);
        let after = split_markdown("doc.md", edited);
        let target_before = before
            .iter()
            .find(|c| c.headings.last().is_some_and(|h| h == "Target"))
            .unwrap();
        let target_after = after
            .iter()
            .find(|c| c.headings.last().is_some_and(|h| h == "Target"))
            .unwrap();
        assert_eq!(
            target_before.metadata["_chunk_id"],
            target_after.metadata["_chunk_id"]
        );
        assert_eq!(
            (
                target_before.metadata["_start_line"].as_u64(),
                target_before.metadata["_end_line"].as_u64()
            ),
            (Some(7), Some(9))
        );
        assert_eq!(
            (
                target_after.metadata["_start_line"].as_u64(),
                target_after.metadata["_end_line"].as_u64()
            ),
            (Some(9), Some(11))
        );
    }

    #[test]
    fn identical_sections_receive_unique_stable_ids() {
        let md = "# Doc\n\n## Same\n\nBody.\n\n## Same\n\nBody.\n";
        let chunks = split_markdown("duplicates.md", md);
        assert_eq!(chunks.len(), 2);
        assert_ne!(
            chunks[0].metadata["_chunk_id"],
            chunks[1].metadata["_chunk_id"]
        );
        assert_eq!(
            chunks[0].metadata["_chunk_id"],
            split_markdown("duplicates.md", md)[0].metadata["_chunk_id"]
        );
    }

    #[test]
    fn multibyte_heading_offsets_produce_valid_ranges() {
        let chunks = split_markdown("unicode.md", "# Café ✅\n\nBody.\n\n###### 深い\n\nText.\n");
        assert_eq!(chunks[0].headings, ["Café ✅"]);
        assert_eq!(chunks[1].headings, ["Café ✅", "深い"]);
        assert_eq!(chunks[0].metadata["_start_line"].as_u64(), Some(1));
        assert_eq!(chunks[0].metadata["_end_line"].as_u64(), Some(3));
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
