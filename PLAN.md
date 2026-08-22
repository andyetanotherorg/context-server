# Design notes

## Goal

A single-binary MCP server that indexes organizational markdown into a local vector database and serves semantic search over stdio. Replace heavyweight Python/torch knowledge-base stacks with something you can distribute as one file.

## Why Rust

- [`ort`](https://github.com/pykeio/ort) can statically link ONNX Runtime (no separate `.so` for end users)
- [`fastembed`](https://github.com/Anush008/fastembed-rs) provides BGE-small-en-v1.5 + tokenizers
- Official [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk) MCP SDK

## CLI

```text
context-server index --input <dir> --db context.db
context-server serve --db context.db
context-server search --db context.db [--mode hybrid|dense|lexical] "<query>"
context-server embed "<query>"   # query-style (BGE instruction)
```

## Modules (`src/`)

| Module | Role |
|--------|------|
| `embed` | fastembed BGESmallENV15 (384-dim, L2-normalized); query instruction on search; `MODEL_ID` constant |
| `index` | Markdown heading chunker (`##` / `###`), oversized split + overlap, per-file content hashes |
| `store` | SQLite: documents + embeddings + `files` (content hashes) + `meta` (model_id, dim, chunker_version) |
| `bm25` | In-memory BM25 + reciprocal rank fusion |
| `search` | Hybrid dense + BM25 (default), or dense/lexical only |
| `mcp` | rmcp stdio tools |
| `main` | clap CLI |

## Schema

- `documents`: id, source_path, chunk_index, text, headings (JSON), metadata (JSON)
- `embeddings`: id, dim, vector (little-endian float32 blob)
- `files`: source_path, content_hash (SHA-256 of that file's chunks)
- `meta`: key/value (`model_id`, `dim`, `chunker_version`) — refuse search if model incompatible; re-embed all files if chunker/model changed

`index` upserts changed files and deletes paths that left `--input`. `--full` re-embeds collected files; `--update` skips the prune.

## Chunking

1. Split on `#` / `##` / `###`
2. Prefix each chunk with `Title > H2 > H3`
3. If embedded text exceeds ~1800 chars, split the body with ~200-char overlap (BGE truncates ~512 tokens)

## Search

Default **hybrid**: rank by dense cosine and by BM25, fuse with reciprocal rank fusion (`k=60`). Exact tokens (usernames, acronyms, IDs) come from BM25; paraphrase matching from dense.

## Input contract

Only `.md` / `.markdown` files are indexed. Structured sources (team YAML, etc.) must be converted to prose markdown **upstream**. Putting YAML in fenced code blocks produces poor retrieval; a summary paragraph that keeps entity, roles, and parent together in one chunk works much better.

## MCP tools

- `semantic_search(query, limit, path_prefix?, heading?, tag?)` — hybrid
- `list_documents(limit, path_prefix?)`
- `get_document(source_path, chunk_index?)` — full chunk(s) by citation

## Status

- [x] Index / search / embed / serve
- [x] Markdown chunking + oversized splits + unit tests
- [x] Hybrid BM25 + dense (RRF)
- [x] Model id / dim recorded in DB
- [x] Static ORT (no `libonnxruntime.so` in `ldd`)
- [x] Claude Code stdio MCP verified against a local knowledge base
- [x] PyPI wheels via Containerfile + maturin
- [x] Search filters (`path_prefix`, `heading`, `tag`) + `get_document` citations
- [x] Incremental re-index (per-file content hash; prune missing paths)

## Roadmap

1. Optional stronger embedding models behind a flag
2. Optional musl / static OpenSSL builds for fewer system deps
