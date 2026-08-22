# context-server Project Assessment

Assessment date: 2026-08-22

## Executive summary

`context-server` is a promising foundation. Its core product shape is sound:

- One distributable Rust binary
- Local SQLite database
- Incremental indexing
- Hybrid dense/BM25 retrieval
- MCP and CLI interfaces
- Source citations
- No external service required after the embedding model is downloaded

For the intended scale—hundreds to a few thousand Markdown files—the current brute-force architecture is appropriate. There is no reason yet to introduce a vector database or approximate-nearest-neighbor index.

The recommended product direction is:

> A local, zero-service search appliance for small and medium Markdown knowledge bases, optimized for coding-agent use.

The project should not become a general-purpose distributed vector database. Its competitive strengths should be trustworthy retrieval, safe local indexing, good citations, simple operation, and excellent MCP ergonomics.

Before encouraging wider use, development should focus on four areas:

1. Persisted-index correctness and indexing safety
2. Markdown parsing and chunking correctness
3. Retrieval-quality measurement
4. Packaging and runtime verification

The overall architecture is worth retaining. The most urgent problems are correctness issues around partial index migrations and remote cache validation—not raw scalability.

## Verification performed

The assessment included direct code review, compilation, execution, and synthetic scale testing.

Verified results:

- `cargo test --locked`: 38 passed, 0 failed
- `cargo clippy --locked --all-targets -- -D warnings`: clean
- `cargo fmt --all -- --check`: clean
- Release build: successful
- Statically linked release binary: approximately 39 MiB
- Sample corpus indexing: 6 chunks from 3 files in approximately 0.35 seconds
- Sample searches: approximately 0.22–0.29 seconds per cold CLI invocation
- Repository remained git-clean; project files were not modified during assessment

Synthetic search benchmark using the actual implementation:

| Chunks | SQLite size | Cold CLI search | Peak RSS |
|---:|---:|---:|---:|
| 1,000 | 2.2 MiB | 0.211 s | 218 MiB |
| 10,000 | 21.9 MiB | 0.253 s | 241 MiB |
| 50,000 | 109.6 MiB | 0.409 s | 374 MiB |

The synthetic rows reused short sample documents, so this is a structural performance test rather than a retrieval-quality benchmark. It demonstrates that exhaustive cosine search is not currently the bottleneck at the expected scale.

## Priority findings

### P0: `--update` can create a mixed-model or mixed-chunker index

References:

- `src/main.rs:211-221`
- `src/main.rs:258-300`
- `src/store.rs:182-213`
- `src/store.rs:449-455`

When `needs_full_reembed()` detects a model or chunker change, `force_full` causes every incoming file to be embedded. However, `--update` disables pruning of paths absent from the current input. `apply_index()` then writes the new global model ID, dimension, and chunker version.

The result can contain:

- Newly embedded incoming files using the new model/chunker
- Retained files using the old model/chunker
- Global metadata claiming everything uses the new configuration

This is persistent silent corruption. Future opens consider the database compatible even though dense search may compare query vectors with embeddings produced by different models.

Immediate fix:

- Reject `--update` whenever a global model/chunker migration is required.
- Add a regression test for this exact combination.

Preferred redesign:

- Build a complete new index generation in a staging database.
- Attach model and chunker provenance to that generation.
- Validate it.
- Atomically activate or replace the old generation.

### P0: GCS cache validation trusts a sidecar instead of the cached database

References:

- `src/remote.rs:227-241`
- `src/remote.rs:295-301`

On a cache hit, `read_local_sha256(&checksum_path)` is preferred. The actual cached database is hashed only if the sidecar is absent. A truncated, corrupted, or modified `context.db` can therefore be accepted while a stale sidecar still matches the remote checksum.

Impact:

- Corrupted data can be reused.
- Failures may appear later as misleading SQLite errors.
- Locally altered data could be served despite checksum validation appearing successful.

Fix:

- Hash the actual cached database on every checksum-based cache hit, or
- Store cache entries under immutable, content-addressed paths.

The database and its validation metadata should represent one validated cache generation.

### P0/P1: Default indexing behavior can unexpectedly delete documents

References:

- `src/main.rs:27-51`
- `src/main.rs:211-221`
- `src/store.rs:182-214`

`index` accepts either a file or directory, while pruning is enabled unless `--update` is passed. Consequently:

```bash
context-server index --input one-file.md --db existing.db
```

can remove all other indexed paths.

The behavior is internally consistent with synchronizing an input root, but it is hazardous under a general command named `index`.

Recommended redesign:

```text
context-server sync   --input DIR --db DB
context-server add    --input FILE_OR_DIR --db DB
context-server remove --path PATH --db DB
```

Alternatively, make upsert/no-prune the default and require `--prune` explicitly.

If synchronization remains supported, persist an input-root or collection identity. Relative paths from separate roots can otherwise collide.

### P1: An empty corpus cannot be synchronized, leaving stale data

References:

- `src/main.rs:172-175`
- `src/main.rs:202-221`

`run_index()` aborts before opening the database whenever collection produces no chunks. If every Markdown file is removed, emptied, or contains only headings, pruning never occurs and old content remains searchable.

The application should distinguish between:

- An invalid input path or unsupported input
- A valid collection whose desired state is empty

Support an explicit prune-to-empty transaction, with clear destructive semantics.

### P1: The Markdown parser misinterprets headings inside fenced code blocks

References:

- `src/index.rs:123-230`

The chunker applies a line-oriented heading regex without tracking fenced code blocks. This was reproduced with a shell code fence containing comment lines beginning with `#` and `##`; those comments became Markdown headings and split the document into multiple incorrect chunks.

This affects common technical documentation containing:

- Shell scripts
- Dockerfiles
- Python comments
- Example Markdown
- Configuration snippets

Recommendation:

- Replace the regex parser with a Markdown event parser such as `pulldown-cmark`.
- Track headings structurally.
- Preserve fenced code as content.
- Define behavior for tables, lists, blockquotes, HTML, and H1–H6 headings.

### P1: Front matter parsing is not valid YAML parsing

References:

- `src/index.rs:303-370`

The custom parser treats comma-containing scalars as arrays. For example:

```yaml
title: Hello, world
description: A doc about a, b, and c
```

is misinterpreted.

Other unsupported or surprising cases include:

- Nested mappings
- Multiline values
- Quoted commas
- Comments after values
- Booleans and numbers
- YAML escaping
- Objects inside arrays
- Empty or null values

Recommendation:

- Use a real YAML/front-matter parser.
- Define a deliberately small supported schema, such as `title`, `description`, `tags`, `aliases`, and optional dates.
- Reject or warn about malformed metadata instead of silently inventing a representation.

### P1: Read operations create and mutate SQLite databases

References:

- `src/store.rs:35-74`
- `src/main.rs:324-333`
- `src/main.rs:369-373`
- `src/store.rs:316-352`

`Db::open` always opens read-write/create, enables WAL, and executes `CREATE TABLE IF NOT EXISTS`. A mistyped search or serve path can therefore create a new database.

`CREATE TABLE IF NOT EXISTS` is also not a schema migration strategy. It cannot upgrade an existing table shape safely.

Recommendation:

- Separate `WriterDb::open_or_create` from `ReaderDb::open_read_only`.
- Add a mandatory schema version.
- Introduce ordered migrations or fail fast on unsupported schemas.
- Treat malformed stored JSON, invalid dimensions, and missing embedding rows as integrity errors rather than replacing values with defaults.

### P1: Legacy model compatibility is unsafe

References:

- `src/store.rs:217-251`
- `src/embed.rs:7-13`

When model metadata is missing, compatibility checks only the first embedding dimension. Any unrelated 384-dimensional model can be accepted as BGE-small-en-v1.5.

Recommendation:

- Reject metadata-less databases and require re-indexing.
- Persist an embedding fingerprint containing model repository/revision, pooling behavior, normalization, passage/query transformations, and dimension.

### P1: Lexical-mode score diagnostics are incorrect

References:

- `src/search.rs:213-218`
- `src/search.rs:275-290`
- `src/main.rs:381-388`

Lexical mode passes lexical scores as the generic ranking function's primary score, but that function always stores its primary argument as `dense_score`. A lexical-only result is printed with the BM25 score under `dense` and zero under `lexical`.

Fix:

- Pass explicitly named dense and lexical arrays when constructing results, or
- Have ranking functions return document IDs and perform mode-aware result construction.

Add tests that assert reported score fields, not just ordering.

## Retrieval quality

### No retrieval evaluation exists

Current tests establish mechanical correctness but not product usefulness. They show that handcrafted vectors rank in an expected order, exact BM25 terms can be found, filters work, and chunk counts are stable. They do not establish whether real questions retrieve correct passages.

Create a checked-in evaluation corpus with:

- 50–200 representative documents
- 100–300 queries
- Expected relevant citations or relevance judgments
- Exact identifier searches
- Paraphrased questions
- Acronyms, usernames, paths, and version numbers
- Competing documents with related terminology
- Queries requiring different sections of the same file
- Code and configuration searches
- Negative/out-of-domain questions

Measure:

- Recall@1, @3, @5, and @10
- Mean reciprocal rank
- nDCG
- No-answer precision
- Document diversity
- Per-query latency
- Indexing time
- Resident memory
- Database size

Run dense, lexical, and hybrid variants. This should become the basis for decisions about chunk size, RRF parameters, tokenization, thresholds, and alternative models.

### Irrelevant results are always returned

References:

- `src/search.rs:246-291`

There is no calibrated relevance threshold. Dense cosine scores are often positive even for unrelated content, so the requested result count can be filled with irrelevant chunks.

Recommendation:

- Permit fewer than `limit` results.
- Add a clear no-sufficiently-relevant-results outcome.
- Calibrate thresholds using the evaluation corpus rather than intuition.

### Search lacks document-level diversity and duplicate suppression

Overlapping or adjacent chunks from one source can consume several top positions.

Recommendation:

- Add an optional per-document result cap.
- Collapse overlapping or adjacent hits.
- Support fetching a hit plus bounded neighboring context.
- Preserve the strongest citation while avoiding redundant output.

### RRF ordering can be nondeterministic on ties

References:

- `src/bm25.rs:117-127`

RRF scores are accumulated in a `HashMap`, then sorted only by score. Equal-score ordering can inherit arbitrary map iteration order.

Use deterministic tie-breakers such as:

1. Fused score
2. Best individual rank
3. Dense score
4. Lexical score
5. Stable chunk ID or citation

### BM25 is English/ASCII-centric

References:

- `src/bm25.rs:85-104`

The tokenizer retains only ASCII alphanumerics, `_`, and `-`. Non-Latin text disappears from lexical search. It also keeps hyphenated identifiers only as whole tokens, so `foo-bar` will not lexically match `foo` or `bar` independently.

This is acceptable only if the product explicitly targets English technical Markdown. Otherwise, adopt Unicode-aware and identifier-aware tokenization, retaining both whole identifiers and useful component forms.

### Ranking ignores useful fields

Source paths, titles, tags, and other metadata are filters but are not meaningfully weighted as retrieval fields.

A future lexical implementation should treat these as separate fields so headings and filenames can receive stronger weights than body text. SQLite FTS5 is a reasonable candidate.

## Chunking and citations

### Character-count chunking should become token-aware

References:

- `src/index.rs:11-18`
- `src/index.rs:233-301`

The chunker estimates four characters per token and caps text around 1,800 characters. That is safe but imprecise for code, identifiers, and Unicode.

Recommended split order:

1. Markdown sections
2. Paragraphs, lists, tables, or other block boundaries
3. Sentences
4. Token windows as a fallback

Use the embedding model tokenizer for the final size limit. Preserve code fences atomically where possible and record token counts for diagnostics.

### Chunk provenance needs more structure

Store additional fields such as:

```text
document_id
section_id
stable_chunk_id
start_line
end_line
start_offset
end_offset
split_index
token_count
```

This supports verifiable citations, neighboring-context expansion, deduplication, and diagnostics.

### Positional citations are not durable

`source_path#chunk_index` changes when an earlier section is inserted or chunking rules change. It is suitable as a transient display citation but not a durable identifier.

Use a content-derived or section-derived stable chunk ID internally, while retaining short path-based citations for human readability.

## MCP and CLI usability

### `list_documents` lists chunks, not documents

The MCP tool currently returns chunk rows. A document-oriented listing should instead return one row per source file with title, path, chunk count, tags, and perhaps a short summary.

Expose `list_chunks` separately only if it is useful.

### MCP output is unbounded

References:

- `src/search.rs:164-172`
- `src/search.rs:256-272`
- `src/mcp.rs:180-207`
- `src/mcp.rs:213-240`
- `src/mcp.rs:247-277`

Caller-provided limits are not capped, and requesting a document without a chunk index returns every full chunk in that file.

Recommended hard bounds:

- Search results: maximum 20
- Listings: maximum 200
- Text output: configurable character or token budget
- Full-document access: pagination or chunk ranges

Return `truncated`, `has_more`, and cursor metadata where appropriate.

### MCP responses should be structured

Tools return formatted strings rather than structured result objects. Structured responses would be easier and safer for agents to consume.

A search response should contain fields such as:

```json
{
  "query": "...",
  "results": [
    {
      "citation": "path#3",
      "chunk_id": "...",
      "source_path": "path",
      "chunk_index": 3,
      "heading_path": ["...", "..."],
      "start_line": 40,
      "end_line": 57,
      "rank": 1,
      "score": 0.032,
      "dense_score": 0.71,
      "lexical_score": 2.4,
      "snippet": "..."
    }
  ],
  "truncated": false
}
```

Use MCP-native tool errors with machine-readable codes rather than ordinary strings beginning with `error:`.

### Search and `get_document` overlap unnecessarily

`semantic_search` returns the entire chunk, then instructs the agent to call `get_document` for the same chunk. This wastes context and makes the follow-up tool redundant.

Prefer:

- Search returns bounded snippets and metadata.
- A context tool returns the selected chunk plus optional neighboring chunks.

### Tool guidance is contradictory

The default server instructions are measured, while the `semantic_search` tool description says it is “REQUIRED.” This can cause unnecessary tool use.

Use consistent, conditional language across server and tool descriptions.

### Add diagnostic commands

Useful additions:

```text
context-server status --db context.db
context-server inspect --db context.db
context-server doctor
context-server validate --db context.db
```

Report:

- File and chunk counts
- Database size
- Schema version
- Model fingerprint
- Chunker version
- Last indexing time
- Model cache location
- Files skipped, changed, and pruned
- Token count percentiles
- Empty or oversized documents
- Front-matter parse errors
- Available tags and path prefixes
- Current MCP instructions

Add `--json` output to indexing, search, retrieval, status, and inspection commands.

## Performance and scalability

### The current design is appropriate for the expected scale

References:

- `src/store.rs:316-353`
- `src/search.rs:186-219`
- `src/bm25.rs:8-82`

All documents and vectors are loaded into memory, dense search scans all vectors, and BM25 stores per-document term-frequency hash maps. This remains simple and fast at the expected scale.

Suggested operating envelope:

- Primary target: up to 10,000 chunks
- Supported after regular benchmarking: up to 50,000 chunks
- Re-evaluate architecture around 100,000 chunks or approximately 500 MiB+ serving RSS

The relevant unit is chunks, not documents. One thousand long documents may create tens of thousands of chunks.

### BM25 memory is likely to become the first structural bottleneck

Raw 384-dimensional vectors require approximately:

- 10,000 chunks: 14.6 MiB
- 50,000 chunks: 73.2 MiB
- 100,000 chunks: 146.5 MiB

Strings, metadata, text duplication, and per-document hash maps account for much of the additional memory. Observed peak RSS was approximately 374 MiB at 50,000 short synthetic chunks.

Optimize lexical storage before adding ANN:

- Consider SQLite FTS5 for lexical candidate generation.
- Store vectors in one contiguous `Vec<f32>` or matrix instead of separate allocations.
- Avoid cloning all text during BM25 construction.
- Profile actual corpora before redesigning.

### Do not add ANN yet

Approximate vector search adds complexity, persistence concerns, tuning, and dependencies without evidence that it is needed. Exact dense scanning is fast enough for the current target.

Only consider ANN after measured p95 latency at realistic corpus sizes is unacceptable.

### Embedding inference is serialized

References:

- `src/mcp.rs:56-63`
- `src/mcp.rs:196-204`

A mutex serializes embedding inference. This is fine for a local stdio server with one coding-agent client, but it would not suit a multi-user HTTP service.

Document the current concurrency contract. If HTTP or multi-user support is added, use a bounded inference worker pool and avoid blocking Tokio worker threads.

### Indexing retains all changed vectors before committing

References:

- `src/main.rs:260-300`

Changed chunks and vectors accumulate in memory before `apply_index()`. This is acceptable at the expected scale but increases full-rebuild memory.

A future generation-based design can stream batches into a staging database and atomically replace the active database after validation.

## Packaging and operations

### Packaged artifacts need installation and runtime smoke tests

The README claims a self-contained ONNX Runtime binary. During assessment, an ordinary local release build initially detected a host `libonnxruntime` through pkg-config and linked an incompatible ONNX Runtime 1.22. Runtime then aborted because the crate required API version 24.

A correctly configured build produced a statically linked working binary, but the incident demonstrates that build success alone does not prove artifact correctness.

Every release artifact should be installed and executed after packaging:

```bash
uv venv
uv pip install dist/*.whl
context-server --help
context-server index --input examples/sample-docs --db /tmp/test.db
context-server search --db /tmp/test.db "password reset"
ldd "$(command -v context-server)"
```

On Linux, assert that no dynamic `libonnxruntime` dependency exists. Test in an environment that already contains an older system ONNX Runtime.

### The documented Rust 1.75 minimum is incorrect or unenforced

References:

- `README.md:148-155`
- `Cargo.toml:1-4`
- `src/search.rs:53-58`
- `.github/workflows/ci.yml:33-44`

`Cargo.toml` has no `rust-version`, CI only uses current stable, source uses APIs newer than Rust 1.75, and the resolved `ort-sys` package requires a newer compiler.

Recommendation:

- Declare the actual MSRV in `Cargo.toml`.
- Add an MSRV CI job.
- Update the README.
- Or explicitly support current stable Rust only.

### Reduce unnecessary fastembed features

The resolved fastembed defaults include image-model functionality although this project uses text embeddings only. Investigate disabling default features and enabling only required text, model-download, and TLS features.

Verify the resulting feature set and binaries on every release platform.

### First-run model download needs explicit UX

The binary is self-contained, but the embedding model is not. First use downloads roughly 127 MiB and depends on network, proxy, CA, and cache behavior.

Consider:

```text
context-server model download
context-server model status
context-server doctor
```

Also support or document:

- Offline mode
- Expected download size
- Download URL/domain requirements
- Hash verification
- Proxy and custom CA behavior
- Cache location
- Clear incomplete-cache recovery

## Hot reload concerns

References:

- `src/mcp.rs:56-85`
- `src/mcp.rs:122-160`

Hot reload compares file/WAL sizes and modification times. These are filesystem heuristics, not database generation identities, and may miss changes on coarse-timestamp filesystems or react to WAL maintenance.

Database, index, and instructions are also replaced under separate mutexes, so concurrent calls could observe different generations.

Recommendation:

- Store an explicit monotonically increasing generation in the committed transaction or use an appropriate SQLite change identity.
- Publish one immutable search snapshot atomically, such as an `Arc` containing all generation state.
- Treat MCP server instructions as initialization-time metadata and document when reconnecting is required.

## Recommended implementation sequence

### Phase 1: Persisted-index correctness and safety

1. Reject `--update` during model/chunker migrations.
2. Add the mixed-generation regression test.
3. Hash the actual cached GCS database on cache hits.
4. Separate `add` from destructive `sync` semantics.
5. Support explicit synchronization to an empty corpus.
6. Separate reader and writer database opening.
7. Add schema versioning and integrity validation.
8. Reject legacy databases without trustworthy model provenance.
9. Fix lexical score reporting.
10. Add deterministic ranking tie-breakers.

### Phase 2: Ingestion correctness

1. Replace regex Markdown parsing with structural parsing.
2. Replace ad hoc YAML parsing.
3. Make chunk sizes tokenizer-aware.
4. Preserve Markdown block boundaries and code fences.
5. Store stable chunk identities and source line ranges.
6. Add edge-case tests for code fences, tables, lists, complex front matter, empty documents, and long sections.

### Phase 3: Retrieval evaluation

1. Create a realistic benchmark corpus and relevance judgments.
2. Add Recall@k, MRR, nDCG, no-answer, latency, memory, and index-size reports.
3. Evaluate chunk sizes and overlap.
4. Calibrate no-result thresholds.
5. Add document diversity and adjacent-context expansion.
6. Compare custom BM25 with SQLite FTS5.
7. Only then evaluate alternative embedding models or fusion parameters.

### Phase 4: MCP and CLI usability

1. Return structured MCP results and native errors.
2. Make `list_documents` truly document-oriented.
3. Add bounded snippet and context retrieval.
4. Cap and paginate all potentially large responses.
5. Add `status`, `inspect`, `doctor`, and `validate`.
6. Add JSON CLI output.
7. Improve model download and offline behavior.
8. Align MCP instructions and tool descriptions.

### Phase 5: Distribution and scale

1. Install and execute every built wheel in CI.
2. Verify the installed binary has no dynamic ONNX Runtime dependency.
3. Declare and test the actual MSRV.
4. Benchmark 1k, 10k, 50k, and 100k realistic chunks regularly.
5. Move lexical search to FTS5 or an inverted index if memory warrants it.
6. Store vectors contiguously if profiling shows meaningful benefit.
7. Add ANN only if measured exact-search latency becomes unacceptable.

## Implementation completion matrix

The findings in this assessment are addressed by the following verified changes:

- Mixed index migrations: rejected unless complete `--sync` is used.
- GCS cache integrity: cached database bytes are hashed directly.
- Index semantics: safe upsert default, explicit destructive synchronization.
- Empty corpus: explicit empty synchronization prunes stale rows.
- Database access: read-only serve/search/get/hot-reload readers and explicit schema compatibility.
- Embedding provenance: versioned, fail-closed fingerprint.
- Markdown/YAML ingestion: structural Markdown parser and bounded real YAML decoding.
- Citations: source-namespaced stable IDs and source line ranges.
- Ranking correctness: lexical score attribution and deterministic ties.
- Retrieval evaluation: golden cases with recall@k and MRR.
- Lexical quality: Unicode and identifier component tokenization.
- Relevance safeguards: dense threshold and per-source diversity cap.
- MCP safety: hard result limits and offset pagination.
- Diagnostics: status, JSON status, validation, model cache status/download.
- Hot reload: committed index generation recorded transactionally.
- Packaging: declared/tested MSRV, release and installed-wheel smoke tests.
- Performance: repeatable 1k/10k/50k benchmark script and scale triggers.

ANN/vector-database and SQLite FTS5 adoption remain conditional optimizations, not
unresolved findings: benchmarks show the exact implementation is appropriate for
the stated operating envelope. They should only be introduced when measured
latency or memory crosses the documented trigger.

## Bottom line

The project should retain its current overall architecture. It is simple, understandable, fast enough, and well matched to the expected workload.

The most important immediate work is:

1. Prevent mixed-model/chunker indexes.
2. Correct GCS cache validation.
3. Make destructive synchronization explicit.
4. Replace the Markdown and front-matter ingestion pipeline.
5. Build a retrieval evaluation harness.

The project does not currently have a scale problem. It has a trust and measurement problem. Fixing persisted-index correctness, ingestion fidelity, retrieval evaluation, and artifact verification will make it substantially more useful without sacrificing its strongest quality: low operational complexity.
