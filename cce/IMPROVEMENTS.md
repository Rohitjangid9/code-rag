# CCE — Industry-Level Robustness Audit

> Date: 2026-04-23 • Baseline run: `cce/agent_run_20260423_102232.txt` (70/90 ≈ 78 %)
> Scope: every module under `cce/` audited for gaps between the current
> implementation and what a production code-context engine needs in order to
> answer arbitrary easy / medium / hard / expert questions reliably.

This document is the **single source of truth** for the roadmap. It supersedes
`cce/changes.txt` (which only covered the P0 hot-fix set). Everything in P0 is
now implemented and verified; this file focuses on what's left.

---

## 1. Question Taxonomy — what the system must handle

Classifying every question the agent can receive into four tiers. The target
accuracy per tier drives which improvements are mandatory vs nice-to-have.

### Tier 1 — EASY (one deterministic tool call)
_Target: **95 %+ accuracy**, no LLM reasoning needed at all._

| Pattern | Tool | Status |
|---|---|---|
| "Where is `X` defined?" | `get_symbol(qname)` / `search_code(x, mode="symbols")` top-1 | ✅ P0 fixed (Q1 9/10) |
| "List routes in router Y" | `list_routes(file=...)` | ✅ P0 fixed (Q3 10/10) |
| "List all CLI commands" | `list_cli_commands()` / `list_symbols(file="cli.py")` | ⚠ partial (Q7 8/10 — missing 3/18) |
| "What is `X`'s signature / docstring?" | `get_symbol(qname)` | ✅ works |
| "What files contain `X`?" | `search_code(x)` → file aggregate | ✅ works |
| "Show me all classes in file Y" | `list_symbols(file=Y, kind="Class")` | ✅ P0 added |
| "What does file Y import?" | ❌ **no tool** | ❌ **gap** |

### Tier 2 — MEDIUM (2–3 tool hops, light synthesis)
_Target: **80-85 % accuracy**._

| Pattern | Tool chain | Status |
|---|---|---|
| "Who calls `X`?" | `find_callers(x)` → fallback `grep_code(x)` | ⚠ Q2 8/10; edges are incomplete |
| "What happens when POST `/a/b`?" | `list_routes` → match effective path → `get_symbol` → body slice | ❌ Q4 fails on prefix-mounted routes |
| "List all agent / graph / pipeline nodes" | `list_symbols(file=..., kind="Function")` or `grep "def .*_node\("` | ❌ Q6 2/10 — planner picks wrong tool |
| "What imports module Y?" | needs an `IMPORTS` edge kind | ❌ **gap** |
| "Which tests cover function X?" | `find_callers(x) WHERE file LIKE '%test%'` | ⚠ works if edge exists |
| "Show full body of symbol X" | ❌ no `get_file_slice(path, start, end)` tool | ❌ **gap** |
| "What env vars does this codebase use?" | `grep_code("os.environ|getenv")` | ✅ works via grep |

### Tier 3 — HARD (multi-hop, narrative synthesis)
_Target: **60-70 % accuracy**._

| Pattern | Tool chain | Status |
|---|---|---|
| "Trace request from HTTP to DB" | route → view → service → repo, reading bodies along the way | ⚠ possible but line-heavy; no "follow CALLS chain" helper |
| "How is the codebase indexed end-to-end?" | pipeline narration | ⚠ Q8 7/10 — content good, lines all `?` (validator strips) |
| "How does auth work in this codebase?" | broad concept retrieval + middleware scan | ✅ Q9 9/10 (honest "no auth") |
| "How does retriever rank results?" | symbol + neighborhood + body | ✅ Q5 9/10 |
| "What would break if I rename X?" | callers + references + text grep + test grep | ⚠ partial (REFERENCES edges missing) |
| "What config / flags exist?" | AST scan of pydantic models + CLI params | ❌ no config-specific tool |

### Tier 4 — EXPERT (cross-module, reasoning-heavy)
_Target: **40-55 % accuracy** (LLM-bound, beyond purely deterministic retrieval)._

| Pattern | What's needed |
|---|---|
| "Where's the race condition in X?" | bodies of all involved symbols + threading primitives |
| "Compare implementation A vs B" | side-by-side diff + semantic reasoning |
| "Suggest where to add feature F" | impact analysis on callers + architecture conventions |
| "What's the SLA / timeout for endpoint E?" | route → middleware → config stack |
| "Explain the caching strategy" | cross-module concept retrieval |
| "Draw a diagram of module Z" | structural dump + relationship edges |

**Headline gap:** tiers 1 and 2 should be *mechanical* — the current system
still leans on the LLM to pick tools correctly. Every tier-2 miss in our runs
was a **tool-routing failure** rather than a retrieval failure. The rest of
this document details how to close those gaps.

---

## 2. Current State — what P0 fixed and what remains

**P0 delivered (verified in run 20260423_102232):**
- Tool-call leak parsing → CJK garbage eliminated from responses.
- Strict citation validation → no more fabricated line numbers.
- Deterministic enumeration tools (`list_routes`, `list_symbols`, `list_files`,
  `list_cli_commands`, `grep_code`).
- `cce doctor` warns on tool-incompatible LLMs.
- Score moved 36 → 70 / 90 on the same 9-question harness.

**Still failing on the live run:**
1. **Q4** — route prefix not stored on the Route node → "POST /agent/query" looks absent.
2. **Q6** — planner chose `search_code` for "list all agent nodes" instead of `list_symbols(file_path="agents/nodes.py", kind="Function")`.
3. **Q8** — every citation collapses to `:?` because retrieved hits are
   whole-file lexical rows (`line_start=0`) that the validator can't match against.
4. **Q7** — 15 / 18 commands (missing `get-api-flow`, `info`, `inspect-qdrant`) — `list_symbols` top-k limit hit.

The per-module audit below surfaces every other latent issue.

---

## 3. Per-Module Audit

Issues are tagged **[SEV]** as **🔴 critical** (correctness bug), **🟠 major**
(limits answer quality), **🟡 minor** (nice to have). Each one is actionable.

### 3.1 `cce/indexer.py` — pipeline orchestration

- 🔴 **Partial-failure mode is silent.** If the `symbols` layer fails for a
  file, downstream `graph` and `framework` still run on stale / empty data.
  `stats.errors` captures the string, but nothing prevents bad graph edges
  from being emitted for half-parsed files. **Fix:** short-circuit on
  per-file parser error; mark that file as "symbols-only" or skip.
- 🟠 **No parallelism.** Per-file indexing is serial; a 5 k-file repo takes
  minutes. Parsers, hashing, and embedding are all I/O / CPU bound and
  trivially parallelizable with `ProcessPoolExecutor`.
- 🟠 **Semantic layer re-embeds every changed file whole.** There's no
  per-chunk content hash, so a one-line edit costs 50 embedding calls for a
  large module. **Fix:** store `chunk_content_hash` in Qdrant payload and
  skip upsert when unchanged.
- 🟠 **`_resolve_references` runs Jedi on every file without caching** the
  Jedi `Project` across files. Jedi warms its resolver per-project; throwing
  it away between files is 5-10× slower than it needs to be.
- 🟡 Layers are order-independent by contract but in practice `framework`
  needs `symbols` to be fully written (it looks up symbols by qname). Not
  documented, not enforced — a `--layers graph,framework` invocation succeeds
  silently with empty framework output.
- 🟡 No transaction boundary — an interrupted index leaves SQLite mid-write
  with orphaned symbols but no edges.

### 3.2 `cce/agents/nodes.py` — planner / retriever / reasoner / responder

- 🔴 **`_validate_citations` has no file-level fallback** (cause of Q8's
  `:?` storm). When a retrieved hit is a whole-file lexical match with
  `line_start=0, line_end=0`, any line cited in that file gets dropped.
  **Fix (5 min):** treat `line_start=0 AND line_end=0` as "any line in this
  file".
- 🔴 **`retriever_node` swallows tool errors as strings.** If `get_symbol`
  raises, the planner sees `"Tool error: ..."` as a successful ToolMessage
  and has no structured signal to retry with a different tool. **Fix:**
  attach `status=error` to the ToolMessage and expose it to `reasoner_node`
  so the loop gate can re-plan.
- 🟠 **`reasoner_node` only counts hits.** It appends
  `"loop N: K items retrieved"` and stops. It doesn't actually reason about
  *coverage* — whether the retrieved items answer the question. A real
  reasoner would score: "do we have a symbol that matches the question's
  subject? do we have its body? do we have its call sites?" and force another
  loop if any are missing.
- 🟠 **No tool-result summarization between loops.** After three retrieval
  loops the message list grows to 60+ items and tokens explode.
  `_trim_messages` trims at 15 k words, which is both too high for small
  LLMs and too crude (keeps the oldest turns over the most recent).
  **Fix:** compress previous `ToolMessage` contents to 1-line summaries
  once a newer retrieval for the same tool has landed.
- 🟠 **Planner has no "reflection" prompt on re-entry.** Every loop calls
  the LLM with the same system prompt; the LLM has no explicit instruction
  to look at what it already retrieved and pick *different* tools next. Q6
  looped three times getting increasingly similar `search_code` results.
- 🟡 `_hit_key` uses `(path, line_start)` for dict hits, which collides
  for symbol hits vs lexical hits at line 0 of the same file.
- 🟡 `_format_context_block` caps snippet at 160 chars — not enough for the
  responder to paraphrase; bump to 400-800 when budget allows.
- 🟡 `planner_node` always emits `loop_count += 1` even when the LLM fails
  and a fallback search runs; arguably the fallback should count as "turn 1"
  only, not burn a loop budget.

### 3.3 `cce/agents/llm.py` — model wiring

- 🔴 **Single LLM for planner + responder.** A small fast model
  (`gpt-4o-mini`) is great for tool selection but mediocre for narrative
  synthesis, and a large model is overkill (slow + expensive) for tool
  selection. **Fix:** split config into `agent.planner_model` and
  `agent.responder_model` and resolve separately in `get_llm()` /
  `get_responder_llm()`.
- 🟠 No retry / rate-limit handling. On a 429, the whole turn fails.
- 🟠 API key load is eager at import time — key rotation requires a server
  restart.
- 🟡 `SYSTEM_PROMPT` playbook is static text; should be generated from the
  actual `ALL_TOOLS` metadata so docstring changes propagate automatically.

### 3.4 `cce/agents/tools.py` + `cce/retrieval/tools.py` — tool surface

- 🔴 **`list_routes` doesn't return effective path.** The Route node stores
  `path="/query"` but not the router prefix (`/agent`). **Fix:** resolve
  `include_router(router, prefix="/X")` at indexing time and add
  `effective_path` to the Route node's `meta`. Q4 depends on this.
- 🔴 **`find_callers` / `find_references` only consult the CALLS/INHERITS
  edges.** The Python resolver only emits CALLS for call sites — so
  `builder.add_node("planner", planner_node)` (identifier-as-argument) never
  becomes an edge. Q2's "who calls planner_node" currently lives only in
  `grep_code`. **Fix:** extend `python_resolver` to emit REFERENCES edges
  for identifier arguments, decorator targets, and dict values.
- 🟠 **No `get_file_slice(path, start, end)` tool.** When an answer needs
  the body of a symbol the planner has only `get_symbol` (header + sig +
  docstring) or `grep_code` (single-line hits). Reading a 15-line body
  requires a tool.
- 🟠 **No `list_imports(file)` tool.** Every "what does X depend on?"
  question falls back to a noisy `grep_code("^import|^from")`.
- 🟠 **No `find_route_handler(method, path)` tool.** Trivial to implement
  (SELECT on Route nodes with effective_path) and would nail Q4
  deterministically.
- 🟠 **No `get_symbol_full(qname)` variant.** `get_symbol` returns header
  only; the body requires a separate file read. Combine them.
- 🟠 **`search_code` has no auto-fallback.** `mode="vector"` on an
  un-embedded index returns `[]` silently; the planner then concludes "no
  results" and stops. **Fix:** if `mode=vector` and collection is missing,
  degrade to `mode=hybrid` (or lexical) and log a warning.
- 🟠 **Top-k limits cause Q7-style misses.** `list_symbols` limit defaults
  to 200, but `list_cli_commands` wraps it with default 50. **Fix:** expose
  per-tool `limit` and have the planner pass `limit=500` when it sees "all"
  in the question.
- 🟡 No MMR / diversity — `search_code` top-10 can be 10 hits from the same
  file, which wastes context budget.
- 🟡 `grep_code` re-tokenizes the FTS content per-call; for a watched repo
  a memory-mapped rg binary would be 100× faster.

### 3.5 `cce/retrieval/hybrid.py` — RRF fusion

- 🟠 **Fixed RRF weights.** The formula is `1/(k+rank)` with `k=60`
  hard-coded. Tuning `k` per question type (short keyword vs long
  conceptual) measurably moves quality.
- 🟠 **No query rewriting / expansion.** "how does auth work" hits
  `authenticate`, `authorize`, `oauth`, `jwt`, `session`, `bearer` in
  concept but BM25 only matches literal tokens. **Fix:** add a tiny LLM
  query-expansion step or a synonym map for common domains.
- 🟠 **Graph expansion is single-hop.** For "trace from HTTP to DB" you
  want 2-3 hops along CALLS edges. Expose `depth` via tool args.
- 🟠 **No cross-encoder reranker.** Top-k in RRF is a decent candidate
  generator but the final ordering benefits from a small cross-encoder
  (`ms-marco-MiniLM`) on the top 50 candidates.
- 🟡 Snippets returned as "signature" only; no surrounding-body preview.

### 3.6 `cce/parsers/python_resolver.py` — reference resolution

- 🔴 **Only call-expressions are resolved.** Decorators, identifier
  arguments, dict values, type hints, `getattr` dynamic lookups — none of
  these produce edges. Directly causes Q2's failure mode.
- 🟠 **Jedi is slow and not cached.** 30-70 ms per file × N files. **Fix:**
  share a single `jedi.Project` across the whole pipeline run.
- 🟠 **No handling of re-exports.** `from .nodes import planner_node` in
  `__init__.py` doesn't produce an alias edge; queries on the aliased name
  fail.
- 🟠 **Binary confidence.** Jedi resolved vs name-matched is all the signal
  there is. Callsite heuristics (enclosing class, is-self-dotted, capitalized)
  could give 5-level confidence for reranking.
- 🟡 No support for PEP-695 `type X = ...` aliases.

### 3.7 `cce/parsers/tree_sitter_parser.py` — symbol extraction

- 🟠 **Languages limited to Python / JS / TS / TSX.** No Go, Rust, Java,
  C# — a non-starter for many enterprise repos. Tree-sitter grammars exist
  for all of these; the registry wiring is the gap.
- 🟠 **Module-level side-effect code is invisible.** `app = FastAPI()` at
  module scope isn't a symbol, so "where is `app` created" has no answer.
  **Fix:** emit a `ModuleVar` kind for top-level assignments that produce
  named values used by other files.
- 🟠 **Decorator metadata not attached.** `@app.command()` on a function
  doesn't leave a trace on the Function node. Every extractor has to re-walk
  the AST to find decorators. Store them on `Node.meta["decorators"]`.
- 🟡 **Nested functions / closures inside methods aren't always captured**
  depending on the containing class structure.

### 3.8 `cce/extractors/` — framework extractors

- 🔴 **FastAPI extractor ignores router composition.** `include_router(r,
  prefix="/agent")` in `server/app.py` is not fed back into the Route nodes
  indexed from `routes/agent.py`. Direct cause of Q4. **Fix:** add a
  second-pass resolver that reads all `include_router` calls and stamps
  `effective_path` onto each child Route.
- 🟠 **FastAPI extractor misses**: `app.add_api_route(...)`,
  `@app.websocket(...)`, `Mount(...)`, sub-apps.
- 🟠 **FastAPI `Depends` captured but not linked.** The Depends function is
  indexed as a Route parameter but no DEPENDS_ON edge is emitted, so
  "what dependencies does route X have" is unanswerable via tools.
- 🟠 **No Typer / Click / argparse extractor** — the P0 `list_cli_commands`
  falls back to `list_symbols(file="cli.py")` which picks up helper
  functions too (Q7 included `_check_tool_call_support` as candidate before
  dedup). The proper fix is a dedicated extractor that only emits nodes
  where `@app.command()` / `@click.command()` decorators are present.
- 🟠 **No LangGraph extractor.** `StateGraph().add_node("x", fn)` /
  `add_edge("a","b")` carries enormous structural signal — exactly what Q6
  wanted — but there's no extractor that turns it into graph-node metadata.
- 🟡 Django URL resolver doesn't follow `include(...)` into sub-url-confs.
- 🟡 React extractor doesn't follow `React.lazy(() => import(...))`.
- 🟡 **Framework detection is single-label per file.** A file that
  imports both `fastapi` and `sqlalchemy` gets one tag; queries for
  ORM-specific concepts miss.

### 3.9 `cce/index/symbol_store.py` + `lexical_store.py`

- 🟠 **FTS5 porter stemmer has poor recall on code tokens.** `planner_node`
  stems to `planner_nod`, `list_routes` to `list_rout`. Works for English but
  fragile for snake_case / camelCase / typos. **Fix:** add an `unicode61`
  tokenizer variant or a bigram index for exact identifier match.
- 🟠 **No n-gram / trigram fallback** for fuzzy name search (useful when the
  user misspells or uses a different case convention).
- 🟠 **Lexical store is whole-file.** A single row per file means BM25
  ranks big files high regardless of *where* the match lives. **Fix:**
  chunk by function / 50-line windows before indexing into FTS.
- 🟡 `search(q, k=10)` has no filter params (kind, framework_tag, language).

### 3.10 `cce/index/vector_store.py`

- 🔴 **Stale vectors after re-index.** `delete_for_node_ids` exists but
  isn't called automatically when a file's symbols change. After a big
  refactor, Qdrant can hold vectors for deleted symbols; they surface as
  phantom results.
- 🟠 **Full re-embed on any change** (see §3.1).
- 🟠 **No filterable payload fields** beyond `node_id` / `path` / `kind`.
  Can't filter by `framework_tag`, `language`, or `visibility` efficiently.
- 🟡 `_chunk_uuid` uses md5 → collisions theoretically possible.

### 3.11 `cce/graph/sqlite_store.py` vs `kuzu_store.py`

- 🟠 **Two backends, subtly different surfaces.** Consumers branch on
  availability. Freeze one (SQLite) as the canonical backend; keep Kùzu
  behind an opt-in flag until the surface is fully parity.
- 🟠 **`find_references` on Kùzu is stubbed** (returns `[]`); SQLite
  implements it. A user switching backends silently loses the feature.
- 🟡 No bulk-insert API — each edge is its own `INSERT`; slow for large repos.

### 3.12 `cce/walker.py` + `cce/hashing.py`

- 🟠 **`sha256_file` re-reads every file even when mtime+size match.** On
  a clean incremental run this is the bottleneck. **Fix:** short-circuit
  with `(mtime, size)` before reading, and only hash if either changed.
- 🟠 **No `.cceignore`.** Users can't exclude paths beyond `.gitignore`.
- 🟡 `walk_repo` reloads `.gitignore` on every call; cache per-root.
- 🟡 Large-file cap is 1 MB in the watcher but unlimited in the full
  pipeline — inconsistent.

### 3.13 `cce/watcher/`

- 🟡 `file_watcher.py` has dead code (`stats = self._pipeline._index_file.__func__`
  assignment never used).
- 🟡 No backpressure — a burst of 10k file events floods the debounce queue.
- 🟡 `git_watcher.py` timeouts inconsistent (10 s vs 15 s vs none).

### 3.14 `cce/server/` — HTTP surface

- 🔴 **No auth, no rate limiting.** CORS wildcard with
  `allow_credentials=True` is a browser CSRF vector.
- 🟠 **No request-size / timeout limits** on `/agent/query`. A malicious
  client can send a 10 MB query and lock a worker.
- 🟠 **No per-`thread_id` concurrency guard.** Two parallel POSTs sharing a
  thread_id race on the LangGraph checkpointer — state can corrupt.
- 🟡 SSE stream has no keepalive; long-running reasoners may drop on proxy
  idle timeouts.
- 🟡 `lifespan` is a no-op; embedder and DB are lazily initialized on first
  request (cold-start latency).

### 3.15 `cce/config.py`

- 🟠 **No validation of `llm_model` vs `llm_provider`.** `provider=anthropic
  model=gpt-4o-mini` silently fails deep inside langchain.
- 🟠 **No decoupled planner / responder model settings** (see §3.3).
- 🟡 `ensure_dirs` misses the `agent_checkpoint` parent.
- 🟡 No profile support — `dev`, `ci`, `prod` configs have to be set via env
  manually.

### 3.16 `cce/embeddings/`

- 🟠 **No query cache.** Repeated questions re-embed the same query string.
  A small LRU on the query embedding would halve vector-search latency.
- 🟠 **Word-based chunk budgets are imprecise** relative to token count.
  Long strings, URLs, or base64 blobs can blow past a model's context. Use
  `tiktoken` or the embedder's own tokenizer for accurate budgets.
- 🟡 Chunker treats every symbol identically; for classes, splitting by
  method (with shared header context) gives better retrievability than one
  big blob.

### 3.17 `cce/scip/` (partially implemented)

- 🟡 Emitter exists; not wired into the watcher or incremental flow.
- 🟡 REFERENCES edges (once added — §3.6) should flow into SCIP output so
  external tools (Sourcegraph etc.) can consume CCE's index.

---

## 4. Cross-Cutting Concerns

These aren't single-module bugs — they're system-level gaps that matter for
industry-level robustness.

### 4.1 Observability

- **No structured per-turn trace log.** `reasoning_steps` is a free-text list
  the LLM reads. A JSONL trace (`logs/agent_trace.jsonl`) with
  `{thread_id, turn, tool, args, elapsed_ms, hit_count, error?}` is needed
  for debugging production runs. Already half-there; needs machine-readable
  format.
- **No retrieval metrics.** Per-query recorded RRF scores, top-k candidates
  before fusion, cross-encoder outputs (when added) — none of this is
  captured. Without it, tuning is blind.
- **No OpenTelemetry.** Every hop (planner → retriever → tool → DB) should
  emit a span. Retrofitting later is much harder than starting now.
- **No index manifest.** `.cce/index.json` with `{root, indexed_at,
  commit_sha, file_count, layers, schema_version}` would let the server
  detect stale indexes at startup and let `cce doctor` surface drift.

### 4.2 Evaluation

- **Ad-hoc eval runs** against a hand-picked 9-question set. A real system
  needs:
  - A versioned golden dataset at `cce/eval/datasets/*.yaml` with
    `{query, expected_symbol?, expected_file?, expected_lines?,
    must_cite?, must_not_cite?, tier}`.
  - `cce eval run --dataset X` that produces `scorecard.json` with metrics
    (exact-symbol hit, file hit, line IoU, citation validity,
    completeness vs enumeration expected count).
  - CI job that fails a PR if **any tier-1** question regresses.
- **No LLM-judge mode.** For tier-3/4 narrative answers, deterministic
  scoring doesn't work. Add an optional LLM-judge step with a rubric.

### 4.3 Testing

- Unit tests are phase-keyed (`test_phase9_agent.py`) and cover happy paths.
  Missing:
  - `_parse_leaked_tool_calls` — all three shapes + malformed input.
  - `_validate_citations` — valid / out-of-range / file-suffix mismatch
    (Windows vs Unix) / whole-file hit fallback.
  - `list_routes` with / without router prefix resolution.
  - `grep_code` with regex special chars in the pattern.
  - End-to-end: build a tiny fixture repo, index it, run all 9 question
    patterns, assert scorecard ≥ N / 90.

### 4.4 Determinism & Reproducibility

- **`cce answer <query>` CLI** that invokes the exact same agent pipeline as
  `/agent/query` but prints a deterministic JSON blob. Enables reproducing a
  user-reported bug without spinning up the server.
- **Pinned LLM + seed.** `temperature=0` is set; there's no explicit seed.
  For OpenAI add `seed=...`.
- **Snapshot testing.** Record (query, tool-call-sequence, final-answer)
  tuples; diff on every change. Any regression in tool routing pops out
  immediately.

### 4.5 Security & Multi-tenancy

- **No auth on server endpoints.** Anyone who can reach the port can query
  arbitrary code + index.
- **No path-traversal hardening.** Tool args like `get_file_slice(path=...)`
  must reject `..` segments and absolute paths outside the indexed root.
- **No per-user isolation.** If the server ever runs multi-tenant, the
  single SQLite DB mixes all repos together. Needs tenant_id in every row.
- **Outbound call policy.** Embedder / LLM make external HTTPS calls
  unconditionally. Ship an `offline=true` mode that refuses any network.

### 4.6 Scale

- **Serial indexer** (§3.1) caps throughput at ~50 files/sec on a typical
  laptop. Production repos run 10-100k files.
- **SQLite for symbols + edges** is fine to ~500k rows; beyond that,
  switch to DuckDB or Postgres. A pluggable store layer is already half
  there — formalize it.
- **Embedded Qdrant** uses the same filesystem; multi-process server
  workers will fight for the lock. Deploy remote Qdrant for anything
  above a single worker.

---

## 5. Prioritized Roadmap

**P0 (shipped):** tool-call leak parser · strict citations ·
enumeration tools · grep_code · doctor check · system-prompt playbook.

### P1 — fixes the remaining 4 failures in the live run (≤ 1 day)

| ID | Item | Effort | Fixes |
|---|---|---|---|
| P1-1 | File-level citation fallback in `_validate_citations` | 15 min | Q8 `:?` storm |
| P1-2 | Effective route path + router-prefix resolver pass | 2 h | Q4 "/agent/query not found" |
| P1-3 | REFERENCES edges in `python_resolver` (identifier-as-arg, decorator, dict value) | 3 h | Q2 proper (replaces grep fallback) |
| P1-4 | Structured tool errors + re-plan on error | 1 h | silent tool failures |
| P1-5 | `get_file_slice(path, start, end)` tool | 30 min | tier-2 "show body of X" |
| P1-6 | Sharper playbook example for "list nodes in file" | 20 min | Q6 tool-routing fix |
| P1-7 | Pytest coverage for leak parser + citation validator | 1 h | regression guard |
| P1-8 | `find_route_handler(method, path)` deterministic tool | 30 min | Q4 backup |

**Expected delta on 9-question eval:** 70 → ~85 / 90.

### P2 — industry-level quality (1-2 weeks)

| ID | Item | Effort | Benefit |
|---|---|---|---|
| P2-1 | Decouple planner / responder LLMs in config | 1 h | cost + quality |
| P2-2 | Index manifest `.cce/index.json` + server startup check | 2 h | staleness detection |
| P2-3 | Structured JSONL trace log | 2 h | debuggability |
| P2-4 | Query rewriting / expansion step | 3 h | tier-3 recall |
| P2-5 | Cross-encoder reranker (`ms-marco-MiniLM`) | 4 h | top-k precision |
| P2-6 | Typer / Click extractor | 3 h | Q7 100% + tier-2 CLI questions |
| P2-7 | LangGraph extractor (StateGraph structural edges) | 4 h | tier-3 "explain this agent" |
| P2-8 | Golden dataset + `cce eval run` + CI gate | 1 day | regression-proof |
| P2-9 | Reasoner that scores coverage, not just count | 4 h | fewer wasted loops |
| P2-10 | Tool result summarization between loops | 3 h | token budget |
| P2-11 | Per-chunk content-hash → skip re-embed | 2 h | 10× faster incremental |
| P2-12 | `cce answer <query>` CLI for reproducibility | 1 h | bug reporting |
| P2-13 | Chunk lexical store by function / window | 3 h | BM25 precision |
| P2-14 | MMR / diversity in hybrid retriever | 2 h | context budget |
| P2-15 | Code-aware chunker (tree-sitter boundaries + method splits) | 4 h | vector recall |

### P3 — scale + ecosystem (ongoing)

| ID | Item |
|---|---|
| P3-1 | Parallel per-file indexing with `ProcessPoolExecutor` |
| P3-2 | Go / Rust / Java tree-sitter grammars wired in |
| P3-3 | Pluggable symbol/graph store (Postgres / DuckDB) |
| P3-4 | Auth + rate limit on server (API key header + token bucket) |
| P3-5 | OpenTelemetry spans end-to-end |
| P3-6 | SCIP REFERENCES export for Sourcegraph consumers |
| P3-7 | `.cceignore` support |
| P3-8 | Embedding query cache (LRU) |
| P3-9 | Per-thread_id concurrency guard in agent route |
| P3-10 | `offline=true` mode (no outbound calls) |

---

## 6. Concrete Next Steps

1. **Today (≤ 30 min total)**: P1-1, P1-6. Both are <20-line diffs; regains
   Q6 + Q8.
2. **This week**: P1-2, P1-3, P1-4, P1-5, P1-7, P1-8. Takes the live eval
   to ~85 / 90 with a regression test suite guarding every P0/P1 fix.
3. **This sprint**: P2-1, P2-2, P2-3, P2-8. Observability + eval harness +
   decoupled LLMs — the foundation for everything else.
4. **Next sprint**: P2-4, P2-5, P2-6, P2-7. Quality step-change for tier-3
   answers.
5. **Backlog**: P3 items prioritized by user demand + deployment targets.

---

## 7. How to read the next agent run

When reviewing a future `cce/agent_run_*.txt`, score each answer on:

- **Tier match:** is the question tier-1/2/3/4? Tier-1 misses are bugs;
  tier-4 misses are acceptable below ~55 %.
- **Tool routing correctness:** did the planner pick the deterministic tool
  where one exists (tier-1/2)? Use the reasoning_steps `loop N: K items`
  to check.
- **Citation validity:** every `file:line` in the answer must be backed by
  a citation-table entry (or replaced with `:?`). Any raw line number that
  isn't verified is a bug in the validator or a model that emitted text
  unsupported by context.
- **Leak artefacts:** no `to=functions.*`, no CJK garbage, no markdown-style
  tool bodies. If any appear, the leak parser failed — open an issue with
  the raw AIMessage (set `CCE_AGENT__DEBUG=true`).
- **Completeness:** for enumeration answers ("list all X"), cross-check
  against `SELECT COUNT(*) FROM symbols WHERE kind='X'`. Missing items
  point at top-k / limit issues.

Anything that fails two or more of those checks should get a ticket
referencing the section above.
