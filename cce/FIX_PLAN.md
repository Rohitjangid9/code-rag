# CCE — Fix Plan (Step-by-Step Execution List)

> Source: `cce/IMPROVEMENTS.md` (audit) → this file is the **ordered execution
> list**. Each item is self-contained: files to touch, change to make, test to
> add, done-when criteria. Work through them top-to-bottom.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` skipped

Severity: 🔴 critical · 🟠 major · 🟡 minor

---

## PHASE 1 — Quick wins (today, ≤ 1 hour)

Unblocks Q6 and Q8 from the last eval run. No new modules; tiny diffs.

### [ ] F1 · File-level citation fallback 🔴
- **Why:** Q8 citations all collapse to `:?` because whole-file lexical hits
  carry `line_start=0, line_end=0` and the validator rejects every line in
  those files.
- **File:** `cce/agents/nodes.py` → `_validate_citations._ok`
- **Change:** when a citation entry has `line_start == 0 AND line_end == 0`,
  treat *any* line in that file as valid.
- **Test:** add `tests/test_phase9_agent.py::test_validate_citations_file_level`
  — citation `{file:"x.py", line_start:0, line_end:0}` accepts `x.py:42`.
- **Done when:** existing tests pass + new test passes + Q8 rerun keeps real
  line numbers.

### [ ] F2 · Planner playbook rule for "list nodes in file" 🔴
- **Why:** Q6 called `search_code("agent nodes")` instead of
  `list_symbols(file_path="agents/nodes.py", kind="Function")`.
- **File:** `cce/agents/llm.py` → `SYSTEM_PROMPT`
- **Change:** add explicit mapping: *"list/enumerate all X in file Y" →
  `list_symbols(file_path=Y, kind=…)`*, with a concrete example for
  LangGraph / agent nodes.
- **Test:** none (prompt change) — verify via re-run.
- **Done when:** Q6 returns ≥ 4 real `*_node` functions from `agents/nodes.py`.

---

## PHASE 2 — Core correctness (this week, ~1 day)

Closes the remaining failures in the live 9-question run. Target: 85/90.

### [ ] F3 · Structured tool errors + re-plan signal 🔴
- **Why:** `retriever_node` turns exceptions into plain `"Tool error: …"`
  strings; the planner can't distinguish success from failure.
- **File:** `cce/agents/nodes.py` → `retriever_node`, `reasoner_node`
- **Change:**
  - Attach `status="error"` + `error_class` to the `ToolMessage` via a
    payload wrapper; keep `content` human-readable.
  - In `reasoner_node`, count errors and force another loop when error-rate
    > 0 and retries left.
- **Test:** mock a tool that raises; assert the state shows `tool_errors > 0`
  and the loop continues.
- **Done when:** a failing tool no longer silently produces "I couldn't find…"
  answers.

### [ ] F4 · Router-prefix resolution → `effective_path` on Route nodes 🔴
- **Why:** Q4 says "Route not found" for `/agent/query` because the Route
  node stores `"/query"` without the `/agent` prefix from
  `include_router(agent.router, prefix="/agent")`.
- **Files:**
  - `cce/extractors/fastapi_extractor.py` — extract `include_router(...)`
    calls into a second pass producing `{router_symbol → prefix}` map.
  - `cce/indexer.py` — after framework pass, walk routes and stamp
    `meta["effective_path"]` on each Route node.
- **Test:** `tests/test_phase8_framework.py` — fixture app with
  `include_router(r, prefix="/agent")`; assert Route has
  `effective_path == "/agent/query"`.
- **Done when:** `list_routes()` returns both `path` and `effective_path`;
  Q4 resolves the correct handler.

### [ ] F5 · `find_route_handler(method, path)` deterministic tool 🟠
- **Why:** Even after F4, the planner needs one-shot tool for "what happens
  when I POST /X" instead of `list_routes` → filter in the LLM.
- **Files:**
  - `cce/retrieval/tools.py` — new function querying Route nodes by
    `method` + `effective_path` (fallback to `path`).
  - `cce/agents/tools.py` — wrap as `@tool`, register in `ALL_TOOLS`.
  - `cce/agents/llm.py` — add playbook entry.
- **Test:** index fixture → `find_route_handler("POST","/agent/query")`
  returns the `query` handler Node.
- **Done when:** Q4 returns the real handler symbol + line range.

### [ ] F6 · REFERENCES edges in Python resolver 🔴
- **Why:** `find_callers("planner_node")` returns `[]` because
  `builder.add_node("planner", planner_node)` is an *identifier argument*,
  not a call expression. Currently only grep can find it.
- **File:** `cce/parsers/python_resolver.py`
- **Change:** extend AST walk to emit `EdgeKind.REFERENCES` (confidence 0.7)
  for:
  1. Function/class identifiers passed as arguments to calls.
  2. Decorator targets (`@something` where something is a known symbol).
  3. Dict/list values that are identifiers.
  4. `getattr(module, "name")` dynamic lookups (confidence 0.4).
- **Test:** fixture file `def foo(): pass\nbar = {"x": foo}\nhandler(foo)`;
  assert two REFERENCES edges into `foo`.
- **Done when:** re-index → `find_callers("planner_node")` returns
  `agents.graph.*` without needing grep fallback.

### [ ] F7 · Include REFERENCES in `find_callers` (flag-gated) 🟠
- **File:** `cce/retrieval/tools.py` → `find_callers`
- **Change:** new arg `include_refs: bool = True`; when true, SELECT also
  where `kind IN ('CALLS','REFERENCES')`.
- **Test:** extend F6 fixture; assert `find_callers(foo)` returns 2 results.
- **Done when:** Q2 answers from edge table alone.

### [ ] F8 · `get_file_slice(path, start, end)` tool 🟠
- **Why:** Planner has no way to read a symbol's body; `get_symbol` returns
  only signature + docstring.
- **Files:**
  - `cce/retrieval/tools.py` — new function, reads file, returns
    `{path, start, end, lines[]}` with hard cap (max 200 lines, max 10 KB).
  - Path-traversal guard: reject `..`, absolute paths, paths outside repo.
  - `cce/agents/tools.py` — wrap as `@tool`.
  - `cce/agents/llm.py` — playbook rule: *after `get_symbol`, call
    `get_file_slice` if the question needs the body*.
- **Test:** `get_file_slice("cce/agents/nodes.py", 224, 280)` returns
  ~56 lines; `get_file_slice("../../etc/passwd", 1, 10)` raises.
- **Done when:** tier-3 questions citing function bodies work.

### [ ] F9 · Planner `limit` escalation for "all X" queries 🟠
- **Why:** Q7 caps at 15/18 because `list_cli_commands` default limit is
  50 but the tool response is truncated further.
- **Files:** `cce/retrieval/tools.py`, `cce/agents/tools.py`
- **Change:** expose `limit` on all `list_*` tools (default 200, cap 1000).
  Prompt rule: *if question contains "all/every/every single", pass
  `limit=500`*.
- **Test:** `list_cli_commands(limit=500)` returns ≥ 18 commands.
- **Done when:** Q7 returns 18/18.

### [ ] F10 · Pytest coverage for P0 + P1 fixes 🟠
- **Files:** new `tests/test_p0_p1_fixes.py` (or extend existing phase tests)
- **Cover:**
  - `_parse_leaked_tool_calls` — all 3 shapes + empty + malformed.
  - `_validate_citations` — valid / out-of-range / file-level / Windows
    vs Unix path / multiple citations in one sentence.
  - `list_routes` — with and without router prefix (after F4).
  - `grep_code` — regex metachars, long pattern, no matches.
  - `find_callers` — with REFERENCES included (after F7).
- **Done when:** `pytest -q` shows ≥ 12 new assertions all green.

---

## PHASE 3 — Industry-level quality (1-2 weeks)

Makes the agent robust and measurable.

### [ ] F11 · Decouple planner / responder LLMs 🔴
- **Files:** `cce/config.py`, `cce/agents/llm.py`, `cce/agents/nodes.py`
- **Change:**
  - Add `AgentSettings.planner_model`, `AgentSettings.responder_model`
    (both optional; fall back to `llm_model`).
  - `get_llm()` → `get_planner_llm()` + `get_responder_llm()`.
  - `responder_node` uses responder LLM.
- **Test:** settings override works; each node calls the right model.
- **Done when:** one env config pairs `gpt-4o-mini` (planner) with
  `gpt-4o` (responder).

### [ ] F12 · Index manifest `.cce/index.json` 🟠
- **Files:** `cce/indexer.py` (write), `cce/server/app.py` (read at startup)
- **Schema:** `{root, indexed_at, commit_sha, file_count, layers,
  schema_version, db_path, qdrant_path}`
- **Change:** write manifest at end of `IndexPipeline.run`; server logs a
  warning on startup if manifest missing or `schema_version` mismatches.
- **Test:** after `cce index .`, manifest exists with correct fields.
- **Done when:** `cce doctor` shows a new "Index manifest" row.

### [ ] F13 · Structured JSONL trace log 🟠
- **File:** new `cce/agents/trace.py`; wire into `planner_node`,
  `retriever_node`, `reasoner_node`.
- **Schema (one line per event):**
  `{ts, thread_id, turn, node, tool?, args?, hits?, elapsed_ms, error?}`
- **Output:** `logs/agent_trace.jsonl` (rotated daily, 7 days retention).
- **Test:** one agent run produces ≥ 4 events per question.
- **Done when:** failures reproducible from the JSONL alone.

### [ ] F14 · Golden-dataset eval + CI gate 🔴
- **Files:**
  - `cce/eval/datasets/core.yaml` — versioned golden set (extend existing
    `cce/eval/dataset.py`).
  - `cce/eval/harness.py` — metrics: symbol hit, file hit, line-range IoU,
    citation validity, completeness.
  - `cli.py` → `cce eval run --dataset core`.
  - `.github/workflows/eval.yml` — run on PR; fail if any tier-1 regresses.
- **Test:** `cce eval run --dataset core` emits `scorecard.json`.
- **Done when:** PR that regresses a tier-1 question is rejected by CI.

### [ ] F15 · Coverage-aware reasoner 🟠
- **File:** `cce/agents/nodes.py` → `reasoner_node`
- **Change:** score state on three axes —
  `(has_subject_symbol, has_symbol_body, has_callers)` — and set
  `should_continue=True` if any required axis is missing and
  `loop_count < max_retrieval_loops`.
- **Test:** mock state with only a symbol hit; assert reasoner requests
  a body fetch.
- **Done when:** Q2/Q4 no longer quit on loop 1 with zero hits.

### [ ] F16 · Tool-result summarization between loops 🟠
- **File:** `cce/agents/nodes.py`
- **Change:** before the planner's next LLM call, replace older
  `ToolMessage` contents with one-line summaries
  (`"<tool>: N hits, top=[...]"`); keep the latest ToolMessage verbatim.
- **Test:** 3-loop run stays under 10k tokens with 50 hits per loop.
- **Done when:** message-history word count stops growing past loop 2.

### [ ] F17 · Typer / Click / argparse extractor 🟠
- **File:** new `cce/extractors/cli_extractor.py`
- **Change:** detect `@app.command()` / `@click.command()` /
  `argparse.ArgumentParser().add_subparsers()` → emit
  `NodeKind.CLI_COMMAND` with name, help, arg list.
- **Updates:** `cce/graph/schema.py` add `NodeKind.CLI_COMMAND`;
  `cce/retrieval/tools.py::list_cli_commands` now SELECTs this kind.
- **Test:** fixture with 3 `@app.command()` → `list_cli_commands()` returns
  exactly 3.
- **Done when:** Q7 enumerates only real user-facing commands (no helper
  functions).

### [ ] F18 · LangGraph extractor 🟠
- **File:** new `cce/extractors/langgraph_extractor.py`
- **Change:** detect `StateGraph().add_node("x", fn)` and
  `.add_edge("a","b")` → emit a REFERENCES edge `builder → fn` and store
  `meta["graph_node_name"]="x"` on the `fn` symbol.
- **Test:** fixture with 4 `add_node` calls → `list_symbols(kind="Function")
  + meta.graph_node_name` returns 4 entries.
- **Done when:** Q6 has a deterministic path: grep for `add_node` in the
  repo → resolve each → enumerate.

### [ ] F19 · Query rewriting / synonym expansion 🟠
- **File:** `cce/retrieval/hybrid.py`
- **Change:** before lex + vector calls, expand query via small synonym map
  (auth ↔ authenticate/authorize/oauth/jwt/session/bearer; db ↔
  database/sql/sqlite/postgres/orm). Optional LLM-based expansion behind
  a flag.
- **Test:** query "auth" hits `OAuth2PasswordBearer` even when that token
  isn't in the original query.
- **Done when:** Q9 / conceptual questions score ≥ 8/10 on small repos.

### [ ] F20 · Cross-encoder reranker 🟠
- **File:** `cce/retrieval/hybrid.py` (new `_rerank` helper)
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU-friendly).
- **Change:** apply to top 50 candidates from RRF; re-score and keep top k.
  Gated by `CCE_RETRIEVAL__RERANK=true`.
- **Test:** toggling rerank on improves retrieval precision on eval.
- **Done when:** tier-3 questions score ≥ 80 % on golden set.

### [ ] F21 · Per-chunk content hash → skip re-embed 🟠
- **Files:** `cce/indexer.py::_index_semantic`, `cce/index/vector_store.py`
- **Change:** store `chunk_content_hash` in Qdrant payload; skip upsert
  when hash matches existing point's hash.
- **Test:** re-run `cce index .` with no file changes → 0 embedder calls.
- **Done when:** second-run embedder calls drop to ~0.

### [ ] F22 · Auto-delete stale vectors on symbol change 🔴
- **File:** `cce/indexer.py::_index_semantic`
- **Change:** before upsert, call
  `vector_store.delete_for_node_ids(old_ids_for_file)` for files in the
  change set.
- **Test:** rename a symbol → old vector not retrievable anymore.
- **Done when:** `search_code("OldName", mode="vector")` returns 0 hits
  after rename + re-index.

### [ ] F23 · `cce answer <query>` deterministic CLI 🟡
- **File:** `cce/cli.py`
- **Change:** new command that spins up the same LangGraph pipeline as
  `/agent/query` but prints a JSON trace. `--thread-id`, `--max-loops`,
  `--debug` flags.
- **Test:** `cce answer "where is planner_node?"` produces JSON with
  `answer`, `citations`, `trace`.
- **Done when:** bug reports can be reproduced from CLI alone.

### [ ] F24 · Code-aware chunker 🟡
- **File:** `cce/embeddings/chunker.py`
- **Change:** chunk at tree-sitter symbol boundaries (one chunk per
  function / method / class header + shared context); use `tiktoken` for
  token-accurate budgets.
- **Test:** a 200-line class produces N method chunks (not one blob).
- **Done when:** vector recall on method-level queries improves ≥ 10 %.

### [ ] F25 · MMR diversity in hybrid retriever 🟡
- **File:** `cce/retrieval/hybrid.py`
- **Change:** after RRF, apply MMR (λ = 0.6) over vectors of the top 2·k
  candidates; return k diverse items.
- **Test:** a query that previously returned 10 hits from 1 file now
  returns hits from ≥ 3 files.
- **Done when:** context block has at least 3 distinct files for broad
  queries.

### [ ] F26 · Chunk lexical store by function / window 🟡
- **Files:** `cce/index/lexical_store.py`, `cce/indexer.py`
- **Change:** instead of one FTS row per file, emit one row per symbol
  (with `(path, line_start, line_end)`) and per 50-line window for non-
  symbol code.
- **Test:** BM25 on a rare keyword returns the exact function, not the
  whole file.
- **Done when:** `search_code(q, mode="lexical")` returns line-accurate
  hits.

### [ ] F27 · Jedi project-scope caching 🟡
- **File:** `cce/parsers/python_resolver.py`
- **Change:** instantiate one `jedi.Project(root)` per pipeline run; reuse
  across files.
- **Test:** indexing time for a 200-file repo drops ≥ 30 %.
- **Done when:** timer on `_resolve_references` halved.

---

## PHASE 4 — Scale, security, ecosystem (ongoing)

### [ ] F28 · Parallel per-file indexing 🟠
- **File:** `cce/indexer.py`
- **Change:** `ProcessPoolExecutor` over `_index_file`; merge results
  serially into stores. Configurable worker count.
- **Done when:** 5k-file index runs ≥ 4× faster on an 8-core laptop.

### [ ] F29 · Additional language support 🟠
- **Files:** `cce/parsers/tree_sitter_parser.py`, new `go_resolver.py`,
  `java_resolver.py`, `rust_resolver.py`.
- **Grammars:** `tree-sitter-go`, `tree-sitter-java`, `tree-sitter-rust`.
- **Done when:** Go / Java / Rust repos index and answer tier-1 questions.

### [ ] F30 · Server auth + rate limit 🔴
- **File:** `cce/server/app.py`, new `cce/server/auth.py`
- **Change:**
  - API-key header middleware (`X-CCE-Key`) validated against
    `ServerSettings.api_keys`.
  - Token-bucket rate limiter per key (50 req/min default).
  - Tighten CORS: reject `["*"]` + `allow_credentials=True`.
- **Done when:** unauthenticated requests get 401; bursts get 429.

### [ ] F31 · Path-traversal hardening on file tools 🔴
- **File:** `cce/retrieval/tools.py` (`get_file_slice`, `grep_code`)
- **Change:** resolve requested path; reject if not under indexed root.
- **Done when:** `get_file_slice("../../etc/passwd", …)` raises `ValueError`.

### [ ] F32 · Per-thread_id concurrency guard 🟠
- **File:** `cce/server/routes/agent.py`
- **Change:** per-`thread_id` `asyncio.Lock`; parallel POSTs serialize.
- **Done when:** no checkpointer corruption under parallel load test.

### [ ] F33 · OpenTelemetry spans 🟠
- **Files:** all agent nodes + retriever tools + HTTP routes
- **Change:** wrap each with `tracer.start_as_current_span(...)`; export
  OTLP.
- **Done when:** a Jaeger trace shows planner → retriever → tool hops.

### [ ] F34 · Pluggable symbol/graph store (DuckDB / Postgres) 🟡
- **Files:** `cce/index/`, `cce/graph/` — extract interface, add
  `DuckDBStore`, `PostgresStore`.
- **Done when:** settings-driven backend swap works.

### [ ] F35 · `.cceignore` support 🟡
- **File:** `cce/walker.py`
- **Change:** load `.cceignore` at repo root; merge with `.gitignore`.
- **Done when:** excluded paths don't appear in `list_files()`.

### [ ] F36 · Embedding query cache (LRU) 🟡
- **File:** `cce/embeddings/embedder.py`
- **Change:** `functools.lru_cache`-like wrapper keyed on
  `(backend, model, text)`.
- **Done when:** repeated `search_code(q, mode="vector")` for same `q`
  makes 1 embedder call.

### [ ] F37 · Offline mode 🟡
- **File:** `cce/config.py` + any outbound caller
- **Change:** `Settings.offline: bool`; when true, embedder / LLM refuse
  to make network calls (raise + log).
- **Done when:** `CCE_OFFLINE=true cce query …` runs end-to-end on local
  models.

### [ ] F38 · SCIP REFERENCES export 🟡
- **File:** `cce/scip/emitter.py`
- **Change:** emit REFERENCES edges (from F6) into SCIP
  `SymbolInformation.relationships`.
- **Done when:** generated SCIP file round-trips through
  `scip validate`.

---

## Execution order (TL;DR)

```
Today:           F1, F2
This week:       F3, F4, F5, F6, F7, F8, F9, F10
Next week:       F11, F12, F13, F14
Sprint 2:        F15, F16, F17, F18
Sprint 3:        F19, F20, F21, F22
Sprint 4:        F23, F24, F25, F26, F27
Backlog:         F28 … F38 (prioritize by deployment needs)
```

Check items off as we go. After each phase, re-run
`python -m cce.cli query` against `cce/examples.txt` and record the new
scorecard in `cce/agent_run_<date>.txt`.
