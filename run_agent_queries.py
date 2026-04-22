"""Automation: read queries from cce/examples.txt, POST each to /agent/query, log results.

Usage:
    # 1. Start the server in another terminal:
    cce serve

    # 2. Run this script:
    python run_agent_queries.py

Output: a timestamped log file `cce/agent_run_YYYYMMDD_HHMMSS.txt`.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

SERVER_URL = "http://127.0.0.1:8765/agent/query"
EXAMPLES_FILE = Path("cce/examples.txt")
LOG_DIR = Path("cce")
TIMEOUT_S = 180.0

_QUERY_RE = re.compile(r'^Question:\s*(.+?)$', re.MULTILINE)


def extract_queries(text: str) -> list[str]:
    """Pull every `Question: ...` line from the file (order preserved, deduped)."""
    found = [m.group(1).strip() for m in _QUERY_RE.finditer(text) if m.group(1).strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for q in found:
        if q in seen:
            continue
        seen.add(q)
        unique.append(q)
    return unique


def ask_agent(query: str, thread_id: str) -> dict:
    body = json.dumps({"query": query, "thread_id": thread_id}).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def format_block(idx: int, total: int, thread_id: str, query: str, result: dict, dt: float) -> str:
    answer = result.get("answer", "")
    citations = result.get("citations", []) or []
    steps = result.get("reasoning_steps", []) or []

    lines = [
        "=" * 78,
        f"[{idx}/{total}]  thread_id={thread_id}  elapsed={dt:.2f}s",
        "-" * 78,
        f"Q: {query}",
        "",
        "A:",
        str(answer).rstrip() or "(empty)",
        "",
        f"Citations ({len(citations)}):",
    ]
    if citations:
        for c in citations:
            sym = c.get("symbol", "")
            file = c.get("file", "")
            line = c.get("line", "")
            lines.append(f"  - {sym}  @  {file}:{line}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"Reasoning steps ({len(steps)}):")
    if steps:
        for s in steps:
            lines.append(f"  - {s}")
    else:
        lines.append("  (none)")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not EXAMPLES_FILE.exists():
        print(f"error: {EXAMPLES_FILE} not found", file=sys.stderr)
        return 2

    text = EXAMPLES_FILE.read_text(encoding="utf-8")
    queries = extract_queries(text)
    if not queries:
        print(f"error: no `Question: ...` entries found in {EXAMPLES_FILE}", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"agent_run_{ts}.txt"

    header = (
        f"# Agent query run @ {ts}\n"
        f"# Endpoint: {SERVER_URL}\n"
        f"# Source:   {EXAMPLES_FILE}\n"
        f"# Queries:  {len(queries)}\n\n"
    )
    print(header, end="")

    with log_path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.flush()
        for i, q in enumerate(queries, 1):
            thread_id = f"auto-{ts}-{i:02d}"
            print(f"[{i}/{len(queries)}] {q}")
            t0 = time.monotonic()
            try:
                result = ask_agent(q, thread_id)
                dt = time.monotonic() - t0
                block = format_block(i, len(queries), thread_id, q, result, dt)
            except urllib.error.URLError as e:
                dt = time.monotonic() - t0
                block = (
                    f"{'=' * 78}\n[{i}/{len(queries)}]  thread_id={thread_id}  "
                    f"elapsed={dt:.2f}s\n{'-' * 78}\nQ: {q}\n\nERROR (URLError): {e}\n"
                    "Is the server running? Start it with `cce serve` in another terminal.\n\n"
                )
            except Exception as e:  # noqa: BLE001
                dt = time.monotonic() - t0
                block = (
                    f"{'=' * 78}\n[{i}/{len(queries)}]  thread_id={thread_id}  "
                    f"elapsed={dt:.2f}s\n{'-' * 78}\nQ: {q}\n\n"
                    f"ERROR ({type(e).__name__}): {e}\n\n"
                )
            f.write(block)
            f.flush()

    print(f"\nDone. Log saved → {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
