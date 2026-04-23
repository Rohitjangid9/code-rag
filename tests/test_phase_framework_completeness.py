"""Phase-3 regression tests for F-M11 … F-M15.

These tests mint fresh fixtures on disk rather than extending the existing
``tests/fixtures`` trees so the Phase-3 features are exercised in isolation
from earlier phase assertions.
"""

from __future__ import annotations

from pathlib import Path

from cce.config import Settings
from cce.extractors.django_extractor import DjangoExtractor
from cce.extractors.fastapi_extractor import FastAPIExtractor
from cce.graph.schema import EdgeKind, NodeKind
from cce.indexer import IndexPipeline


# ── F-M11 — FastAPI completeness ──────────────────────────────────────────────

def test_fastapi_websocket_route_extracted(tmp_path: Path) -> None:
    src = tmp_path / "ws.py"
    src.write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n\n"
        "@app.websocket('/ws/chat')\n"
        "async def chat(ws):\n"
        "    pass\n",
        encoding="utf-8",
    )
    data = FastAPIExtractor().extract(src, "ws.py", src.read_text())
    routes = [n for n in data.nodes if n.kind == NodeKind.ROUTE]
    assert any(
        r.meta.get("methods") == ["WEBSOCKET"] and "/ws/chat" in r.meta.get("path", "")
        for r in routes
    )


def test_fastapi_add_api_route_extracted(tmp_path: Path) -> None:
    src = tmp_path / "prog.py"
    src.write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter(prefix='/admin')\n"
        "def ping():\n    return {}\n"
        "router.add_api_route('/ping', ping, methods=['POST'])\n",
        encoding="utf-8",
    )
    data = FastAPIExtractor().extract(src, "prog.py", src.read_text())
    prog = [n for n in data.nodes
            if n.kind == NodeKind.ROUTE
            and n.meta.get("registration") == "add_api_route"]
    assert len(prog) == 1
    assert prog[0].meta["path"] == "/admin/ping"
    assert prog[0].meta["methods"] == ["POST"]


def test_fastapi_mount_emits_mounts_router_edge(tmp_path: Path) -> None:
    src = tmp_path / "mnt.py"
    src.write_text(
        "from fastapi import FastAPI\n"
        "from starlette.staticfiles import StaticFiles\n"
        "app = FastAPI()\n"
        "app.mount('/static', StaticFiles(directory='s'))\n",
        encoding="utf-8",
    )
    data = FastAPIExtractor().extract(src, "mnt.py", src.read_text())
    mount_edges = [e for e in data.raw_edges if e.kind == EdgeKind.MOUNTS_ROUTER]
    assert any("StaticFiles" in e.dst_qualified_name for e in mount_edges)


# ── F-M12 — Django / DRF completeness ────────────────────────────────────────

def test_drf_router_register_emits_url_patterns(tmp_path: Path) -> None:
    src = tmp_path / "urls.py"
    src.write_text(
        "from django.urls import include, path\n"
        "from rest_framework.routers import DefaultRouter\n"
        "from app.views import BookViewSet\n\n"
        "router = DefaultRouter()\n"
        "router.register('books', BookViewSet)\n\n"
        "urlpatterns = [path('api/', include(router.urls))]\n",
        encoding="utf-8",
    )
    data = DjangoExtractor().extract(src, "urls.py", src.read_text())
    urls = [n for n in data.nodes if n.kind == NodeKind.URL_PATTERN]
    actions = {u.meta.get("action") for u in urls if u.meta.get("action")}
    assert {"list", "create", "retrieve", "update", "destroy"}.issubset(actions)
    # Effective path should include the include() prefix ("api/") + viewset ("books").
    assert any("api/books" in u.meta.get("pattern", "") for u in urls)


def test_django_action_decorator_emits_url_pattern(tmp_path: Path) -> None:
    src = tmp_path / "views.py"
    src.write_text(
        "from rest_framework import viewsets\n"
        "from rest_framework.decorators import action\n\n"
        "class BookViewSet(viewsets.ModelViewSet):\n"
        "    @action(detail=True, methods=['post'])\n"
        "    def publish(self, request, pk=None):\n"
        "        return None\n",
        encoding="utf-8",
    )
    data = DjangoExtractor().extract(src, "views.py", src.read_text())
    publishes = [n for n in data.nodes
                 if n.kind == NodeKind.URL_PATTERN
                 and n.meta.get("action") == "publish"]
    assert len(publishes) == 1
    assert publishes[0].meta["methods"] == ["POST"]
    assert publishes[0].meta["detail"] is True


# ── F-M13 — API linker (end-to-end) ──────────────────────────────────────────

def test_api_linker_creates_calls_api_edges(tmp_path: Path) -> None:
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n\n"
        "@app.get('/api/v1/users/')\n"
        "def list_users(): return []\n",
        encoding="utf-8",
    )
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "App.tsx").write_text(
        "import axios from 'axios';\n"
        "export function App() { return axios.get('/api/v1/users/'); }\n",
        encoding="utf-8",
    )

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "link.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(tmp_path, layers=["lexical", "symbols", "graph", "framework"])

    calls_api = pipeline.symbol_store._db.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind=?", (EdgeKind.CALLS_API.value,),
    ).fetchone()[0]
    assert calls_api >= 1


# ── F-M14 — JS resolver import-table ─────────────────────────────────────────

def test_js_resolver_qualifies_imports_via_import_table(tmp_path: Path) -> None:
    from cce.graph.schema import Language  # noqa: PLC0415
    from cce.parsers.base import ParsedFile  # noqa: PLC0415
    from cce.parsers.js_resolver import resolve_js_file  # noqa: PLC0415

    src = tmp_path / "app.tsx"
    src.write_text(
        "import { UserCard } from './UserCard';\n"
        "import axios from 'axios';\n\n"
        "export function App() {\n"
        "  axios.get('/other');\n"
        "  return <UserCard userId='1' />;\n"
        "}\n",
        encoding="utf-8",
    )
    parsed = ParsedFile(path=src, rel_path="app.tsx", language=Language.TSX,
                        source=src.read_text())
    edges = resolve_js_file(parsed, tmp_path)
    # The <UserCard/> JSX render should now resolve to the import target.
    renders = [e for e in edges if e.kind == EdgeKind.RENDERS]
    assert any(
        e.dst_qualified_name.endswith(".UserCard") and e.resolver_method == "import"
        for e in renders
    )
    # ``axios.get`` is a namespace-member call; the import table rewrites the
    # ``axios`` head to the package name.
    calls = [e for e in edges if e.kind == EdgeKind.CALLS]
    assert any(
        e.dst_qualified_name == "axios.get" and e.resolver_method == "import"
        for e in calls
    )


# ── F-M15 — Node ID salting ──────────────────────────────────────────────────

def test_node_id_salt_isolates_repos() -> None:
    from cce.parsers.tree_sitter_parser import (  # noqa: PLC0415
        _node_id_from_qname,
        reset_repo_salt,
        set_repo_salt,
    )

    # Baseline — no salt: same qname ⇒ same id (existing behaviour).
    baseline = _node_id_from_qname("app.main.handler")

    tok_a = set_repo_salt("/repos/alpha")
    try:
        id_a = _node_id_from_qname("app.main.handler")
    finally:
        reset_repo_salt(tok_a)

    tok_b = set_repo_salt("/repos/beta")
    try:
        id_b = _node_id_from_qname("app.main.handler")
    finally:
        reset_repo_salt(tok_b)

    assert id_a != baseline
    assert id_b != baseline
    assert id_a != id_b
    # After reset the salt should be empty again and match baseline.
    assert _node_id_from_qname("app.main.handler") == baseline
