"""HTTP REST API + MCP SSE transport for RAG system.

Запуск:
    rag-server --http                  # stdio + HTTP (REST + MCP SSE)
    rag-server --http --port 8080      # на конкретном порту

REST эндпоинты:
    GET     /health
    GET     /stats
    GET     /graph/stats
    POST    /search          {query, k, mode, alpha, metadata_filter, ...}
    POST    /documents       {text, meta}
    GET     /documents       {limit, offset, metadata_filter}
    GET     /documents/{id}  {offset, limit, relations_load_depth}
    PUT     /documents/{id}  {text?, meta?}
    DELETE  /documents/{id}
    POST    /relations       {source_id, target_id, relation, weight}
    DELETE  /relations       {source_id, target_id, relation}
    GET     /relations/{id}  {max_depth, metadata_filter}
    POST    /clear

MCP:
    GET     /mcp    (SSE)
    POST    /mcp    (JSON-RPC messages)
"""

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from pydantic import BaseModel, Field

from src._meta_filter import normalize_metadata_filter, parse_meta
from src.config import RAGConfig
from src.mcp_server import build_mcp_server, initialization_options
from src.rag import RAGSystem
from src.result_utils import enrich_with_links, format_results

logger = logging.getLogger(__name__)


# -- Pydantic models ---------------------------------------------------------


MAX_TEXT_BYTES = 10 * 1024 * 1024


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8192)
    k: int = Field(5, ge=1, le=100)
    mode: str = Field("hybrid", pattern="^(semantic|bm25|hybrid)$")
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata_filter: Optional[dict] = None
    max_chars: Optional[int] = Field(None, ge=1)
    rerank: Optional[bool] = None
    query_expansion: Optional[bool] = None
    relations_load_depth: int = Field(1, ge=0, le=5)
    relations_load_type_filter: Optional[list[str]] = None
    relations_load_meta_filter: Optional[dict] = None


class AddDocumentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_BYTES)
    meta: Any = None
    extract_graph: bool = False


class AddRelationRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=256)
    target_id: str = Field(min_length=1, max_length=256)
    relation: str = Field(min_length=1, max_length=256)
    weight: float = Field(1.0, ge=0.0, le=1e6)


class ClearResponse(BaseModel):
    status: str = "ok"


class StructuredRequest(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_TEXT_BYTES)
    extract_graph: bool = False


class UpdateDocumentRequest(BaseModel):
    text: Optional[str] = Field(None, min_length=1, max_length=MAX_TEXT_BYTES)
    meta: Any = None


class DeleteRelationRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=256)
    target_id: str = Field(min_length=1, max_length=256)
    relation: str = Field(min_length=1, max_length=256)


class FindCommunitiesRequest(BaseModel):
    resolution: float = Field(1.0, gt=0.0, le=100.0)
    k_nn: int = Field(15, ge=1, le=1000)


class SetCommunityNamesRequest(BaseModel):
    names: dict


# -- RAGSystem holder --------------------------------------------------------


class RAGHolder:
    """Хранилище для RAGSystem, инициализируемое при старте."""

    def __init__(self):
        self.instance: Optional[RAGSystem] = None

    def init(self, config: Optional[RAGConfig] = None) -> RAGSystem:
        cfg = config or RAGConfig.from_env()
        self.instance = RAGSystem(config=cfg)
        return self.instance

    def close(self):
        if self.instance:
            self.instance.close()
            self.instance = None


rag_holder = RAGHolder()


# -- lifespan ----------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag = rag_holder.init()
    _mount_mcp(app, rag)
    logger.info("RAGSystem initialised for HTTP API")
    yield
    rag_holder.close()
    logger.info("RAGSystem shut down")


app = FastAPI(
    lifespan=lifespan,
    title="graphrag",
    version="0.2.0",
    description="Production-grade RAG MCP server with hybrid search, knowledge graph, and HTTP API",
)

# -- Security ----------------------------------------------------------------

PUBLIC_PATHS = frozenset({"/health"})


def _api_token() -> str:
    """Токен из окружения (читается на каждый запрос — удобно для тестов)."""
    return os.environ.get("API_TOKEN", "")


def _presented_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "")


@app.middleware("http")
async def require_api_token(request: Request, call_next):
    """Bearer-аутентификация всех эндпоинтов, если задан API_TOKEN.

    Без токена в окружении аутентификация выключена — рассчитано на локальный
    (loopback) запуск; при биндинге на внешний интерфейс CLI требует токен.
    """
    token = _api_token()
    if token and request.url.path not in PUBLIC_PATHS and request.method != "OPTIONS":
        if not hmac.compare_digest(_presented_token(request), token):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


_cors_origins = RAGConfig.from_env().resolve_cors_origins()
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        # allow_credentials=False: API не использует куки/сессии, авторизация —
        # через заголовок, поэтому браузерные креденшелы не нужны.
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount("/ui", StaticFiles(directory=str(Path(__file__).parent / "webui"), html=True), name="webui")


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """ValueError из доменного слоя — это невалидный запрос, а не 500.

    Без этого обработчика предсказуемые ошибки валидации (несуществующий
    узел в /relations, битый JSON в /structured, нечисловой id сообщества)
    возвращались клиенту как Internal Server Error.
    """
    logger.warning("%s %s → 400: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# -- helpers -----------------------------------------------------------------


def _rag() -> RAGSystem:
    if rag_holder.instance is None:
        raise HTTPException(status_code=503, detail="RAGSystem not initialised")
    return rag_holder.instance


def _relations_query_params(
    relations_load_depth: int,
    relations_load_type_filter: Optional[str],
    relations_load_meta_filter: Optional[str],
) -> dict:
    """Query-параметры relations_load_* → аргументы RAGSystem."""
    return {
        "relations_load_depth": relations_load_depth,
        "relations_load_type_filter": (
            relations_load_type_filter.split(",") if relations_load_type_filter else None
        ),
        "relations_load_meta_filter": normalize_metadata_filter(relations_load_meta_filter),
    }


# -- REST endpoints ----------------------------------------------------------


@app.get("/")
async def root():
    return RedirectResponse(url="/ui")

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    s = _rag().stats()
    return {"total_documents": s["total_documents"], "store_path": s["store_path"], "dimension": s["dimension"]}


@app.get("/graph/stats")
def graph_stats():
    s = _rag().stats()
    return {"total_nodes": s["total_nodes"], "total_edges": s["total_edges"], "relation_types": s["relation_types"]}


@app.post("/search")
def search(req: SearchRequest):
    rag = _rag()
    mode = req.mode or "hybrid"
    mf = normalize_metadata_filter(req.metadata_filter)

    if mode == "semantic":
        results = rag.search(req.query, k=req.k, metadata_filter=mf, rerank=req.rerank)
    elif mode == "bm25":
        results = rag.bm25_search(req.query, k=req.k, metadata_filter=mf)
    else:
        results = rag.search_hybrid(req.query, k=req.k, alpha=req.alpha, metadata_filter=mf, rerank=req.rerank, query_expansion=req.query_expansion)

    docs = format_results(results, max_chars=req.max_chars)
    enrich_with_links(rag, docs, req.model_dump())
    return {"query": req.query, "k": len(docs), "results": docs}


@app.post("/documents")
def add_document(req: AddDocumentRequest):
    rag = _rag()
    existing = rag.is_duplicate(req.text)
    meta = parse_meta(req.meta)
    try:
        doc_id = rag.add_document(req.text, meta, extract_graph=req.extract_graph)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"doc_id": doc_id, "duplicate": existing is not None}


@app.get("/documents")
def list_documents(
    limit: int = Query(20, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    max_chars: Optional[int] = Query(None, ge=1),
    metadata_filter: Optional[str] = Query(None),
    relations_load_depth: int = Query(1, ge=0, le=5),
    relations_load_type_filter: Optional[str] = Query(None),
    relations_load_meta_filter: Optional[str] = Query(None),
):
    return _rag().list_documents(
        limit=limit, offset=offset, max_chars=max_chars,
        metadata_filter=normalize_metadata_filter(metadata_filter),
        **_relations_query_params(
            relations_load_depth, relations_load_type_filter, relations_load_meta_filter
        ),
    )


@app.get("/documents/{doc_id}")
def get_document(
    doc_id: str,
    offset: int = Query(0, ge=0),
    limit: Optional[int] = Query(None, ge=1),
    relations_load_depth: int = Query(1, ge=0, le=5),
    relations_load_type_filter: Optional[str] = Query(None),
    relations_load_meta_filter: Optional[str] = Query(None),
):
    doc = _rag().get_document(
        doc_id, offset=offset, limit=limit,
        **_relations_query_params(
            relations_load_depth, relations_load_type_filter, relations_load_meta_filter
        ),
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    deleted = _rag().delete_document(doc_id)
    return {"status": "ok", "doc_id": doc_id, "deleted": deleted}


@app.post("/relations")
def add_relation(req: AddRelationRequest):
    _rag().add_relation(req.source_id, req.target_id, req.relation, req.weight)
    return {"status": "ok"}


@app.get("/relations/{node_id}")
def get_related(
    node_id: str,
    max_depth: int = Query(1, ge=1, le=5),
    metadata_filter: Optional[str] = Query(None),
):
    relations = _rag().get_related(
        node_id, max_depth=max_depth, metadata_filter=normalize_metadata_filter(metadata_filter)
    )
    return {
        "relations": [
            {"source": r[0], "target": r[1], "relation": r[2], "weight": r[3], "direction": r[4]}
            for r in relations
        ]
    }


@app.post("/clear")
def clear():
    _rag().clear()
    return {"status": "ok"}


@app.post("/reindex")
def reindex(force: bool = False):
    count = _rag().reindex(force=force)
    return {"status": "ok", "reindexed": count}


@app.post("/structured")
def add_structured(req: StructuredRequest):
    result = _rag().index_structured(req.content, extract_graph=req.extract_graph)
    return result


@app.put("/documents/{doc_id}")
def update_document(doc_id: str, req: UpdateDocumentRequest):
    meta = parse_meta(req.meta)
    try:
        result = _rag().update_document(doc_id, text=req.text, meta=meta)
        return result
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@app.delete("/relations")
def delete_relation(req: DeleteRelationRequest):
    result = _rag().delete_relation(req.source_id, req.target_id, req.relation)
    return result


@app.post("/communities")
def find_communities(req: FindCommunitiesRequest):
    return _rag().find_communities(
        resolution=req.resolution,
        k_nn=req.k_nn,
    )


@app.put("/communities/names")
def set_community_names(req: SetCommunityNamesRequest):
    return _rag().set_community_names(req.names)


@app.get("/communities")
async def get_communities():
    return _rag().get_communities()


# -- MCP SSE transport -------------------------------------------------------


class _MCPSSEApp:
    """ASGI-приложение MCP over SSE.

    GET  → SSE handshake (connect_sse) + server.run
    POST → JSON-RPC message (handle_post_message)
    """

    def __init__(self, transport: SseServerTransport, mcp_server: Server):
        self._transport = transport
        self._server = mcp_server

    async def __call__(self, scope, receive, send):
        if scope["method"] == "GET":
            async with self._transport.connect_sse(scope, receive, send) as streams:
                read_stream, write_stream = streams
                await self._server.run(
                    read_stream, write_stream, initialization_options(self._server)
                )
        elif scope["method"] == "POST":
            await self._transport.handle_post_message(scope, receive, send)


def _mount_mcp(app: FastAPI, rag_instance) -> None:
    """Создать и смонтировать MCP SSE хендлер."""
    transport = SseServerTransport("/mcp")
    app.mount("/mcp", _MCPSSEApp(transport, build_mcp_server(rag_instance)))
