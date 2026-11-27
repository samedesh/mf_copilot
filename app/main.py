from __future__ import annotations
import hashlib
from contextlib import asynccontextmanager

import diskcache
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agent import state
from app.agent.graph import run_agent
from app.config import settings
from app.observability import init_tracing
from app.rag.retriever import get_retriever
from app.schemas import ChatRequest, ChatResponse

limiter = Limiter(key_func=get_remote_address)
cache = diskcache.Cache(str(settings.cache_dir))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing()
    has_index = (settings.index_dir / "chunks.db").exists() or (settings.index_dir / "chunks.json").exists()
    if has_index:
        try:
            get_retriever().load()
            print("Retriever index preloaded from", settings.index_dir)
        except Exception as e:
            print("Index preload failed:", e)
    else:
        print("Index missing at", settings.index_dir, "— run offline: python -m app.rag.ingest")
    yield


app = FastAPI(title="Mutual Fund Scheme Information Document (SID) Q&A Copilot", version="1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health")
def health():
    return {"status": "ok", "llm_mode": settings.llm_mode, "max_steps": settings.max_steps}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit)
def chat(request: Request, body: ChatRequest) -> ChatResponse:
    # Cache only first-turn questions (answers to follow-ups depend on session history)
    first_turn = not state.history(body.session_id, 1)
    key = hashlib.sha256(f"{settings.planner_strategy}|{body.question.strip().lower()}".encode()).hexdigest()
    if first_turn and (hit := cache.get(key)) is not None:
        state.add_turn(body.session_id, "user", body.question)
        state.add_turn(body.session_id, "assistant", hit["answer"])
        return ChatResponse(**{**hit, "cached": True, "latency_ms": 0})

    res = run_agent(body.question, body.session_id)
    resp = ChatResponse(answer=res.answer, status=res.status, citations=res.citations, steps=res.steps,
                        tool_calls=res.tool_calls, tokens=res.tokens, latency_ms=res.latency_ms, notes=res.notes,
                        trace_id=res.trace_id)
    state.add_turn(body.session_id, "user", body.question)
    state.add_turn(body.session_id, "assistant", res.answer)
    if first_turn and res.status in ("answered", "refused"):
        cache.set(key, resp.model_dump(), expire=settings.cache_ttl)
    return resp


@app.get("/sessions/{session_id}/history")
def session_history(session_id: str):
    return state.history(session_id, 50)
