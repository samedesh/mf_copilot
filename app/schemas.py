from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=1000)


class Citation(BaseModel):
    doc_id: str
    section: str | None = None
    snippet: str = ""


class ChatResponse(BaseModel):
    answer: str
    status: Literal["answered", "refused", "needs_verification", "error"]
    citations: list[Citation] = []
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    latency_ms: int = 0
    cached: bool = False
    notes: list[str] = []


class PlanStep(BaseModel):
    tool: Literal["search_scheme_docs", "get_fund_metric", "calculator"]
    args: dict = {}


class Plan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list, max_length=5)
    refuse: bool = False
    reason: str = ""
    reasoning: str = ""            # used by the "cot" strategy
    sub_questions: list[str] = []  # used by the "decomposition" strategy
