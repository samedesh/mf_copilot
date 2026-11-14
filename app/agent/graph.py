"""Orchestrator: guardrail -> planner -> executor -> composer -> critic, with bounded replanning."""
from __future__ import annotations
import time
from dataclasses import dataclass, field

from app.agent import critic, state
from app.agent.composer import NOT_FOUND, citations_from, compose
from app.agent.executor import Observation, run_plan
from app.agent.planner import make_plan
from app.budget import Budget
from app.config import settings
from app.errors import BudgetExceeded, LLMError
from app.guardrails import REFUSAL_TEXT, check_scope, redact_pii
from app.llm import LLMClient
from app.observability import span
from app.schemas import Citation


@dataclass
class AgentResult:
    answer: str
    status: str
    citations: list[Citation] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    latency_ms: int = 0
    notes: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def run_agent(question: str, session_id: str, strategy: str | None = None, use_memory: bool = True) -> AgentResult:
    t0, budget = time.time(), Budget()
    client = LLMClient(budget)
    notes: list[str] = []

    def done(answer, status, cites=None, obs=None, unsupported=None) -> AgentResult:
        return AgentResult(answer, status, cites or [], obs or [], budget.steps, budget.tool_calls,
                           budget.tokens, int((time.time() - t0) * 1000), notes, unsupported or [])

    with span("agent.request", session_id=session_id):
        q, n_pii = redact_pii(question)
        if n_pii:
            notes.append(f"{n_pii} sensitive item(s) redacted from your question")
        refuse, why = check_scope(q)
        if refuse:
            notes.append(f"refused: {why}")
            return done(REFUSAL_TEXT, "refused")

        hist = state.history(session_id) if use_memory else []
        feedback, obs, answer, cites, verdict = None, [], NOT_FOUND, [], None
        try:
            for attempt in range(settings.max_replans + 1):
                budget.step()
                with span("planner", attempt=attempt):
                    plan = make_plan(q, hist, client, feedback, strategy)
                if plan.refuse:
                    notes.append(f"refused by planner: {plan.reason}")
                    return done(REFUSAL_TEXT, "refused")
                with span("executor", n_steps=len(plan.steps)):
                    obs = run_plan(plan, budget)
                with span("composer"):
                    answer = compose(q, obs, client)
                    cites = citations_from(obs)
                with span("critic") as sp:
                    verdict = critic.review(q, obs, answer, cites, attempt < settings.max_replans)
                    sp.set_attribute("decision", verdict.decision)
                if verdict.decision == "approve":
                    return done(answer, "answered", cites, obs)
                if verdict.decision == "retry":
                    feedback = verdict.feedback
                    notes.append(f"replan {attempt + 1}: {feedback}")
                    continue
                break
        except BudgetExceeded as e:
            notes.append(f"budget: {e}")
            return done("I couldn't finish within the allowed limits. Please rephrase or narrow the question.",
                        "needs_verification", cites, obs)
        except LLMError as e:
            notes.append(f"llm_error: {e}")
            return done("The language model endpoint failed. Please try again.", "error", cites, obs)

        why = verdict.feedback if verdict else "unknown"
        notes.append(f"flagged: {why}")
        return done(f"{answer}\n\nNeeds analyst verification: {why}.", "needs_verification", cites, obs,
                    verdict.unsupported if verdict else [])
