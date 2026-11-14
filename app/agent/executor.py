"""Executor: validates tool args, runs steps, resolves $N references, detects loops and tool failures."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.budget import Budget
from app.errors import ToolError
from app.observability import span
from app.schemas import Plan
from app.tools import calculator, fund_metrics, search_docs


class SearchArgs(BaseModel):
    query: str = Field(min_length=2)
    scheme: str | None = None
    top_k: int = Field(default=5, ge=1, le=8)


class MetricArgs(BaseModel):
    scheme: str
    metric: str
    plan: Literal["Direct", "Regular"] = "Direct"
    option: Literal["Growth", "IDCW"] = "Growth"
    month: str | None = None


class CalcArgs(BaseModel):
    op: Literal["pct_change", "cagr", "sip_fv", "expense_cost", "diff"]
    params: dict[str, float | str]


TOOLS = {"search_scheme_docs": (SearchArgs, search_docs.run),
         "get_fund_metric": (MetricArgs, fund_metrics.run),
         "calculator": (CalcArgs, calculator.run)}


@dataclass
class Observation:
    step: int
    tool: str
    args: dict
    ok: bool
    result: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {"step": self.step, "tool": self.tool, "args": self.args, "ok": self.ok,
                "result": self.result, "error": self.error}


_REF = re.compile(r"^\$(\d+)(?:\.(\w+))?$")


def _resolve(value, obs: list[Observation]):
    if isinstance(value, dict):
        return {k: _resolve(v, obs) for k, v in value.items()}
    if isinstance(value, str) and (m := _REF.match(value)):
        idx = int(m.group(1)) - 1
        if idx < 0 or idx >= len(obs) or not obs[idx].ok:
            raise ToolError(f"reference {value} points to a missing or failed step")
        res = obs[idx].result or {}
        field_name = m.group(2)
        v = res.get(field_name) if field_name else (res.get("numeric") if res.get("numeric") is not None else res.get("value"))
        if v is None:
            raise ToolError(f"reference {value} has no numeric value")
        return v
    return value


def run_plan(plan: Plan, budget: Budget) -> list[Observation]:
    obs: list[Observation] = []
    seen: set[tuple] = set()
    for i, st in enumerate(plan.steps, 1):
        key = (st.tool, json.dumps(st.args, sort_keys=True, default=str))
        if key in seen:  # loop detector: same tool + same args twice
            obs.append(Observation(i, st.tool, st.args, False, error="loop detected: repeated identical tool call"))
            break
        seen.add(key)
        budget.step(tool=True)  # raises BudgetExceeded past the 10-step cap
        with span(f"tool.{st.tool}", step=i, args=st.args) as sp:
            try:
                args = _resolve(st.args, obs)
                model, fn = TOOLS[st.tool]
                result = fn(**model(**args).model_dump())
                obs.append(Observation(i, st.tool, args, True, result=result))
            except (ValidationError, ToolError, TypeError) as e:
                msg = str(e).replace("\n", " ")[:300]
                sp.set_attribute("error", msg)
                obs.append(Observation(i, st.tool, st.args, False, error=msg))
    return obs
