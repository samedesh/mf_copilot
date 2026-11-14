"""Deterministic arithmetic. The LLM never does maths."""
from __future__ import annotations
from app.errors import ToolError


def _f(params: dict, *names: str) -> list[float]:
    try:
        return [float(str(params[n]).replace(",", "")) for n in names]
    except KeyError as e:
        raise ToolError(f"missing parameter {e}; needs {list(names)}") from e
    except (TypeError, ValueError) as e:
        raise ToolError(f"non-numeric parameter in {names}: {e}") from e


def run(op: str, params: dict) -> dict:
    if op == "pct_change":
        old, new = _f(params, "old", "new")
        if old == 0:
            raise ToolError("old value is zero")
        return _out(op, (new - old) / old * 100, f"({new} - {old}) / {old} * 100", "percent")
    if op == "cagr":
        start, end, years = _f(params, "start", "end", "years")
        if start <= 0 or years <= 0:
            raise ToolError("start and years must be positive")
        return _out(op, ((end / start) ** (1 / years) - 1) * 100, f"(({end}/{start})^(1/{years}) - 1) * 100", "percent")
    if op == "sip_fv":
        p, rate, years = _f(params, "monthly", "annual_rate_pct", "years")
        r, n = rate / 12 / 100, int(years * 12)
        fv = p * n if r == 0 else p * (((1 + r) ** n - 1) / r) * (1 + r)
        return _out(op, fv, f"{p} * (((1+{r:.6f})^{n} - 1) / {r:.6f}) * (1+{r:.6f}) [payments at start of month]", "INR")
    if op == "expense_cost":
        amt, exp, gross, years = _f(params, "amount", "expense_pct", "gross_return_pct", "years")
        fv_g = amt * (1 + gross / 100) ** years
        fv_n = amt * (1 + (gross - exp) / 100) ** years
        return _out(op, fv_g - fv_n,
                    f"{amt}*(1+{gross}%)^{years} - {amt}*(1+({gross}-{exp})%)^{years}", "INR")
    if op == "diff":
        a, b = _f(params, "a", "b")
        return _out(op, a - b, f"{a} - {b}", "")
    raise ToolError(f"unknown op '{op}'")


def _out(op: str, value: float, formula: str, unit: str) -> dict:
    return {"op": op, "value": round(value, 4), "numeric": round(value, 4), "formula": formula, "unit": unit}
