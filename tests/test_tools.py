import pytest
from app.errors import ToolError
from app.tools import calculator, fund_metrics


def test_pct_change():
    assert calculator.run("pct_change", {"old": 100, "new": 120})["value"] == 20.0


def test_cagr():
    assert round(calculator.run("cagr", {"start": 100, "end": 200, "years": 5})["value"], 2) == 14.87


def test_expense_cost_positive():
    assert calculator.run("expense_cost", {"amount": 500000, "expense_pct": 1, "gross_return_pct": 10, "years": 10})["value"] > 0


def test_bad_input():
    with pytest.raises(ToolError):
        calculator.run("pct_change", {"old": 0, "new": 5})
    with pytest.raises(ToolError):
        calculator.run("cagr", {"start": "abc", "end": 1, "years": 1})


def test_metric_plan_not_mixed():
    d = fund_metrics.run("Sample Large Cap Fund", "expense_ratio", "Direct")
    r = fund_metrics.run("Sample Large Cap Fund", "expense_ratio", "Regular")
    assert d["numeric"] == 0.85 and r["numeric"] == 1.75


def test_metric_latest_month():
    assert fund_metrics.run("Sample Large Cap Fund", "expense_ratio")["month"] == "2025-09"


def test_metric_missing():
    with pytest.raises(ToolError):
        fund_metrics.run("Nope Fund", "exit_load")
