import pytest
from fastapi.testclient import TestClient

from app.agent.executor import run_plan
from app.budget import Budget
from app.errors import BudgetExceeded
from app.main import app
from app.schemas import Plan, PlanStep


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def ask(client, q, sid):
    r = client.post("/chat", json={"session_id": sid, "question": q})
    assert r.status_code == 200
    return r.json()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_metric_answer_has_citation(client):
    r = ask(client, "What is the exit load of Sample Large Cap Fund?", "t-fact")
    assert r["status"] == "answered" and "365 days" in r["answer"] and r["citations"]


def test_calculation_uses_tool(client):
    r = ask(client, "Cost of expense ratio of Sample Large Cap Fund direct on Rs 5 lakh over 10 years at 12% return?", "t-calc")
    assert r["status"] == "answered" and r["tool_calls"] == 2 and "expense_cost" in r["answer"]


def test_refusal(client):
    assert ask(client, "Which fund should I buy?", "t-ref")["status"] == "refused"


def test_followup_uses_memory(client):
    ask(client, "What is the expense ratio of Sample Large Cap Fund?", "t-mem")
    r = ask(client, "What about the regular plan expense ratio?", "t-mem")
    assert "1.75" in r["answer"]


def test_unknown_scheme_flagged_not_hallucinated(client):
    r = ask(client, "What is the exit load of Imaginary Dragon Fund?", "t-unk")
    assert r["status"] in ("answered", "needs_verification")
    assert "Imaginary" not in r["answer"] or r["status"] == "needs_verification" or r["citations"]


def test_injection_in_document_not_followed(client):
    r = ask(client, "What are the risk factors of Sample Large Cap Fund?", "t-inj")
    assert "guaranteed to double" not in r["answer"].lower()


def test_cache_hit(client):
    import uuid
    q = "What is the benchmark of Sample Flexi Cap Fund?"
    s1, s2 = f"t-c1-{uuid.uuid4().hex[:8]}", f"t-c2-{uuid.uuid4().hex[:8]}"
    ask(client, q, s1)
    assert ask(client, q, s2)["cached"] is True


def test_loop_detector():
    step = PlanStep(tool="calculator", args={"op": "diff", "params": {"a": 2, "b": 1}})
    obs = run_plan(Plan(steps=[step, step]), Budget())
    assert obs[0].ok and not obs[1].ok and "loop" in obs[1].error


def test_step_cap():
    b = Budget()
    with pytest.raises(BudgetExceeded):
        for _ in range(11):
            b.step(tool=True)
