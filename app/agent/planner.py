"""Planner: turns a question into a validated JSON tool plan. Three prompting strategies (Track B item II)."""
from __future__ import annotations
import json

from app.agent.mock import mock_plan
from app.config import settings
from app.llm import LLMClient
from app.schemas import Plan

TOOL_SPEC = """You plan tool calls to answer questions about mutual fund factsheets and scheme documents.
Tools:
1. search_scheme_docs {"query": str, "scheme": str|null, "top_k": int}   -> passages with doc_id/section
2. get_fund_metric {"scheme": str, "metric": "exit_load|expense_ratio|benchmark|riskometer|aum|min_sip",
                    "plan": "Direct|Regular", "option": "Growth|IDCW", "month": "YYYY-MM"|null}
3. calculator {"op": "pct_change|cagr|sip_fv|expense_cost|diff", "params": {...}}
   pct_change: old,new | cagr: start,end,years | sip_fv: monthly,annual_rate_pct,years
   expense_cost: amount,expense_pct,gross_return_pct,years | diff: a,b
Use "$N.numeric" inside params to reference the numeric result of step N. Never do arithmetic yourself.
Always pick the right plan (Direct/Regular) and option (Growth/IDCW).
If a scheme is not in get_fund_metric or if get_fund_metric fails, search for it in scheme documents using search_scheme_docs.
If the question asks for advice, recommendations or predictions, set "refuse": true with no steps."""

FEWSHOT = """Examples:
Q: What is the exit load of Sample Large Cap Fund (Direct, Growth)?
{"steps":[{"tool":"get_fund_metric","args":{"scheme":"Sample Large Cap Fund","metric":"exit_load","plan":"Direct","option":"Growth"}}],"refuse":false,"reason":""}
Q: What is the exit load of Nippon India ELSS Tax Saver Fund?
{"steps":[{"tool":"search_scheme_docs","args":{"query":"exit load","scheme":"Nippon India ELSS Tax Saver Fund","top_k":3}}],"refuse":false,"reason":""}
Q: How much does a 1.5% expense ratio cost me on 5 lakh over 10 years at 10% gross return?
{"steps":[{"tool":"calculator","args":{"op":"expense_cost","params":{"amount":500000,"expense_pct":1.5,"gross_return_pct":10,"years":10}}}],"refuse":false,"reason":""}
Q: Which fund should I buy?
{"steps":[],"refuse":true,"reason":"investment advice"}"""

STRATEGIES = {
    "structured": TOOL_SPEC + "\n" + FEWSHOT + '\nReply with ONLY JSON: {"steps":[...],"refuse":bool,"reason":str}',
    "cot": TOOL_SPEC + '\nThink step by step first. Reply with ONLY JSON: {"reasoning":str,"steps":[...],"refuse":bool,"reason":str}',
    "decomposition": TOOL_SPEC + '\nFirst split the question into atomic sub-questions, then plan one step per need. '
                     'Reply with ONLY JSON: {"sub_questions":[str],"steps":[...],"refuse":bool,"reason":str}',
}


def make_plan(question: str, history: list[dict], client: LLMClient, feedback: str | None = None,
              strategy: str | None = None) -> Plan:
    if settings.llm_mode == "mock":
        return mock_plan(question, history)
    msgs = [{"role": "system", "content": STRATEGIES[strategy or settings.planner_strategy]}]
    if history:
        msgs.append({"role": "user", "content": "Conversation so far:\n" + json.dumps(history[-4:])})
    content = f"Q: {question}"
    if feedback:
        content += f"\nYour previous plan failed review: {feedback}\nProduce a corrected plan."
    msgs.append({"role": "user", "content": content})
    return client.chat_json(msgs, Plan)
