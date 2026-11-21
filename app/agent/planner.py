"""Planner: turns a question into a validated JSON tool plan. Three prompting strategies (Track B item II)."""
from __future__ import annotations
import json

from app.agent.mock import mock_plan
from app.config import settings
from app.llm import LLMClient
from app.schemas import Plan

TOOL_SPEC = """You plan tool calls to answer questions about mutual fund Scheme Information Documents (SIDs).
Tools:
1. search_scheme_docs {"query": str, "scheme": str|null, "top_k": int}   -> PRIMARY tool for ALL real AMC Scheme Information Documents (Axis, Bandhan, Bank of India, Baroda BNP Paribas, Canara Robeco, DSP, DWS, Nippon India, etc.)
2. get_fund_metric {"scheme": str, "metric": "exit_load|expense_ratio|benchmark|riskometer|aum|min_sip",
                    "plan": "Direct|Regular", "option": "Growth|IDCW", "month": "YYYY-MM"|null}   -> ONLY use for "Sample Large Cap Fund" or "Sample Flexi Cap Fund". NEVER call get_fund_metric for any other fund!
3. calculator {"op": "pct_change|cagr|sip_fv|expense_cost|diff", "params": {...}}
   pct_change: old,new | cagr: start,end,years | sip_fv: monthly,annual_rate_pct,years
   expense_cost: amount,expense_pct,gross_return_pct,years | diff: a,b
Use "$N.numeric" inside params to reference the numeric result of step N. Never do arithmetic yourself.
Always pick the right plan (Direct/Regular) and option (Growth/IDCW).
For questions comparing TWO funds, emit TWO separate tool steps — one per fund.
Default to search_scheme_docs (with top_k=5) for real AMC Scheme Information Documents (SIDs). If get_fund_metric fails, fall back to search_scheme_docs.
If the question asks for advice, recommendations or predictions, set "refuse": true with no steps."""

FEWSHOT = """Examples:
Q: What is the exit load of Sample Large Cap Fund (Direct, Growth)?
{"steps":[{"tool":"get_fund_metric","args":{"scheme":"Sample Large Cap Fund","metric":"exit_load","plan":"Direct","option":"Growth"}}],"refuse":false,"reason":""}
Q: What is the exit load of Nippon India ELSS Tax Saver Fund?
{"steps":[{"tool":"search_scheme_docs","args":{"query":"exit load load structure","scheme":"Nippon India ELSS Tax Saver Fund","top_k":5}}],"refuse":false,"reason":""}
Q: Compare the exit loads of Fund A and Fund B.
{"steps":[{"tool":"search_scheme_docs","args":{"query":"exit load load structure","scheme":"Fund A","top_k":5}},{"tool":"search_scheme_docs","args":{"query":"exit load load structure","scheme":"Fund B","top_k":5}}],"refuse":false,"reason":""}
Q: What is the difference between Sample Large Cap Fund Regular and Direct expense ratio?
{"steps":[{"tool":"get_fund_metric","args":{"scheme":"Sample Large Cap Fund","metric":"expense_ratio","plan":"Regular","option":"Growth"}},{"tool":"get_fund_metric","args":{"scheme":"Sample Large Cap Fund","metric":"expense_ratio","plan":"Direct","option":"Growth"}},{"tool":"calculator","args":{"op":"diff","params":{"a":"$1.numeric","b":"$2.numeric"}}}],"refuse":false,"reason":""}
Q: What is the future value of a monthly SIP of Rs 10,000 for 10 years at 12% annual return?
{"steps":[{"tool":"calculator","args":{"op":"sip_fv","params":{"monthly":10000,"annual_rate_pct":12,"years":10}}}],"refuse":false,"reason":""}
Q: How much does a 1.5% expense ratio cost me on 5 lakh over 10 years at 10% gross return?
{"steps":[{"tool":"calculator","args":{"op":"expense_cost","params":{"amount":500000,"expense_pct":1.5,"gross_return_pct":10,"years":10}}}],"refuse":false,"reason":""}
Q: Which fund should I buy?
{"steps":[],"refuse":true,"reason":"investment advice"}"""

STRATEGIES = {
    "structured": TOOL_SPEC + "\n" + FEWSHOT + '\nReply with ONLY JSON: {"steps":[...],"refuse":bool,"reason":str}',
    "cot": TOOL_SPEC + '\nThink step by step in the "reasoning" field, then put valid tool objects {"tool":..., "args":{...}} in "steps". '
           'Reply with ONLY JSON: {"reasoning":str,"steps":[{"tool":str,"args":dict}],"refuse":bool,"reason":str}',
    "decomposition": TOOL_SPEC + '\nFirst split the question into atomic sub-questions in "sub_questions", then plan one valid tool step {"tool":..., "args":{...}} per need. '
                     'Reply with ONLY JSON: {"sub_questions":[str],"steps":[{"tool":str,"args":dict}],"refuse":bool,"reason":str}',
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
