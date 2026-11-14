"""OpenAI-compatible chat + embeddings client with JSON validation and retry."""
from __future__ import annotations
import hashlib
import json
import re

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

from app.budget import Budget
from app.config import settings
from app.errors import LLMError
import ssl

EMBED_DIM = 256

def extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        raise ValueError("no JSON object found")
    return text[a:b + 1]


class LLMClient:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget

    def chat(self, messages: list[dict], json_mode: bool = False, max_tokens: int = 700) -> str:
        payload = {"model": settings.llm_model, "messages": messages,
                   "temperature": 0, "max_tokens": max_tokens}
        if json_mode and settings.json_mode:
            payload["response_format"] = {"type": "json_object"}  # constrained decoding if supported
        data = _post("/chat/completions", payload)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {e}") from e
        used = (data.get("usage") or {}).get("total_tokens") or (len(json.dumps(messages)) + len(text)) // 4
        self.budget.add_tokens(used)
        return text

    def chat_json(self, messages: list[dict], model_cls: type[BaseModel], retries: int = 2):
        """Ask for JSON, validate against a pydantic model, feed the error back on failure."""
        msgs, last = list(messages), None
        for _ in range(retries + 1):
            text = self.chat(msgs, json_mode=True)
            try:
                return model_cls.model_validate_json(extract_json(text))
            except (ValidationError, ValueError) as e:
                last = e
                msgs += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That was invalid ({str(e)[:300]}). "
                                                     "Reply with ONLY valid JSON matching the schema."}]
        raise LLMError(f"invalid JSON after {retries + 1} attempts: {str(last)[:200]}")


def _post(path: str, payload: dict) -> dict:
    try:
        if path.startswith("/embeddings") and settings.llm_mode == "api":
            apiurl = settings.embed_base_url
            api_key = settings.embed_api_key
            api_cert = settings.embed_ca_cert
        elif path.startswith("/chat/completions") and settings.llm_mode == "api":
            apiurl = settings.llm_base_url
            api_key = settings.llm_api_key
            api_cert = settings.llm_ca_cert

        r = httpx.post(f"{apiurl}{path}", json=payload, timeout=settings.llm_timeout,
                       headers={"Authorization": f"Bearer {api_key}"}, verify=api_cert)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise LLMError(f"endpoint error {e.response.status_code}: {e.response.text}") from e
    except (httpx.HTTPError, ValueError) as e:
        raise LLMError(f"endpoint error: {e}") from e


def _hash_embed(text: str) -> np.ndarray:
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % EMBED_DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def _clean_text_for_embed(text: str) -> str:
    # Collapse repeated dots/dashes (e.g. table-of-contents leader dots) that inflate token count
    t = re.sub(r"[.\-_~*]{3,}", " ", text)
    # Limit to 200 words to strictly respect the server's 512 token limit
    words = t.split()
    if len(words) > 200:
        t = " ".join(words[:200])
    return t.strip() or "empty"


def embed(texts: list[str]) -> np.ndarray:
    """Embeddings endpoint in api mode; hashed bag-of-words in mock mode."""
    if settings.llm_mode == "mock":
        return np.vstack([_hash_embed(t) for t in texts])
    cleaned = [_clean_text_for_embed(t) for t in texts]
    data = _post("/embeddings", {"model": settings.embed_model, "input": cleaned})
    arr = np.array([d["embedding"] for d in data["data"]], dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
