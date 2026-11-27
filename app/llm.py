"""OpenAI-compatible chat + embeddings client with JSON validation, pooling, and retry."""
from __future__ import annotations
import hashlib
import json
import re
import time
from pathlib import Path

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

from app.budget import Budget
from app.config import settings
from app.errors import LLMError

EMBED_DIM = 384
_CLIENTS: dict[tuple[str | bool, float], httpx.Client] = {}
_RETRY_STATUS = {429, 502, 503, 504}


def extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        raise ValueError("no JSON object found")
    return text[a:b + 1]


def _get_client(verify: str | bool, timeout: float) -> httpx.Client:
    """Reuse persistent keep-alive HTTPX clients per CA certificate & timeout."""
    verify_arg: str | bool = verify if verify else True
    if isinstance(verify_arg, str):
        try:
            if not Path(verify_arg).is_file():
                verify_arg = False
        except (OSError, PermissionError):
            verify_arg = False
    key = (verify_arg, timeout)
    client = _CLIENTS.get(key)
    if client is None or client.is_closed:
        client = httpx.Client(
            verify=verify_arg,
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
        _CLIENTS[key] = client
    return client


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


def _post(path: str, payload: dict, retries: int = 5) -> dict:
    if path.startswith("/embeddings"):
        apiurl = settings.embed_base_url
        api_key = settings.embed_api_key
        api_cert = settings.embed_ca_cert
    else:
        apiurl = settings.llm_base_url
        api_key = settings.llm_api_key
        api_cert = settings.llm_ca_cert

    client = _get_client(api_cert, settings.llm_timeout)
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        try:
            r = client.post(
                f"{apiurl}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if r.status_code in _RETRY_STATUS and attempt < retries:
                ra = r.headers.get("Retry-After")
                try:
                    wait = min(float(ra), 15.0) if ra else min(2.0 * (attempt + 1), 12.0)
                except ValueError:
                    wait = min(2.0 * (attempt + 1), 12.0)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"endpoint error {e.response.status_code}: {e.response.text}") from e
        except (httpx.TransportError, ValueError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(min(2.0 * (attempt + 1), 10.0))
                continue
            raise LLMError(f"endpoint error: {e}") from e

    raise LLMError(f"endpoint error: {last_err}")


def _hash_embed(text: str) -> np.ndarray:
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % EMBED_DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def _clean_text_for_embed(text: str, max_words: int = 135, max_chars: int = 650) -> str:
    """Sanitize and bound text length to stay safely under the 512-token embedding limit."""
    # Collapse repeated dots/dashes/table borders (e.g. TOC leader dots) that blow up WordPiece tokens
    t = re.sub(r"[.\-_~*|=]{3,}", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0] or t[:max_chars]
    return t.strip() or "empty"


def _embed_single_safe(text: str) -> np.ndarray:
    """Fallback for a single chunk that hit HTTP 422: progressively truncate until accepted."""
    for max_chars in (400, 250, 120):
        cleaned = _clean_text_for_embed(text, max_words=max_chars // 4, max_chars=max_chars)
        try:
            data = _post("/embeddings", {"model": settings.embed_model, "input": [cleaned]})
            return np.array(data["data"][0]["embedding"], dtype=np.float32)
        except LLMError as e:
            if "422" in str(e) and max_chars > 120:
                continue
            raise
    raise LLMError("failed to embed chunk even after truncation")


def embed(texts: list[str]) -> np.ndarray:
    """Embeddings endpoint in api mode; hashed bag-of-words in mock mode."""
    if settings.llm_mode == "mock":
        return np.vstack([_hash_embed(t) for t in texts])
    cleaned = [_clean_text_for_embed(t) for t in texts]
    try:
        data = _post("/embeddings", {"model": settings.embed_model, "input": cleaned})
        arr = np.array([d["embedding"] for d in data["data"]], dtype=np.float32)
    except LLMError as e:
        if "422" not in str(e):
            raise
        # One or more chunks in the batch exceeded 512 tokens; embed individually with progressive truncation
        arr = np.vstack([_embed_single_safe(t) for t in texts])
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
