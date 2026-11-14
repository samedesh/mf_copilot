from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(ROOT / ".env")


def _e(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    llm_mode: str = _e("LLM_MODE", "mock")
    llm_base_url: str = _e("LLM_BASE_URL", "http://localhost:8001/v1/").rstrip("/")
    llm_api_key: str = _e("LLM_API_KEY", "changeme")
    llm_model: str = _e("LLM_MODEL", "llama-3.1-8b-instruct")
    llm_ca_cert: str = _e("LLM_CA_CERT", "")
    embed_base_url: str = _e("EMBED_BASE_URL", "http://localhost:8001/v1/").rstrip("/")
    embed_api_key: str = _e("EMBED_API_KEY", "changeme")
    embed_model: str = _e("EMBED_MODEL", "bge-small-en")
    embed_ca_cert: str = _e("EMBED_CA_CERT", "")
    llm_timeout: float = float(_e("LLM_TIMEOUT", "60"))
    json_mode: bool = _e("JSON_MODE", "true").lower() in ("1", "true", "yes")
    planner_strategy: str = _e("PLANNER_STRATEGY", "structured")
    reranker: str = _e("RERANKER", "none")

    # Track B guardrails
    max_steps: int = int(_e("MAX_STEPS", "10"))
    max_replans: int = int(_e("MAX_REPLANS", "2"))
    token_budget: int = int(_e("TOKEN_BUDGET", "8000"))
    rate_limit: str = _e("RATE_LIMIT", "20/minute")
    cache_ttl: int = int(_e("CACHE_TTL", "3600"))

    # RAG
    top_k: int = int(_e("TOP_K", "5"))
    chunk_size: int = int(_e("CHUNK_SIZE", "400"))      # words
    chunk_overlap: int = int(_e("CHUNK_OVERLAP", "50"))

    data_dir: Path = Path(_e("DATA_DIR", str(ROOT / "data")))
    log_dir: Path = Path(_e("LOG_DIR", str(ROOT / "logs")))

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def metrics_csv(self) -> Path:
        return self.data_dir / "fund_metrics.csv"

    @property
    def session_db(self) -> Path:
        return self.data_dir / "sessions.db"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


settings = Settings()
