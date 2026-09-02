"""Central configuration, read from .env.

Phase 5 stages a deliberate eval regression by changing values here (top_k,
system_prompt) — so nothing in the retrieval or generation path may hardcode
them.
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ at import time. pydantic-settings reads the file
# itself, but the Langfuse SDK reads os.environ directly — without this its
# credentials are invisible and tracing silently disables itself.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROUNDED_SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's \
question using ONLY the provided context.

Rules:
- If the context does not contain the answer, say "The documentation provided \
does not cover this." Do not use outside knowledge to fill the gap.
- Do not speculate or infer beyond what the context states.
- Quote specific details (names, numbers, flags) from the context where relevant.
- Be concise: two to five sentences unless the question demands more."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Credentials (names must match what the SDKs auto-read) ---
    anthropic_api_key: str = ""
    database_url: str = ""

    # --- Retrieval ---
    # Phase 5 regression: drop to 1 and eval scores should fall.
    rag_top_k: int = 5
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768  # must match embedding_model's output width

    # --- Generation ---
    rag_model: str = "claude-haiku-4-5"
    rag_max_tokens: int = 1024
    system_prompt: str = GROUNDED_SYSTEM_PROMPT

    # --- Ingestion ---
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
