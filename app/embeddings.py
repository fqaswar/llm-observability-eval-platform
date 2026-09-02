"""Local sentence-transformers embeddings — no API, no credentials.

The model is loaded once and reused. Loading per request would take seconds and
would dominate the latency traces Phase 2 exists to measure.
"""

import logging
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_lock = threading.Lock()

# BGE models were trained with this prefix on *queries* only; documents are
# embedded bare. Skipping it measurably degrades retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                logger.info("Loading embedding model %s…", settings.embedding_model)
                _model = SentenceTransformer(settings.embedding_model)
                dim = _model.get_sentence_embedding_dimension()
                if dim != settings.embedding_dim:
                    raise RuntimeError(
                        f"{settings.embedding_model} outputs {dim} dims but "
                        f"EMBEDDING_DIM is {settings.embedding_dim}. Update config "
                        f"and recreate the rag_chunks table — the vector(n) column "
                        f"width is fixed at creation."
                    )
    return _model


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed corpus chunks for storage. Normalized for cosine distance."""
    return get_model().encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=len(texts) > 64
    )


def embed_query(text: str) -> np.ndarray:
    """Embed a user question for retrieval."""
    return get_model().encode(
        QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
    )
