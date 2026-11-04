"""
rag_engine.py – Retrieval-Augmented Generation engine using FAISS.

Embeds the knowledge base with sentence-transformers and retrieves
top-k relevant passages for Granite to use as context.
Falls back to keyword search when sentence-transformers is unavailable.
"""
from __future__ import annotations

import re
from typing import Any

import config
from knowledge_base.interview_kb import ALL_DOCUMENTS, ALL_METADATA

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss  # type: ignore

    FAISS_OK = True
except ImportError:
    FAISS_OK = False


class RAGEngine:
    """FAISS-backed semantic retrieval over the interview knowledge base."""

    def __init__(self) -> None:
        self._docs = ALL_DOCUMENTS
        self._meta = ALL_METADATA
        self._index: Any = None
        self._model: Any = None
        self._use_faiss = FAISS_OK

        if self._use_faiss:
            self._build_index()

    def _build_index(self) -> None:
        try:
            self._model = SentenceTransformer(config.EMBEDDING_MODEL)
            embeddings = self._model.encode(
                self._docs, convert_to_numpy=True, normalize_embeddings=True
            )
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)  # inner-product = cosine on normalized
            self._index.add(embeddings.astype("float32"))
        except Exception:  # noqa: BLE001
            self._use_faiss = False

    def retrieve(self, query: str, top_k: int = config.TOP_K_DOCS) -> list[dict]:
        """Return top-k relevant knowledge-base entries for a query."""
        if self._use_faiss and self._index is not None:
            return self._semantic_search(query, top_k)
        return self._keyword_search(query, top_k)

    def retrieve_context(self, query: str, top_k: int = config.TOP_K_DOCS) -> str:
        """Return retrieved passages as a single formatted string."""
        results = self.retrieve(query, top_k)
        if not results:
            return ""
        return "\n\n---\n\n".join(r["content"] for r in results)

    # ── Semantic search ──────────────────────────────────────────────────────
    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        import numpy as np  # local import for safety

        q_emb = self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self._index.search(q_emb, min(top_k, len(self._docs)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "content": self._docs[idx],
                "metadata": self._meta[idx],
                "score": float(score),
            })
        return results

    # ── Keyword fallback ─────────────────────────────────────────────────────
    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        tokens = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._docs):
            doc_tokens = set(re.findall(r"\w+", doc.lower()))
            overlap = len(tokens & doc_tokens)
            if overlap:
                scored.append((overlap / len(tokens | doc_tokens), i))
        scored.sort(reverse=True)
        return [
            {"content": self._docs[i], "metadata": self._meta[i], "score": s}
            for s, i in scored[:top_k]
        ]


# Singleton
_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
