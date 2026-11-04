"""
evaluator.py – Evaluate mock interview answers using RAG + Granite.
"""
from __future__ import annotations

from typing import Any

from modules.granite_client import get_client
from modules.rag_engine import get_engine


def evaluate_answer(
    question: str,
    answer: str,
    role: str,
) -> dict[str, Any]:
    """
    Evaluate a single answer. Returns dict with:
    - score (int 0–10)
    - max_score (10)
    - strengths (list[str])
    - improvements (list[str])
    - improved_answer (str)
    - evaluation_mode ("demo" | "ibm_granite" | "demo_fallback")

    Empty answers are passed directly to the evaluator so the
    rule-based scoring (score=0) and feedback are consistently applied.
    """
    client = get_client()
    engine = get_engine()
    rag_ctx = engine.retrieve_context(f"{role} {question}")
    return client.evaluate_answer(
        question=question,
        answer=answer,
        role=role,
        rag_context=rag_ctx,
    )


def evaluate_session(
    qa_pairs: list[dict],  # [{"question": str, "answer": str}]
    role: str,
) -> list[dict[str, Any]]:
    """Evaluate a full mock interview session."""
    results = []
    for item in qa_pairs:
        evaluation = evaluate_answer(
            question=item["question"],
            answer=item["answer"],
            role=role,
        )
        results.append({
            "question": item["question"],
            "answer": item["answer"],
            **evaluation,
        })
    return results


def compute_session_stats(evaluations: list[dict]) -> dict[str, Any]:
    """Aggregate stats across a full session."""
    if not evaluations:
        return {"total": 0, "avg_score": 0, "max_score": 10, "percentage": 0}

    scores = [e.get("score", 0) for e in evaluations]
    return {
        "total": len(scores),
        "avg_score": round(sum(scores) / len(scores), 1),
        "max_score": 10,
        "percentage": round(sum(scores) / (len(scores) * 10) * 100, 1),
        "scores": scores,
    }
