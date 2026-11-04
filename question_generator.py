"""
question_generator.py – Orchestrate RAG + Granite to produce interview questions.
"""
from __future__ import annotations

from typing import Any

from modules.granite_client import get_client
from modules.rag_engine import get_engine

QUESTION_TYPES = ["technical", "hr", "behavioral", "project"]


def generate_all_question_sets(
    role: str,
    job_description: str,
    resume_skills: list[str],
    counts: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    """
    Generate question sets for all four categories.

    Parameters
    ----------
    role : Job role title
    job_description : Full JD text
    resume_skills : Skills extracted from the candidate's resume
    counts : How many questions per type (default 5 each)

    Returns
    -------
    Dict mapping question_type → list[question_str]
    """
    if counts is None:
        counts = {qt: 5 for qt in QUESTION_TYPES}

    client = get_client()
    engine = get_engine()
    results: dict[str, list[str]] = {}

    for q_type in QUESTION_TYPES:
        query = f"{role} {q_type} interview questions {' '.join(resume_skills[:5])}"
        rag_ctx = engine.retrieve_context(query)
        questions = client.generate_questions(
            role=role,
            job_description=job_description,
            skills=resume_skills,
            question_type=q_type,
            count=counts.get(q_type, 5),
            rag_context=rag_ctx,
        )
        results[q_type] = questions

    return results


def generate_single_type(
    role: str,
    job_description: str,
    resume_skills: list[str],
    question_type: str,
    count: int = 5,
) -> list[str]:
    client = get_client()
    engine = get_engine()
    query = f"{role} {question_type} interview questions"
    rag_ctx = engine.retrieve_context(query)
    return client.generate_questions(
        role=role,
        job_description=job_description,
        skills=resume_skills,
        question_type=question_type,
        count=count,
        rag_context=rag_ctx,
    )
