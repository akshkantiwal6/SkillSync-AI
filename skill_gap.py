"""
skill_gap.py – Skill-gap analysis using the job_resume_fit.csv dataset.

Dataset access is delegated to modules/dataset_loader.py.
Resume parsing is delegated to modules/resume_parser.py.
These two concerns are intentionally kept separate.

NOTE: The dataset is used for matching and benchmarking only.
      It was NOT used to train IBM Granite.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from modules.dataset_loader import (
    get_required_skills_from_dataset,
    find_similar_candidates,
    compute_dataset_match_score,
    get_dataset,
)

# ── Static fallback role → required skills map ────────────────────────────────
# Used when the dataset doesn't contain a matching category.
ROLE_SKILLS_MAP: dict[str, list[str]] = {
    "data scientist": [
        "python", "machine learning", "deep learning", "pandas", "numpy",
        "scikit-learn", "sql", "tensorflow", "pytorch", "statistics",
        "data visualization", "nlp",
    ],
    "software engineer": [
        "python", "java", "javascript", "rest", "sql", "git", "docker",
        "algorithms", "data structures", "testing", "agile",
    ],
    "ml engineer": [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "mlops", "docker", "kubernetes", "sql", "spark",
    ],
    "data engineer": [
        "python", "sql", "spark", "kafka", "airflow", "aws", "gcp",
        "etl", "data pipelines", "hadoop", "scala",
    ],
    "frontend developer": [
        "javascript", "react", "angular", "vue", "html", "css",
        "typescript", "rest", "git", "testing",
    ],
    "backend developer": [
        "python", "java", "node.js", "sql", "rest", "docker", "git",
        "microservices", "redis", "postgresql",
    ],
    "full stack developer": [
        "javascript", "react", "node.js", "python", "sql", "rest",
        "docker", "git", "html", "css", "typescript",
    ],
    "devops engineer": [
        "docker", "kubernetes", "jenkins", "terraform", "ansible",
        "aws", "azure", "linux", "bash", "ci/cd", "monitoring",
    ],
    "cloud engineer": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "networking", "security", "linux", "python",
    ],
    "data analyst": [
        "sql", "python", "excel", "tableau", "power bi", "statistics",
        "pandas", "data visualization", "r",
    ],
    "information technology": [
        "python", "sql", "networking", "linux", "windows", "aws",
        "docker", "security", "git", "rest",
    ],
    "finance": [
        "excel", "financial modeling", "accounting", "sql", "python",
        "tableau", "statistics", "risk analysis",
    ],
    "banking": [
        "financial analysis", "excel", "sql", "risk management",
        "accounting", "compliance", "python",
    ],
    "healthcare": [
        "medical terminology", "patient care", "ehr", "hipaa",
        "clinical research", "data analysis",
    ],
    "sales": [
        "crm", "communication", "negotiation", "salesforce", "excel",
        "market research", "presentation",
    ],
    "consultant": [
        "problem solving", "communication", "data analysis", "excel",
        "project management", "presentation", "python",
    ],
}


def _normalize_role(role: str) -> str:
    return role.lower().strip()


def get_required_skills(role: str, job_description: str = "") -> list[str]:
    """
    Return required skills for a role.

    Priority order:
    1. Dataset (job_required_skills aggregated by category)
    2. Static ROLE_SKILLS_MAP fallback
    3. Extract from job_description text
    """
    # 1. Dataset
    dataset_skills = get_required_skills_from_dataset(role)
    if dataset_skills:
        return dataset_skills

    # 2. Static map
    norm = _normalize_role(role)
    for key, skills in ROLE_SKILLS_MAP.items():
        if key in norm or norm in key:
            return skills

    # 3. Extract from job description
    if job_description:
        from modules.resume_parser import extract_skills
        return extract_skills(job_description)

    return []


def compute_match_score(resume_skills: list[str], required_skills: list[str]) -> float:
    """Return 0.0–1.0 overlap score (precision against required skills)."""
    if not required_skills:
        return 0.5
    r = set(s.lower() for s in resume_skills)
    req = set(s.lower() for s in required_skills)
    return len(r & req) / len(req)


def get_similar_candidates(
    resume_skills: list[str], role: str, top_n: int = 5
) -> pd.DataFrame:
    """
    Find similar profiles in the dataset for benchmarking.
    Delegates to dataset_loader.find_similar_candidates.
    """
    return find_similar_candidates(resume_skills, role, top_n)


def full_skill_gap_report(
    resume_skills: list[str],
    role: str,
    job_description: str = "",
) -> dict[str, Any]:
    """
    Comprehensive skill-gap report combining dataset + static analysis.

    Returns
    -------
    dict with keys:
      required_skills, matching_skills, missing_skills, extra_skills,
      fit_score, recommendations, dataset_benchmark (if available)
    """
    required = get_required_skills(role, job_description)
    resume_set = set(s.lower() for s in resume_skills)
    req_set = set(s.lower() for s in required)

    matching = sorted(resume_set & req_set)
    missing = sorted(req_set - resume_set)
    extra = sorted(resume_set - req_set)
    fit_score = compute_match_score(resume_skills, required)

    recommendations: list[str] = []
    for skill in missing[:6]:
        recommendations.append(
            f"Build **{skill}** via hands-on projects or targeted online courses."
        )

    # Dataset benchmark (non-blocking — silently empty if dataset unavailable)
    benchmark: dict[str, Any] = {}
    try:
        benchmark = compute_dataset_match_score(resume_skills, role)
    except Exception:  # noqa: BLE001
        pass

    return {
        "required_skills": required,
        "matching_skills": matching,
        "missing_skills": missing,
        "extra_skills": extra,
        "fit_score": fit_score,
        "recommendations": recommendations,
        "dataset_benchmark": benchmark,
    }
