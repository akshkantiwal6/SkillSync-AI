"""
dataset_loader.py – Loads, validates, and caches the job_resume_fit.csv dataset.

This module is the single point of contact for the dataset.
It is kept completely separate from the resume-upload feature:
  - Resume parsing  → modules/resume_parser.py (live user upload)
  - Dataset access  → modules/dataset_loader.py (static reference corpus)

Dataset columns (10 expected):
  ID                       – int, row identifier
  resume_text              – str, full resume text
  job_text                 – str, full job description text
  category                 – str, job role / occupation category
  job_required_skills      – str, Python-list representation of required skills
  resume_skill_list        – str, Python-list representation of resume skills
  ai_matched_skills        – str, Python-list of pre-computed matched skills
  ai_match_score           – float, AI-based match score (0–100)
  skill_string_match_score – float, string-similarity match score (0–100)
  fuzzy_match_score        – float, fuzzy-match score (0–100)

NOTE: This dataset was used for resume/job matching and skill-gap analysis.
      It was NOT used to train IBM Granite — it is a reference corpus only.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from typing import Any

import pandas as pd

import config

# ── Expected schema ───────────────────────────────────────────────────────────
EXPECTED_COLUMNS: list[str] = [
    "ID",
    "resume_text",
    "job_text",
    "category",
    "job_required_skills",
    "resume_skill_list",
    "ai_matched_skills",
    "ai_match_score",
    "skill_string_match_score",
    "fuzzy_match_score",
]

FLOAT_COLUMNS: list[str] = [
    "ai_match_score",
    "skill_string_match_score",
    "fuzzy_match_score",
]

TEXT_COLUMNS: list[str] = [
    "resume_text",
    "job_text",
    "category",
    "job_required_skills",
    "resume_skill_list",
    "ai_matched_skills",
]


# ── Status dataclass ──────────────────────────────────────────────────────────
class DatasetStatus:
    """Carries load result metadata for display in the UI."""

    def __init__(
        self,
        loaded: bool,
        row_count: int = 0,
        columns: list[str] | None = None,
        missing_columns: list[str] | None = None,
        error: str = "",
        path: str = "",
    ) -> None:
        self.loaded = loaded
        self.row_count = row_count
        self.columns = columns or []
        self.missing_columns = missing_columns or []
        self.error = error
        self.path = path

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DatasetStatus(loaded={self.loaded}, rows={self.row_count}, "
            f"cols={len(self.columns)}, missing={self.missing_columns})"
        )


# ── Skill-list parser ─────────────────────────────────────────────────────────
def parse_skill_string(value: Any) -> list[str]:
    """
    Convert the dataset's skill string to a Python list.

    Handles:
      - Python list literals:  "['skill a', 'skill b']"
      - Comma-separated text:  "skill a, skill b"
      - Semicolon-separated:   "skill a; skill b"
      - Already a list (passthrough)
    """
    if isinstance(value, list):
        return [str(s).strip().lower() for s in value if str(s).strip()]
    if not isinstance(value, str) or not value.strip() or value.strip() in ("nan", "[]", ""):
        return []
    val = value.strip()
    # Try Python literal eval first (safe: only parses literals)
    if val.startswith("["):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(s).strip().lower() for s in parsed if str(s).strip()]
        except (ValueError, SyntaxError):
            pass
    # Fallback: split on comma or semicolon
    sep = ";" if ";" in val and "," not in val else ","
    parts = [p.strip().strip("'\"[] ").lower() for p in val.split(sep)]
    return [p for p in parts if len(p) > 1]


# ── Loader & validator ────────────────────────────────────────────────────────
def _validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate columns, fill missing values, and normalise types.
    Returns (cleaned_df, list_of_missing_columns).
    """
    missing: list[str] = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    # Fill missing text columns with empty string
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Fill missing numeric columns with 0.0
    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Normalise category: uppercase, strip
    if "category" in df.columns:
        df["category"] = df["category"].str.strip().str.upper()

    # Ensure ID is integer where possible
    if "ID" in df.columns:
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce").fillna(0).astype(int)

    return df, missing


def load_dataset(path: str | None = None) -> tuple[pd.DataFrame | None, DatasetStatus]:
    """
    Load and validate the dataset from *path* (defaults to config.DATASET_PATH).

    Returns
    -------
    (DataFrame, DatasetStatus)
      DataFrame is None when loading fails.
    """
    target_path = path or config.DATASET_PATH

    import os
    if not os.path.exists(target_path):
        return None, DatasetStatus(
            loaded=False,
            error=f"File not found: {target_path}",
            path=target_path,
        )

    try:
        df = pd.read_csv(target_path, encoding="utf-8", encoding_errors="replace")
    except Exception as exc:  # noqa: BLE001
        return None, DatasetStatus(
            loaded=False,
            error=f"Read error: {exc}",
            path=target_path,
        )

    df, missing_cols = _validate_and_clean(df)

    status = DatasetStatus(
        loaded=True,
        row_count=len(df),
        columns=list(df.columns),
        missing_columns=missing_cols,
        path=target_path,
    )
    return df, status


# ── Cached singleton ──────────────────────────────────────────────────────────
_cache: tuple[pd.DataFrame | None, DatasetStatus] | None = None


def get_dataset() -> tuple[pd.DataFrame | None, DatasetStatus]:
    """Return (df, status). Loads once and caches in process memory."""
    global _cache
    if _cache is None:
        _cache = load_dataset()
    return _cache


def reset_cache() -> None:
    """Force a reload on the next get_dataset() call."""
    global _cache
    _cache = None


# ── Domain helpers (used by skill_gap.py) ────────────────────────────────────
def get_categories() -> list[str]:
    """Return sorted list of unique job categories in the dataset."""
    df, status = get_dataset()
    if df is None or "category" not in df.columns:
        return []
    return sorted(df["category"].dropna().unique().tolist())


def get_required_skills_from_dataset(role: str) -> list[str]:
    """
    Return aggregated required skills for a role from the dataset.
    Matches on the `category` column using a cascade:
      1. Exact hyphenated form  (DATA-SCIENTIST)
      2. Plain upper            (DATA SCIENTIST)
      3. Any-word substring     (SCIENTIST -> *SCIENTIST*)
    """
    df, status = get_dataset()
    if df is None:
        return []

    role_upper = role.strip().upper()
    role_hyphen = role_upper.replace(" ", "-")
    words = [w for w in re.split(r"[\s\-_]+", role_upper) if len(w) > 2]

    # Cascade matching
    subset = df[df["category"] == role_hyphen]
    if subset.empty:
        subset = df[df["category"] == role_upper]
    if subset.empty and words:
        for word in words:
            m = df["category"].str.contains(re.escape(word), na=False)
            if m.sum() > 0:
                subset = df[m]
                break

    if subset.empty:
        return []

    # Aggregate required skills from matching rows
    all_skills: dict[str, int] = {}
    for val in subset["job_required_skills"].head(50):
        for skill in parse_skill_string(val):
            all_skills[skill] = all_skills.get(skill, 0) + 1

    # Return skills ordered by frequency, top 20
    return [s for s, _ in sorted(all_skills.items(), key=lambda x: -x[1])][:20]


def find_similar_candidates(
    resume_skills: list[str],
    role: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Find the top-N most similar dataset entries to a candidate's skill set.

    Similarity = Jaccard index between resume skills and resume_skill_list.
    Optionally pre-filtered by role/category.

    Returns a display-ready DataFrame with columns:
    [category, resume_skill_list, ai_match_score, skill_string_match_score, fuzzy_match_score, similarity]
    """
    df, status = get_dataset()
    if df is None or df.empty:
        return pd.DataFrame()

    resume_set = set(s.strip().lower() for s in resume_skills if s.strip())
    if not resume_set:
        return pd.DataFrame()

    norm = role.strip().upper().replace(" ", "-")
    mask_exact = df["category"] == norm
    mask_sub = df["category"].str.contains(
        re.escape(role.strip().split()[0].upper()), na=False
    )
    subset = df[mask_exact].copy() if mask_exact.sum() > 0 else df[mask_sub].copy()
    if subset.empty:
        subset = df.copy()  # fall back to full dataset

    def _jaccard(val: Any) -> float:
        candidate = set(parse_skill_string(val))
        if not candidate:
            return 0.0
        return len(resume_set & candidate) / len(resume_set | candidate)

    subset["similarity"] = subset["resume_skill_list"].apply(_jaccard)
    top = subset.nlargest(top_n, "similarity").reset_index(drop=True)

    # Select display columns only
    display_cols = [
        c for c in ["category", "resume_skill_list", "ai_match_score",
                    "skill_string_match_score", "fuzzy_match_score", "similarity"]
        if c in top.columns
    ]
    return top[display_cols]


def compute_dataset_match_score(
    resume_skills: list[str],
    role: str,
) -> dict[str, Any]:
    """
    Use the dataset's pre-computed scores to benchmark a candidate.

    Returns percentile rank and mean scores for similar profiles.
    All values are plain Python floats/ints for safe JSON serialisation.
    """
    df, status = get_dataset()
    if df is None or df.empty:
        return {}

    role_upper = role.strip().upper()
    role_hyphen = role_upper.replace(" ", "-")
    words = [w for w in re.split(r"[\s\-_]+", role_upper) if len(w) > 2]

    # Same cascade as get_required_skills_from_dataset
    subset = df[df["category"] == role_hyphen]
    if subset.empty:
        subset = df[df["category"] == role_upper]
    if subset.empty and words:
        for word in words:
            m = df["category"].str.contains(re.escape(word), na=False)
            if m.sum() > 0:
                subset = df[m]
                break
    if subset.empty:
        subset = df  # global fallback

    resume_set = set(s.strip().lower() for s in resume_skills)

    # Precision: fraction of each job's required skills the candidate has (0-100)
    def _score(val: Any) -> float:
        req = set(parse_skill_string(val))
        if not req:
            return 0.0
        return len(resume_set & req) / len(req) * 100

    candidate_score = float(subset["job_required_skills"].apply(_score).mean())

    return {
        "candidate_score": round(candidate_score, 1),
        "dataset_mean_ai_score": round(float(subset["ai_match_score"].mean()), 1),
        "dataset_mean_fuzzy_score": round(float(subset["fuzzy_match_score"].mean()), 1),
        "role_row_count": int(len(subset)),
        "percentile": round(
            float((subset["ai_match_score"] < candidate_score).sum()) / max(len(subset), 1) * 100,
            1,
        ),
    }
