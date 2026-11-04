"""
resume_parser.py – Extract structured information from PDF and DOCX resumes.
"""
from __future__ import annotations

import io
import re
from typing import Any

# PDF – use the modern pymupdf API (fitz is deprecated in PyMuPDF ≥ 2.x)
try:
    import pymupdf as fitz  # PyMuPDF ≥ 2.0
    PYMUPDF_OK = True
except ImportError:
    try:
        import fitz  # PyMuPDF 1.x fallback
        PYMUPDF_OK = True
    except ImportError:
        PYMUPDF_OK = False

# DOCX
try:
    from docx import Document
    DOCX_OK = True
except ImportError:
    DOCX_OK = False


# ── Skill taxonomy (expandable) ──────────────────────────────────────────────
SKILL_KEYWORDS: list[str] = [
    # Programming
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "r", "scala", "sql", "bash", "shell",
    # Web
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "spring", "html", "css", "rest", "graphql",
    # Data / ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "plotly", "opencv", "hugging face",
    # Cloud / Infra
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "ci/cd", "jenkins", "git", "linux", "ibm cloud", "watsonx",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle",
    "sqlite", "cassandra", "dynamodb",
    # Data Engineering
    "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery",
    # Soft skills
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "agile", "scrum",
]


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    if not PYMUPDF_OK:
        return ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def _extract_text_from_docx(file_bytes: bytes) -> str:
    if not DOCX_OK:
        return ""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(file_bytes: bytes, file_name: str) -> str:
    """Return raw text from a PDF or DOCX file."""
    ext = file_name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return _extract_text_from_pdf(file_bytes)
    if ext in ("docx", "doc"):
        return _extract_text_from_docx(file_bytes)
    return ""


def extract_email(text: str) -> str:
    m = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    return m.group(0) if m else ""


def extract_phone(text: str) -> str:
    m = re.search(r"(\+?\d[\d\s\-().]{7,}\d)", text)
    return m.group(0).strip() if m else ""


def extract_skills(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def extract_education(text: str) -> list[str]:
    degrees = re.findall(
        r"(b\.?tech|m\.?tech|b\.?e|m\.?e|b\.?sc|m\.?sc|phd|mba|bca|mca"
        r"|bachelor|master|doctorate)[^\n]*",
        text,
        re.IGNORECASE,
    )
    return list(dict.fromkeys(d.strip() for d in degrees))


def extract_experience_years(text: str) -> float:
    """Heuristic: sum up year ranges or look for 'X years' patterns."""
    # e.g. "3 years of experience", "2+ years"
    matches = re.findall(r"(\d+\.?\d*)\+?\s*years?", text, re.IGNORECASE)
    if matches:
        return max(float(m) for m in matches)
    return 0.0


def parse_resume(file_bytes: bytes, file_name: str) -> dict[str, Any]:
    """Full resume parse → structured dict."""
    text = extract_text(file_bytes, file_name)
    return {
        "raw_text": text,
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "education": extract_education(text),
        "experience_years": extract_experience_years(text),
        "word_count": len(text.split()),
    }
