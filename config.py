"""
config.py – Central configuration loader for SkillSync AI.
All secrets come from environment variables; never hardcoded.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── IBM watsonx.ai ──────────────────────────────────────────────────────────
WATSONX_API_KEY: str = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL: str = os.getenv(
    "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
)
GRANITE_MODEL_ID: str = os.getenv(
    "GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2"
)

# ── Demo mode ────────────────────────────────────────────────────────────────
# Automatically enabled when IBM credentials are absent.
DEMO_MODE: bool = not bool(WATSONX_API_KEY and WATSONX_PROJECT_ID)

# ── Generation parameters ────────────────────────────────────────────────────
GRANITE_PARAMS: dict = {
    "decoding_method": "greedy",
    "max_new_tokens": 1024,
    "min_new_tokens": 10,
    "stop_sequences": [],
    "repetition_penalty": 1.1,
    "temperature": 0.7,
}

# ── RAG / FAISS ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
TOP_K_DOCS: int = 5

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_PATH: str = os.getenv(
    "DATASET_PATH",
    str(Path(__file__).resolve().parent / "data" / "job_resume_fit.csv")
)

# ── App ──────────────────────────────────────────────────────────────────────
APP_TITLE: str = "SkillSync AI – Interview Trainer"
APP_ICON: str = "🎯"
