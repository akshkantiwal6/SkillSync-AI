# SkillSync AI – AI-Powered Interview Trainer Agent

> **AICTE × IBM Edunet Internship · Problem Statement No. 22**

SkillSync AI is a full-featured, AI-powered mock interview trainer built with **Streamlit** and **IBM Granite** (via IBM watsonx.ai Runtime). It analyses your resume, generates personalised interview questions, evaluates your answers in real-time, identifies skill gaps, and shows a performance dashboard.

---

## Features

| Feature | Description |
|---|---|
| 📄 Resume Parsing | Upload PDF/DOCX — extracts skills, education, experience |
| 🤖 IBM Granite AI | Generates personalised questions & evaluates answers |
| 🔍 RAG Pipeline | FAISS-backed retrieval augments every Granite prompt |
| 🎤 Mock Interview | Interactive Q&A with live feedback after each answer |
| 📊 Dashboard | Plotly charts: score per question, radar, gauge, distribution |
| 🔎 Skill Gap | Matches your skills vs. role requirements; study plan |
| 📂 Dataset | Uses `job_resume_fit.csv` (2 385 records) for benchmarking |
| 🛡️ Demo Mode | Full UI works without IBM credentials (canned responses) |

---

## Quick Start

### 1. Clone / open the project

```bash
cd "AICTE-SkillSync-AI"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure IBM credentials (optional)

```bash
cp .env.example .env
# Edit .env and fill in WATSONX_API_KEY and WATSONX_PROJECT_ID
```

> **Demo Mode:** If you skip this step the app runs with pre-built responses — all UI flows still work.

### 5. Add the dataset

Place `job_resume_fit.csv` in the `data/` folder:

```
data/
└── job_resume_fit.csv   ← drop it here
```

### 6. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Project Structure

```
AICTE-SkillSync-AI/
├── app.py                      # Main Streamlit UI (5 pages)
├── config.py                   # Env-var configuration
├── requirements.txt
├── .env.example                # Credential template
├── README.md
├── data/
│   └── job_resume_fit.csv      # Dataset (place here — not modified)
├── knowledge_base/
│   ├── __init__.py
│   └── interview_kb.py         # Curated interview Q&A for RAG
└── modules/
    ├── __init__.py
    ├── resume_parser.py        # PDF/DOCX → structured dict
    ├── granite_client.py       # IBM watsonx.ai / demo fallback
    ├── rag_engine.py           # FAISS semantic retrieval
    ├── question_generator.py   # Orchestrates RAG + Granite
    ├── evaluator.py            # Answer scoring + feedback
    ├── skill_gap.py            # Skill-gap report + dataset match
    └── dashboard.py            # Plotly chart builders
```

---

## Pages

| Page | Description |
|---|---|
| 🏠 Setup | Upload resume, enter job role/description, choose question counts |
| 📋 Questions | Review generated questions by category (Technical / HR / Behavioral / Project) |
| 🎤 Mock Interview | Answer questions one-by-one; get immediate AI feedback |
| 📊 Dashboard | Visual performance summary: charts, scores, export |
| 🔍 Skill Gap | AI skill-gap analysis, recommendations, dataset benchmarking |

---

## IBM Granite Integration

The app uses `ibm-watsonx-ai` SDK:

```python
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
client = APIClient(credentials)
model = ModelInference(model_id="ibm/granite-13b-instruct-v2", ...)
```

Credentials are **only** read from environment variables — never hardcoded.

---

## RAG Architecture

```
User query
    ↓
FAISS index (sentence-transformers embeddings of interview_kb.py)
    ↓
Top-K relevant passages retrieved
    ↓
Appended to Granite prompt as context
    ↓
IBM Granite generates personalised response
```

Falls back to keyword search if `sentence-transformers` / `faiss-cpu` are unavailable.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `WATSONX_API_KEY` | For live AI | IBM Cloud API key |
| `WATSONX_PROJECT_ID` | For live AI | watsonx.ai project ID |
| `WATSONX_URL` | Optional | Defaults to `us-south.ml.cloud.ibm.com` |
| `GRANITE_MODEL_ID` | Optional | Defaults to `ibm/granite-13b-instruct-v2` |
| `EMBEDDING_MODEL` | Optional | Defaults to `all-MiniLM-L6-v2` |
| `DATASET_PATH` | Optional | Defaults to `data/job_resume_fit.csv` |

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **IBM Granite** — via `ibm-watsonx-ai`
- **FAISS** + **sentence-transformers** — RAG retrieval
- **PyMuPDF** — PDF parsing
- **python-docx** — DOCX parsing
- **Pandas + scikit-learn** — Dataset analysis
- **Plotly** — Interactive charts

---

## License

MIT — see `LICENSE` for details.
