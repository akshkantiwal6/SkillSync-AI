"""
app.py – SkillSync AI: AI-Powered Interview Trainer Agent
Main Streamlit application.

Run: streamlit run app.py
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import streamlit as st

import config
from modules.resume_parser import parse_resume
from modules.question_generator import generate_all_question_sets, generate_single_type
from modules.evaluator import evaluate_answer, evaluate_session, compute_session_stats
from modules.skill_gap import full_skill_gap_report, get_similar_candidates
from modules.granite_client import get_client
from modules.dataset_loader import get_dataset, get_categories
from modules import dashboard

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════
   ROOT TOKENS
═══════════════════════════════════════════════════ */
:root {
    --bg:         #0B1020;
    --surface:    #151D32;
    --surface2:   #1E2744;
    --border:     #2A3558;
    --accent:     #6366F1;
    --accent2:    #22D3EE;
    --text:       #F8FAFC;
    --text-muted: #CBD5E1;
    --success:    #22C55E;
    --warning:    #F59E0B;
    --danger:     #EF4444;
    --radius:     12px;
    --radius-sm:  8px;
}

/* ═══════════════════════════════════════════════════
   GLOBAL BACKGROUND & TEXT
═══════════════════════════════════════════════════ */
.stApp,
.main .block-container {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.main .block-container {
    padding-top: 1.5rem !important;
    max-width: 1400px;
}

/* All markdown text */
.stMarkdown, .stMarkdown p, .stMarkdown li,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li { color: var(--text) !important; }

div[data-testid="stMarkdownContainer"] strong { color: var(--text) !important; }
div[data-testid="stMarkdownContainer"] code {
    background: var(--surface2) !important;
    color: var(--accent2) !important;
    padding: 1px 5px; border-radius: 4px;
}

/* ═══════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1528 0%, #111A35 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-muted) !important; }
section[data-testid="stSidebar"] strong { color: var(--text) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--text) !important; }

/* Sidebar radio — nav items */
section[data-testid="stSidebar"] .stRadio label {
    color: var(--text-muted) !important;
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    display: block;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,102,241,0.15) !important;
    color: var(--text) !important;
}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:has(input:checked) + div label,
section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* Status boxes inside sidebar */
section[data-testid="stSidebar"] .stAlert {
    border-radius: var(--radius-sm) !important;
    margin: 4px 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stNotificationContentSuccess"] * { color: #052e0e !important; }
section[data-testid="stSidebar"] [data-testid="stNotificationContentError"]   * { color: #450a0a !important; }
section[data-testid="stSidebar"] [data-testid="stNotificationContentWarning"] * { color: #431407 !important; }

/* Divider in sidebar */
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 0.5rem 0 !important;
}

/* ═══════════════════════════════════════════════════
   INPUTS, TEXTAREAS, SELECTS
═══════════════════════════════════════════════════ */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox select,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background-color: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder { color: #64748B !important; }
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
}

/* Labels for inputs */
.stTextInput label,
.stTextArea label,
.stNumberInput label,
.stFileUploader label,
.stSelectbox label { color: var(--text-muted) !important; font-size: 0.88rem !important; }

/* Number input buttons */
div[data-baseweb="input"] button { background: var(--surface2) !important; color: var(--text) !important; }

/* ═══════════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════════ */
.stButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(99,102,241,0.1) !important;
}
/* Primary button */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--accent) 0%, #818CF8 100%) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #10B981) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
}

/* ═══════════════════════════════════════════════════
   METRIC CARDS (st.metric)
═══════════════════════════════════════════════════ */
div[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.82rem !important; }
div[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.6rem !important; font-weight: 700 !important; }
div[data-testid="stMetricDelta"] { color: var(--success) !important; }

/* ═══════════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════════ */
div[data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    gap: 0 !important;
    padding: 4px !important;
    border-bottom: 1px solid var(--border) !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 18px !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"]:hover { color: var(--text) !important; background: var(--surface2) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--accent) !important;
    color: #fff !important;
    font-weight: 600 !important;
}
div[data-baseweb="tab-panel"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    padding: 1rem !important;
}

/* ═══════════════════════════════════════════════════
   EXPANDER
═══════════════════════════════════════════════════ */
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
div[data-testid="stExpander"] summary {
    color: var(--text) !important;
    background: var(--surface) !important;
    border-radius: var(--radius-sm) !important;
}
div[data-testid="stExpander"] summary:hover { background: var(--surface2) !important; }
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
    background: var(--surface2) !important;
    border-top: 1px solid var(--border) !important;
}

/* ═══════════════════════════════════════════════════
   ALERTS / INFO / WARNING / SUCCESS / ERROR
═══════════════════════════════════════════════════ */
div[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
}
/* Override info alert for dark bg */
div[data-testid="stNotificationContentInfo"]   { background: rgba(34,211,238,0.08) !important; border-left: 3px solid var(--accent2) !important; }
div[data-testid="stNotificationContentSuccess"]{ background: rgba(34,197,94,0.08) !important;  border-left: 3px solid var(--success) !important; }
div[data-testid="stNotificationContentWarning"]{ background: rgba(245,158,11,0.10) !important; border-left: 3px solid var(--warning) !important; }
div[data-testid="stNotificationContentError"]  { background: rgba(239,68,68,0.08) !important;  border-left: 3px solid var(--danger) !important; }

div[data-testid="stNotificationContentInfo"] *    { color: var(--accent2) !important; }
div[data-testid="stNotificationContentSuccess"] * { color: var(--success) !important; }
div[data-testid="stNotificationContentWarning"] * { color: var(--warning) !important; }
div[data-testid="stNotificationContentError"] *   { color: var(--danger) !important; }

/* ═══════════════════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════════════════ */
div[data-testid="stProgress"] > div > div {
    background: var(--surface2) !important;
    border-radius: 99px !important;
}
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%) !important;
    border-radius: 99px !important;
}

/* ═══════════════════════════════════════════════════
   FILE UPLOADER
═══════════════════════════════════════════════════ */
section[data-testid="stFileUploadDropzone"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s !important;
}
section[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent) !important;
    background: rgba(99,102,241,0.05) !important;
}
section[data-testid="stFileUploadDropzone"] * { color: var(--text-muted) !important; }
section[data-testid="stFileUploadDropzone"] button { color: var(--accent) !important; }

/* ═══════════════════════════════════════════════════
   DATAFRAME
═══════════════════════════════════════════════════ */
div[data-testid="stDataFrame"] iframe,
div[data-testid="stDataFrameGlideDataEditor"] {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}

/* ═══════════════════════════════════════════════════
   DIVIDER
═══════════════════════════════════════════════════ */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

/* ═══════════════════════════════════════════════════
   SPINNER
═══════════════════════════════════════════════════ */
div[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ═══════════════════════════════════════════════════
   HEADINGS
═══════════════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }

/* ═══════════════════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ═══════════════════════════════════════════════════
   CUSTOM COMPONENTS (used via st.markdown unsafe)
═══════════════════════════════════════════════════ */

/* ── App header ─────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #151D32 0%, #1E2744 60%, #1a1f4e 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: "";
    position: absolute; top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.app-header h1 {
    margin: 0; font-size: 1.9rem; font-weight: 700;
    color: var(--text) !important;
    background: linear-gradient(90deg, #fff 60%, var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-header p { margin: 0.35rem 0 0 0; color: var(--text-muted) !important; font-size: 0.95rem; }
.app-header .badge {
    display: inline-block;
    margin-top: 0.5rem;
    padding: 3px 10px;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 99px;
    font-size: 0.75rem;
    color: #a5b4fc !important;
}

/* ── Demo mode banner ───────────────────── */
.demo-banner {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.35);
    border-left: 3px solid var(--warning);
    border-radius: var(--radius-sm);
    padding: 0.7rem 1rem;
    margin-bottom: 1.2rem;
    font-size: 0.88rem;
    color: #FDE68A !important;
}
.demo-banner strong { color: var(--warning) !important; }
.demo-banner code {
    background: rgba(245,158,11,0.15);
    color: var(--warning) !important;
    padding: 1px 5px; border-radius: 4px;
}

/* ── Section heading ────────────────────── */
.section-heading {
    font-size: 1.05rem; font-weight: 700;
    color: var(--text) !important;
    margin: 0 0 1rem 0;
    display: flex; align-items: center; gap: 8px;
}
.section-heading::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Card ────────────────────────────────── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}
.card-accent-left {
    border-left: 3px solid var(--accent);
}

/* ── Question card ───────────────────────── */
.q-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-sm);
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.q-card:hover {
    border-left-color: var(--accent2);
    box-shadow: 0 2px 12px rgba(99,102,241,0.15);
}
.q-number {
    font-size: 0.72rem; font-weight: 700;
    color: var(--accent) !important;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.q-text {
    font-size: 0.97rem; color: var(--text) !important;
    line-height: 1.6;
}

/* ── Interview question card ─────────────── */
.interview-q-card {
    background: linear-gradient(135deg, #1E2744 0%, #151D32 100%);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(99,102,241,0.12);
}
.interview-q-type {
    font-size: 0.72rem; font-weight: 700;
    color: var(--accent2) !important;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.interview-q-text {
    font-size: 1.1rem; font-weight: 600;
    color: var(--text) !important;
    line-height: 1.55;
}
.interview-q-counter {
    float: right;
    font-size: 0.75rem;
    color: var(--text-muted) !important;
    background: var(--surface2);
    padding: 2px 8px;
    border-radius: 99px;
    border: 1px solid var(--border);
}

/* ── Score badge ─────────────────────────── */
.score-badge {
    display: inline-flex; align-items: center;
    padding: 3px 12px;
    border-radius: 99px;
    font-weight: 700; font-size: 0.88rem;
}
.score-high  { background: rgba(34,197,94,0.15);  color: #4ADE80 !important; border: 1px solid rgba(34,197,94,0.3); }
.score-mid   { background: rgba(245,158,11,0.15); color: #FCD34D !important; border: 1px solid rgba(245,158,11,0.3); }
.score-low   { background: rgba(239,68,68,0.15);  color: #FCA5A5 !important; border: 1px solid rgba(239,68,68,0.3); }

/* ── Feedback cards ──────────────────────── */
.feedback-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem 1.2rem;
    margin-top: 0.6rem;
}
.feedback-card h4 { margin: 0 0 0.5rem 0; font-size: 0.88rem; color: var(--text-muted) !important; text-transform: uppercase; letter-spacing: 0.06em; }
.feedback-card ul { margin: 0; padding-left: 1.2rem; }
.feedback-card li { color: var(--text) !important; font-size: 0.93rem; line-height: 1.6; }

/* ── Skill badge ─────────────────────────── */
.skill-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.78rem; font-weight: 500;
    margin: 2px 3px;
    white-space: nowrap;
}
.skill-match   { background: rgba(34,197,94,0.12);  color: #4ADE80 !important; border: 1px solid rgba(34,197,94,0.25); }
.skill-missing { background: rgba(239,68,68,0.12);   color: #FCA5A5 !important; border: 1px solid rgba(239,68,68,0.25); }
.skill-neutral { background: rgba(99,102,241,0.12);  color: #A5B4FC !important; border: 1px solid rgba(99,102,241,0.25); }

/* ── Recommendation card ─────────────────── */
.rec-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.7rem;
    color: var(--text) !important;
    font-size: 0.93rem; line-height: 1.6;
}
.rec-card strong { color: var(--accent2) !important; }

/* ── Study plan item ─────────────────────── */
.study-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.8rem 1.1rem;
    margin-bottom: 0.5rem;
    display: flex; align-items: flex-start; gap: 12px;
}
.study-week {
    min-width: 60px;
    background: var(--accent);
    color: #fff !important;
    border-radius: 6px;
    text-align: center;
    padding: 2px 0;
    font-size: 0.75rem; font-weight: 700;
    flex-shrink: 0;
}
.study-text { color: var(--text) !important; font-size: 0.9rem; line-height: 1.5; }
.study-text strong { color: var(--accent2) !important; }

/* ── Benchmark metric card ───────────────── */
.bm-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.bm-card .bm-value { font-size: 1.7rem; font-weight: 700; color: var(--accent2) !important; }
.bm-card .bm-label { font-size: 0.78rem; color: var(--text-muted) !important; margin-top: 2px; }

/* ── Interview complete card ─────────────── */
.complete-card {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(99,102,241,0.1));
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: var(--radius);
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.complete-card h2 { color: var(--success) !important; font-size: 1.6rem; margin-bottom: 0.4rem; }
.complete-card p  { color: var(--text-muted) !important; font-size: 0.95rem; }

/* ── Page title ──────────────────────────── */
.page-title {
    font-size: 1.3rem; font-weight: 700;
    color: var(--text) !important;
    margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 10px;
}
.page-title-accent { color: var(--accent) !important; }

/* ── Sidebar logo area ───────────────────── */
.sidebar-logo {
    padding: 0.5rem 0 1rem 0;
    text-align: center;
}
.sidebar-logo h2 {
    font-size: 1.25rem; font-weight: 800;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0.3rem 0 0 0;
}
.sidebar-logo p { font-size: 0.75rem; color: var(--text-muted) !important; margin: 0; }

/* ── Session pill ───────────────────────── */
.session-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.55rem 0.9rem;
    font-size: 0.82rem;
    color: var(--text-muted) !important;
    margin-bottom: 0.4rem;
    display: flex; align-items: center; gap: 6px;
}
.session-pill span { color: var(--text) !important; font-weight: 600; }

/* ── Misc ────────────────────────────────── */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* prevent horizontal overflow on badges */
.stMarkdown div { overflow-wrap: break-word; word-break: break-word; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict[str, Any] = {
        "resume_data": None,
        "job_role": "",
        "job_description": "",
        "questions": {},
        "current_q_type": "technical",
        "current_q_idx": 0,
        "answers": {},
        "evaluations": {},
        "session_stats": {},
        "skill_gap": None,
        "interview_active": False,
        "interview_done": False,
        "page": "Setup",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Helper: score badge HTML ──────────────────────────────────────────────────
def _score_badge(score: int) -> str:
    cls = "score-high" if score >= 8 else ("score-mid" if score >= 5 else "score-low")
    return f'<span class="score-badge {cls}">{score} / 10</span>'

# ── Helper: skill badge HTML ──────────────────────────────────────────────────
def _skill_badges(skills: list[str], kind: str = "neutral") -> str:
    return " ".join(
        f'<span class="skill-badge skill-{kind}">{s}</span>' for s in skills
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2rem;">🎯</div>
        <h2>SkillSync AI</h2>
        <p>Interview Trainer Agent</p>
    </div>
    """, unsafe_allow_html=True)

    # ── AI status ────────────────────────────────────────────────────────────
    client = get_client()
    if client.is_demo:
        st.warning("**Demo Mode** — IBM credentials not detected. AI responses are pre-built examples.", icon="⚠️")
    else:
        st.success("**IBM Granite** connected.", icon="🤖")

    # ── Dataset status ────────────────────────────────────────────────────────
    _ds_df, _ds_status = get_dataset()
    if _ds_status.loaded:
        st.success(f"**Dataset** — {_ds_status.row_count:,} records · {len(_ds_status.columns)} cols", icon="📂")
    else:
        st.error("**Dataset not found** — place `job_resume_fit.csv` in `data/`", icon="📂")

    st.divider()

    # ── Navigation ────────────────────────────────────────────────────────────
    pages = ["🏠 Setup", "📋 Questions", "🎤 Mock Interview", "📊 Dashboard", "🔍 Skill Gap"]
    # Sync radio to programmatic page changes (from navigation buttons)
    _page_labels = {p.split(" ", 1)[1]: p for p in pages}
    _desired = st.session_state.get("_nav_target")
    if _desired and _desired in _page_labels:
        st.session_state["nav_radio"] = _page_labels[_desired]
        st.session_state.pop("_nav_target", None)
    selected = st.radio("Navigation", pages, key="nav_radio", label_visibility="collapsed")
    page = selected.split(" ", 1)[1]
    st.session_state.page = page

    st.divider()

    # ── Session summary ───────────────────────────────────────────────────────
    st.markdown('<div style="font-size:0.78rem;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Session</div>', unsafe_allow_html=True)

    if st.session_state.resume_data:
        rd = st.session_state.resume_data
        st.markdown(f'<div class="session-pill">📄 Resume <span>{rd["word_count"]} words · {len(rd["skills"])} skills</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="session-pill">📄 Resume <span>Not uploaded</span></div>', unsafe_allow_html=True)

    if st.session_state.job_role:
        st.markdown(f'<div class="session-pill">💼 Role <span>{st.session_state.job_role[:22]}</span></div>', unsafe_allow_html=True)

    total_q = sum(len(v) for v in st.session_state.questions.values())
    total_ans = sum(len(v) for v in st.session_state.answers.values() if isinstance(v, dict))
    if total_q:
        st.markdown(f'<div class="session-pill">🎤 Progress <span>{total_ans} / {total_q} answered</span></div>', unsafe_allow_html=True)

    st.divider()
    if st.button("↺  Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# APP HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <h1>🎯 SkillSync AI</h1>
    <p>AI-Powered Interview Trainer Agent &nbsp;·&nbsp; AICTE × IBM Edunet Internship</p>
    <span class="badge">Problem Statement No. 22</span>
</div>
""", unsafe_allow_html=True)

if config.DEMO_MODE:
    st.markdown(
        '<div class="demo-banner">⚡ <strong>Demo Mode Active</strong> — '
        'IBM Granite responses are simulated. Set <code>WATSONX_API_KEY</code> and '
        '<code>WATSONX_PROJECT_ID</code> in <code>.env</code> to enable live AI.</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETUP
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Setup":
    st.markdown('<div class="page-title">🏠 <span class="page-title-accent">Setup</span> — Resume & Job Details</div>', unsafe_allow_html=True)

    col_resume, col_job = st.columns([1, 1], gap="large")

    # ── Resume ────────────────────────────────────────────────────────────────
    with col_resume:
        st.markdown('<div class="section-heading">📄 Upload Resume</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "PDF or DOCX — parsed locally, nothing leaves your machine",
            type=["pdf", "docx", "doc"],
            label_visibility="visible",
        )
        if uploaded:
            with st.spinner("Parsing resume…"):
                resume_data = parse_resume(uploaded.read(), uploaded.name)
            st.session_state.resume_data = resume_data
            st.success(f"Resume parsed — **{len(resume_data['skills'])} skills** extracted.")

        if st.session_state.resume_data:
            rd = st.session_state.resume_data
            with st.expander("📋 Extracted Profile", expanded=True):
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Skills", len(rd["skills"]))
                mc2.metric("Experience", f"{rd['experience_years']} yrs")
                mc3.metric("Words", rd["word_count"])
                if rd["email"]:
                    st.markdown(f'<div style="color:var(--text-muted);font-size:0.85rem;margin:6px 0;">📧 {rd["email"]}</div>', unsafe_allow_html=True)
                if rd["education"]:
                    st.markdown(f'<div style="color:var(--text-muted);font-size:0.85rem;margin:4px 0;">🎓 {" · ".join(rd["education"][:2])}</div>', unsafe_allow_html=True)
                if rd["skills"]:
                    st.markdown('<div style="margin-top:10px;font-size:0.82rem;color:#64748B;margin-bottom:4px;">SKILLS DETECTED</div>', unsafe_allow_html=True)
                    st.markdown(_skill_badges(rd["skills"][:24], "neutral"), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">📂</div>
                <div style="color:var(--text-muted);font-size:0.92rem;">Upload your resume above, or use the demo profile below.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("▶  Use Demo Resume", use_container_width=True):
                st.session_state.resume_data = {
                    "raw_text": "Demo resume text.",
                    "email": "candidate@example.com",
                    "phone": "+1-555-0100",
                    "skills": ["python", "machine learning", "pandas", "sql",
                               "scikit-learn", "docker", "git"],
                    "education": ["B.Tech Computer Science"],
                    "experience_years": 2.0,
                    "word_count": 320,
                }
                st.rerun()

    # ── Job details ───────────────────────────────────────────────────────────
    with col_job:
        st.markdown('<div class="section-heading">💼 Job Details</div>', unsafe_allow_html=True)
        job_role = st.text_input(
            "Job Role / Position",
            value=st.session_state.job_role or "Data Scientist",
            placeholder="e.g. Data Scientist, Software Engineer, DevOps Engineer",
        )
        job_description = st.text_area(
            "Job Description",
            value=st.session_state.job_description or "",
            height=220,
            placeholder=(
                "Paste the full job description here.\n\n"
                "IBM Granite + RAG will tailor every question to this role and description."
            ),
        )
        st.session_state.job_role = job_role
        st.session_state.job_description = job_description

        st.markdown('<div class="section-heading" style="margin-top:1rem;">⚙️ Question Settings</div>', unsafe_allow_html=True)
        q_counts: dict[str, int] = {}
        g1, g2, g3, g4 = st.columns(4)
        q_counts["technical"]  = g1.number_input("⚙️ Technical",  1, 10, 5, key="qc_tech")
        q_counts["hr"]         = g2.number_input("🤝 HR",         1, 10, 5, key="qc_hr")
        q_counts["behavioral"] = g3.number_input("🧠 Behavioral", 1, 10, 5, key="qc_beh")
        q_counts["project"]    = g4.number_input("🗂️ Project",    1, 10, 5, key="qc_proj")

    st.divider()
    if st.button("🚀  Generate Personalised Interview Questions", type="primary", use_container_width=True):
        if not st.session_state.job_role.strip():
            st.error("Please enter a job role before generating questions.")
        else:
            skills = st.session_state.resume_data["skills"] if st.session_state.resume_data else []
            with st.spinner("Generating questions with IBM Granite + RAG knowledge base…"):
                questions = generate_all_question_sets(
                    role=st.session_state.job_role,
                    job_description=st.session_state.job_description,
                    resume_skills=skills,
                    counts=q_counts,
                )
            st.session_state.questions = questions
            st.session_state.answers = {}
            st.session_state.evaluations = {}
            st.session_state.session_stats = {}
            st.session_state.interview_active = False
            st.session_state.interview_done = False
            total = sum(len(v) for v in questions.values())
            st.success(f"**{total} questions** generated across 4 categories. Navigate to **📋 Questions** to review.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Questions":
    st.markdown('<div class="page-title">📋 <span class="page-title-accent">Interview Questions</span></div>', unsafe_allow_html=True)

    if not st.session_state.questions:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:2.5rem;margin-bottom:0.6rem;">⚠️</div>
            <div style="color:var(--text);font-size:1rem;font-weight:600;">No questions generated yet</div>
            <div style="color:var(--text-muted);font-size:0.9rem;margin-top:0.3rem;">Go to <strong>🏠 Setup</strong> and click Generate.</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    TYPE_META = {
        "technical":  {"label": "⚙️ Technical",    "color": "#6366F1"},
        "hr":         {"label": "🤝 HR",            "color": "#22D3EE"},
        "behavioral": {"label": "🧠 Behavioral",    "color": "#F59E0B"},
        "project":    {"label": "🗂️ Project-Based", "color": "#22C55E"},
    }

    # Summary row
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, qt in zip([sc1, sc2, sc3, sc4], ["technical", "hr", "behavioral", "project"]):
        n = len(st.session_state.questions.get(qt, []))
        col.metric(TYPE_META[qt]["label"], f"{n} questions")

    st.markdown("")
    tabs = st.tabs([TYPE_META[t]["label"] for t in ["technical", "hr", "behavioral", "project"]])

    for tab, q_type in zip(tabs, ["technical", "hr", "behavioral", "project"]):
        with tab:
            qs = st.session_state.questions.get(q_type, [])
            if not qs:
                st.info("No questions for this category.")
                continue
            for i, q in enumerate(qs):
                answered = (
                    q_type in st.session_state.answers and
                    i in st.session_state.answers[q_type] and
                    st.session_state.answers[q_type][i]
                )
                tick = '<span style="color:#22C55E;margin-left:8px;font-size:0.75rem;">✓ Answered</span>' if answered else ""
                st.markdown(
                    f'<div class="q-card">'
                    f'<div class="q-number">{TYPE_META[q_type]["label"]} · Q{i+1}{tick}</div>'
                    f'<div class="q-text">{q}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()
    if st.button("🎤  Start Mock Interview", type="primary", use_container_width=True):
        st.session_state.interview_active = True
        st.session_state.current_q_type = "technical"
        st.session_state.current_q_idx = 0
        st.session_state.answers = {t: {} for t in ["technical", "hr", "behavioral", "project"]}
        st.session_state.evaluations = {}
        st.session_state["_nav_target"] = "Mock Interview"
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MOCK INTERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Mock Interview":
    st.markdown('<div class="page-title">🎤 <span class="page-title-accent">Mock Interview</span></div>', unsafe_allow_html=True)

    if not st.session_state.questions:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:2.5rem;margin-bottom:0.6rem;">⚠️</div>
            <div style="color:var(--text);font-size:1rem;font-weight:600;">No questions yet</div>
            <div style="color:var(--text-muted);font-size:0.9rem;margin-top:0.3rem;">Go to <strong>🏠 Setup</strong> to generate questions first.</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    TYPES_ORDER = ["technical", "hr", "behavioral", "project"]
    TYPE_LABELS = {"technical": "⚙️ Technical", "hr": "🤝 HR", "behavioral": "🧠 Behavioral", "project": "🗂️ Project"}

    all_qs: list[tuple[str, int, str]] = []
    for qt in TYPES_ORDER:
        for i, q in enumerate(st.session_state.questions.get(qt, [])):
            all_qs.append((qt, i, q))

    total_qs = len(all_qs)

    # Count only questions that were submitted (non-empty answer recorded)
    submitted_count = sum(
        1 for qt_evals in st.session_state.evaluations.values()
        for _ in qt_evals
    )
    # Total touched = submitted + skipped
    answered_count = sum(len(v) for v in st.session_state.answers.values() if isinstance(v, dict))
    progress_val = answered_count / total_qs if total_qs else 0

    # Progress bar
    pct_done = int(progress_val * 100)
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1.2rem;">
        <div style="flex:1;background:var(--surface2);border-radius:99px;height:8px;overflow:hidden;">
            <div style="width:{pct_done}%;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:99px;transition:width 0.4s;"></div>
        </div>
        <div style="font-size:0.82rem;color:var(--text-muted);white-space:nowrap;">
            {submitted_count} evaluated · {answered_count - submitted_count} skipped · {total_qs - answered_count} remaining
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Complete ──────────────────────────────────────────────────────────────
    if st.session_state.interview_done:
        skipped = answered_count - submitted_count
        st.markdown(f"""
        <div class="complete-card">
            <h2>🎉 Interview Complete!</h2>
            <p>{submitted_count} answers evaluated · {skipped} skipped · Head to the Dashboard for detailed results.</p>
        </div>
        """, unsafe_allow_html=True)
        if submitted_count == 0:
            st.warning("No answers were submitted. The Dashboard will be empty. Try answering at least one question.")
        db_col, sk_col = st.columns([2, 1])
        with db_col:
            if st.button("📊  View Dashboard", type="primary", use_container_width=True):
                st.session_state["_nav_target"] = "Dashboard"
                st.rerun()
        with sk_col:
            if st.button("🔍  Skill Gap", use_container_width=True):
                st.session_state["_nav_target"] = "Skill Gap"
                st.rerun()
        st.stop()

    # ── Find next unanswered (not yet in answers dict at all) ─────────────────
    # A question is "answered" once it appears in st.session_state.answers[qt],
    # whether that's a real answer (evaluated) or "" (skipped).
    next_flat_idx: int | None = None
    for flat_idx, (qt, qi, _) in enumerate(all_qs):
        if qi not in st.session_state.answers.get(qt, {}):
            next_flat_idx = flat_idx
            break

    if next_flat_idx is None:
        # All questions have been touched (answered or skipped)
        st.session_state.interview_done = True
        st.rerun()

    cur_type, cur_qi, cur_question = all_qs[next_flat_idx]

    # Question card
    st.markdown(
        f'<div class="interview-q-card">'
        f'<span class="interview-q-counter">Q {next_flat_idx + 1} of {total_qs}</span>'
        f'<div class="interview-q-type">{TYPE_LABELS[cur_type]}</div>'
        f'<div class="interview-q-text">{cur_question}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Answer textarea — key includes flat index so it resets between questions
    answer_key = f"interview_answer_{next_flat_idx}"
    answer = st.text_area(
        "Your Answer",
        key=answer_key,
        height=160,
        placeholder="Type your answer here. Be specific — use examples, numbers, and the STAR method for behavioral questions.",
    )

    col_submit, col_skip = st.columns([3, 1])
    with col_submit:
        submit_clicked = st.button("✅  Submit & Get AI Feedback", type="primary", use_container_width=True)
    with col_skip:
        skip_clicked = st.button("⏭  Skip", use_container_width=True)

    if submit_clicked:
        if not answer.strip():
            st.warning("Please write an answer before submitting.")
        else:
            eval_label = "IBM Granite" if not get_client().is_demo else "Rule-Based Demo Evaluator"
            with st.spinner(f"Evaluating your answer with {eval_label}…"):
                evaluation = evaluate_answer(
                    question=cur_question,
                    answer=answer,
                    role=st.session_state.job_role,
                )
            if cur_type not in st.session_state.answers:
                st.session_state.answers[cur_type] = {}
            st.session_state.answers[cur_type][cur_qi] = answer
            if cur_type not in st.session_state.evaluations:
                st.session_state.evaluations[cur_type] = {}
            st.session_state.evaluations[cur_type][cur_qi] = {
                "question": cur_question, "answer": answer, **evaluation
            }
            st.rerun()

    if skip_clicked:
        if cur_type not in st.session_state.answers:
            st.session_state.answers[cur_type] = {}
        st.session_state.answers[cur_type][cur_qi] = ""  # mark as skipped (not evaluated)
        st.rerun()

    # ── Show feedback for already-submitted question ──────────────────────────
    # This only fires when the current question has already been evaluated
    # (e.g., the user re-navigates back to this page or after a rerun shows
    # the feedback before advancing).
    last_evals = st.session_state.evaluations.get(cur_type, {})
    if cur_qi in last_evals:
        e = last_evals[cur_qi]
        score = e.get("score", 0)
        eval_mode = e.get("evaluation_mode", "demo")
        mode_label = {
            "demo": "⚙️ Rule-Based Demo Evaluator",
            "ibm_granite": "🤖 IBM Granite",
            "demo_fallback": "⚙️ Demo Fallback (Granite parse failed)",
        }.get(eval_mode, "⚙️ Demo Evaluator")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin-bottom:0.8rem;display:flex;align-items:center;gap:10px;">'
            f'Score: {_score_badge(score)}'
            f'<span style="font-size:0.75rem;color:var(--text-muted);background:var(--surface2);'
            f'padding:2px 8px;border-radius:99px;border:1px solid var(--border);">{mode_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        fb1, fb2 = st.columns(2)
        with fb1:
            items = "".join(f"<li>{s}</li>" for s in e.get("strengths", []) if s)
            if not items:
                items = "<li style='color:var(--text-muted)'>No specific strengths identified.</li>"
            st.markdown(f'<div class="feedback-card"><h4>✅ Strengths</h4><ul>{items}</ul></div>', unsafe_allow_html=True)
        with fb2:
            items = "".join(f"<li>{s}</li>" for s in e.get("improvements", []) if s)
            if not items:
                items = "<li style='color:var(--text-muted)'>No specific improvements identified.</li>"
            st.markdown(f'<div class="feedback-card"><h4>💡 Improvements</h4><ul>{items}</ul></div>', unsafe_allow_html=True)
        with st.expander("📝 Model Answer", expanded=False):
            improved = e.get("improved_answer", "")
            if improved:
                st.markdown(f'<div style="color:var(--text);font-size:0.93rem;line-height:1.6;">{improved}</div>', unsafe_allow_html=True)
            else:
                st.info("No model answer available.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard":
    st.markdown('<div class="page-title">📊 <span class="page-title-accent">Performance Dashboard</span></div>', unsafe_allow_html=True)

    evaluations_flat: list[dict] = []
    category_scores: dict[str, float] = {}

    for qt in ["technical", "hr", "behavioral", "project"]:
        qt_evals = list(st.session_state.evaluations.get(qt, {}).values())
        if qt_evals:
            scores = [e.get("score", 0) for e in qt_evals]
            category_scores[qt.capitalize()] = round(sum(scores) / len(scores), 1)
            evaluations_flat.extend(qt_evals)

    if not evaluations_flat:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:2.5rem;margin-bottom:0.6rem;">📭</div>
            <div style="color:var(--text);font-size:1rem;font-weight:600;">No interview data yet</div>
            <div style="color:var(--text-muted);font-size:0.9rem;margin-top:0.3rem;">Complete the <strong>🎤 Mock Interview</strong> to see your results here.</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    stats = compute_session_stats(evaluations_flat)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Questions Answered", stats["total"])
    k2.metric("Average Score", f"{stats['avg_score']} / 10")
    k3.metric("Overall Performance", f"{stats['percentage']}%")
    k4.metric("Categories Covered", len(category_scores))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart row 1 ───────────────────────────────────────────────────────────
    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(dashboard.scores_bar_chart(evaluations_flat), use_container_width=True)
    with cb:
        if len(category_scores) >= 3:
            st.plotly_chart(dashboard.radar_chart(category_scores), use_container_width=True)
        else:
            st.plotly_chart(
                dashboard.score_distribution_histogram([e.get("score", 0) for e in evaluations_flat]),
                use_container_width=True,
            )

    # ── Chart row 2 ───────────────────────────────────────────────────────────
    cg, ch = st.columns([1, 2])
    with cg:
        st.plotly_chart(dashboard.progress_gauge(stats["percentage"]), use_container_width=True)
    with ch:
        st.markdown('<div class="section-heading" style="margin-top:0.5rem;">📋 Question Breakdown</div>', unsafe_allow_html=True)
        for i, e in enumerate(evaluations_flat):
            score = e.get("score", 0)
            q_short = e.get("question", "")[:80]
            with st.expander(f"Q{i+1}  ·  {q_short}{'…' if len(e.get('question',''))>80 else ''}", expanded=False):
                st.markdown(f'Score: {_score_badge(score)}', unsafe_allow_html=True)
                st.markdown(f'<div style="color:var(--text-muted);font-size:0.85rem;margin:8px 0;">Your answer:</div>'
                            f'<div style="color:var(--text);font-size:0.9rem;">{e.get("answer","")[:300]}</div>', unsafe_allow_html=True)
                if e.get("improvements"):
                    st.markdown(f'<div style="color:#FCD34D;font-size:0.88rem;margin-top:8px;">💡 {e["improvements"][0]}</div>', unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    col_exp, _ = st.columns([1, 3])
    with col_exp:
        export_data = {
            "role": st.session_state.job_role,
            "stats": stats,
            "category_scores": category_scores,
            "evaluations": evaluations_flat,
        }
        st.download_button(
            "📥  Download Results (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name="skillsync_results.json",
            mime="application/json",
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SKILL GAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Skill Gap":
    st.markdown('<div class="page-title">🔍 <span class="page-title-accent">Skill Gap Analysis</span></div>', unsafe_allow_html=True)

    resume_skills = (
        st.session_state.resume_data["skills"] if st.session_state.resume_data else []
    )
    role = st.session_state.job_role or "Data Scientist"
    job_desc = st.session_state.job_description or ""

    if not resume_skills:
        st.warning("Upload your resume in **🏠 Setup** for a personalised analysis. You can still run the analysis with the target role.")

    if st.button("🔄  Run Skill Gap Analysis", type="primary"):
        with st.spinner("Analysing skills against dataset and role requirements…"):
            report = full_skill_gap_report(
                resume_skills=resume_skills,
                role=role,
                job_description=job_desc,
            )
            ai_analysis = get_client().analyze_skill_gaps(
                resume_skills=resume_skills,
                required_skills=report["required_skills"],
                role=role,
            )
        st.session_state.skill_gap = {"report": report, "ai": ai_analysis}

    if st.session_state.skill_gap:
        rep = st.session_state.skill_gap["report"]
        ai  = st.session_state.skill_gap["ai"]
        fit_pct = round(rep["fit_score"] * 100, 1)

        # ── KPI row ───────────────────────────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Role Fit Score",    f"{fit_pct}%")
        s2.metric("Matching Skills",   len(rep["matching_skills"]))
        s3.metric("Missing Skills",    len(rep["missing_skills"]))
        s4.metric("Bonus Skills",      len(rep["extra_skills"]))

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Skills donut + badge grid ─────────────────────────────────────────
        dc, ds = st.columns([1, 1])
        with dc:
            st.plotly_chart(
                dashboard.skill_gap_donut(len(rep["matching_skills"]), len(rep["missing_skills"])),
                use_container_width=True,
            )
        with ds:
            st.markdown('<div class="section-heading">✅ Matching Skills</div>', unsafe_allow_html=True)
            if rep["matching_skills"]:
                st.markdown(_skill_badges(rep["matching_skills"], "match"), unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:var(--text-muted);font-size:0.9rem;">No matching skills detected for this role.</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-heading" style="margin-top:1rem;">❌ Missing Skills</div>', unsafe_allow_html=True)
            if rep["missing_skills"]:
                st.markdown(_skill_badges(rep["missing_skills"], "missing"), unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:var(--success);font-size:0.9rem;">No skill gaps detected — great fit!</div>', unsafe_allow_html=True)

            if rep["extra_skills"]:
                st.markdown('<div class="section-heading" style="margin-top:1rem;">⭐ Bonus Skills</div>', unsafe_allow_html=True)
                st.markdown(_skill_badges(rep["extra_skills"][:12], "neutral"), unsafe_allow_html=True)

        # ── AI Recommendations ────────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-heading">🤖 AI-Powered Preparation Recommendations</div>', unsafe_allow_html=True)
        recs = ai.get("recommendations") or rep.get("recommendations") or []
        if recs:
            for i, rec in enumerate(recs, 1):
                st.markdown(f'<div class="rec-card"><strong>Tip {i}:</strong> {rec}</div>', unsafe_allow_html=True)
        else:
            st.info("Complete the mock interview to receive personalised recommendations.")

        # ── Dataset benchmark ─────────────────────────────────────────────────
        bm = rep.get("dataset_benchmark", {})
        _, ds_status = get_dataset()
        st.divider()
        st.markdown('<div class="section-heading">📂 Dataset Benchmark</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:1rem;">'
            'Computed from <code style="color:var(--accent2);">job_resume_fit.csv</code> '
            '(2,385 resume-job records). This dataset is used for matching and benchmarking only — '
            'it was <strong>not</strong> used to train IBM Granite.'
            '</div>',
            unsafe_allow_html=True,
        )
        if ds_status.loaded and bm:
            bm1, bm2, bm3, bm4 = st.columns(4)
            bm1.metric("Your Skill Match",      f"{bm.get('candidate_score', 0):.1f} / 100",
                       help="How well your skills cover this role's requirements in the dataset.")
            bm2.metric("Dataset Avg (AI)",       f"{bm.get('dataset_mean_ai_score', 0):.1f} / 100",
                       help="Mean AI match score for this role across dataset rows.")
            bm3.metric("Percentile Rank",        f"{bm.get('percentile', 0):.0f}%",
                       help="You score better than this % of profiles in the dataset.")
            bm4.metric("Role Dataset Records",   f"{bm.get('role_row_count', 0):,}",
                       help="Number of dataset records matching this role.")
        elif not ds_status.loaded:
            st.info("Dataset not loaded — place `job_resume_fit.csv` in `data/` for benchmarking.")

        # ── Similar profiles ──────────────────────────────────────────────────
        st.markdown('<div class="section-heading" style="margin-top:1.2rem;">👥 Similar Profiles in Dataset</div>', unsafe_allow_html=True)
        similar = get_similar_candidates(resume_skills=resume_skills, role=role, top_n=5)
        if not similar.empty:
            similar = similar.copy()
            if "resume_skill_list" in similar.columns:
                similar["resume_skill_list"] = similar["resume_skill_list"].apply(
                    lambda v: ", ".join(str(v)[:120].split(",")[:6]) + ("…" if len(str(v)) > 120 else "")
                )
            for col in ["similarity", "ai_match_score", "fuzzy_match_score", "skill_string_match_score"]:
                if col in similar.columns:
                    if col == "similarity":
                        similar[col] = similar[col].apply(lambda x: f"{x:.1%}")
                    else:
                        similar[col] = similar[col].apply(lambda x: f"{float(x):.1f}")
            st.dataframe(similar, use_container_width=True, hide_index=True)
        else:
            st.info("No similar profiles found for this role in the dataset.")

        # ── Study plan ────────────────────────────────────────────────────────
        if rep["missing_skills"]:
            st.divider()
            st.markdown('<div class="section-heading">📚 Suggested Study Plan</div>', unsafe_allow_html=True)
            weeks = min(len(rep["missing_skills"]), 6)
            plan_html = ""
            for w in range(1, weeks + 1):
                skill = rep["missing_skills"][w - 1]
                plan_html += (
                    f'<div class="study-item">'
                    f'<div class="study-week">Week {w}</div>'
                    f'<div class="study-text">Focus on <strong>{skill}</strong> — '
                    f'build a hands-on mini-project, review official docs, '
                    f'and solve 3–5 practice problems or exercises.</div>'
                    f'</div>'
                )
            st.markdown(plan_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:2.5rem;margin-bottom:0.6rem;">🔍</div>
            <div style="color:var(--text);font-size:1rem;font-weight:600;">Ready to analyse your skill fit</div>
            <div style="color:var(--text-muted);font-size:0.9rem;margin-top:0.3rem;">
                Click <strong>Run Skill Gap Analysis</strong> above to see your skill alignment,
                gaps, AI recommendations, and dataset benchmarking.
            </div>
        </div>
        """, unsafe_allow_html=True)
