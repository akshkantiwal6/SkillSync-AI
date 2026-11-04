"""
test_pipeline.py – Headless end-to-end test for SkillSync AI.
Run: python test_pipeline.py
"""
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("SkillSync AI – Pipeline Test")
print("=" * 60)

# ── 1. Config ────────────────────────────────────────────────────────────────
import config
print(f"\n[config] DEMO_MODE={config.DEMO_MODE}, model={config.GRANITE_MODEL_ID}")

# ── 2. Resume parser ─────────────────────────────────────────────────────────
from modules.resume_parser import extract_skills, extract_email, extract_experience_years
sample_text = """
Jane Doe | jane@example.com | +1-555-0100
B.Tech Computer Science
3 years of experience in Python, machine learning, pandas, scikit-learn, SQL, Docker.
"""
skills = extract_skills(sample_text)
email = extract_email(sample_text)
exp = extract_experience_years(sample_text)
print(f"\n[resume_parser] email={email}, exp_years={exp}, skills={skills[:5]}")
assert email == "jane@example.com", "Email extraction failed"
assert exp == 3.0, f"Expected 3.0 got {exp}"
assert "python" in skills
print("  PASS")

# ── 3. RAG engine ────────────────────────────────────────────────────────────
from modules.rag_engine import get_engine
engine = get_engine()
results = engine.retrieve("Python data science machine learning interview questions", top_k=3)
print(f"\n[rag_engine] FAISS={engine._use_faiss}, retrieved {len(results)} docs")
assert len(results) > 0
print(f"  top doc score: {results[0]['score']:.3f}")
ctx = engine.retrieve_context("behavioral interview STAR method", top_k=2)
assert len(ctx) > 50
print("  PASS")

# ── 4. Granite client (demo) ─────────────────────────────────────────────────
from modules.granite_client import get_client
client = get_client()
print(f"\n[granite_client] demo={client.is_demo}")
qs = client.generate_questions(
    role="Data Scientist",
    job_description="ML model development",
    skills=["python", "pandas", "sklearn"],
    question_type="technical",
    count=5,
)
assert len(qs) == 5
print(f"  generated {len(qs)} questions, first: {qs[0][:60]}")
ev = client.evaluate_answer("What is overfitting?", "Overfitting means the model memorizes training data.", "Data Scientist")
assert "score" in ev and isinstance(ev["score"], int)
print(f"  evaluation score: {ev['score']}/10")
print("  PASS")

# ── 5. Question generator ────────────────────────────────────────────────────
from modules.question_generator import generate_all_question_sets
all_qs = generate_all_question_sets(
    role="Software Engineer",
    job_description="FastAPI backend with PostgreSQL",
    resume_skills=["python", "sql", "docker"],
    counts={"technical": 3, "hr": 2, "behavioral": 2, "project": 2},
)
total_q = sum(len(v) for v in all_qs.values())
print(f"\n[question_generator] {total_q} total questions across {len(all_qs)} types")
for qtype, qlist in all_qs.items():
    print(f"  {qtype}: {len(qlist)}  -> {qlist[0][:55]}")
assert total_q == 9
print("  PASS")

# ── 6. Evaluator ─────────────────────────────────────────────────────────────
from modules.evaluator import evaluate_answer, evaluate_session, compute_session_stats
qa_pairs = [
    {"question": all_qs["technical"][0], "answer": "I would use proper indexing and caching."},
    {"question": all_qs["hr"][0], "answer": "I am a backend engineer with 2 years of Python experience."},
    {"question": "", "answer": ""},   # edge: empty
]
results = evaluate_session(qa_pairs, "Software Engineer")
stats = compute_session_stats(results)
print(f"\n[evaluator] avg_score={stats['avg_score']}, total={stats['total']}, pct={stats['percentage']}%")
assert stats["total"] == 3
assert 0 <= stats["avg_score"] <= 10
print("  PASS")

# ── 7. Skill gap ─────────────────────────────────────────────────────────────
from modules.skill_gap import full_skill_gap_report, compute_match_score
rep = full_skill_gap_report(["python", "sql", "docker"], "software engineer")
print(f"\n[skill_gap] fit={rep['fit_score']:.0%}, matching={rep['matching_skills']}, missing={rep['missing_skills'][:3]}")
assert 0.0 <= rep["fit_score"] <= 1.0
assert isinstance(rep["recommendations"], list)
score = compute_match_score(["python", "sql"], ["python", "java", "sql", "docker"])
assert score == 0.5, f"Expected 0.5 got {score}"
print("  PASS")

# ── 8. Dashboard ─────────────────────────────────────────────────────────────
from modules.dashboard import scores_bar_chart, radar_chart, skill_gap_donut, progress_gauge, score_distribution_histogram
evals = [{"score": s} for s in [6, 8, 9, 4, 7]]
fig1 = scores_bar_chart(evals)
fig2 = radar_chart({"Technical": 7.0, "HR": 6.5, "Behavioral": 8.5, "Project": 7.5})
fig3 = skill_gap_donut(4, 6)
fig4 = progress_gauge(71.0)
fig5 = score_distribution_histogram([5, 7, 8, 9, 6])
for f in [fig1, fig2, fig3, fig4, fig5]:
    assert hasattr(f, "data"), "Figure missing data attribute"
print(f"\n[dashboard] 5 charts constructed successfully")
print("  PASS")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("ALL TESTS PASSED - OK")
print("=" * 60)
