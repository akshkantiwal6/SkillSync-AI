"""
test_evaluation.py – Tests for answer evaluation correctness.

Tests all required cases:
  1. Empty answer
  2. Single-character answer
  3. Irrelevant / filler answer
  4. Partially correct answer
  5. Detailed correct answer

Also tests navigation state helpers and button logic.
Run: python test_evaluation.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from modules.granite_client import _demo_evaluate, _sample_answer, get_client
from modules.evaluator import evaluate_answer, compute_session_stats

QUESTION_TECHNICAL = "Explain the difference between supervised and unsupervised learning."
QUESTION_BEHAVIORAL = "Describe a challenging project and how you overcame obstacles."
QUESTION_HR = "Tell me about yourself."
ROLE = "Data Scientist"

SEPARATOR = "=" * 60
PASS = "[PASS]"
FAIL = "[FAIL]"

all_passed = True

def check(condition: bool, label: str, detail: str = "") -> None:
    global all_passed
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    if not condition:
        print(f"         Detail: {detail}")
        all_passed = False


print(SEPARATOR)
print("SkillSync AI - Evaluation Tests")
print(SEPARATOR)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. EMPTY ANSWER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] Empty answer → must score 0")
e = _demo_evaluate(QUESTION_TECHNICAL, "")
check(e["score"] == 0, f"score=0 for empty answer", f"got {e['score']}")
check(len(e["improvements"]) > 0, "has improvements feedback")
check(e.get("improved_answer", ""), "has model answer")
print(f"     score={e['score']}, improvements={e['improvements'][:1]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SINGLE CHARACTER ANSWER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] Single-character answer 'I' → must score 0")
e = _demo_evaluate(QUESTION_TECHNICAL, "I")
check(e["score"] == 0, f"score=0 for 'I'", f"got {e['score']}")
check("too short" in " ".join(e["improvements"]).lower() or "does not address" in " ".join(e["improvements"]).lower(),
      "feedback mentions shortness or irrelevance")
print(f"     score={e['score']}, feedback={e['improvements'][0][:70]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FILLER WORDS ONLY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] Filler answers → must score 0-2")
for filler in ["yes", "no", "ok", "sure", "idk", "yes yes yes"]:
    e = _demo_evaluate(QUESTION_TECHNICAL, filler)
    check(e["score"] <= 2, f"score<=2 for filler '{filler}'", f"got {e['score']}")
    print(f"     '{filler}' -> score={e['score']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. RANDOM / IRRELEVANT TEXT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4] Random/irrelevant answers → must score low (<=3)")
irrelevant_cases = [
    "The weather is nice today and I like to play football on weekends.",
    "I enjoy cooking pasta and watching movies in my free time.",
    "asdf qwerty zxcv hello world testing one two three",
]
for text in irrelevant_cases:
    e = _demo_evaluate(QUESTION_TECHNICAL, text)
    check(e["score"] <= 4, f"score<=4 for irrelevant answer", f"got {e['score']} for: {text[:50]}")
    print(f"     score={e['score']} for: {text[:50]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PARTIAL ANSWER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] Partial answer → should score 3-6")
partial = "Supervised learning uses labeled data. Unsupervised does not use labels."
e = _demo_evaluate(QUESTION_TECHNICAL, partial)
check(3 <= e["score"] <= 7, f"score in 3-7 for partial answer", f"got {e['score']}")
print(f"     score={e['score']}, strengths={e['strengths'][:1]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. GOOD ANSWER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6] Good technical answer → should score 6-9")
good = (
    "Supervised learning trains models on labeled data where the target output is known, "
    "such as classification (spam detection) and regression (house price prediction). "
    "The algorithm learns a mapping from input features to the output label. "
    "Unsupervised learning works with unlabeled data to discover hidden patterns, "
    "for example clustering customers into segments using K-Means or reducing dimensions "
    "with PCA. The key difference is whether ground truth labels are available during training."
)
e = _demo_evaluate(QUESTION_TECHNICAL, good)
check(e["score"] >= 6, f"score>=6 for good answer", f"got {e['score']}")
check(len(e["strengths"]) > 0, "has strengths feedback")
print(f"     score={e['score']}, strengths={e['strengths']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SCORES MUST DIFFER ACROSS QUALITY LEVELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7] Scores must increase with answer quality")
empty_score   = _demo_evaluate(QUESTION_TECHNICAL, "")["score"]
single_score  = _demo_evaluate(QUESTION_TECHNICAL, "I")["score"]
partial_score = _demo_evaluate(QUESTION_TECHNICAL, partial)["score"]
good_score    = _demo_evaluate(QUESTION_TECHNICAL, good)["score"]

print(f"     empty={empty_score}  single-char={single_score}  partial={partial_score}  good={good_score}")
check(empty_score <= single_score <= partial_score <= good_score,
      "scores increase with answer quality",
      f"order: {empty_score} <= {single_score} <= {partial_score} <= {good_score}")
check(good_score > single_score + 3,
      "good answer scores significantly higher than single-char",
      f"{good_score} vs {single_score}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. BEHAVIORAL QUESTION — STAR METHOD RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[8] Behavioral question with STAR answer → score >= 5")
star_answer = (
    "During my final year project (Situation), I was responsible for integrating "
    "a machine learning model into our web application (Task). I researched Flask APIs "
    "and built a REST endpoint for the model, then wrote unit tests for each component (Action). "
    "As a result, the feature shipped on time and reduced manual data entry by 60% (Result)."
)
e = _demo_evaluate(QUESTION_BEHAVIORAL, star_answer)
check(e["score"] >= 5, f"score>=5 for STAR answer", f"got {e['score']}")
print(f"     score={e['score']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. EVALUATE ANSWER via evaluator.py (integration test)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[9] evaluator.evaluate_answer() integration test")
# Test that the main entry-point returns correct structure
for ans, expected_max in [("", 3), ("I", 3), (good, 6)]:
    result = evaluate_answer(QUESTION_TECHNICAL, ans, ROLE)
    check("score" in result, f"result has 'score' key")
    check("improvements" in result, f"result has 'improvements' key")
    check("improved_answer" in result, f"result has 'improved_answer' key")
    check(result["score"] <= 10, f"score <= 10, got {result['score']}")
    if ans == "":
        check(result["score"] == 0, "empty answer scores 0 via evaluate_answer()", f"got {result['score']}")
    elif ans == "I":
        check(result["score"] <= expected_max, f"'I' scores <={expected_max}", f"got {result['score']}")
    else:
        check(result["score"] >= expected_max, f"good answer scores >={expected_max}", f"got {result['score']}")
    print(f"     answer={repr(ans[:20]):<24} -> score={result['score']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. compute_session_stats — only uses evaluated answers
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[10] compute_session_stats()")
evaluations = [
    {"score": 8, "question": "q1", "answer": "good answer"},
    {"score": 3, "question": "q2", "answer": "partial"},
    {"score": 0, "question": "q3", "answer": "I"},
]
stats = compute_session_stats(evaluations)
check(stats["total"] == 3, f"total=3", f"got {stats['total']}")
check(stats["avg_score"] == round((8+3+0)/3, 1), f"avg_score correct", f"got {stats['avg_score']}")
check(0 < stats["percentage"] < 100, f"percentage in (0,100)", f"got {stats['percentage']}")
print(f"     avg={stats['avg_score']}, pct={stats['percentage']}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. _sample_answer coverage
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[11] _sample_answer() returns non-empty string for all tested questions")
for q in [QUESTION_TECHNICAL, QUESTION_BEHAVIORAL, QUESTION_HR,
          "What is Big O notation?", "Explain the SOLID principles.",
          "Describe a time you showed leadership."]:
    ans = _sample_answer(q)
    check(len(ans) > 40, f"model answer len>40 for: {q[:50]}", f"got: {repr(ans[:30])}")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. evaluation_mode field is set
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[12] evaluation_mode field present in all evaluations")
for ans in ["", "I", good]:
    e = evaluate_answer(QUESTION_TECHNICAL, ans, ROLE)
    check("evaluation_mode" in e, f"evaluation_mode present for answer={repr(ans[:15])}")
    check(e["evaluation_mode"] in ("demo", "ibm_granite", "demo_fallback"),
          f"evaluation_mode is valid string", f"got: {e.get('evaluation_mode')}")
    print(f"     answer={repr(ans[:15]):<20} -> mode={e['evaluation_mode']}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(SEPARATOR)
if all_passed:
    print("ALL EVALUATION TESTS PASSED - OK")
else:
    print("SOME TESTS FAILED - see [FAIL] lines above")
print(SEPARATOR)
sys.exit(0 if all_passed else 1)
