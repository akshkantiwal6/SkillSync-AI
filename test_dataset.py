"""
test_dataset.py – End-to-end test for dataset integration.
Run: python test_dataset.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 60)
print("SkillSync AI - Dataset Integration Test")
print("=" * 60)

# ── 1. DatasetStatus / load ──────────────────────────────────────────────────
print("\n[1] dataset_loader.load_dataset")
from modules.dataset_loader import (
    load_dataset, get_dataset, get_categories,
    get_required_skills_from_dataset, find_similar_candidates,
    compute_dataset_match_score, parse_skill_string, EXPECTED_COLUMNS,
)

df, status = load_dataset()
assert status.loaded, f"Dataset failed to load: {status.error}"
assert status.row_count == 2385, f"Expected 2385 rows, got {status.row_count}"
assert len(status.missing_columns) == 0, f"Missing columns: {status.missing_columns}"
assert set(EXPECTED_COLUMNS).issubset(set(status.columns)), "Schema mismatch"
print(f"  rows={status.row_count}, cols={len(status.columns)}, missing={status.missing_columns}")
print(f"  columns: {status.columns}")
print("  PASS")

# ── 2. Column types and nulls ────────────────────────────────────────────────
print("\n[2] Column validation and null handling")
null_counts = df.isnull().sum()
assert null_counts.sum() == 0, f"Nulls remain after cleaning: {null_counts[null_counts > 0].to_dict()}"
assert df["ai_match_score"].dtype in ("float64", "float32"), "ai_match_score not float"
assert df["category"].str.isupper().all(), "category not uppercased"
print(f"  null cells after cleaning: {null_counts.sum()}")
print(f"  ai_match_score range: {df['ai_match_score'].min():.1f} - {df['ai_match_score'].max():.1f}")
print(f"  unique categories: {df['category'].nunique()}")
print("  PASS")

# ── 3. parse_skill_string ────────────────────────────────────────────────────
print("\n[3] parse_skill_string")
cases = [
    ("['python', 'sql', 'pandas']", ["python", "sql", "pandas"]),
    ("python, sql, pandas",        ["python", "sql", "pandas"]),
    ("python; sql; pandas",        ["python", "sql", "pandas"]),
    ("",                           []),
    ("nan",                        []),
    ("[]",                         []),
]
for raw, expected in cases:
    result = parse_skill_string(raw)
    assert result == expected, f"parse_skill_string({raw!r}) -> {result}, expected {expected}"
print("  All 6 parse cases passed")
print("  PASS")

# ── 4. get_categories ────────────────────────────────────────────────────────
print("\n[4] get_categories")
cats = get_categories()
assert len(cats) >= 20, f"Expected >=20 categories, got {len(cats)}"
assert "INFORMATION-TECHNOLOGY" in cats
assert "FINANCE" in cats
print(f"  {len(cats)} unique categories, sample: {cats[:5]}")
print("  PASS")

# ── 5. get_required_skills_from_dataset ──────────────────────────────────────
print("\n[5] get_required_skills_from_dataset")
for role_test in ["Information Technology", "Finance", "data scientist", "banking"]:
    skills = get_required_skills_from_dataset(role_test)
    print(f"  {role_test!r:30} -> {len(skills)} skills: {skills[:4]}")
    assert isinstance(skills, list)
print("  PASS")

# ── 6. find_similar_candidates ───────────────────────────────────────────────
print("\n[6] find_similar_candidates")
similar = find_similar_candidates(
    ["python", "sql", "machine learning", "pandas"],
    "information technology",
    top_n=5,
)
assert not similar.empty, "Expected results, got empty DataFrame"
assert "similarity" in similar.columns
assert "category" in similar.columns
print(f"  found {len(similar)} similar profiles")
print(f"  top similarity: {similar['similarity'].iloc[0]}")
print(f"  columns: {list(similar.columns)}")
print("  PASS")

# ── 7. compute_dataset_match_score ───────────────────────────────────────────
print("\n[7] compute_dataset_match_score")
bm = compute_dataset_match_score(["python", "sql", "pandas"], "data scientist")
print(f"  benchmark result: {bm}")
assert "candidate_score" in bm
assert "dataset_mean_ai_score" in bm
assert "percentile" in bm
assert "role_row_count" in bm
assert 0 <= bm["candidate_score"] <= 100
assert 0 <= bm["percentile"] <= 100
print("  PASS")

# ── 8. skill_gap.py uses dataset ─────────────────────────────────────────────
print("\n[8] skill_gap.full_skill_gap_report with real dataset")
from modules.skill_gap import full_skill_gap_report
rep = full_skill_gap_report(["python", "sql", "pandas", "machine learning"], "information technology")
print(f"  required_skills (from dataset): {rep['required_skills'][:5]}")
print(f"  matching: {rep['matching_skills']}")
print(f"  missing:  {rep['missing_skills'][:4]}")
print(f"  fit_score: {rep['fit_score']:.1%}")
print(f"  benchmark: {rep['dataset_benchmark']}")
assert len(rep["required_skills"]) > 0, "Should get skills from dataset"
assert 0.0 <= rep["fit_score"] <= 1.0
assert "dataset_benchmark" in rep
print("  PASS")

# ── 9. get_dataset() cache ────────────────────────────────────────────────────
print("\n[9] Caching: get_dataset() returns same object on repeated calls")
df1, s1 = get_dataset()
df2, s2 = get_dataset()
assert df1 is df2, "Cache not working - different objects returned"
print("  PASS")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("ALL DATASET TESTS PASSED - OK")
print("=" * 60)
