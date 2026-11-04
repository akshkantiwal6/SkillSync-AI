"""
test_scoring_fix.py - Quick targeted test for the exact failing cases.
Run: python test_scoring_fix.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Reset singleton to pick up latest code
import modules.granite_client as gc
gc._client = None

from modules.evaluator import evaluate_answer

Q = "Explain how a hash table works internally."
ROLE = "Software Engineer"

cases = [
    # (answer_text, min_score, max_score, label)
    ("",         0, 0,  "empty"),
    ("sdfghj",   0, 1,  "gibberish single token"),
    ("asdf qwerty zxcv mnop lkjh", 0, 1, "gibberish multi-token"),
    ("It stores data.", 1, 3, "vague one sentence"),
    ("It uses hash functions to map keys to indexes.", 3, 6, "partial correct"),
    ("The weather is nice today. I like playing football.", 0, 2, "completely irrelevant"),
    (
        "A hash table uses a hash function to compute an array index (bucket) for each key. "
        "On collision two keys hash to the same bucket, handled by chaining with a linked list "
        "or open addressing. Average O(1) lookup, insertion and deletion.",
        6, 10,
        "technically correct explanation"
    ),
    (
        "A hash table maps keys to values using a hash function that computes an array index. "
        "The index points to a bucket. Collisions occur when two keys map to the same bucket. "
        "Resolution strategies include separate chaining (linked list per bucket) and open "
        "addressing (linear/quadratic probing). Load factor determines when to resize. "
        "Python dict is a hash table with O(1) average lookup.",
        7, 10,
        "comprehensive answer with examples"
    ),
]

print("=" * 65)
print("Hash Table Evaluation Scoring Tests")
print("=" * 65)

all_ok = True
for answer, lo, hi, label in cases:
    result = evaluate_answer(Q, answer, ROLE)
    score = result["score"]
    ok = lo <= score <= hi
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"[{status}] {label}")
    print(f"       answer   : {repr(answer[:55])}")
    print(f"       expected : {lo}-{hi}  |  got: {score}")
    if not ok:
        imps = result.get("improvements", [])
        print(f"       feedback : {imps[0] if imps else '(none)'}")
    print()

print("=" * 65)
print("ALL PASS" if all_ok else "SOME TESTS FAILED")
print("=" * 65)
sys.exit(0 if all_ok else 1)
