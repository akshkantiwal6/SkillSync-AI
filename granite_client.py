"""
granite_client.py – IBM Granite via watsonx.ai Runtime.

Falls back to a transparent rule-based evaluator when credentials are absent.
The demo evaluator genuinely analyses the answer — it never returns a fixed score.
"""
from __future__ import annotations

import json
import re
import textwrap
from typing import Any

import config

# ── Optional IBM SDK import ──────────────────────────────────────────────────
try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

    IBM_SDK_OK = True
except ImportError:
    IBM_SDK_OK = False


# ── Demo-mode question pools ──────────────────────────────────────────────────
_DEMO_QUESTIONS = {
    "technical": [
        "Explain the difference between supervised and unsupervised learning.",
        "What is a REST API and how does it differ from GraphQL?",
        "Describe the concept of Big O notation with an example.",
        "What are SOLID principles in software engineering?",
        "Explain how a hash table works internally.",
    ],
    "hr": [
        "Tell me about yourself.",
        "Where do you see yourself in 5 years?",
        "Why do you want to work at our company?",
        "What are your biggest strengths and weaknesses?",
        "Describe a time you dealt with a difficult coworker.",
    ],
    "behavioral": [
        "Describe a challenging project and how you overcame obstacles.",
        "Give an example of when you showed leadership.",
        "Tell me about a time you failed and what you learned.",
        "Describe a situation where you had to meet a tight deadline.",
        "How do you prioritize tasks when everything seems urgent?",
    ],
    "project": [
        "Walk me through your most impactful project.",
        "What technologies did you use and why did you choose them?",
        "What were the biggest technical challenges and how did you resolve them?",
        "How did you measure the success of your project?",
        "What would you do differently if you rebuilt this project today?",
    ],
}


# ── Rule-based demo evaluator ─────────────────────────────────────────────────
# Used transparently when IBM credentials are not set.
# Scores are computed from real signals — NOT a fixed value.

# Common filler / irrelevant tokens that add no value
_FILLER_TOKENS = {
    "i", "yes", "no", "ok", "okay", "sure", "maybe", "idk", "hmm",
    "dunno", "nothing", "none", "na", "hi", "hello", "hey",
    "test", "testing", "abc", "asdf", "qwerty", "random",
}

# Technical concept signals — ALL entries must be ≥ 3 characters.
# Keys are checked with whole-word regex against the question.
# Signals are also checked with whole-word regex against the answer.
_TECH_SIGNALS: dict[str, list[str]] = {
    "supervised":   ["labeled", "label", "training", "classification", "regression", "target"],
    "unsupervised": ["cluster", "unlabeled", "pattern", "grouping", "dimensionality", "kmeans"],
    "rest":         ["http", "endpoint", "status", "json", "verb", "resource", "stateless", "request"],
    "graphql":      ["query", "schema", "resolver", "mutation", "subscription", "field"],
    "notation":     ["complexity", "logarithm", "linear", "quadratic", "constant", "algorithm", "time"],
    "solid":        ["responsibility", "liskov", "segregation", "dependency", "inversion", "open", "closed"],
    # Hash table: tier 1 = partial credit (functions, keys, map, index)
    #             tier 2 = full credit (bucket, collision, chaining, probing)
    "hash":         ["functions", "function", "keys", "index", "indexes", "indices", "maps",
                     "bucket", "collision", "chaining", "probing", "lookup", "hashing", "load"],
    "overfitting":  ["generalize", "regularization", "validation", "variance", "dropout", "training"],
    "docker":       ["container", "image", "dockerfile", "volume", "layer", "daemon"],
    "kubernetes":   ["cluster", "deployment", "service", "ingress", "namespace", "replica", "pod"],
    "agile":        ["sprint", "scrum", "backlog", "iteration", "standup", "velocity", "story"],
    "behavioral":   ["situation", "task", "action", "result", "outcome", "star", "challenge"],
    "machine":      ["learning", "model", "training", "dataset", "feature", "prediction", "accuracy"],
    "database":     ["query", "index", "transaction", "schema", "foreign", "primary", "normal", "table"],
    "neural":       ["layer", "activation", "backprop", "gradient", "weight", "neuron", "network"],
    "recursion":    ["base", "call", "stack", "return", "self", "depth", "case"],
    "sorting":      ["compare", "swap", "pivot", "merge", "partition", "order", "time"],
    "concurrency":  ["thread", "lock", "mutex", "async", "await", "race", "deadlock", "sync"],
    "memory":       ["heap", "stack", "pointer", "reference", "garbage", "allocation", "free"],
}

# Common English words (≥ 4 chars) — used for gibberish detection.
# A word is "real" if it appears in this set OR contains a known pattern.
_COMMON_WORDS = {
    "that", "this", "with", "have", "from", "they", "will", "been",
    "were", "when", "what", "your", "more", "also", "into", "than",
    "then", "some", "time", "data", "used", "uses", "work", "works",
    "each", "such", "both", "much", "most", "only", "very", "over",
    "after", "which", "their", "there", "about", "would", "other",
    "these", "those", "where", "while", "using", "based", "makes",
    "takes", "gives", "store", "stores", "map", "maps", "maps",
    "keys", "values", "index", "array", "table", "hash", "function",
    "node", "list", "tree", "graph", "sort", "search", "insert",
    "delete", "update", "create", "read", "write", "return", "call",
    "class", "method", "object", "type", "code", "program", "file",
    "user", "data", "system", "service", "server", "client", "query",
    "model", "train", "test", "layer", "loss", "score", "result",
    "error", "output", "input", "feature", "label", "class", "example",
}


def _demo_evaluate(question: str, answer: str) -> dict[str, Any]:
    """
    Transparent rule-based evaluation used in demo mode.

    Scoring rubric (0–10):
      0    – empty
      0–1  – single word / pure filler / random characters
      1–3  – extremely short (< 15 words) with no relevant content
      3–5  – partial: shows some relevant knowledge but incomplete
      5–7  – adequate: covers the main idea, lacks depth/examples
      7–9  – good: specific, correct, some examples
      9–10 – excellent: thorough, technically precise, well-structured
    """
    answer_stripped = answer.strip()

    # ── 0: empty ──────────────────────────────────────────────────────────────
    if not answer_stripped:
        return {
            "score": 0, "max_score": 10,
            "strengths": [],
            "improvements": ["No answer was provided. Please write a response to be evaluated."],
            "improved_answer": _sample_answer(question),
            "evaluation_mode": "demo",
        }

    words = re.findall(r"\b\w+\b", answer_stripped.lower())
    word_count = len(words)
    unique_words = set(words)

    # ── 1: single word or pure filler ────────────────────────────────────────
    if word_count <= 1 or (word_count <= 3 and unique_words.issubset(_FILLER_TOKENS)):
        return {
            "score": 0, "max_score": 10,
            "strengths": [],
            "improvements": [
                f"Your answer '{answer_stripped[:30]}' is too short and does not address the question.",
                "Provide a substantive explanation with at least 3–5 sentences.",
                "Use specific examples, concepts, or experiences relevant to the question.",
            ],
            "improved_answer": _sample_answer(question),
            "evaluation_mode": "demo",
        }

    # ── Detect filler-only answers ────────────────────────────────────────────
    non_filler = unique_words - _FILLER_TOKENS - {"the", "a", "an", "is", "it", "to", "of", "and", "or"}
    if len(non_filler) <= 2:
        return {
            "score": 1, "max_score": 10,
            "strengths": [],
            "improvements": [
                "Your answer does not contain any meaningful content related to the question.",
                "Avoid one-word or vague responses in an interview setting.",
                "Demonstrate your knowledge by explaining concepts, giving examples, or describing experiences.",
            ],
            "improved_answer": _sample_answer(question),
            "evaluation_mode": "demo",
        }

    # ── Gibberish detection ───────────────────────────────────────────────────
    # If most multi-char words are gibberish (no vowels / consonant clusters),
    # score 0–1 regardless of word count.
    g_ratio = _gibberish_ratio(words)
    if g_ratio >= 0.6:
        return {
            "score": 0 if g_ratio >= 0.9 else 1,
            "max_score": 10,
            "strengths": [],
            "improvements": [
                f"Your answer appears to contain random or meaningless characters ('{answer_stripped[:30]}').",
                "Write a substantive explanation that addresses the question directly.",
                "Use proper English words and technical terms relevant to the topic.",
            ],
            "improved_answer": _sample_answer(question),
            "evaluation_mode": "demo",
        }

    # ── Relevance check ───────────────────────────────────────────────────────
    q_lower = question.lower()
    relevance_score = _check_relevance(q_lower, answer_stripped.lower(), words)

    # ── Length / substance signal ─────────────────────────────────────────────
    # Target: 50–200 words for a good answer
    if word_count < 15:
        length_score = 1
    elif word_count < 30:
        length_score = 2
    elif word_count < 60:
        length_score = 3
    elif word_count <= 200:
        length_score = 4
    else:
        length_score = 4  # no bonus for padding

    # ── Structure signals ─────────────────────────────────────────────────────
    structure_score = 0
    if any(w in answer_stripped.lower() for w in ["for example", "such as", "e.g.", "instance", "specifically"]):
        structure_score += 1
    if any(w in answer_stripped.lower() for w in ["because", "therefore", "this means", "which allows", "as a result"]):
        structure_score += 1
    if word_count >= 40 and len(re.findall(r"[.!?]", answer_stripped)) >= 2:
        structure_score += 1  # multiple complete sentences

    # ── Combined score ────────────────────────────────────────────────────────
    # Relevance is a hard gate: zero relevance → max score 2 (irrelevant answer).
    # This prevents a long irrelevant essay from scoring above 2.
    raw = relevance_score + length_score + structure_score
    if relevance_score == 0:
        raw = min(raw, 2)
    elif relevance_score == 1:
        raw = min(raw, 4)  # only partial relevance → cap at 4
    score = max(0, min(10, raw))

    # ── Build feedback ────────────────────────────────────────────────────────
    strengths: list[str] = []
    improvements: list[str] = []

    if relevance_score >= 3:
        strengths.append("Your answer is relevant to the question topic.")
    if relevance_score >= 4:
        strengths.append("You demonstrate awareness of key concepts in your response.")
    if length_score >= 3:
        strengths.append("You provided a reasonably detailed response.")
    if structure_score >= 2:
        strengths.append("Your answer uses examples or clear reasoning.")

    if relevance_score < 2:
        improvements.append("Your answer does not clearly address what was asked. Re-read the question and focus on the specific concept.")
    if relevance_score < 4:
        improvements.append("Include more technical keywords and specific concepts related to the question.")
    if word_count < 50:
        improvements.append(f"Your answer is only {word_count} words. Aim for 50–150 words with concrete examples.")
    if structure_score < 2:
        improvements.append("Add specific examples (e.g., 'For example, when I...') and explain your reasoning.")
    if score < 6:
        improvements.append("Study the underlying concept more deeply before your interview.")

    if not strengths:
        strengths.append("You attempted to answer the question.")
    if not improvements:
        improvements.append("Consider adding quantified outcomes or metrics to strengthen your answer.")

    return {
        "score": score,
        "max_score": 10,
        "strengths": strengths[:3],
        "improvements": improvements[:3],
        "improved_answer": _sample_answer(question),
        "evaluation_mode": "demo",
    }


def _is_gibberish_word(word: str) -> bool:
    """
    Heuristic: a word is likely gibberish if it has poor consonant/vowel structure.
    Real English words tend to have vowels mixed with consonants.
    Pure consonant clusters of 4+ chars are almost always gibberish.
    """
    if len(word) < 3:
        return False  # too short to judge
    vowels = set("aeiou")
    vowel_count = sum(1 for c in word if c in vowels)
    # No vowels at all in a word ≥ 4 chars → gibberish
    if len(word) >= 4 and vowel_count == 0:
        return True
    # Consecutive consonants ≥ 4 → likely gibberish (e.g. "sdfg", "qwrt")
    consonant_run = 0
    for c in word:
        if c not in vowels:
            consonant_run += 1
            if consonant_run >= 4:
                return True
        else:
            consonant_run = 0
    return False


def _gibberish_ratio(words: list[str]) -> float:
    """Return fraction of words that appear to be gibberish (0.0–1.0)."""
    if not words:
        return 1.0
    # Only judge words ≥ 3 chars
    candidates = [w for w in words if len(w) >= 3]
    if not candidates:
        return 0.0
    gibberish_count = sum(1 for w in candidates if _is_gibberish_word(w))
    return gibberish_count / len(candidates)


def _check_relevance(question: str, answer: str, answer_words: list[str]) -> int:
    """
    Return a relevance score 0–4 based on keyword overlap and concept signals.

    Uses whole-word matching (not substring) to prevent false positives
    like 'n' matching inside 'mnop'.
    """
    answer_set = set(answer_words)

    # Direct keyword overlap: words from question that also appear in answer
    # Exclude stop words and very short words (≥ 4 chars only)
    stop = {"what", "when", "where", "which", "explain", "describe", "tell",
            "your", "have", "does", "work", "works", "used", "uses", "make"}
    q_words = set(re.findall(r"\b\w{4,}\b", question)) - stop
    overlap = len(q_words & answer_set)

    # Domain concept signals — check with WHOLE-WORD prefix regex in answer string.
    # Using \bsignal (prefix) instead of \bsignal\b to handle inflections:
    #   "challenging" matches signal "challenge"
    #   "buckets" matches signal "bucket"
    # Still requires word-boundary start to avoid matching inside longer unrelated words.
    concept_hits = 0
    for key, signals in _TECH_SIGNALS.items():
        # Check if key (whole word) appears in question
        key_match = any(
            re.search(r"\b" + re.escape(k), question)
            for k in key.split()
            if len(k) >= 3  # skip single-char / two-char keys
        )
        if key_match:
            hits = sum(
                1 for s in signals
                if re.search(r"\b" + re.escape(s), answer)
            )
            concept_hits += min(hits, 2)

    # ── Behavioral / situational question detector ────────────────────────────
    # Questions about experiences, challenges, or STAR-format scenarios don't
    # contain the word "behavioral" — detect them by their content keywords
    # and check for STAR method signals in the answer.
    _BEHAVIORAL_Q_PATTERNS = [
        "challenging", "challenge", "obstacle", "overcame", "overcome",
        "time you", "tell me about", "describe a", "example of",
        "situation", "experience", "conflict", "leadership", "teamwork",
        "failure", "success", "proud", "difficult",
    ]
    _STAR_SIGNALS = ["situation", "task", "action", "result", "outcome", "challenge"]

    is_behavioral_q = any(pat in question for pat in _BEHAVIORAL_Q_PATTERNS)
    if is_behavioral_q:
        star_hits = sum(
            1 for s in _STAR_SIGNALS
            if re.search(r"\b" + re.escape(s), answer)
        )
        concept_hits += min(star_hits, 2)

    raw = min(overlap, 2) + min(concept_hits, 2)
    return min(raw, 4)


def _sample_answer(question: str) -> str:
    """Return a generic model-answer hint based on the question content."""
    q = question.lower()
    if "supervised" in q or "unsupervised" in q:
        return ("Supervised learning uses labeled data where the target output is known (e.g., classification, regression). "
                "Unsupervised learning finds hidden patterns in unlabeled data (e.g., clustering with K-Means, dimensionality reduction). "
                "Example: predicting house prices is supervised; grouping customers by behaviour is unsupervised.")
    if "rest" in q or "graphql" in q:
        return ("REST uses multiple endpoints (one per resource) and relies on HTTP verbs (GET/POST/PUT/DELETE). "
                "GraphQL exposes a single endpoint and lets clients request exactly the fields they need, "
                "reducing over-fetching. REST is simpler; GraphQL is more flexible for complex, nested data.")
    if "big o" in q or "complexity" in q:
        return ("Big O notation describes the worst-case time or space complexity of an algorithm as input grows. "
                "O(1) is constant, O(log n) is logarithmic (binary search), O(n) is linear (traversal), "
                "O(n²) is quadratic (nested loops). It helps compare algorithms and predict performance at scale.")
    if "solid" in q:
        return ("SOLID stands for: Single Responsibility (a class has one reason to change), "
                "Open/Closed (open for extension, closed for modification), "
                "Liskov Substitution (subtypes must be substitutable for base types), "
                "Interface Segregation (small focused interfaces), "
                "Dependency Inversion (depend on abstractions, not concretions).")
    if "hash" in q:
        return ("A hash table maps keys to values using a hash function that computes an array index. "
                "On collision (two keys map to the same index), it uses chaining (linked list) or open addressing. "
                "Average O(1) lookup, insertion, and deletion. Python's dict is a hash table.")
    if "tell me about yourself" in q:
        return ("Start with a concise professional summary (current role/education), highlight 2–3 relevant skills or "
                "achievements, then connect them to why you're excited about this specific role. Keep it to 90 seconds.")
    if "strength" in q or "weakness" in q:
        return ("For strengths, pick one directly relevant to the job with a concrete example. "
                "For weaknesses, choose a genuine area you're improving (not a disguised strength), "
                "and explain the steps you're actively taking — e.g., 'I sometimes over-engineer solutions; "
                "I now set time-boxes and review scope before coding.'")
    if "star" in q or "situation" in q or "behavioral" in q or "challenge" in q or "time you" in q:
        return ("Use the STAR method: Situation (brief context), Task (your specific responsibility), "
                "Action (concrete steps you personally took), Result (measurable outcome). "
                "Example: 'In my internship (S), I was tasked with reducing API latency (T). "
                "I profiled the queries and added caching (A), cutting response time by 40% (R).'")
    # Generic fallback
    return ("A strong answer directly addresses the question, explains the core concept accurately, "
            "provides a specific real-world example, and closes with the impact or lesson learned. "
            "Aim for 3–5 clear sentences using the STAR method where applicable.")


class GraniteClient:
    """Thin wrapper around IBM watsonx.ai ModelInference."""

    def __init__(self) -> None:
        self._model: Any = None
        self._demo = config.DEMO_MODE or not IBM_SDK_OK

        if not self._demo:
            try:
                credentials = Credentials(
                    url=config.WATSONX_URL,
                    api_key=config.WATSONX_API_KEY,
                )
                client = APIClient(credentials)
                self._model = ModelInference(
                    model_id=config.GRANITE_MODEL_ID,
                    api_client=client,
                    project_id=config.WATSONX_PROJECT_ID,
                    params=config.GRANITE_PARAMS,
                )
            except Exception as exc:  # noqa: BLE001
                self._demo = True
                self._init_error = str(exc)

    @property
    def is_demo(self) -> bool:
        return self._demo

    # ── Core generate ────────────────────────────────────────────────────────
    def generate(self, prompt: str) -> str:
        if self._demo:
            return ""  # demo path bypasses generate() entirely
        try:
            response = self._model.generate_text(prompt=prompt)
            return response
        except Exception as exc:  # noqa: BLE001
            return f"[Error calling IBM Granite: {exc}]"

    # ── High-level helpers ───────────────────────────────────────────────────
    def generate_questions(
        self,
        role: str,
        job_description: str,
        skills: list[str],
        question_type: str = "technical",
        count: int = 5,
        rag_context: str = "",
    ) -> list[str]:
        if self._demo:
            pool = _DEMO_QUESTIONS.get(question_type, _DEMO_QUESTIONS["technical"])
            return pool[:count]

        skills_str = ", ".join(skills[:15]) if skills else "general skills"
        prompt = textwrap.dedent(f"""
            You are an expert technical interviewer.
            
            Context from knowledge base:
            {rag_context or "No additional context."}
            
            Job Role: {role}
            Job Description: {job_description[:600]}
            Candidate Skills: {skills_str}
            Question Type: {question_type}
            
            Generate exactly {count} unique, specific {question_type} interview questions
            for the above role. Each question should be on its own line starting with a number.
            Focus on practical, insightful questions that assess real competence.
            
            Questions:
        """).strip()

        raw = self.generate(prompt)
        return self._parse_numbered_list(raw, count)

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        role: str,
        rag_context: str = "",
    ) -> dict[str, Any]:
        # Demo mode: genuine rule-based evaluation — NOT a fixed score
        if self._demo:
            result = _demo_evaluate(question, answer)
            result["evaluation_mode"] = "demo"
            return result

        # IBM Granite: strict evaluation with explicit scoring rubric
        answer_words = len(answer.strip().split())
        prompt = textwrap.dedent(f"""
            You are a strict interview coach. Evaluate the candidate's answer honestly.

            SCORING RUBRIC (score integer 0–10):
              0   = empty, single character, or completely irrelevant (e.g. "I", "yes", "ok")
              1–2 = answer exists but shows no understanding of the question
              3–4 = partial: mentions some relevant concepts but incomplete or inaccurate
              5–6 = adequate: covers main idea but lacks examples or depth
              7–8 = good: accurate, specific, includes examples or reasoning
              9–10 = excellent: thorough, technically precise, well-structured with examples

            Context from knowledge base:
            {rag_context or "No additional context."}

            Role: {role}
            Question: {question}
            Candidate's Answer ({answer_words} words): {answer}

            IMPORTANT:
            - If the answer is "I", "yes", "no", or any single-word/filler response, score it 0–1.
            - Do NOT reward length alone. A padded irrelevant answer should score 1–3.
            - Be specific in feedback: name the exact concept that is missing or incorrect.

            Respond ONLY with valid JSON — no explanation outside the JSON:
            {{
              "score": <integer 0-10>,
              "max_score": 10,
              "strengths": ["<specific strength or empty list if none>"],
              "improvements": ["<specific missing concept or error>", "<another improvement>"],
              "improved_answer": "<A concise model answer in 3-5 sentences>",
              "evaluation_mode": "ibm_granite"
            }}
        """).strip()

        raw = self.generate(prompt)
        _fallback = _demo_evaluate(question, answer)  # fall back to rule-based if parse fails
        _fallback["evaluation_mode"] = "demo_fallback"
        return self._parse_json_response(raw, _fallback)

    def analyze_skill_gaps(
        self,
        resume_skills: list[str],
        required_skills: list[str],
        role: str,
    ) -> dict[str, Any]:
        if self._demo:
            missing = [s for s in required_skills if s not in resume_skills]
            return {
                "matching_skills": [s for s in resume_skills if s in required_skills],
                "missing_skills": missing[:8],
                "recommendations": [
                    f"Study {s} through online courses or projects." for s in missing[:4]
                ],
                "overall_fit_score": max(
                    0,
                    min(100, int(100 * len([s for s in resume_skills if s in required_skills]) / max(len(required_skills), 1))),
                ),
            }

        prompt = textwrap.dedent(f"""
            You are a career counselor performing a skill gap analysis.
            
            Target Role: {role}
            Candidate Skills: {", ".join(resume_skills[:20])}
            Required Skills for Role: {", ".join(required_skills[:20])}
            
            Respond ONLY with valid JSON:
            {{
              "matching_skills": ["<skill>"],
              "missing_skills": ["<skill>"],
              "recommendations": ["<specific action to close gap>"],
              "overall_fit_score": <integer 0-100>
            }}
        """).strip()

        raw = self.generate(prompt)
        return self._parse_json_response(raw, {
            "matching_skills": [],
            "missing_skills": [],
            "recommendations": [],
            "overall_fit_score": 50,
        })

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_numbered_list(text: str, count: int) -> list[str]:
        lines = text.strip().splitlines()
        questions: list[str] = []
        for line in lines:
            line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
            if len(line) > 15:
                questions.append(line)
            if len(questions) >= count:
                break
        # Pad with fallback if model was stingy
        fallback = _DEMO_QUESTIONS["technical"]
        while len(questions) < count:
            questions.append(fallback[len(questions) % len(fallback)])
        return questions[:count]

    @staticmethod
    def _parse_json_response(text: str, fallback: dict) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                # Clamp score to valid range
                if "score" in parsed:
                    parsed["score"] = max(0, min(10, int(parsed["score"])))
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return fallback.copy()


# Singleton
_client: GraniteClient | None = None


def get_client() -> GraniteClient:
    global _client
    if _client is None:
        _client = GraniteClient()
    return _client
