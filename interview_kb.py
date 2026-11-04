"""
interview_kb.py – Curated interview knowledge base content.

This module provides structured Q&A pairs and preparation tips for common
interview topics. The RAG engine indexes this content so Granite can
retrieve relevant passages before generating questions or evaluating answers.
"""
from __future__ import annotations

# Each entry: {"id": str, "category": str, "role_tags": [str], "content": str}
KNOWLEDGE_BASE: list[dict] = [
    # ── Data Science / ML ────────────────────────────────────────────────────
    {
        "id": "ds_001",
        "category": "technical",
        "role_tags": ["data scientist", "ml engineer", "ai engineer"],
        "content": (
            "Bias-Variance Tradeoff: High bias causes underfitting; high variance "
            "causes overfitting. The goal is to find the sweet spot via cross-validation, "
            "regularization (L1/L2), and ensemble methods. "
            "Regularization techniques: Lasso (L1) drives sparse solutions; "
            "Ridge (L2) shrinks coefficients uniformly."
        ),
    },
    {
        "id": "ds_002",
        "category": "technical",
        "role_tags": ["data scientist", "ml engineer"],
        "content": (
            "Cross-validation: k-fold CV splits data into k subsets, trains on k-1 folds, "
            "validates on 1. Helps estimate generalization. Stratified CV preserves class "
            "distribution—critical for imbalanced datasets. "
            "Evaluation metrics: Accuracy for balanced classes; F1/AUC-ROC for imbalanced."
        ),
    },
    {
        "id": "ds_003",
        "category": "technical",
        "role_tags": ["data scientist", "data engineer", "ml engineer"],
        "content": (
            "Feature engineering: Scaling (StandardScaler, MinMaxScaler), encoding "
            "(one-hot, ordinal, target encoding), and feature selection (RFE, mutual info). "
            "Handling missing data: mean/median/mode imputation vs. model-based vs. deletion. "
            "Outlier detection: IQR method, Z-score, Isolation Forest."
        ),
    },
    # ── Software Engineering ─────────────────────────────────────────────────
    {
        "id": "swe_001",
        "category": "technical",
        "role_tags": ["software engineer", "backend engineer", "full stack"],
        "content": (
            "SOLID Principles: Single Responsibility, Open/Closed, Liskov Substitution, "
            "Interface Segregation, Dependency Inversion. "
            "Design Patterns: Factory, Singleton, Observer, Strategy, Decorator. "
            "Use dependency injection to decouple modules and improve testability."
        ),
    },
    {
        "id": "swe_002",
        "category": "technical",
        "role_tags": ["software engineer", "backend engineer"],
        "content": (
            "REST API design: Use nouns for endpoints (/users, /orders), HTTP verbs for "
            "actions (GET, POST, PUT, DELETE), status codes (200, 201, 400, 401, 404, 500). "
            "Pagination with limit/offset or cursor. Versioning via URL path (/v1/) or headers."
        ),
    },
    {
        "id": "swe_003",
        "category": "technical",
        "role_tags": ["software engineer", "backend engineer", "full stack"],
        "content": (
            "Database design: Normalization (1NF–3NF) reduces redundancy. "
            "Indexing: B-tree for range queries, Hash for equality. "
            "ACID properties: Atomicity, Consistency, Isolation, Durability. "
            "SQL vs NoSQL: Relational for structured data with complex queries; "
            "NoSQL for flexible schemas and horizontal scaling."
        ),
    },
    # ── Behavioral / HR ──────────────────────────────────────────────────────
    {
        "id": "beh_001",
        "category": "behavioral",
        "role_tags": ["all"],
        "content": (
            "STAR Method for behavioral questions: Situation (set context briefly), "
            "Task (your specific responsibility), Action (concrete steps YOU took), "
            "Result (quantifiable outcome). "
            "Common behavioral themes: conflict resolution, failure & learning, "
            "leadership, collaboration, handling ambiguity, meeting deadlines."
        ),
    },
    {
        "id": "hr_001",
        "category": "hr",
        "role_tags": ["all"],
        "content": (
            "Common HR questions and strong answer frameworks: "
            "'Tell me about yourself' → 2-min pitch: education → experience → current goal. "
            "'Strengths' → pick a real strength relevant to the role, give evidence. "
            "'Weakness' → genuine area for growth + concrete steps you're taking. "
            "'Why this company' → research mission, products, culture; align with your goals."
        ),
    },
    # ── Cloud / DevOps ───────────────────────────────────────────────────────
    {
        "id": "cloud_001",
        "category": "technical",
        "role_tags": ["cloud engineer", "devops", "sre", "platform engineer"],
        "content": (
            "Kubernetes: Pods, Deployments, Services (ClusterIP, NodePort, LoadBalancer), "
            "ConfigMaps, Secrets, PersistentVolumes. "
            "Docker: Images, containers, multi-stage builds, Docker Compose. "
            "CI/CD: GitHub Actions, Jenkins pipelines—build → test → lint → deploy stages."
        ),
    },
    {
        "id": "cloud_002",
        "category": "technical",
        "role_tags": ["cloud engineer", "devops", "data engineer"],
        "content": (
            "AWS core services: EC2 (compute), S3 (object storage), RDS (managed SQL), "
            "Lambda (serverless), ECS/EKS (containers), CloudFormation (IaC). "
            "IBM Cloud: Watson Studio, watsonx.ai, Cloud Object Storage, Kubernetes Service. "
            "Security: IAM policies, least-privilege, secrets management (Vault, AWS Secrets Manager)."
        ),
    },
    # ── Python / Programming ─────────────────────────────────────────────────
    {
        "id": "py_001",
        "category": "technical",
        "role_tags": ["python developer", "backend engineer", "data scientist", "ml engineer"],
        "content": (
            "Python internals: GIL limits true threading for CPU-bound tasks; use multiprocessing. "
            "Generators and itertools for memory-efficient data pipelines. "
            "Decorators for cross-cutting concerns (logging, retry, caching). "
            "Context managers (__enter__/__exit__) for resource cleanup. "
            "Type hints + mypy for static analysis."
        ),
    },
    {
        "id": "py_002",
        "category": "technical",
        "role_tags": ["python developer", "data scientist"],
        "content": (
            "Pandas tips: Use vectorized operations over loops. "
            "groupby + agg for aggregations. merge/join for combining DataFrames. "
            "apply is slower than built-in methods. Use .loc for label-based, .iloc for positional access. "
            "NumPy: broadcasting, universal functions (ufuncs), vectorized math."
        ),
    },
    # ── Project-based ────────────────────────────────────────────────────────
    {
        "id": "proj_001",
        "category": "project",
        "role_tags": ["all"],
        "content": (
            "Project interview questions framework: "
            "1. Architecture overview – explain components and data flow. "
            "2. Technology choices – why did you pick X over Y? "
            "3. Challenges faced and technical decisions made. "
            "4. Scalability and performance considerations. "
            "5. Testing strategy – unit, integration, end-to-end. "
            "6. Outcome / impact – quantify if possible."
        ),
    },
    # ── System Design ────────────────────────────────────────────────────────
    {
        "id": "sys_001",
        "category": "technical",
        "role_tags": ["software engineer", "backend engineer", "full stack", "sre"],
        "content": (
            "System design framework: Requirements (functional + non-functional) → "
            "Capacity estimation → High-level design → Detailed design → "
            "Scalability (horizontal scaling, caching, CDN) → Reliability (replication, failover) → "
            "Monitoring (metrics, logs, alerts). "
            "Common patterns: Load balancer, message queue, event-driven architecture, CQRS."
        ),
    },
    # ── Interview preparation tips ───────────────────────────────────────────
    {
        "id": "prep_001",
        "category": "preparation",
        "role_tags": ["all"],
        "content": (
            "Interview preparation checklist: "
            "- Research the company (products, recent news, culture). "
            "- Prepare 3-5 STAR stories covering leadership, failure, success, collaboration. "
            "- Review fundamentals for your target role. "
            "- Practice LeetCode (Easy/Medium) for coding rounds. "
            "- Prepare questions to ask the interviewer. "
            "- Mock interviews with peers or tools like SkillSync AI."
        ),
    },
]

# Flat list of just the text content for easy embedding
ALL_DOCUMENTS: list[str] = [item["content"] for item in KNOWLEDGE_BASE]
ALL_METADATA: list[dict] = [
    {"id": item["id"], "category": item["category"], "role_tags": item["role_tags"]}
    for item in KNOWLEDGE_BASE
]
