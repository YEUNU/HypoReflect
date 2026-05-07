import os

class RAGConfig:
    DATASET = os.environ.get("RAG_DATASET", "financebench").strip().lower() or "financebench"

    # --- Infrastructure (Actual ports identified) ---
    VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:28000/v1")
    VLLM_EMBED_URL = os.environ.get("VLLM_EMBED_URL", "http://localhost:18082/v1")
    VLLM_OCR_URL = os.environ.get("VLLM_OCR_URL", "http://localhost:28001/v1")
    VLLM_RERANK_URL = os.environ.get("VLLM_RERANK_URL", "http://localhost:18083/v1")
    
    # --- LLM Settings ---
    DEFAULT_MODEL = os.environ.get("VLLM_SERVED_MODEL_NAME", "generation-model")
    EMBEDDING_MODEL = os.environ.get("VLLM_SERVED_EMBED_MODEL_NAME", "embedding-model")
    OCR_MODEL = os.environ.get("VLLM_SERVED_OCR_MODEL_NAME", "ocr-model")
    
    # --- Evaluation (LLM-as-a-judge) ---
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    EVAL_MODEL = os.environ.get(
        "EVAL_MODEL",
        os.environ.get("VLLM_SERVED_MODEL_NAME", "generation-model")
    )  # Default to local served model unless explicitly overridden.
    HALLUCINATION_EVAL_MODEL = os.environ.get("HALLUCINATION_EVAL_MODEL", "gpt-5.5-2026-04-23")

    # --- Per-stage model overrides (empty string = use DEFAULT_MODEL) ---
    # Supports local vLLM model names or OpenAI model names (gpt-*, o1-*, o3-*, o4-*)
    PERCEPTION_MODEL  = os.environ.get("PERCEPTION_MODEL",  "")
    PLANNING_MODEL    = os.environ.get("PLANNING_MODEL",    "")
    EXECUTION_MODEL   = os.environ.get("EXECUTION_MODEL",   "")
    REFLECTION_MODEL  = os.environ.get("REFLECTION_MODEL",  "")
    REFINEMENT_MODEL  = os.environ.get("REFINEMENT_MODEL",  "")
    
    # --- Common Service Settings ---
    RETRY_COUNT = int(os.environ.get("RAG_RETRY_COUNT", "3"))
    RETRY_DELAY = float(os.environ.get("RAG_RETRY_DELAY", "2.0"))
    LLM_REQUEST_TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT", "300"))
    LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "5"))
    LLM_RETRY_DELAY = float(os.environ.get("LLM_RETRY_DELAY", "2.0"))
    MAX_CONTEXT_LENGTH = 65536
    MAX_OUTPUT_TOKENS = 16384
    MAX_EMBEDDING_LENGTH = int(os.environ.get("MAX_EMBEDDING_LENGTH", "16384"))
    
    # --- RAG & Indexing Settings ---
    MAX_CONCURRENT_LLM_CALLS = int(os.environ.get("MAX_CONCURRENT_LLM_CALLS", "10"))
    EMBEDDING_BATCH_SIZE = int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "32"))
    EMBEDDING_DIMENSIONS = int(os.environ.get("NEO4J_VECTOR_DIMENSIONS", "1024"))
    NEO4J_BATCH_SIZE = int(os.environ.get("NEO4J_BATCH_SIZE", "10"))
    
    # --- Search & Ranking (RRF) ---
    RRF_K_CONSTANT = int(os.environ.get("RAG_RRF_K", "60"))
    # After the reranker fix (raw-string Qwen3-Reranker prompt) the vector
    # channel actually carries semantic signal again; bumping vector above
    # text restores the calibration we lost when the reranker collapsed to
    # near-uniform scores.
    RRF_VECTOR_WEIGHT = float(os.environ.get("RAG_RRF_VECTOR_WEIGHT", "1.3"))
    RRF_TEXT_WEIGHT = float(os.environ.get("RAG_RRF_TEXT_WEIGHT", "1.0"))
    VECTOR_SEARCH_LIMIT = int(os.environ.get("RAG_VECTOR_SEARCH_LIMIT", "20"))
    TEXT_SEARCH_LIMIT = int(os.environ.get("RAG_TEXT_SEARCH_LIMIT", "20"))
    
    # --- Thresholds & Traversal ---
    HOP_THRESHOLD = float(os.environ.get("RAG_HOP_THRESHOLD", "0.82"))
    SIMILARITY_THRESHOLD = float(os.environ.get("RAG_SIMILARITY_THRESHOLD", "0.65"))
    HOP_DECAY = float(os.environ.get("RAG_HOP_DECAY", "0.85"))
    RERANKER_THRESHOLD = float(os.environ.get("RERANKER_THRESHOLD", "0.4"))
    RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "4"))
    RERANK_QUERY_MAX_TOKENS = int(os.environ.get("RERANK_QUERY_MAX_TOKENS", "256"))
    RERANK_DOC_MAX_TOKENS = int(os.environ.get("RERANK_DOC_MAX_TOKENS", "2800"))
    RERANK_OVERFLOW_DOC_MAX_TOKENS = int(os.environ.get("RERANK_OVERFLOW_DOC_MAX_TOKENS", "1800"))
    
    # --- Agent Settings ---
    MAX_AGENT_TURNS = 6
    MAX_REFINEMENT_ATTEMPTS = int(os.environ.get("RAG_MAX_REFINEMENT", "2"))
    STRICT_CITATION_CHECK = os.environ.get("RAG_STRICT_CITATION", "True").lower() == "true"
    
    # --- OCR & PDF Processing Settings ---
    OCR_TEMPERATURE = float(os.environ.get("VLLM_OCR_TEMPERATURE", "0.2"))
    OCR_TOP_P = float(os.environ.get("VLLM_OCR_TOP_P", "0.9"))
    PDF_DPI = int(os.environ.get("RAG_PDF_DPI", "200"))
    PDF_MAX_DIM = int(os.environ.get("RAG_PDF_MAX_DIM", "1540"))
    PDF_BATCH_SIZE = int(os.environ.get("RAG_PDF_BATCH_SIZE", "5"))
    PDF_CONVERT_THREADS = int(os.environ.get("RAG_PDF_THREADS", "4"))
    MAX_PARALLEL_PAGES = int(os.environ.get("RAG_MAX_PARALLEL_PAGES", "4"))
    MAX_PARALLEL_PDFS = int(os.environ.get("RAG_MAX_PARALLEL_PDFS", "4"))
    STREAMING_WINDOW_SIZE = int(os.environ.get("RAG_STREAMING_WINDOW_SIZE", "10"))
    
    # --- Indexing Pipeline Settings ---
    PAGE_SIMILARITY_THRESHOLD = float(os.environ.get("RAG_PAGE_SIMILARITY_THRESHOLD", "0.5"))
    SENTENCE_COHESION_THRESHOLD = float(os.environ.get("RAG_SENTENCE_COHESION_THRESHOLD", "0.65"))
    MILESTONE_INTERVAL = int(os.environ.get("RAG_MILESTONE_INTERVAL", "5"))
    INDEXING_TEMPERATURE = float(os.environ.get("RAG_INDEXING_TEMPERATURE", "0.1"))
    MIN_CHUNK_SENTENCES = int(os.environ.get("RAG_MIN_CHUNK_SENTENCES", "2"))
    HOP_LINK_LIMIT = int(os.environ.get("RAG_HOP_LINK_LIMIT", "5"))
    CONTEXT_FETCH_LIMIT = int(os.environ.get("RAG_CONTEXT_FETCH_LIMIT", "10"))
    GRAPH_SEARCH_LIMIT = int(os.environ.get("RAG_GRAPH_SEARCH_LIMIT", "10"))
    DEFAULT_TOP_K = int(os.environ.get("RAG_DEFAULT_TOP_K", "8"))
    FULLTEXT_ANALYZER = os.environ.get("NEO4J_FULLTEXT_ANALYZER", "english")
    RECREATE_TEXT_INDEX = os.environ.get("RAG_RECREATE_TEXT_INDEX", "False").lower() == "true"

    # --- Retrieval Robustness ---
    ENABLE_QUERY_REWRITE = os.environ.get("RAG_ENABLE_QUERY_REWRITE", "True").lower() == "true"
    QUERY_REWRITE_COUNT = int(os.environ.get("RAG_QUERY_REWRITE_COUNT", "2"))
    QUERY_REWRITE_WEIGHT = float(os.environ.get("RAG_QUERY_REWRITE_WEIGHT", "0.85"))
    BOILERPLATE_PENALTY_WEIGHT = float(os.environ.get("RAG_BOILERPLATE_PENALTY_WEIGHT", "0.25"))
    META_BOOST_WEIGHT = float(os.environ.get("RAG_META_BOOST_WEIGHT", "0.50"))

    # --- Benchmark Gate (Optional Quality Guardrail) ---
    BENCHMARK_GATE_ENABLED = os.environ.get("RAG_BENCHMARK_GATE", "False").lower() == "true"
    BENCHMARK_MAX_AVG_LATENCY = float(os.environ.get("RAG_GATE_MAX_LATENCY", "45.0"))
    BENCHMARK_MIN_LLM_JUDGE = float(os.environ.get("RAG_GATE_MIN_LLM_JUDGE", "0.55"))
    BENCHMARK_MIN_DOC_MATCH = float(os.environ.get("RAG_GATE_MIN_DOC_MATCH", "0.60"))
    BENCHMARK_MIN_F1 = float(os.environ.get("RAG_GATE_MIN_F1", "0.35"))
    BENCHMARK_MIN_SP_F1 = float(os.environ.get("RAG_GATE_MIN_SP_F1", "0.25"))

    # --- Ablation & Experimental Toggles ---
    ABLATION_TABLE_TO_TEXT = os.environ.get("RAG_ABLATION_TABLE", "True").lower() == "true"
    ABLATION_ADAPTIVE_CHUNKING = os.environ.get("RAG_ABLATION_CHUNKING", "True").lower() == "true"
    ABLATION_ROLLING_SUMMARY = os.environ.get("RAG_ABLATION_SUMMARY", "True").lower() == "true"
    ENABLE_AGENT_REFLECTION = os.environ.get("RAG_ENABLE_REFLECTION", "True").lower() == "true"

    # Predictive Knowledge Mapping channel ablations.
    # ABLATION_Q_MINUS / ABLATION_Q_PLUS gate whether the Q-/Q+ channels
    # participate in indexing (embedding storage) and retrieval (channel use).
    # Disabling Q+ also disables offline HOP edge construction, since HOP
    # selection is anchored on Q+ embeddings.
    ABLATION_Q_MINUS = os.environ.get("RAG_ABLATION_Q_MINUS", "True").lower() == "true"
    ABLATION_Q_PLUS = os.environ.get("RAG_ABLATION_Q_PLUS", "True").lower() == "true"

    # HOP construction mode: "offline" pre-builds edges at indexing time
    # (default, paper config). "runtime" skips offline HOP construction and
    # expands the frontier via Q+ ANN + cross-encoder rerank at query time.
    HOP_MODE = os.environ.get("RAG_HOP_MODE", "offline").strip().lower() or "offline"

    # --- Project Metadata ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
