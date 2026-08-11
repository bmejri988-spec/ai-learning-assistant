PROJECT_NAME = "AI Learning Assistant API"
PROJECT_VERSION = "0.3.0"
LOG_LEVEL = "INFO"


# ============================================================
# RAG
# ============================================================

VECTOR_DB_PATH = "data/vector_store"
RAG_COLLECTION_NAME = "study_documents"

RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 150
RAG_DEFAULT_TOP_K = 3

UPLOADS_DIR = "data/uploads"


# ============================================================
# Ollama
# ============================================================

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Model used for RAG answers
OLLAMA_MODEL = "llama3.2:3b"

# Model used by the Agent
AGENT_OLLAMA_MODEL = "medragondot/llama-3.2-3b-thinking"


# ============================================================
# Agent
# ============================================================

RAG_API_BASE_URL = "http://127.0.0.1:8000"
AGENT_PLANNER_TYPE = "llm"