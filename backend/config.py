from os import getenv

from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = getenv("PROJECT_NAME", "AI Learning Assistant API")
PROJECT_VERSION = getenv("PROJECT_VERSION", "0.3.0")
LOG_LEVEL = getenv("LOG_LEVEL", "INFO")
OPENAI_API_KEY = getenv("OPENAI_API_KEY", "")
MODEL_NAME = getenv("MODEL_NAME", "")
VECTOR_DB_PATH = getenv("VECTOR_DB_PATH", "data/vector_store")
DATABASE_PATH = getenv("DATABASE_PATH", "data/app.db")
RAG_COLLECTION_NAME = getenv("RAG_COLLECTION_NAME", "study_documents")
RAG_EMBEDDING_MODEL = getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_CHUNK_SIZE = int(getenv("RAG_CHUNK_SIZE", "1000"))
RAG_CHUNK_OVERLAP = int(getenv("RAG_CHUNK_OVERLAP", "150"))
UPLOADS_DIR = getenv("UPLOADS_DIR", "data/uploads")
RAG_DEFAULT_TOP_K = int(getenv("RAG_DEFAULT_TOP_K", "3"))
OLLAMA_BASE_URL = getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = getenv("OLLAMA_MODEL", "llama3.2:3b")
AGENT_OLLAMA_MODEL = getenv("AGENT_OLLAMA_MODEL", "medragondot/llama-3.2-3b-thinking")
RAG_API_BASE_URL = getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")
AGENT_PLANNER_TYPE = getenv("AGENT_PLANNER_TYPE", "deterministic")
