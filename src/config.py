from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "problem-statement" / "data"
INDEX_DIR: Path = BASE_DIR / "storage" / "chroma"

ModelBackend = Literal["openai", "ollama", "gemini"]

MODEL_BACKEND: ModelBackend = "ollama"

# Default model names per backend.
OPENAI_CHAT_MODEL: str = "gpt-4.1-mini"
OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

OLLAMA_CHAT_MODEL: str = "qwen2.5:7b"
OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

GEMINI_CHAT_MODEL: str = "gemini-2.0-flash"
GEMINI_EMBED_MODEL: str = "models/text-embedding-004"

# Convenience aliases resolved based on MODEL_BACKEND.
if MODEL_BACKEND == "openai":
    LLM_MODEL_NAME: str = OPENAI_CHAT_MODEL
    EMBED_MODEL_NAME: str = OPENAI_EMBED_MODEL
elif MODEL_BACKEND == "gemini":
    LLM_MODEL_NAME = GEMINI_CHAT_MODEL
    EMBED_MODEL_NAME = GEMINI_EMBED_MODEL
else:
    LLM_MODEL_NAME = OLLAMA_CHAT_MODEL
    EMBED_MODEL_NAME = OLLAMA_EMBED_MODEL

CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 120

RETRIEVAL_TOP_K: int = 8
