from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "problem-statement" / "data"
INDEX_DIR: Path = BASE_DIR / "storage" / "chroma"

ModelBackend = Literal["openai", "ollama"]

MODEL_BACKEND: ModelBackend = "openai"
LLM_MODEL_NAME: str = "gpt-4.1-mini"
EMBED_MODEL_NAME: str = "text-embedding-3-small"

CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 120

RETRIEVAL_TOP_K: int = 8
