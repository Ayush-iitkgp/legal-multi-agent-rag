from __future__ import annotations

from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src import config
from src.llm import ollama_api, openai_api


Backend = Literal["openai", "ollama"]


def get_backend() -> Backend:
    return config.MODEL_BACKEND  # type: ignore[return-value]


def make_chat_model() -> BaseChatModel:
    backend = get_backend()
    if backend == "openai":
        return openai_api.make_chat_model()
    return ollama_api.make_chat_model()


def make_embeddings() -> Embeddings:
    backend = get_backend()
    if backend == "openai":
        return openai_api.make_embeddings()
    return ollama_api.make_embeddings()

