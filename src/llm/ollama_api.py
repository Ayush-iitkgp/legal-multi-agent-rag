from __future__ import annotations

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src import config


def make_chat_model() -> BaseChatModel:
    return ChatOllama(
        model=config.LLM_MODEL_NAME,
        temperature=0.1,
    )


def make_embeddings() -> Embeddings:
    return OllamaEmbeddings(model=config.EMBED_MODEL_NAME)

