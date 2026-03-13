from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src import config


def make_chat_model() -> BaseChatModel:
    return ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        temperature=0.1,
    )


def make_embeddings() -> Embeddings:
    return OpenAIEmbeddings(model=config.EMBED_MODEL_NAME)

