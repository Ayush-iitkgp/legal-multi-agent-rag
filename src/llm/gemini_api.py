from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src import config


def make_chat_model() -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL_NAME,
        temperature=0.1,
    )


def make_embeddings() -> Embeddings:
    return GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL_NAME)
