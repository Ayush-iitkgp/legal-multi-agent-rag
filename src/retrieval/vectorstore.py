from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import INDEX_DIR


def build_vectorstore(
    docs: Iterable[Document],
    embeddings: Embeddings,
    persist_dir: Path | None = None,
) -> Chroma:
    path = persist_dir or INDEX_DIR
    path.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=list(docs),
        embedding=embeddings,
        persist_directory=str(path),
    )


def load_vectorstore(embeddings: Embeddings, persist_dir: Path | None = None) -> Chroma:
    path = persist_dir or INDEX_DIR
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(path),
    )
