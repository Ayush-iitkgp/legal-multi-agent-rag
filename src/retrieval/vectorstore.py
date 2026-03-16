import logging
from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import INDEX_DIR

logger = logging.getLogger(__name__)


def build_vectorstore(
    docs: Iterable[Document],
    embeddings: Embeddings,
    persist_dir: Path | None = None,
) -> Chroma:
    path = persist_dir or INDEX_DIR
    path.mkdir(parents=True, exist_ok=True)
    doc_list = list(docs)

    logger.info("Building vector store with %d chunks. Sample entries:", len(doc_list))
    for i, d in enumerate(doc_list[:3]):
        preview = d.page_content[:120].replace("\n", " ")
        logger.info(
            "  [%d] source=%s | document_type=%s | section_index=%s | section_title=%s\n"
            "       content: %s...",
            i + 1,
            d.metadata.get("source"),
            d.metadata.get("document_type"),
            d.metadata.get("section_index"),
            d.metadata.get("section_title"),
            preview,
        )

    return Chroma.from_documents(
        documents=doc_list,
        embedding=embeddings,
        persist_directory=str(path),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vectorstore(embeddings: Embeddings, persist_dir: Path | None = None) -> Chroma:
    path = persist_dir or INDEX_DIR
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(path),
        collection_metadata={"hnsw:space": "cosine"},
    )
