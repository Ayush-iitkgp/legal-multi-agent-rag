from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import asyncio
from langchain_core.documents import Document

from src.config import DATA_DIR


@dataclass
class RawDoc:
    path: Path
    text: str


async def load_raw_docs(data_dir: Path | None = None) -> List[RawDoc]:
    base = data_dir or DATA_DIR
    docs: List[RawDoc] = []
    for path in sorted(base.glob("*.txt")):
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        docs.append(RawDoc(path=path, text=text))
    return docs


def simple_clause_chunk(text: str, source_name: str) -> List[Document]:
    lines = text.splitlines()
    chunks: List[Document] = []
    current_lines: List[str] = []

    def flush() -> None:
        if not current_lines:
            return
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append(
                Document(
                    page_content=content,
                    metadata={"source": source_name},
                )
            )

    for line in lines:
        if line.strip().startswith(tuple(str(i) + "." for i in range(1, 10))):
            flush()
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()
    return chunks


def chunk_corpus(raw_docs: Iterable[RawDoc]) -> List[Document]:
    all_chunks: List[Document] = []
    for doc in raw_docs:
        all_chunks.extend(simple_clause_chunk(doc.text, source_name=doc.path.name))
    return all_chunks

