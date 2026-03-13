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
    document_type: str | None = None


def _infer_document_type(path: Path, text: str) -> str:
    """
    Use the filename (without extension) as the document_type.

    This makes it easy to reason about specific agreements like
    'vendor_services_agreement' or 'data_processing_agreement' directly
    in routing and analysis.
    """
    return path.stem


async def load_raw_docs(data_dir: Path | None = None) -> List[RawDoc]:
    """
    Load all raw text documents from the given data directory.

    The project structure under `problem-statement/data` can contain multiple
    agreements and helper files spread across subdirectories. To ensure that
    retrieval can surface relevant context from *all* of these sources, we
    recurse through the directory tree instead of only looking at the top level.
    """
    base = data_dir or DATA_DIR
    docs: List[RawDoc] = []
    for path in sorted(base.rglob("*.txt")):
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        document_type = _infer_document_type(path, text)
        docs.append(RawDoc(path=path, text=text, document_type=document_type))
    return docs


def simple_clause_chunk(
    text: str,
    source_name: str,
    document_type: str | None = None,
) -> List[Document]:
    lines = text.splitlines()
    chunks: List[Document] = []
    current_lines: List[str] = []

    def is_section_header(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        # Numbered headings: "1.", "2.1", etc.
        if stripped[0].isdigit() and "." in stripped[:4]:
            return True
        # Markdown-style headings: "#", "##", etc.
        if stripped.startswith(("#", "##", "###")):
            return True
        # Short all-caps headings: "TERMINATION", "GOVERNING LAW"
        if stripped.isupper() and 3 <= len(stripped) <= 80:
            return True
        return False

    for line in lines:
        if is_section_header(line):
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    first_line = content.splitlines()[0].strip()
                    # Derive a human-readable section title from the header line.
                    import re

                    heading = (
                        re.sub(r"^\d+(\.\d+)*\s*\.?\s*", "", first_line).strip()
                        or first_line
                    )
                    chunks.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": source_name,
                                "document_type": document_type,
                                "section_index": len(chunks) + 1,
                                "section_title": heading,
                            },
                        )
                    )
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            first_line = content.splitlines()[0].strip()
            import re

            heading = (
                re.sub(r"^\d+(\.\d+)*\s*\.?\s*", "", first_line).strip() or first_line
            )
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source_name,
                        "document_type": document_type,
                        "section_index": len(chunks) + 1,
                        "section_title": heading,
                    },
                )
            )
    return chunks


def chunk_corpus(raw_docs: Iterable[RawDoc]) -> List[Document]:
    all_chunks: List[Document] = []
    for doc in raw_docs:
        all_chunks.extend(
            simple_clause_chunk(
                doc.text,
                source_name=doc.path.name,
                document_type=doc.document_type,
            )
        )
    return all_chunks
