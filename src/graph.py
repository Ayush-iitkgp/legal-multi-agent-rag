from dataclasses import dataclass, field
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.agents.core import (
    QueryAnalysis,
    analyze_query,
    answer_with_optional_risks,
)
from src.config import RETRIEVAL_TOP_K
from src.ingest.core import chunk_corpus, load_raw_docs
from src.llm.factory import make_embeddings
from src.retrieval.vectorstore import build_vectorstore, load_vectorstore


@dataclass
class GraphState:
    messages: List[BaseMessage] = field(default_factory=list)
    question: str | None = None
    query_analysis: QueryAnalysis | None = None
    is_risk_query: bool = False
    routed_doc_types: List[str] = field(default_factory=list)
    retrieved: List[Document] = field(default_factory=list)
    clauses: List[Document] = field(default_factory=list)
    final_answer: str | None = None


async def node_router(state: GraphState) -> GraphState:
    """Router / orchestrator: analyse query and decide which docs to focus on."""
    if not state.question:
        return state

    qa = await analyze_query(
        question=state.question,
        history=[m for m in state.messages],
    )
    state.query_analysis = qa
    state.is_risk_query = qa.query_type == "risk_summary"

    # Map high-level doc_targets from the classifier onto concrete document_type values.
    targets: set[str] = set()
    for label in qa.doc_targets:
        label_norm = label.strip().lower()
        if label_norm == "nda":
            targets.add("nda_acme_vendor")
        elif label_norm == "vendor_services":
            targets.add("vendor_services_agreement")
        elif label_norm == "service_level":
            targets.add("service_level_agreement")
        elif label_norm == "dpa":
            targets.add("data_processing_agreement")
        elif label_norm == "all":
            targets.update(
                {
                    "nda_acme_vendor",
                    "vendor_services_agreement",
                    "service_level_agreement",
                    "data_processing_agreement",
                }
            )

    if not targets:
        # Fallback: consider all agreement types when routing is ambiguous.
        targets.update(
            {
                "nda_acme_vendor",
                "vendor_services_agreement",
                "service_level_agreement",
                "data_processing_agreement",
            }
        )

    state.routed_doc_types = sorted(targets)
    return state


async def node_clause_extractor(state: GraphState) -> GraphState:
    """Clause extractor: map-reduce style retrieval over routed documents."""
    embeddings = make_embeddings()
    try:
        vs = load_vectorstore(embeddings)
    except Exception:
        raw = await load_raw_docs()
        chunks = chunk_corpus(raw)
        vs = build_vectorstore(chunks, embeddings)

    # Use a wider retrieval fan-out for risk queries to allow cross-agreement analysis.
    is_risk = state.is_risk_query
    top_k = RETRIEVAL_TOP_K * 2 if is_risk else RETRIEVAL_TOP_K
    retriever = vs.as_retriever(search_kwargs={"k": top_k})

    base_query = state.question or ""
    all_docs: List[Document] = retriever.invoke(base_query)

    # Reduce phase: de-duplicate and filter to routed document types, if provided.
    seen_keys: set[tuple[str, str]] = set()
    reduced: List[Document] = []
    # For risk queries, allow clauses from all agreements; otherwise, respect routing.
    allowed_types = set() if is_risk else set(state.routed_doc_types)

    def _key(d: Document) -> tuple[str, str]:
        source = str(d.metadata.get("source", "unknown"))
        section = str(d.metadata.get("section_index", "0"))
        return source, section

    for d in all_docs:
        if allowed_types:
            doc_type = d.metadata.get("document_type")
            if not doc_type or doc_type not in allowed_types:
                continue
        k = _key(d)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        reduced.append(d)

    # Safety net: if routing filters everything out, fall back to all_docs (deduped)
    # instead of leaving retrieval empty, which would force the LLM to hallucinate.
    if not reduced and all_docs:
        seen_keys.clear()
        for d in all_docs:
            k = _key(d)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            reduced.append(d)

    state.retrieved = reduced
    state.clauses = reduced
    return state


async def node_answer_and_risk(state: GraphState) -> GraphState:
    if not state.question:
        return state
    docs: Sequence[Document] = state.clauses or state.retrieved
    answer = await answer_with_optional_risks(
        question=state.question,
        docs=docs,
        is_risk_query=state.is_risk_query,
    )
    state.final_answer = answer
    state.messages.append(AIMessage(content=answer))
    return state


def build_graph() -> StateGraph:
    builder: StateGraph = StateGraph(GraphState)
    builder.add_node("router", node_router)
    builder.add_node("clause_extractor", node_clause_extractor)
    builder.add_node("answer_and_risk", node_answer_and_risk)
    builder.set_entry_point("router")
    builder.add_edge("router", "clause_extractor")
    builder.add_edge("clause_extractor", "answer_and_risk")
    builder.add_edge("answer_and_risk", END)
    return builder
