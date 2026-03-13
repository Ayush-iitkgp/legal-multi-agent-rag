from dataclasses import dataclass, field
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.agents.core import (
    QueryAnalysis,
    analyze_query,
    assess_risks,
    compose_final_answer,
    draft_answer,
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
    answer_body: str | None = None
    risk_summary: str | None = None
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

    q = state.question.lower()
    targets: set[str] = set()

    if any(
        k in q
        for k in ("dpa", "data processing", "sub-processor", "72 hours", "breach")
    ):
        targets.add("dpa")
    if any(k in q for k in ("nda", "non-disclosure", "confidential information")):
        targets.add("nda")
    if any(
        k in q
        for k in ("service", "sla", "uptime", "availability", "termination", "fees")
    ):
        targets.add("services_agreement")

    if not targets:
        # Fallback: consider all agreement types when routing is ambiguous.
        targets.update(("nda", "dpa", "services_agreement"))

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
    queries: List[str] = [base_query]

    if state.query_analysis and state.query_analysis.focus_areas:
        for focus in state.query_analysis.focus_areas:
            queries.append(f"{focus} – {base_query}")

    all_docs: List[Document] = []
    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

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

    state.retrieved = reduced
    state.clauses = reduced
    return state


async def node_legal_analyst(state: GraphState) -> GraphState:
    if not state.question:
        return state
    docs: Sequence[Document] = state.clauses or state.retrieved
    state.answer_body = await draft_answer(state.question, docs)
    state.messages.append(AIMessage(content=state.answer_body or ""))
    return state


async def node_legal_auditor(state: GraphState) -> GraphState:
    if not state.question:
        return state
    docs: Sequence[Document] = state.clauses or state.retrieved
    state.risk_summary = await assess_risks(state.question, docs)
    return state


async def node_answer_composer(state: GraphState) -> GraphState:
    if not state.question:
        return state
    state.final_answer = await compose_final_answer(
        question=state.question,
        answer_body=state.answer_body or "",
        risk_summary=state.risk_summary or "",
    )
    return state


def _route_after_legal_analyst(state: GraphState) -> str:
    """
    Decide whether to run the risk auditor.

    Risk analysis is optional and only executed when the query is primarily
    about risk/coverage. For other query types, we skip the risk node and go
    directly to answer composition.
    """
    if state.is_risk_query:
        return "do_risk"
    return "skip_risk"


def build_graph() -> StateGraph:
    builder: StateGraph = StateGraph(GraphState)
    builder.add_node("router", node_router)
    builder.add_node("clause_extractor", node_clause_extractor)
    builder.add_node("legal_analyst", node_legal_analyst)
    builder.add_node("legal_auditor", node_legal_auditor)
    builder.add_node("answer_composer", node_answer_composer)
    builder.set_entry_point("router")
    builder.add_edge("router", "clause_extractor")
    builder.add_edge("clause_extractor", "legal_analyst")
    # Conditionally run the risk auditor: only for explicit risk_summary queries.
    builder.add_conditional_edges(
        "legal_analyst",
        _route_after_legal_analyst,
        {
            "do_risk": "legal_auditor",
            "skip_risk": "answer_composer",
        },
    )
    builder.add_edge("legal_auditor", "answer_composer")
    builder.add_edge("answer_composer", END)
    return builder
