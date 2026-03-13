from dataclasses import dataclass, field
from typing import List

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
    retrieved: List[Document] = field(default_factory=list)
    answer_body: str | None = None
    risk_summary: str | None = None
    final_answer: str | None = None


async def node_query_analyzer(state: GraphState) -> GraphState:
    if not state.question:
        return state
    qa = await analyze_query(
        question=state.question,
        history=[m for m in state.messages],
    )
    state.query_analysis = qa
    return state


async def node_retrieve(state: GraphState) -> GraphState:
    embeddings = make_embeddings()
    try:
        vs = load_vectorstore(embeddings)
    except Exception:
        raw = await load_raw_docs()
        chunks = chunk_corpus(raw)
        vs = build_vectorstore(chunks, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    docs = retriever.invoke(state.question or "")
    state.retrieved = docs
    return state


async def node_legal_analyst(state: GraphState) -> GraphState:
    if not state.question:
        return state
    state.answer_body = await draft_answer(state.question, state.retrieved)
    state.messages.append(AIMessage(content=state.answer_body or ""))
    return state


async def node_risk_assessor(state: GraphState) -> GraphState:
    if not state.question:
        return state
    state.risk_summary = await assess_risks(state.question, state.retrieved)
    return state


async def node_answer_composer(state: GraphState) -> GraphState:
    if not state.question:
        return state
    state.final_answer = await compose_final_answer(
        question=state.question,
        answer_body=state.answer_body or "",
        risk_summary=state.risk_summary or "None identified.",
    )
    return state


def build_graph() -> StateGraph:
    builder: StateGraph = StateGraph(GraphState)
    builder.add_node("query_analyzer", node_query_analyzer)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("legal_analyst", node_legal_analyst)
    builder.add_node("risk_assessor", node_risk_assessor)
    builder.add_node("answer_composer", node_answer_composer)
    builder.set_entry_point("query_analyzer")
    builder.add_edge("query_analyzer", "retrieve")
    builder.add_edge("retrieve", "legal_analyst")
    builder.add_edge("legal_analyst", "risk_assessor")
    builder.add_edge("risk_assessor", "answer_composer")
    builder.add_edge("answer_composer", END)
    return builder
