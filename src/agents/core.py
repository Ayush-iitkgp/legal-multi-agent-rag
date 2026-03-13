from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.llm.factory import make_chat_model


QueryType = Literal[
    "fact_lookup",
    "cross_agreement_compare",
    "risk_summary",
    "other",
]


@dataclass
class QueryAnalysis:
    query_type: QueryType
    focus_areas: List[str]


def analyze_query(question: str, history: Sequence[HumanMessage | AIMessage]) -> QueryAnalysis:
    model = make_chat_model()
    history_summary = "\n".join(m.content for m in history[-4:]) if history else "None"
    prompt = (
        "You are a legal query classifier for contract Q&A.\n"
        "Classify the user's question into one of: fact_lookup, cross_agreement_compare, "
        "risk_summary, other. Also list 1-3 focus areas from:\n"
        "termination, confidentiality, liability, indemnification, governing_law, "
        "data_breach, uptime_sla, remedies, other.\n\n"
        f"Recent history:\n{history_summary}\n\n"
        f"Question: {question}\n\n"
        "Respond as JSON with keys query_type and focus_areas."
    )
    msg = model.invoke(prompt)
    import json

    try:
        data = json.loads(msg.content)  # type: ignore[arg-type]
        qtype: QueryType = data.get("query_type", "fact_lookup")
        focus = data.get("focus_areas") or []
    except Exception:
        qtype = "fact_lookup"
        focus = []
    return QueryAnalysis(query_type=qtype, focus_areas=list(focus))


def draft_answer(question: str, docs: Sequence[Document]) -> str:
    model = make_chat_model()
    context = "\n\n".join(
        f"[{i+1}] {d.metadata.get('source')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    prompt = (
        "You are a legal contracts analyst.\n"
        "Answer the question using ONLY the context clauses below.\n"
        "Quote key clauses briefly and refer to them as [1], [2], etc. where relevant.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    msg = model.invoke(prompt)
    return msg.content  # type: ignore[return-value]


def assess_risks(question: str, docs: Sequence[Document]) -> str:
    model = make_chat_model()
    context = "\n\n".join(
        f"[{i+1}] {d.metadata.get('source')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    prompt = (
        "You are a legal risk reviewer.\n"
        "Given the context clauses and the user's question, identify any material legal or "
        "financial risks to Acme Corp.\n"
        "Return a short bullet list, each item mentioning severity (LOW/MEDIUM/HIGH), "
        "risk type (liability, confidentiality, data_breach, governing_law, uptime_sla, other), "
        "and reference clause numbers [1], [2], etc.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    msg = model.invoke(prompt)
    return msg.content  # type: ignore[return-value]


def compose_final_answer(
    question: str,
    answer_body: str,
    risk_summary: str,
) -> str:
    return (
        f"Question: {question}\n\n"
        f"Answer:\n{answer_body}\n\n"
        f"Risk flags:\n{risk_summary}"
    )

