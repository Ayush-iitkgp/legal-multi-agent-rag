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


async def analyze_query(
    question: str,
    history: Sequence[HumanMessage | AIMessage],
) -> QueryAnalysis:
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
    msg = await model.ainvoke(prompt)
    import json

    try:
        data = json.loads(msg.content)  # type: ignore[arg-type]
        qtype: QueryType = data.get("query_type", "fact_lookup")
        focus = data.get("focus_areas") or []
    except Exception:
        qtype = "fact_lookup"
        focus = []
    return QueryAnalysis(query_type=qtype, focus_areas=list(focus))


async def draft_answer(question: str, docs: Sequence[Document]) -> str:
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
    msg = await model.ainvoke(prompt)
    return msg.content  # type: ignore[return-value]


async def assess_risks(question: str, docs: Sequence[Document]) -> str:
    model = make_chat_model()
    context = "\n\n".join(
        f"[{i+1}] {d.metadata.get('source')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    prompt = (
        "You are a senior legal auditor reviewing a portfolio of related agreements "
        "(e.g., NDA, Vendor Services Agreement, SLA, DPA) on behalf of Acme Corp.\n"
        "Given the context clauses and the user's question, identify any material legal, "
        "financial, or operational risks.\n\n"
        "Interpret 'risk indicators' as specific flags or warnings in the contracts, including:\n"
        "- Financial exposure: unlimited liability, very low liability caps (e.g., 12-month fee caps), "
        "or caps that might not cover reasonably foreseeable damages.\n"
        "- Legal inconsistencies: conflicting terms across agreements, such as different governing "
        "laws (e.g., California in the NDA vs. England and Wales in the Vendor Services Agreement) "
        "or incompatible dispute resolution frameworks.\n"
        "- Compliance gaps: strict timelines or obligations with significant consequences if missed, "
        "such as the 72-hour data breach notification window in the DPA, and whether other documents "
        "align with or ignore those obligations.\n"
        "- Unfavorable terms: 'sole and exclusive' remedies (e.g., in the SLA) that prevent Acme "
        "from seeking additional damages or alternative remedies.\n"
        "- Omissions: missing protections, such as the absence of an explicit limitation of liability "
        "in the NDA or missing confidentiality survival language.\n\n"
        "Pay special attention to inconsistencies across documents and situations where obligations "
        "cannot all be satisfied at once.\n"
        "Use metadata such as document_type and section_index and refer to clauses as [1], [2], etc.\n"
        "Return a concise bullet list. For each bullet, include: severity (LOW/MEDIUM/HIGH), "
        "risk category (financial_exposure, legal_inconsistency, compliance_gap, unfavorable_term, omission, other), "
        "a short description, and the relevant clause references [1], [2], etc.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    msg = await model.ainvoke(prompt)
    return msg.content  # type: ignore[return-value]


async def compose_final_answer(
    question: str,
    answer_body: str,
    risk_summary: str,
) -> str:
    return (
        f"Question: {question}\n\n"
        f"Answer:\n{answer_body}\n\n"
        f"Risk flags:\n{risk_summary}"
    )
