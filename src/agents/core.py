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
        "You are a legal query classifier for contract Q&A over the following corpus:\n"
        "- NDA between Acme and Vendor\n"
        "- Vendor Services Agreement\n"
        "- Service Level Agreement (SLA)\n"
        "- Data Processing Agreement (DPA)\n\n"
        "Your job is to (1) classify the user's question into a coarse query_type and "
        "(2) identify 1–3 focus_areas that will be used to steer retrieval.\n\n"
        "Set query_type as exactly one of:\n"
        "- fact_lookup: direct questions answerable from a single clause in one agreement.\n"
        '  Examples: "What is the notice period for terminating the NDA?", '
        '"What is the uptime commitment in the SLA?", '
        '"Which law governs the Vendor Services Agreement?".\n'
        "- cross_agreement_compare: questions that require comparing or reconciling terms "
        "across multiple agreements.\n"
        '  Examples: "Which agreement governs data breach notification timelines?", '
        '"Are there conflicting governing laws across agreements?".\n'
        "- risk_summary: questions that primarily ask about risk, exposure, or legal strategy.\n"
        '  Examples: "Are there any legal risks related to liability exposure?", '
        '"Identify any clauses that could pose financial risk to Acme Corp.", '
        '"Is there any unlimited liability in these agreements?", '
        '"What happens if Vendor delays breach notification beyond 72 hours?".\n'
        "- other: anything else (chitchat, drafting entire new contracts, etc.).\n\n"
        "For focus_areas, choose 1–3 of the following that best match the question:\n"
        "- termination\n"
        "- confidentiality\n"
        "- liability\n"
        "- indemnification\n"
        "- governing_law\n"
        "- data_breach\n"
        "- uptime_sla\n"
        "- remedies\n"
        "- subprocessors\n"
        "- other\n\n"
        "Examples of focus_areas mapping:\n"
        "- Questions about notice periods or contract end → termination\n"
        "- Questions about secrecy, sharing data, or subcontractors → confidentiality, subprocessors\n"
        "- Questions about caps, unlimited liability, or financial risk → liability\n"
        "- Questions about which law applies → governing_law\n"
        "- Questions about breach notification or 72 hours → data_breach\n"
        "- Questions about uptime or credits if SLA is not met → uptime_sla, remedies\n\n"
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
    parts: List[str] = [
        f"Question: {question}",
        "",
        f"Answer:\n{answer_body}",
    ]
    if risk_summary.strip():
        parts.extend(
            [
                "",
                f"Risk flags:\n{risk_summary}",
            ]
        )
    return "\n".join(parts)
