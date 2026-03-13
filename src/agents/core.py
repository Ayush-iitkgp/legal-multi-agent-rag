from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.llm.factory import make_chat_model


QueryType = Literal["fact_lookup", "cross_agreement_compare", "risk_summary", "other"]


@dataclass
class QueryAnalysis:
    query_type: QueryType
    doc_targets: List[str]


async def analyze_query(
    question: str,
    history: Sequence[HumanMessage | AIMessage],
) -> QueryAnalysis:
    model = make_chat_model()
    history_summary = "\n".join(m.content for m in history[-4:]) if history else "None"
    prompt = (
        "You are a routing agent for a legal contract Q&A system.\n"
        "The corpus contains four agreements:\n"
        "- NDA between Acme and Vendor\n"
        "- Vendor Services Agreement\n"
        "- Service Level Agreement (SLA)\n"
        "- Data Processing Agreement (DPA)\n\n"
        "Your job is to:\n"
        "1) classify the user's question into a query_type, and\n"
        "2) choose which agreement(s) are most relevant via doc_targets.\n\n"
        "Set query_type as exactly one of:\n"
        "- fact_lookup: direct questions answerable from a clause in one agreement.\n"
        "- cross_agreement_compare: questions that compare or reconcile terms across multiple agreements.\n"
        "- risk_summary: questions primarily about risk, exposure, or legal strategy.\n"
        "- other: anything else.\n\n"
        "Set doc_targets to a list containing one or more of:\n"
        "- nda             (for the NDA between Acme and Vendor)\n"
        "- vendor_services (for the Vendor Services Agreement)\n"
        "- service_level   (for the SLA)\n"
        "- dpa             (for the Data Processing Agreement)\n"
        "- all             (for questions that clearly span all agreements, especially risk_summary).\n\n"
        "Examples:\n"
        '- "What is the notice period for terminating the NDA?" → query_type: fact_lookup, doc_targets: ["nda"]\n'
        '- "What is the uptime commitment in the SLA?" → query_type: fact_lookup, doc_targets: ["service_level"]\n'
        '- "Which law governs the Vendor Services Agreement?" → query_type: fact_lookup, doc_targets: ["vendor_services"]\n'
        '- "Which agreement governs data breach notification timelines?" → query_type: cross_agreement_compare, '
        'doc_targets: ["dpa", "vendor_services", "service_level"]\n'
        '- "Are there any legal risks related to liability exposure?" → query_type: risk_summary, doc_targets: ["all"]\n\n'
        f"Recent history:\n{history_summary}\n\n"
        f"Question: {question}\n\n"
        "Respond as JSON with keys query_type and doc_targets."
    )
    msg = await model.ainvoke(prompt)
    import json

    try:
        data = json.loads(msg.content)  # type: ignore[arg-type]
        qtype: QueryType = data.get("query_type", "fact_lookup")
        doc_targets = data.get("doc_targets") or []
    except Exception:
        qtype = "fact_lookup"
        doc_targets = ["all"]

    if isinstance(doc_targets, str):
        doc_targets_list: List[str] = [doc_targets]
    else:
        doc_targets_list = [str(t) for t in doc_targets]
    if not doc_targets_list:
        doc_targets_list = ["all"]

    return QueryAnalysis(query_type=qtype, doc_targets=doc_targets_list)


async def answer_with_optional_risks(
    question: str,
    docs: Sequence[Document],
    is_risk_query: bool,
) -> str:
    model = make_chat_model()
    context = "\n\n".join(
        f"[{i+1}] {d.metadata.get('source')}, section: {d.metadata.get('section_title')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    risk_instructions = (
        "Because this question is primarily about risk or exposure, you MUST also add a "
        "'Risk flags:' section after the main answer. In that section, list concise bullet "
        "points describing any material legal, financial, or operational risks, referencing "
        "clauses with [1], [2], etc., where relevant.\n"
    )
    no_risk_instructions = (
        "This question is not primarily about risk; do NOT add any 'Risk flags:' section. "
        "Only answer the question and show citations.\n"
    )
    prompt = (
        "You are a legal contracts analyst for Acme Corp.\n"
        "Use ONLY the context clauses below to answer the question.\n"
        "Quote key clauses briefly and refer to them as [1], [2], etc. where relevant.\n"
        "Your answer MUST:\n"
        "- Provide a clear, direct answer in plain language.\n"
        '- Include bracket references next to the relevant sentences, e.g. "30 days[1]."\n'
        "- At the end, add a 'Citations used:' block, one per line, in the exact format:\n"
        "  [n] filename.txt, section: <section title>\n\n"
        + (risk_instructions if is_risk_query else no_risk_instructions)
        + f"Context:\n{context}\n\nQuestion: {question}"
    )
    msg = await model.ainvoke(prompt)
    return msg.content  # type: ignore[return-value]
