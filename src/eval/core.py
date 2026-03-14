import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document

from src.graph import GraphState, build_graph


SAMPLE_QUERIES: List[str] = [
    "What is the notice period for terminating the NDA?",
    "What is the uptime commitment in the SLA?",
    "Which law governs the Vendor Services Agreement?",
    "Do confidentiality obligations survive termination of the NDA?",
    "Is liability capped for breach of confidentiality?",
]


def _serialize_doc(d: Document) -> Dict[str, Any]:
    return {
        "source": d.metadata.get("source"),
        "document_type": d.metadata.get("document_type"),
        "section_index": d.metadata.get("section_index"),
        "section_title": d.metadata.get("section_title"),
        "text": d.page_content,
    }


def _serialize_state(state: GraphState) -> Dict[str, Any]:
    qa = state.query_analysis
    return {
        "question": state.question,
        "query_analysis": asdict(qa) if qa else None,
        "is_risk_query": state.is_risk_query,
        "routed_doc_types": state.routed_doc_types,
        "retrieved": [_serialize_doc(d) for d in state.retrieved],
        "clauses": [_serialize_doc(d) for d in state.clauses],
        "final_answer": state.final_answer,
    }


async def run_eval(output_path: Path | None = None) -> Path:
    graph = build_graph().compile()
    results = []
    for q in SAMPLE_QUERIES:
        state = GraphState(question=q)
        result_dict = await graph.ainvoke(state.__dict__)
        out = GraphState(**result_dict)
        results.append(_serialize_state(out))

    out_dir = output_path or Path("eval_outputs.json")
    if out_dir.is_dir():
        path = out_dir / "eval_results.json"
    else:
        path = out_dir
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path
