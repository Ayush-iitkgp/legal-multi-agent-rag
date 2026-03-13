from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.graph import GraphState, build_graph


SAMPLE_QUERIES: List[str] = [
    "What is the notice period for terminating the NDA?",
    "What is the uptime commitment in the SLA?",
    "Which law governs the Vendor Services Agreement?",
    "Do confidentiality obligations survive termination of the NDA?",
    "Is liability capped for breach of confidentiality?",
]


async def run_eval(output_path: Path | None = None) -> Path:
    graph = build_graph().compile()
    results = []
    for q in SAMPLE_QUERIES:
        state = GraphState(question=q)
        out: GraphState = await graph.ainvoke(state)
        results.append(
            {
                "question": q,
                "answer": out.final_answer,
                "retrieved_sources": [d.metadata.get("source") for d in out.retrieved],
            }
        )

    out_dir = output_path or Path("eval_outputs.json")
    if out_dir.is_dir():
        path = out_dir / "eval_results.json"
    else:
        path = out_dir
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path

