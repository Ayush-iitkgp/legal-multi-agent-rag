"""
Evaluate Recall@K for the RAG retrieval pipeline.

Compares the retrieved sections from eval_outputs.json against
the manually annotated ground truth in ground_truth.json.

Usage:
    poetry run python -m src.eval.recall [--eval eval_outputs.json] [--k 8]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


SectionKey = Tuple[str, int]


def _section_key(source: str, section_index: int) -> SectionKey:
    return (source, section_index)


def load_ground_truth(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["question"]: entry["relevant_sections"] for entry in data}


def load_eval_outputs(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_text_match(
    retrieved_docs: List[Dict[str, Any]],
    gt_section: Dict[str, Any],
    k: int,
) -> bool:
    """Return True if the expected_text snippet appears in the matching retrieved chunk."""
    expected = gt_section.get("expected_text", "")
    if not expected:
        return True
    key = _section_key(gt_section["source"], gt_section["section_index"])
    for doc in retrieved_docs[:k]:
        if _section_key(doc["source"], doc["section_index"]) == key:
            return expected.lower() in (doc.get("text", "")).lower()
    return False


def compute_recall(
    retrieved: List[Dict[str, Any]],
    relevant: List[Dict[str, Any]],
    k: int,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Returns (section_recall, text_recall, per-section details)."""
    retrieved_keys: Set[SectionKey] = set()
    for doc in retrieved[:k]:
        retrieved_keys.add(_section_key(doc["source"], doc["section_index"]))

    if not relevant:
        return 1.0, 1.0, []

    section_hits = 0
    text_hits = 0
    details: List[Dict[str, Any]] = []

    for sec in relevant:
        key = _section_key(sec["source"], sec["section_index"])
        found = key in retrieved_keys
        text_ok = _check_text_match(retrieved, sec, k) if found else False

        if found:
            section_hits += 1
        if text_ok:
            text_hits += 1

        details.append(
            {
                "source": sec["source"],
                "section_index": sec["section_index"],
                "section_title": sec.get("section_title", "?"),
                "expected_text": sec.get("expected_text", ""),
                "section_found": found,
                "text_found": text_ok,
            }
        )

    return section_hits / len(relevant), text_hits / len(relevant), details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Recall@K")
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("eval_outputs.json"),
        help="Path to eval outputs JSON",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(__file__).resolve().parent / "ground_truth.json",
        help="Path to ground truth JSON",
    )
    parser.add_argument("--k", type=int, default=8, help="K for Recall@K")
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)
    eval_results = load_eval_outputs(args.eval)

    total_section_recall = 0.0
    total_text_recall = 0.0
    matched = 0

    print(f"\n{'=' * 78}")
    print(f"  Recall@{args.k} Evaluation  (section match + text content match)")
    print(f"{'=' * 78}\n")

    for entry in eval_results:
        question = entry["question"]
        if question not in gt:
            print(f"  SKIP  {question}")
            print("         (no ground truth annotation)\n")
            continue

        relevant = gt[question]
        retrieved = entry.get("retrieved", [])
        sec_recall, txt_recall, details = compute_recall(
            retrieved,
            relevant,
            k=args.k,
        )
        total_section_recall += sec_recall
        total_text_recall += txt_recall
        matched += 1

        status = "PASS" if sec_recall == 1.0 and txt_recall == 1.0 else "MISS"
        n_relevant = len(relevant)
        print(f"  [{status}]  {question}")
        print(
            f"         Section Recall@{args.k} = {sec_recall:.2f}  "
            f"({int(sec_recall * n_relevant)}/{n_relevant})"
        )
        print(
            f"         Text    Recall@{args.k} = {txt_recall:.2f}  "
            f"({int(txt_recall * n_relevant)}/{n_relevant})"
        )

        for d in details:
            sec_icon = "✓" if d["section_found"] else "✗"
            txt_icon = "✓" if d["text_found"] else "✗"
            snippet = (
                d["expected_text"][:60] + "..."
                if len(d["expected_text"]) > 60
                else d["expected_text"]
            )
            print(
                f"           {sec_icon} sec | {txt_icon} txt  "
                f"{d['source']} §{d['section_index']} ({d['section_title']})"
            )
            if snippet:
                print(f'                        expected: "{snippet}"')
        print()

    if matched:
        avg_sec = total_section_recall / matched
        avg_txt = total_text_recall / matched
        print(f"{'─' * 78}")
        print(f"  Average Section Recall@{args.k}:  {avg_sec:.2f}  ({matched} queries)")
        print(f"  Average Text    Recall@{args.k}:  {avg_txt:.2f}  ({matched} queries)")
        print(f"{'─' * 78}\n")
    else:
        print("  No matching queries found between eval outputs and ground truth.\n")


if __name__ == "__main__":
    main()
