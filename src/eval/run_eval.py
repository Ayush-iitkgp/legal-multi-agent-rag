from __future__ import annotations

from pathlib import Path

from rich.console import Console

from src.eval.core import run_eval

console = Console()


def main() -> None:
    path = run_eval(Path("eval_outputs.json"))
    console.print(f"[bold green]Wrote evaluation results to {path}[/bold green]")


if __name__ == "__main__":
    main()

