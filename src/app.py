from __future__ import annotations

from rich.console import Console
from rich.prompt import Prompt

from src.graph import GraphState, build_graph
from langchain_core.messages import HumanMessage

console = Console()


async def main() -> None:
    graph = build_graph().compile()
    state = GraphState()

    console.print("[bold green]Legal Multi-Agent RAG CLI[/bold green]")
    console.print("Type your question about the contracts, or /quit to exit.\n")

    while True:
        try:
            text: str = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break

        if not text.strip():
            continue
        if text.strip().lower() in {"/quit", "/exit"}:
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break

        state.question = text
        state.messages.append(HumanMessage(content=text))
        result: GraphState = await graph.ainvoke(state)
        console.print(f"[bold magenta]Assistant[/bold magenta]: {result.final_answer}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
