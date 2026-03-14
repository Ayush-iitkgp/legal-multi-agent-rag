import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from src.graph import GraphState, build_graph
from langchain_core.messages import HumanMessage
import asyncio

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "legal-multi-agent-rag")

console = Console()


async def main() -> None:
    if not os.environ.get("LANGCHAIN_API_KEY"):
        console.print(
            "[yellow]LANGCHAIN_API_KEY not set — LangSmith tracing disabled.[/yellow]"
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

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
        # LangGraph returns the state as a plain dict; rehydrate into GraphState
        result_dict = await graph.ainvoke(state.__dict__)
        state = GraphState(**result_dict)
        console.print(f"[bold magenta]Assistant[/bold magenta]: {state.final_answer}")


if __name__ == "__main__":
    asyncio.run(main())
