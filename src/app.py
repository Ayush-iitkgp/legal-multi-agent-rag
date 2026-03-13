from rich.console import Console
from rich.prompt import Prompt

from src.graph import GraphState, build_graph
from langchain_core.messages import HumanMessage
import asyncio

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
        # LangGraph returns the state as a plain dict; rehydrate into GraphState
        result_dict = await graph.ainvoke(state.__dict__)
        state = GraphState(**result_dict)
        console.print(f"[bold magenta]Assistant[/bold magenta]: {state.final_answer}")


if __name__ == "__main__":
    asyncio.run(main())
