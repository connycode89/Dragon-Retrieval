"""
Dragon Retrieval — CLI entry point.

Usage:
    # First-time setup: ingest data into the vector store
    python app.py ingest --wikipedia          # no API key needed
    python app.py ingest --logainm            # requires LOGAINM_API_KEY
    python app.py ingest --duchas             # requires DUCHAS_API_KEY
    python app.py ingest --all                # everything

    # Ask questions
    python app.py ask "What legends surround Knocknarea in Sligo?"
    python app.py ask "What does the name Knocknarea mean in Irish?"
    python app.py ask "Tell me about fairy forts in County Clare"

    # Interactive REPL
    python app.py chat
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def cmd_ingest(args):
    """Ingest data from selected sources into the vector store."""
    from ingest import DuchasLoader, LogainmLoader, WikipediaIrishPlacesLoader, CensusLoader
    from rag import Embedder
    from config import DUCHAS_COUNTY_IDS

    embedder = Embedder()
    all_docs = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        if args.wikipedia or args.all:
            task = progress.add_task("Loading Wikipedia Irish places...", total=None)
            loader = WikipediaIrishPlacesLoader()
            docs = loader.load_irish_counties()
            all_docs.extend(docs)
            progress.update(task, description=f"Wikipedia: {len(docs)} articles loaded")
            progress.stop_task(task)

        if args.census or args.all:
            task = progress.add_task("Loading historical demographics...", total=None)
            loader = CensusLoader()
            docs = loader.load_all()
            all_docs.extend(docs)
            progress.update(task, description=f"Census/Demographics: {len(docs)} documents loaded")
            progress.stop_task(task)

        if args.logainm or args.all:
            from config import LOGAINM_API_KEY
            if not LOGAINM_API_KEY:
                console.print("[yellow]Skipping Logainm — LOGAINM_API_KEY not set[/yellow]")
            else:
                task = progress.add_task("Loading Logainm placenames...", total=None)
                loader = LogainmLoader()
                docs = []
                # Load a sample of counties
                for county_name, county_id in list(DUCHAS_COUNTY_IDS.items())[:5]:
                    county_docs = loader.load_by_county(county_id)
                    docs.extend(county_docs)
                all_docs.extend(docs)
                progress.update(task, description=f"Logainm: {len(docs)} placenames loaded")
                progress.stop_task(task)

        if args.duchas or args.all:
            from config import DUCHAS_API_KEY
            if not DUCHAS_API_KEY:
                console.print("[yellow]Skipping Dúchas — DUCHAS_API_KEY not set[/yellow]")
            else:
                task = progress.add_task("Loading Dúchas folklore...", total=None)
                loader = DuchasLoader()
                docs = []
                for county_name, county_id in list(DUCHAS_COUNTY_IDS.items())[:3]:
                    console.print(f"  Fetching folklore from {county_name}...")
                    county_docs = loader.load_by_county(county_id, max_items=100)
                    docs.extend(county_docs)
                all_docs.extend(docs)
                progress.update(task, description=f"Dúchas: {len(docs)} folklore items loaded")
                progress.stop_task(task)

    if not all_docs:
        console.print("[red]No documents loaded. Check your API keys and try again.[/red]")
        sys.exit(1)

    console.print(f"\n[green]Total documents to embed: {len(all_docs)}[/green]")
    embedder.embed_documents(all_docs)
    console.print("[bold green]Ingestion complete. Vector store saved.[/bold green]")


def cmd_ask(args):
    """Ask a single question and print the answer."""
    from rag import IrishFolkloreChain

    question = " ".join(args.question)
    console.print(Panel(f"[bold]{question}[/bold]", title="Question", border_style="blue"))

    try:
        chain = IrishFolkloreChain()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print("\n[dim]Thinking...[/dim]\n")
    answer, docs = chain.ask_with_sources(question)

    console.print(Panel(Markdown(answer), title="Answer", border_style="green"))

    if args.show_sources:
        console.print("\n[bold]Sources:[/bold]")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title") or doc.metadata.get("name_en") or "—"
            console.print(f"  [{i}] [cyan]{source}[/cyan] — {title}")
            console.print(f"      {doc.page_content[:120]}...")


def cmd_chat(args):
    """Start an interactive question-answering session."""
    from rag import IrishFolkloreChain

    try:
        chain = IrishFolkloreChain()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print(Panel(
        "[bold green]Dragon Retrieval[/bold green] — Irish Folklore & Place History\n"
        "Ask about any Irish place, legend, or placename. Type [bold]quit[/bold] to exit.",
        border_style="green",
    ))

    while True:
        try:
            question = console.input("\n[bold blue]You:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Slán![/dim]")
            break

        if question.lower() in ("quit", "exit", "q", "slán"):
            console.print("[dim]Slán![/dim]")
            break

        if not question:
            continue

        console.print("\n[bold green]Dragon:[/bold green]")
        try:
            for token in chain.stream(question):
                console.print(token, end="")
            console.print()
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(
        description="Dragon Retrieval — Irish Folklore & Place History RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Load data into the vector store")
    ingest_parser.add_argument("--wikipedia", action="store_true", help="Load Wikipedia articles")
    ingest_parser.add_argument("--logainm", action="store_true", help="Load Logainm placenames")
    ingest_parser.add_argument("--duchas", action="store_true", help="Load Dúchas folklore")
    ingest_parser.add_argument("--census", action="store_true", help="Load historical demographics")
    ingest_parser.add_argument("--all", action="store_true", help="Load all sources")

    # ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", nargs="+", help="The question to ask")
    ask_parser.add_argument("--show-sources", action="store_true", help="Show source documents")

    # chat command
    subparsers.add_parser("chat", help="Start an interactive session")

    args = parser.parse_args()

    commands = {"ingest": cmd_ingest, "ask": cmd_ask, "chat": cmd_chat}
    commands[args.command](args)


if __name__ == "__main__":
    main()
