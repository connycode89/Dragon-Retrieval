"""
Dragon Retrieval — Interactive Marimo Notebook

Run with:
    marimo edit notebook.py     # editable notebook UI
    marimo run notebook.py      # read-only app UI

This notebook lets you:
- Search and explore the Dúchas, Logainm, and Wikipedia sources live
- Inspect individual retrieved documents before they reach the LLM
- Ask full RAG questions and see the answer alongside the sources
- Tune retrieval parameters (top_k, source filters) interactively
"""

import marimo as mo

app = mo.App(width="full")


@app.cell
def _setup():
    import sys
    from pathlib import Path

    # Ensure project root is on the path when running from any directory
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    mo.md("# 🐉 Dragon Retrieval\n### Irish Folklore & Place History — Interactive Explorer")
    return (project_root,)


@app.cell
def _check_env():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    duchas_ok = bool(os.getenv("DUCHAS_API_KEY"))
    logainm_ok = bool(os.getenv("LOGAINM_API_KEY"))

    status_rows = [
        ("ANTHROPIC_API_KEY", "✅ Set" if anthropic_ok else "❌ Missing — set in .env"),
        ("DUCHAS_API_KEY",    "✅ Set" if duchas_ok else "⚠️  Missing — email eolas@duchas.ie"),
        ("LOGAINM_API_KEY",   "✅ Set" if logainm_ok else "⚠️  Missing — register at gaois.ie"),
    ]

    mo.md(
        "## Environment\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in status_rows)
        + ("\n\n> ⚠️ Vector store may not exist yet — run `python app.py ingest --wikipedia` first"
           if not Path("chroma_db").exists() else
           "\n\n> ✅ Vector store found")
    )
    return anthropic_ok, duchas_ok, logainm_ok


@app.cell
def _section_live_search():
    mo.md("---\n## 1. Live Source Search\nQuery individual data sources directly — no vector store needed.")
    return ()


@app.cell
def _wikipedia_search_ui():
    search_input = mo.ui.text(
        placeholder="e.g. Knocknarea Sligo",
        label="Search Wikipedia for Irish places",
        full_width=True,
    )
    search_btn = mo.ui.run_button(label="Search Wikipedia")
    mo.hstack([search_input, search_btn])
    return search_btn, search_input


@app.cell
def _wikipedia_results(search_input, search_btn):
    results_md = ""

    if search_btn.value and search_input.value:
        from ingest import WikipediaIrishPlacesLoader
        loader = WikipediaIrishPlacesLoader()
        docs = loader.search(search_input.value, max_results=3)

        if docs:
            parts = []
            for doc in docs:
                title = doc.metadata.get("title", "")
                url = doc.metadata.get("url", "")
                text = doc.page_content[:500]
                parts.append(f"### [{title}]({url})\n{text}...")
            results_md = "\n\n---\n\n".join(parts)
        else:
            results_md = "_No results found._"

    mo.md(results_md) if results_md else mo.md("_Enter a search term above._")
    return (results_md,)


@app.cell
def _logainm_search_ui():
    logainm_input = mo.ui.text(
        placeholder="e.g. Dun na nGall",
        label="Search Logainm for Irish placenames",
        full_width=True,
    )
    logainm_btn = mo.ui.run_button(label="Search Logainm")
    mo.hstack([logainm_input, logainm_btn])
    return logainm_btn, logainm_input


@app.cell
def _logainm_results(logainm_input, logainm_btn):
    import os
    logainm_key = os.getenv("LOGAINM_API_KEY", "")

    if logainm_btn.value and logainm_input.value:
        if not logainm_key:
            mo.md("⚠️ **LOGAINM_API_KEY not set.** Register at gaois.ie to get a key.")
        else:
            from ingest import LogainmLoader
            loader = LogainmLoader(api_key=logainm_key)
            docs = loader.search(logainm_input.value, max_results=5)
            if docs:
                parts = [f"**{doc.metadata.get('name_en', '')}** ({doc.metadata.get('name_ga', '')})\n\n{doc.page_content}" for doc in docs]
                mo.md("\n\n---\n\n".join(parts))
            else:
                mo.md("_No placenames found._")
    else:
        mo.md("_Enter a placename to look up its Irish etymology._")
    return (logainm_key,)


@app.cell
def _section_rag():
    mo.md("---\n## 2. Full RAG Query\nAsk a question — the system retrieves relevant context and passes it to Claude.")
    return ()


@app.cell
def _rag_ui():
    example_questions = [
        "What legends surround Knocknarea in Sligo?",
        "What does the name Dún na nGall (Donegal) mean in Irish?",
        "Tell me about fairy forts in County Clare",
        "What folk cures were used in rural Connaught?",
        "Describe life in a typical Galway townland around 1900",
        "What are the strangest fairy stories from County Kerry?",
    ]

    question_input = mo.ui.text_area(
        placeholder="Ask anything about Irish folklore, placenames, or local history...",
        label="Your question",
        full_width=True,
        rows=3,
    )
    example_dropdown = mo.ui.dropdown(
        options=["(pick an example)"] + example_questions,
        label="Or pick an example",
    )
    top_k_slider = mo.ui.slider(
        start=2, stop=12, step=1, value=6,
        label="Documents to retrieve (top_k)",
    )
    ask_btn = mo.ui.run_button(label="Ask Dragon")

    mo.vstack([
        mo.hstack([question_input, example_dropdown]),
        mo.hstack([top_k_slider, ask_btn]),
    ])
    return ask_btn, example_dropdown, question_input, top_k_slider, example_questions


@app.cell
def _rag_answer(question_input, example_dropdown, ask_btn, top_k_slider):
    from pathlib import Path

    question = ""
    if example_dropdown.value and example_dropdown.value != "(pick an example)":
        question = example_dropdown.value
    elif question_input.value:
        question = question_input.value

    if ask_btn.value and question:
        if not Path("chroma_db").exists():
            mo.md("⚠️ **Vector store not found.** Run `python app.py ingest --wikipedia` first.")
        else:
            import os
            if not os.getenv("ANTHROPIC_API_KEY"):
                mo.md("⚠️ **ANTHROPIC_API_KEY not set.** Add it to your .env file.")
            else:
                from rag import IrishFolkloreChain, IrishPlacesRetriever
                retriever = IrishPlacesRetriever(top_k=top_k_slider.value)
                chain = IrishFolkloreChain(retriever=retriever)

                answer, source_docs = chain.ask_with_sources(question)

                source_table = mo.ui.table(
                    data=[
                        {
                            "Source": doc.metadata.get("source", ""),
                            "Title": doc.metadata.get("title") or doc.metadata.get("name_en") or "—",
                            "Preview": doc.page_content[:120] + "...",
                        }
                        for doc in source_docs
                    ],
                    label="Retrieved documents",
                )

                mo.vstack([
                    mo.callout(mo.md(answer), kind="info"),
                    source_table,
                ])
    else:
        mo.md("_Ask a question above to see the answer here._")
    return ()


@app.cell
def _section_retrieval_explorer():
    mo.md("---\n## 3. Retrieval Explorer\nInspect raw retrieval results — useful for debugging and tuning.")
    return ()


@app.cell
def _retrieval_ui():
    retrieval_query = mo.ui.text(
        placeholder="e.g. banshee legend Connacht",
        label="Retrieval query",
        full_width=True,
    )
    source_filter = mo.ui.dropdown(
        options=["all", "duchas", "logainm", "wikipedia", "census_wikipedia"],
        value="all",
        label="Filter by source",
    )
    retrieval_k = mo.ui.slider(start=1, stop=20, step=1, value=5, label="Top K")
    retrieve_btn = mo.ui.run_button(label="Retrieve")
    mo.hstack([retrieval_query, source_filter, retrieval_k, retrieve_btn])
    return retrieve_btn, retrieval_query, retrieval_k, source_filter


@app.cell
def _retrieval_results(retrieval_query, source_filter, retrieval_k, retrieve_btn):
    from pathlib import Path

    if retrieve_btn.value and retrieval_query.value:
        if not Path("chroma_db").exists():
            mo.md("⚠️ **Vector store not found.** Run `python app.py ingest --wikipedia` first.")
        else:
            from rag import IrishPlacesRetriever
            retriever = IrishPlacesRetriever(top_k=retrieval_k.value)

            if source_filter.value == "all":
                results = retriever.retrieve_with_scores(retrieval_query.value, top_k=retrieval_k.value)
            else:
                raw = retriever.retrieve_by_source(retrieval_query.value, source_filter.value, top_k=retrieval_k.value)
                results = [(doc, None) for doc in raw]

            if results:
                rows = [
                    {
                        "Score": f"{score:.3f}" if score is not None else "—",
                        "Source": doc.metadata.get("source", ""),
                        "Title": doc.metadata.get("title") or doc.metadata.get("name_en") or "—",
                        "County": doc.metadata.get("county", ""),
                        "Text preview": doc.page_content[:150] + "...",
                    }
                    for doc, score in results
                ]
                mo.ui.table(data=rows, label=f"{len(rows)} results")
            else:
                mo.md("_No results found._")
    else:
        mo.md("_Enter a query above to see raw retrieval results._")
    return ()


if __name__ == "__main__":
    app.run()
