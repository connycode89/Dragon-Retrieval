# Dragon Retrieval

A RAG (Retrieval-Augmented Generation) application for exploring Irish folklore, placename history, and local heritage. Ask questions about any Irish place and get answers drawn from three rich sources: the Dúchas folklore archives, the Logainm placename database, and Wikipedia.

> *"What legends surround Knocknarea in Sligo?"*
> *"What does the name Dún na nGall mean in Irish?"*
> *"Tell me about fairy forts in County Clare."*

---

## Data Sources

| Source | What it contains | Auth |
|---|---|---|
| [Dúchas / Schools' Collection](https://www.duchas.ie) | 740,000+ pages of folklore collected by Irish schoolchildren in the 1930s — fairy stories, local legends, folk cures, curses | API key (free) |
| [Logainm](https://www.logainm.ie) | Authoritative bilingual database of 90,000+ Irish placenames with etymologies and historical forms | API key (free) |
| [Wikipedia](https://en.wikipedia.org) | County and townland history, geography, demographics | None |
| [Wikidata SPARQL](https://query.wikidata.org) | Historical population figures for Irish places | None |
| [data.gov.ie](https://data.gov.ie) | Aggregate historical datasets and townland data | None |

### Getting API Keys

- **Dúchas**: email [eolas@duchas.ie](mailto:eolas@duchas.ie) — mention it's for a personal/educational project
- **Logainm**: register at [gaois.ie](https://gaois.ie) or email [logainm@dcu.ie](mailto:logainm@dcu.ie)

You can start immediately without these keys using `--wikipedia` and `--census` — the Dúchas and Logainm loaders will be skipped gracefully if their keys are absent.

---

## Setup

```bash
git clone <repo>
cd Dragon-Retrieval
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your keys:

```env
ANTHROPIC_API_KEY=sk-ant-...
DUCHAS_API_KEY=...          # optional until you have it
LOGAINM_API_KEY=...         # optional until you have it
```

---

## Usage

### 1. Ingest data into the vector store

```bash
# Start here — no API keys needed
python app.py ingest --wikipedia

# Add historical demographics (Wikidata + data.gov.ie, no key needed)
python app.py ingest --census

# Add Irish placename etymologies (requires LOGAINM_API_KEY)
python app.py ingest --logainm

# Add folklore (requires DUCHAS_API_KEY)
python app.py ingest --duchas

# Everything at once
python app.py ingest --all
```

### 2. Ask a question

```bash
python app.py ask "What legends surround Knocknarea in Sligo?"
python app.py ask "What does the name Connemara mean?" --show-sources
python app.py ask "Tell me about banshees in Connacht"
```

### 3. Interactive chat

```bash
python app.py chat
```

### 4. Marimo notebook (interactive exploration)

```bash
marimo edit notebook.py     # editable notebook
marimo run notebook.py      # read-only app
```

The notebook lets you search sources live, inspect retrieved documents, tune retrieval parameters, and ask full RAG questions — all without touching the CLI.

---

## Project Structure

```
Dragon-Retrieval/
├── config.py                  # env vars, API URLs, model/chunk settings
├── app.py                     # CLI entry point
├── notebook.py                # Marimo interactive notebook
├── requirements.txt
├── .env.example
├── ingest/
│   ├── duchas_loader.py       # Dúchas Schools' Collection API
│   ├── logainm_loader.py      # Logainm placename API
│   ├── wikipedia_loader.py    # Wikipedia MediaWiki API
│   └── census_loader.py       # Wikidata SPARQL + data.gov.ie + Wikipedia demographics
└── rag/
    ├── embedder.py            # Chunk, embed, persist to Chroma
    ├── retriever.py           # Similarity search with source filtering
    └── chain.py               # LangChain RAG chain with streaming
```

---

## Architecture

```
User query
    │
    ▼
IrishFolkloreChain (rag/chain.py)
    │
    ├── IrishPlacesRetriever → Chroma vector store
    │       similarity search across all ingested sources
    │
    └── Claude (claude-sonnet-4-6)
            prompted to weave together folklore, etymology,
            and historical context into a storytelling response
```

Embeddings use `sentence-transformers/all-MiniLM-L6-v2` locally — no OpenAI key needed for the vector store.

---

## Full Census Microdata (optional)

The 1901 and 1911 Irish censuses have no public API, but [IPUMS International](https://international.ipums.org/international/) offers complete individual-level microdata (4.4M person records each year) as a free download after registration and a signed educational-use license. This can be loaded as an additional ingest source if you want household-level detail.
