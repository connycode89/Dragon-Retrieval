"""
LangChain RAG chain.

Combines retrieved folklore, placename, and demographic context with
Claude to answer questions about Irish places and their history.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from .retriever import IrishPlacesRetriever
from config import CLAUDE_MODEL, ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are a knowledgeable guide to Irish folklore, place names, and local history.
You draw on three sources of knowledge:

1. **Dúchas / Schools' Collection** — folklore collected from Irish schoolchildren in the 1930s,
   including fairy stories, local legends, folk cures, and oral traditions.
2. **Logainm** — the authoritative database of Irish place names, with etymologies and
   the meanings behind townland and parish names in Irish and English.
3. **Wikipedia & historical records** — background on the history, geography, and demographics
   of Irish places.

When answering:
- Weave together the different sources into a coherent, engaging response.
- Highlight the etymology of place names when relevant — Irish placenames often contain
  vivid meanings ("the fort of the red slaughter", "the hill of the fairy woman").
- Share folklore and legends in a way that captures their strange, eerie, or humorous quality.
- Ground stories in specific places and people where the sources allow.
- If sources are silent on something, say so honestly rather than inventing.

Answer in a warm, storytelling tone — as if you're sitting by a fire telling someone
about the place they're asking about."""

HUMAN_PROMPT = """Question: {question}

Relevant sources:
{context}

Please answer based on the sources above."""


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string for the prompt."""
    if not docs:
        return "No relevant sources found."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        title = (
            doc.metadata.get("title")
            or doc.metadata.get("name_en")
            or doc.metadata.get("county")
            or f"Source {i}"
        )
        parts.append(f"[{i}] ({source.upper()}) {title}\n{doc.page_content[:600]}")

    return "\n\n---\n\n".join(parts)


class IrishFolkloreChain:
    """End-to-end RAG chain for querying Irish folklore and place history."""

    def __init__(
        self,
        retriever: IrishPlacesRetriever | None = None,
        model: str = CLAUDE_MODEL,
        api_key: str = ANTHROPIC_API_KEY,
    ):
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY in your .env file."
            )

        self.retriever = retriever or IrishPlacesRetriever()
        self.llm = ChatAnthropic(model=model, anthropic_api_key=api_key)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])

        self.chain = (
            {
                "context": RunnableLambda(
                    lambda x: format_docs(self.retriever.retrieve(x["question"]))
                ),
                "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """Ask a question and return the answer as a string."""
        return self.chain.invoke({"question": question})

    def ask_with_sources(self, question: str) -> tuple[str, list[Document]]:
        """Ask a question and return both the answer and the source documents."""
        docs = self.retriever.retrieve(question)
        context = format_docs(docs)

        answer = (
            self.prompt
            | self.llm
            | StrOutputParser()
        ).invoke({"question": question, "context": context})

        return answer, docs

    def stream(self, question: str):
        """Stream the answer token by token."""
        yield from self.chain.stream({"question": question})
