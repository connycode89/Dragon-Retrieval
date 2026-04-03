"""
Retriever.

Wraps Chroma with source-aware retrieval so we can see which data source
each result came from (Dúchas, Logainm, Wikipedia, Census).
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, RETRIEVAL_TOP_K

SOURCE_LABELS = {
    "duchas": "Dúchas Folklore",
    "logainm": "Logainm Placenames",
    "wikipedia": "Wikipedia",
    "census_wikipedia": "Historical Demographics",
    "cso": "CSO Statistics",
}

COLLECTION_NAME = "irish_places"


class IrishPlacesRetriever:
    """Retrieve relevant documents from the Irish places vector store."""

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = RETRIEVAL_TOP_K,
    ):
        self.top_k = top_k
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve the most relevant documents for a query."""
        k = top_k or self.top_k
        return self.vectorstore.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str, top_k: int | None = None) -> list[tuple[Document, float]]:
        """Retrieve documents with their relevance scores."""
        k = top_k or self.top_k
        return self.vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def retrieve_by_source(self, query: str, source: str, top_k: int = 3) -> list[Document]:
        """Retrieve documents from a specific source only."""
        return self.vectorstore.similarity_search(
            query,
            k=top_k,
            filter={"source": source},
        )

    def get_source_label(self, doc: Document) -> str:
        source = doc.metadata.get("source", "unknown")
        return SOURCE_LABELS.get(source, source.capitalize())

    def as_langchain_retriever(self, top_k: int | None = None):
        """Return a standard LangChain retriever interface."""
        return self.vectorstore.as_retriever(
            search_kwargs={"k": top_k or self.top_k}
        )
