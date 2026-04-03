"""
Document embedder.

Splits documents into chunks and embeds them into the Chroma vector store
using a local sentence-transformers model (no API key needed).
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, CHROMA_PERSIST_DIR

COLLECTION_NAME = "irish_places"


class Embedder:
    """Chunk, embed, and persist documents into Chroma."""

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        persist_dir: str = CHROMA_PERSIST_DIR,
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.persist_dir = persist_dir
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def embed_documents(self, documents: list[Document], batch_size: int = 100) -> Chroma:
        """
        Split and embed a list of Documents into Chroma.

        Documents are upserted so re-running is safe — duplicates won't pile up
        as long as the source metadata is stable.
        """
        chunks = self.splitter.split_documents(documents)
        print(f"  Embedding {len(chunks)} chunks from {len(documents)} documents...")

        # Build in batches to avoid OOM on large corpora
        vectorstore = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    collection_name=COLLECTION_NAME,
                    persist_directory=self.persist_dir,
                )
            else:
                vectorstore.add_documents(batch)
            print(f"  ... {min(i + batch_size, len(chunks))}/{len(chunks)} chunks embedded")

        return vectorstore

    def load_existing(self) -> Chroma:
        """Load an already-persisted Chroma collection (no re-embedding)."""
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
