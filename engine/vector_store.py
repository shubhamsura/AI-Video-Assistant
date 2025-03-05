from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from engine.config import config


def get_embeddings():
    """Load CPU-compatible sentence transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(transcript: str) -> Chroma:
    """Chunk transcript text and build persisted Chroma vector database."""
    print("Chunking transcript for vector indexing...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)

    documents = [
        Document(
            page_content=chunk,
            metadata={"chunk_id": i, "source": "meeting_transcript"},
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
    )
    print(f"Successfully indexed {len(documents)} vector chunks into Chroma DB.")
    return vector_db


def load_vector_store() -> Chroma:
    """Load existing persisted Chroma vector database from storage."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = config.DEFAULT_TOP_K):
    """Retrieve top-k relevant document chunks via similarity search."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
