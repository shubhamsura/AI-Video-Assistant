from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from engine.config import config
from engine.vector_store import build_vector_store, load_vector_store, get_retriever


def get_llm():
    """Instantiate ChatMistralAI model instance for RAG response generation."""
    return ChatMistralAI(
        model=config.LLM_MODEL_NAME,
        mistral_api_key=config.MISTRAL_API_KEY,
        temperature=0.3,
    )


def format_context_documents(docs: list) -> str:
    """Format document chunks into a formatted string block with source indices."""
    formatted_chunks = []
    for i, doc in enumerate(docs):
        chunk_idx = doc.metadata.get("chunk_id", i + 1)
        formatted_chunks.append(f"[Chunk #{chunk_idx}]\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)


def build_rag_chain(transcript: str):
    """Construct LCEL RAG execution chain for transcript retrieval Q&A."""
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=config.DEFAULT_TOP_K)
    llm = get_llm()

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI meeting assistant. Answer the question using ONLY the meeting context below.\n\n"
                "If the information is not present in the context, explicitly respond:\n"
                "\"I could not find this information in the meeting transcript context.\"\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    return (
        {
            "context": retriever | RunnableLambda(format_context_documents),
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )


def load_rag_chain():
    """Load existing RAG pipeline from disk vector store."""
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=config.DEFAULT_TOP_K)
    llm = get_llm()

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI meeting assistant. Answer the question using ONLY the meeting context below.\n\n"
                "If the information is not present in the context, explicitly respond:\n"
                "\"I could not find this information in the meeting transcript context.\"\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    return (
        {
            "context": retriever | RunnableLambda(format_context_documents),
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )


def ask_question(rag_chain, question: str) -> str:
    """Execute query prompt against active RAG chain."""
    print(f"Querying RAG Chain: {question}")
    answer = rag_chain.invoke(question)
    return answer
