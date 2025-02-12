from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from engine.config import config


def get_llm(temperature: float = 0.3):
    """Instantiate ChatMistralAI language model instance."""
    if not config.MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable is not configured.")
    return ChatMistralAI(
        model=config.LLM_MODEL_NAME,
        mistral_api_key=config.MISTRAL_API_KEY,
        temperature=temperature,
    )


def split_transcript(transcript: str, chunk_size: int = 3000, overlap: int = 200) -> list:
    """Split transcript text into overlapping chunks for map-reduce processing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str, detail_level: str = "bullet") -> str:
    """Execute map-reduce summarization chain on transcript text."""
    llm = get_llm(temperature=0.2)

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert executive meeting assistant. "
                "Summarize this segment of a meeting transcript clearly and concisely.",
            ),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined_summary_input = "\n\n".join(chunk_summaries)

    reduce_system_prompt = (
        "You are a senior executive AI meeting analyst. Synthesize the provided partial summaries "
        "into a comprehensive, professional meeting summary. Use clear headings and structured bullet points."
    )

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", reduce_system_prompt),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined_summary_input)


def generate_title(transcript: str) -> str:
    """Generate a crisp executive meeting title (maximum 8 words)."""
    llm = get_llm(temperature=0.3)
    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript preview, generate a short, professional meeting title "
                    "(max 8 words). Return only the title text without quotes.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000]).strip()
