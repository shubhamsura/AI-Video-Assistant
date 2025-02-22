from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from engine.config import config


def get_llm(temperature: float = 0.1):
    """Instantiate low-temperature ChatMistralAI model for precise extraction."""
    return ChatMistralAI(
        model=config.LLM_MODEL_NAME,
        mistral_api_key=config.MISTRAL_API_KEY,
        temperature=temperature,
    )


def _build_chain(system_prompt: str):
    """Helper to construct LCEL extraction chain."""
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    """Extract action items, responsible owners, and target deadlines."""
    prompt = (
        "You are an expert meeting analyst. Extract all actionable tasks from the transcript.\n"
        "For each task provide:\n"
        "- Task: Description of what needs to be done\n"
        "- Assigned Owner: Person responsible (if mentioned, else 'Unassigned')\n"
        "- Target Deadline: Specified date/time (if mentioned, else 'Not specified')\n"
        "- Priority Level: High / Medium / Low\n\n"
        "Format as a clean numbered list. If none are found, return 'No action items identified.'"
    )
    chain = _build_chain(prompt)
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    """Extract major consensus decisions made during the meeting."""
    prompt = (
        "You are an expert meeting analyst. Extract all major decisions finalized in the transcript.\n"
        "Format as a clean numbered list with brief context for each decision. "
        "If none are found, return 'No key decisions identified.'"
    )
    chain = _build_chain(prompt)
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    """Extract unresolved questions or follow-up items needing clarification."""
    prompt = (
        "You are an expert meeting analyst. Extract all open questions, unresolved issues, "
        "or topics requiring follow-up.\n"
        "Format as a clean numbered list. If none are found, return 'No open questions identified.'"
    )
    chain = _build_chain(prompt)
    return chain.invoke(transcript)
