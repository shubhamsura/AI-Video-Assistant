from engine.config import config
from engine.models import MeetingAnalysisResult
from helpers.audio_processor import process_input
from engine.transcriber import transcribe_all
from engine.summarizer import summarize, generate_title
from engine.extractor import extract_action_items, extract_key_decisions, extract_questions
from engine.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english") -> MeetingAnalysisResult:
    """Execute end-to-end meeting analysis pipeline on input video/audio."""
    print("=" * 60)
    print("Starting AI Video & Audio Assistant Analysis...")
    print("=" * 60)

    # 1. Preprocess & Chunk Audio
    chunks = process_input(source)

    # 2. Multi-Engine Speech-to-Text
    transcript = transcribe_all(chunks, language)
    print(f"\nExtracted Transcript ({len(transcript.split())} words):\n{transcript[:300]}...\n")

    # 3. LLM Insight Extraction
    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    # 4. RAG Indexing
    rag_chain = build_rag_chain(transcript)

    return MeetingAnalysisResult(
        title=title,
        transcript=transcript,
        summary=summary,
        action_items=action_items,
        key_decisions=decisions,
        open_questions=questions,
        rag_chain=rag_chain,
        chunk_count=len(chunks),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Video & Audio Assistant")
    print("=" * 60)

    # Validate config
    warnings = config.validate()
    for w in warnings:
        print(f"⚠️ Warning: {w}")

    source = input("\nEnter YouTube URL or local video/audio file path: ").strip()
    language = input("Target language mode (english/hinglish) [default: english]: ").strip() or "english"

    if source:
        result = run_pipeline(source, language)

        print("\n" + "=" * 60)
        print(f"📌 Meeting Title: {result.title}")
        print(f"📊 Word Count: {result.word_count} words | Chunks: {result.chunk_count}")
        print(f"\n📋 Summary:\n{result.summary}")
        print(f"\n✅ Action Items:\n{result.action_items}")
        print(f"\n🔑 Key Decisions:\n{result.key_decisions}")
        print(f"\n❓ Open Questions:\n{result.open_questions}")
        print("=" * 60)

        # Interactive RAG Session
        print("\n💬 Chat with your meeting transcript context (type 'exit' or 'q' to quit)\n")
        rag_chain = result.rag_chain
        while True:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Session ended.")
                break
            if not question:
                continue
            answer = ask_question(rag_chain, question)
            print(f"\n🤖 Assistant: {answer}\n")
