import streamlit as st
import time
import os
from engine.config import config
from helpers.audio_processor import process_input
from helpers.exporter import export_markdown_report, export_json_report
from engine.transcriber import transcribe_all
from engine.summarizer import summarize, generate_title
from engine.extractor import extract_action_items, extract_key_decisions, extract_questions
from engine.rag_engine import build_rag_chain, ask_question

# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

:root {
    --bg-dark: #0a0a0f;
    --surface-dark: #111118;
    --surface-card: #181824;
    --border-color: #272738;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --text-primary: #f3f4f6;
    --text-secondary: #9ca3af;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: var(--bg-dark) !important;
}

.brand-header {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.12), rgba(6, 182, 212, 0.08));
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #c084fc 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}

.engine-tag {
    display: inline-block;
    background: rgba(139, 92, 246, 0.2);
    color: #c084fc;
    border: 1px solid rgba(139, 92, 246, 0.4);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 1rem;
}

.insight-card {
    background: var(--surface-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    transition: all 0.2s ease-in-out;
}

.insight-card:hover {
    border-color: var(--accent-purple);
    transform: translateY(-2px);
}

.metric-pill {
    display: inline-block;
    background: rgba(6, 182, 212, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(6, 182, 212, 0.3);
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent-purple), #6d28d9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.5rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

.stButton > button:hover {
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Top Banner ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <h1 class="hero-title">🎬 AI Video & Meeting Assistant</h1>
    <div class="hero-subtitle">Automated Video Summarization, Action Items Extraction & Interactive RAG Assistant</div>
    <div class="engine-tag">AI Video Assistant Engine</div>
</div>
""", unsafe_allow_html=True)

# Config warnings
config_warnings = config.validate()
for warning in config_warnings:
    st.warning(f"⚠️ Configuration Warning: {warning}")

# ─── Sidebar Controls ────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Pipeline Settings")
source_type = st.sidebar.radio("Input Source", ["YouTube URL", "Local Audio/Video File"])
language = st.sidebar.selectbox(
    "Language Engine",
    ["english", "hinglish"],
    help="English uses Whisper locally; Hinglish uses Sarvam AI STT & Translation API.",
)

source_input = None
if source_type == "YouTube URL":
    source_input = st.sidebar.text_input("Enter YouTube Link", placeholder="https://www.youtube.com/watch?v=...")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Video/Audio File", type=["mp4", "mp3", "wav", "m4a", "webm"])
    if uploaded_file:
        os.makedirs(config.DOWNLOAD_DIR, exist_ok=True)
        source_input = os.path.join(config.DOWNLOAD_DIR, uploaded_file.name)
        with open(source_input, "wb") as f:
            f.write(uploaded_file.getbuffer())

if st.sidebar.button("🚀 Analyze Meeting", type="primary") and source_input:
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("⏳ Step 1/4: Audio extraction & chunking...")
    progress_bar.progress(20)
    chunks = process_input(source_input)

    status_text.text("🎙️ Step 2/4: Speech-to-text transcription...")
    progress_bar.progress(45)
    transcript = transcribe_all(chunks, language)

    status_text.text("🧠 Step 3/4: Executive summary & action item extraction...")
    progress_bar.progress(75)
    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    status_text.text("⚡ Step 4/4: Vector indexing into Chroma RAG database...")
    progress_bar.progress(90)
    rag_chain = build_rag_chain(transcript)

    progress_bar.progress(100)
    status_text.text("✅ Pipeline processing complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()

    st.session_state["pipeline_result"] = {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "word_count": len(transcript.split()),
        "chunk_count": len(chunks),
    }

# ─── Results View ────────────────────────────────────────────────────────────────
if "pipeline_result" in st.session_state:
    res = st.session_state["pipeline_result"]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## 📌 {res['title']}")
        st.markdown(
            f"<span class='metric-pill'>📝 Words: {res['word_count']}</span>"
            f"<span class='metric-pill'>🧩 Audio Segments: {res['chunk_count']}</span>",
            unsafe_allow_html=True,
        )
    with col2:
        md_data = export_markdown_report(res)
        json_data = export_json_report(res)
        st.download_button("📥 Report (.md)", md_data, file_name="meeting_summary.md", mime="text/markdown")
        st.download_button("📊 Report (.json)", json_data, file_name="meeting_summary.json", mime="application/json")

    st.write("")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Executive Summary",
        "✅ Action Items",
        "🔑 Key Decisions",
        "❓ Open Questions",
        "📜 Transcript",
        "💬 Interactive RAG Assistant",
    ])

    with tab1:
        st.markdown(f"<div class='insight-card'>{res['summary']}</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown(f"<div class='insight-card'>{res['action_items']}</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown(f"<div class='insight-card'>{res['key_decisions']}</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown(f"<div class='insight-card'>{res['open_questions']}</div>", unsafe_allow_html=True)

    with tab5:
        st.text_area("Full Transcript", res["transcript"], height=420)

    with tab6:
        st.markdown("### 💬 Chat with your Meeting Transcript Context")
        q = st.text_input("Ask any question regarding the meeting context:")
        if st.button("Query Assistant") and q:
            with st.spinner("Searching vector index context..."):
                answer = ask_question(res["rag_chain"], q)
                st.markdown(f"<div class='insight-card'><b>🤖 Assistant:</b><br>{answer}</div>", unsafe_allow_html=True)
