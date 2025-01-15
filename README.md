# AI Video & Audio Meeting Assistant 🎬

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![Chroma DB](https://img.shields.io/badge/VectorDB-Chroma-purple.svg)](https://www.trychroma.com/)

An end-to-end AI-powered video meeting analyzer, automated summarizer, key decision extractor, and RAG conversational assistant.

---

## 🌟 Features

- 📹 **YouTube & Local File Processing**: Seamlessly ingests YouTube video URLs or local `.mp4`, `.mp3`, `.wav`, `.m4a`, and `.webm` files.
- 🎙️ **Hybrid Speech-to-Text Engines**:
  - **OpenAI Whisper (Local)**: High-performance offline transcription for English content.
  - **Sarvam AI STT & Translation**: Specialized transcription and translation for Hinglish / Indian regional languages.
- 🧠 **LangChain & LLM Analytics**:
  - Map-Reduce meeting summarization.
  - Extraction of structured **Action Items** (Task, Assigned Owner, Deadline).
  - Extraction of **Key Decisions** and **Open Questions**.
- ⚡ **RAG Context Querying**: Local **Chroma Vector Store** powered by HuggingFace sentence transformer (`all-MiniLM-L6-v2`) embeddings to chat interactively with meeting transcripts.
- 🎨 **Futuristic UI & Exporters**: Neon Cyberpunk Streamlit web dashboard with direct Markdown and JSON report download options.

---

## 🏗️ Architecture Flow

```
[ Video / Audio Input ] ──► [ Audio Converter & Chunker ] ──► [ Speech-To-Text Engine ]
                                                                      │
                                                                      ▼
[ Interactive RAG Chat ] ◄── [ Chroma Vector DB ] ◄── [ LLM Summarizer & Extractor ]
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- FFmpeg installed and added to system PATH

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shubhamsura/AI-Video-Assistant.git
   cd AI-Video-Assistant
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Set your API keys:
   ```env
   MISTRAL_API_KEY=your_mistral_api_key
   SARVAM_API_KEY=your_sarvam_api_key
   ```

---

## 💻 Running the Project

### Streamlit Web App
Launch the interactive dashboard:
```bash
streamlit run app.py
```

### CLI Mode
Run pipeline in terminal:
```bash
python main.py
```

### Run Unit Tests
Validate helper utilities:
```bash
python -m unittest test.py
```

---

## 📂 Project Structure

```
AI-Video-Assistant/
├── app.py                  # Streamlit web application dashboard
├── main.py                 # Command-line interface runner
├── test.py                 # Unit testing suite
├── requirements.txt        # Package dependencies
├── .env.example            # Environment template file
├── engine/
│   ├── config.py           # Application configuration manager
│   ├── models.py           # Typed dataclasses & metric containers
│   ├── transcriber.py      # Whisper and Sarvam STT engines
│   ├── summarizer.py       # LangChain Map-Reduce meeting summarizer
│   ├── extractor.py        # Action item, decision & question extractor
│   ├── vector_store.py     # Chroma vector database pipeline
│   └── rag_engine.py       # RAG chain implementation
└── helpers/
    ├── audio_processor.py  # YouTube downloader & audio chunker
    └── exporter.py         # Markdown and JSON report generator
```
