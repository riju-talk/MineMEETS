# MineMEETS: Agentic RAG Meeting Assistant

MineMEETS is a fully agentic, human-friendly retrieval-augmented generation (RAG) platform that lets users upload meeting transcripts or audio/video recordings, analyze them for insights with local or cloud language models, and access seamless Q&A and insights—all via a user-friendly Streamlit web app.

---

## Features

- **Transcript/Audio/Video Upload**: Upload any meeting transcript (.txt), audio (.mp3/.wav), or video (.mp4/.webm), processed instantly.
- **Automated Chunking & Vectorization**: Data is chunked and uploaded to Pinecone vector db for efficient retrieval.
- **Query & Summarization**: Ask questions using a local LLM (via Ollama) or fallback to HuggingFace.
- **Email & Insight Extraction**: Extract insights, action items, and email them with one click.
- **Streamlit UI**: All features accessible via a beautiful, accessible web interface.

---

## Quickstart

1. **Install Requirements**

```bash
pip install -r requirements.txt
```

2. **Create `.env` with these minimum variables:**

```
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-env
EMAIL_ADDRESS=your@email.xyz
EMAIL_PASSWORD=your-email-app-password
OLLAMA_MODEL=llama3 (or other local model)
```

3. **[Recommended] Launch Ollama locally:**

- Download and start from https://ollama.com/download
- `ollama pull llama3` (or your favorite model)
- Ollama will serve HTTP at `http://localhost:11434` by default.


4. **Run Streamlit app:**

```bash
streamlit run app.py
```

5. **Enjoy!**
- Upload .txt/.mp3/.mp4 in the sidebar, process your meeting, and ask magic questions!

---

## Architecture and How It Works

- **Agents** in `/agents`:
  - `audio_agent.py`: uses Whisper for transcription
  - `coordinator.py`: orchestrates upload, chunk, store, analyze
  - `llm.py`: dynamically uses Ollama or HF based on your env
  - `qa_agent.py`: RAG querying with prompt-templates
  - `insights_agent.py`: LLM-based summarization
  - `email_agent.py`: handles SMTP 
- **Prompt Templates**: Human-editable .txt prompt scripts in `/prompts`, loaded automatically and used in QA/insights/email.
- **Pinecone Vectorstore**: Stores all chunks for blazing-fast similarity querying.
- **Streamlit UI**: Modern, accessible, and feedback-focused, with all errors and progress clearly shown.

## Troubleshooting
- Missing API keys? Add them in `.env`.
- Ollama not running? Add or remove OLLAMA_MODEL from `.env` to control local/remote LLM use.
- Require more models? Just set your HuggingFace or Ollama preferred model in the `.env`.

---

## Why it Works
- **Agentic**: Each component is pluggable, robustly handles errors, and self-documents its intent.
- **Humanized**: Code and comments are written for human maintainers. Feedback in the UI is actionable, not cryptic.
- **Windows Friendly**: Paths and dependencies work on Windows by default.
- **Extensible & Prompt-driven**: Easily add more .txt prompt templates and agents—no refactor needed.


## Contributors
- Designed and curated by your AI + you. For support and additions, open a pull-request!

---

## License
MIT License

# MineMEETs – AI Meeting Intelligence Agent

MinMEETs is your silent meeting analyst. It listens, learns, and summarizes your meetings, extracting key action items, decisions, and follow-ups. Built using Whisper, vector DBs, and GROQ's blazing-fast LLM APIs, MinMEETs also mails summaries and transcripts to your team.

---

## 🚀 Features

- 🎙️ Transcribes meeting audio (.mp3, .wav) using Whisper
- 🧠 Embeds transcripts with Sentence Transformers and stores in FAISS
- 🔍 RAG-based querying: “What tasks were assigned to me?”
- 📝 Extracts decisions, owners, and deadlines using GROQ's LLM
- 📧 Emails transcripts and summaries to specified recipients

---

## 🛠️ Tech Stack

- Python, FastAPI
- Whisper (openai-whisper or faster alternatives)
- Sentence Transformers (for semantic embeddings)
- FAISS (Vector DB)
- GROQ API (LLM inference)
- SendGrid / SMTP (for emails)

---
