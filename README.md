# MineMEETS: Agentic Multimodal RAG Meeting Assistant

MineMEETS is a multimodal RAG app that ingests text transcripts, transcribes audio/video, and embeds screenshots/images for retrieval. It extracts insights and enables scoped Q&A over your uploaded content via a Streamlit UI using Ollama for all LLM tasks.

---

## Features

- **Text + Audio/Video**: Upload `.txt` transcripts or audio/video for Whisper transcription.
- **Screenshots/Images**: Upload `.png/.jpg/.jpeg/.webp/.bmp`; embedded with CLIP ViT-B/32.
- **Automated Chunking & Vectorization**: Text chunks + image vectors stored in Pinecone.
- **Meeting-Scoped Namespaces**: Each upload is isolated per `meeting_id`.
- **Insights & Q&A**: Summaries, key points, action items, and RAG-based Q&A powered by Ollama.
- **Streamlit UI**: Simple upload → process → ask flow.

---

## Quickstart

1. **Install Requirements**

```bash
pip install -r requirements.txt
```

2. **Create `.env` with these variables:**

```
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-env-or-project-region
PINECONE_INDEX=mine-meets
PINECONE_METRIC=cosine
EMBEDDING_MODEL=sentence-transformers/clip-ViT-B-32
EMBEDDING_DIM=512

# Whisper
WHISPER_MODEL=base
WHISPER_CACHE_DIR=.cache/whisper

# Ollama (required)
OLLAMA_MODEL=llama3
OLLAMA_HOST=http://localhost:11434
```

3. **Launch Ollama locally:**

- Download and start from https://ollama.com/download
- `ollama pull llama3` (or your favorite model)
- Ollama will serve HTTP at `http://localhost:11434` by default.

4. **Run Streamlit app:**

```bash
streamlit run app.py
```

5. **Use it**
- Upload `.txt`, audio/video, and screenshots.
- Click "Process Meeting"; then ask questions in the Q&A tab.

---

## Architecture and How It Works

- **Agents** in `agents/`:
  - `audio_agent.py`: Whisper transcription for audio/video.
  - `image_agent.py`: CLIP ViT-B/32 embeddings for screenshots/images.
  - `coordinator.py`: orchestrates ingestion, chunking, embeddings, Pinecone upserts, insights.
  - `qa_agent.py`: RAG querying over a meeting namespace using Ollama.
  - `insights_agent.py`: Summaries, key points, action items using Ollama.
  - `llm.py`: Provides Ollama LLM provider.
  - `pinecone_db.py`: Pinecone index management and vector store helpers.
- **Vector DB**:
  - Text: embedded via CLIP text encoder (same space as images) using `sentence-transformers/clip-ViT-B-32` (512-d).
  - Images: vectors upserted directly via `upsert_vectors()` to same namespace.
  - Retrieval: meeting-scoped via `namespace=meeting_id` and cosine similarity.
- **UI**: Streamlit app in `app.py`.

## Troubleshooting
- **Pinecone dimension mismatch**: Ensure your index dimension equals `EMBEDDING_DIM` (default 512 for CLIP ViT-B/32).
- **Whisper downloads slowly**: set `WHISPER_CACHE_DIR` to a persistent path.
- **Ollama not available**: Ensure Ollama is running and `OLLAMA_MODEL` is set.

---

## Why it Works
- **Agentic**: Each component is pluggable, robustly handles errors, and self-documents its intent.
- **Humanized**: Code and comments are written for human maintainers. Feedback in the UI is actionable, not cryptic.
- **Windows Friendly**: Paths and dependencies work on Windows by default.
- **Extensible & Prompt-driven**: Easily add more .txt prompt templates and agents—no refactor needed.

---

## License
MIT License
