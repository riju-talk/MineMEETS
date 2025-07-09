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
