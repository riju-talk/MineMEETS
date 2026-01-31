# MineMEETS - Quick Reference Card

**Production MLOps Multimodal RAG Platform**

---

## 🚀 Quick Commands

```bash
# Setup
make install                 # Install dependencies
cp .env.example .env        # Copy config template

# Development
make run                    # Run application
make test                   # Run tests
make lint                   # Check code quality
make format                 # Auto-format code
make check                  # Run all quality checks

# Docker
make docker-build           # Build image
make docker-run             # Run in container
make docker-stop            # Stop containers

# Cleanup
make clean                  # Remove generated files
```

---

## 📦 Project Structure

```
MineMEETS/
├── agents/              # Core pipeline modules
├── tests/               # Test suite
├── data/raw/            # Input files
├── app.py               # Gradio UI
├── requirements.txt     # Dependencies
├── Dockerfile           # Container
├── Makefile             # Commands
└── README.md            # Documentation
```

---

## 🔧 Configuration (.env)

```env
PINECONE_API_KEY=your-key-here
WHISPER_MODEL=base
OLLAMA_MODEL=llama3.1
OLLAMA_HOST=http://localhost:11434
```

---

## 🎯 Core Capabilities

| Feature | Technology | Purpose |
|---------|-----------|---------|
| Text Processing | PyPDF2, python-docx | Extract & chunk |
| Audio Processing | Whisper | Transcription |
| Image Processing | CLIP ViT-B/32 | Visual embeddings |
| Vector Storage | Pinecone | Semantic search |
| LLM Inference | Ollama | Q&A generation |
| UI | Gradio | Web interface |

---

## 📊 Architecture Layers

1. **Ingestion** → Validate & preprocess
2. **Embedding** → Generate vectors (512-dim)
3. **Storage** → Pinecone with namespaces
4. **Retrieval** → Hybrid search
5. **Inference** → Local LLM (Ollama)
6. **UI** → Gradio interface

---

## 🔍 Retrieval Strategies

1. **Semantic Search** (boost: 1.0)
   - Vector similarity via Pinecone
   
2. **Keyword Search** (boost: 0.8)
   - Extract keywords, search each
   
3. **Query Expansion** (boost: 0.6)
   - For general questions

---

## 📝 Pinecone Metadata Schema

```json
{
  "meeting_id": "meeting_20260131",
  "modality": "text|audio|image",
  "type": "text_chunk|audio_segment",
  "chunk_index": 14,
  "position": 14,
  "text": "Content...",
  "timestamp_start": 120,
  "timestamp_end": 145
}
```

---

## 🧪 Testing

```bash
pytest tests/ -v                    # Run all tests
pytest tests/ --cov=agents          # With coverage
pytest tests/test_llm.py -v         # Single file
```

---

## 🐳 Docker

```bash
# Build
docker build -t minemeets:latest .

# Run
docker run -p 7860:7860 \
  --env-file .env \
  minemeets:latest

# Compose
docker-compose up --build
```

---

## 🔒 Security

- ✅ API keys in .env (not committed)
- ✅ Local LLM (no external calls)
- ✅ File validation
- ✅ Container isolation

---

## 📈 Monitoring Points

- Ingestion success/failure rate
- Embedding dimension validation
- Upsert counts per meeting
- Query latency (p50, p95, p99)
- LLM generation time

---

## 🔄 Common Operations

### Reprocess Meeting
```python
db.delete_vectors(
    namespace=meeting_id, 
    delete_all=True
)
coordinator.process_meeting(meeting_data)
```

### Check Database Stats
```python
stats = db.get_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
```

### Query Meeting
```python
result = qa_agent.process(
    question="What were the main points?",
    context={"meeting_id": "meeting_20260131"}
)
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Activate venv, reinstall deps |
| Ollama connection | Check `ollama serve` is running |
| Pinecone errors | Verify API key in .env |
| Docker networking | Use `host.docker.internal:11434` |

---

## 📚 Documentation

- **README.md** - Overview & quick start
- **ARCHITECTURE.md** - System design
- **QUICKSTART.md** - Step-by-step setup
- **CONTRIBUTING.md** - Development guide
- **TRANSFORMATION_SUMMARY.md** - What changed

---

## 🎤 Interview Talking Points

**Key Phrases:**
- "Production ML pipeline"
- "Namespace isolation"
- "Hybrid retrieval"
- "Deterministic preprocessing"
- "Operational observability"
- "Idempotent operations"

**What NOT to say:**
- "Research project"
- "Experimental system"
- "Novel architecture"
- "Cutting-edge model"

---

## ✅ Quality Metrics

- **Test Coverage:** >80% target
- **Code Formatting:** Black (100 char)
- **Linting:** Pylint score >8.0
- **Type Hints:** MyPy checks
- **CI/CD:** All checks green

---

## 🚀 Deployment Checklist

- [ ] Environment variables configured
- [ ] Ollama running locally
- [ ] Pinecone API key valid
- [ ] Dependencies installed
- [ ] Tests passing
- [ ] Docker builds successfully
- [ ] Application accessible at :7860

---

## 📞 Support

- Issues: GitHub Issues
- Docs: README.md, ARCHITECTURE.md
- Tests: Check test files for examples

---

**Remember:** This is an **MLOps platform**, not a research project!

Focus on: Reliability, Observability, Deployment, Testing, Documentation
