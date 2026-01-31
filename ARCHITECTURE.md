# MineMEETS System Architecture

**Production MLOps Architecture for Multimodal RAG System**

---

## 📐 System Overview

MineMEETS is designed as a **production-grade ML pipeline** with clear separation of concerns, deterministic processing, and operational observability. This is **not a research prototype** — it's built to demonstrate MLOps best practices.

---

## 🏗️ Architecture Layers

### 1. **Ingestion Layer**

**Purpose:** Validate, route, and preprocess raw meeting content

**Components:**
- `coordinator.py` - Orchestrates multimodal ingestion
- `document_processor.py` - Text extraction and chunking
- `audio_agent.py` - Audio transcription via Whisper
- `image_agent.py` - Image embedding via CLIP

**Key Design Decisions:**
- ✅ **Idempotent operations** - Can safely rerun without duplication
- ✅ **Validation first** - File type, size, format checks before processing
- ✅ **Fallback paths** - Graceful degradation on partial failures
- ✅ **Meeting-scoped namespaces** - Isolation per `meeting_id`

**Data Flow:**
```
Raw File Input → Validation → Type Detection → Agent Routing
    ↓
[Text] → DocumentProcessor → Chunks
[Audio] → AudioAgent (Whisper) → Transcription → Chunks
[Image] → ImageAgent (CLIP) → Visual Embeddings
```

---

### 2. **Embedding Layer**

**Purpose:** Generate consistent 512-dimensional embeddings across all modalities

**Components:**
- `SentenceTransformer` (CLIP ViT-B/32) for text and images
- Whisper transcription → text embeddings
- Dimensional validation before upsert

**Key Design Decisions:**
- ✅ **Unified embedding space** - All modalities use CLIP's shared space
- ✅ **Batch processing** - Embeddings generated in batches for efficiency
- ✅ **Dimension checks** - Strict validation (must be 512-dim)
- ✅ **Error isolation** - Failed embeddings don't block entire batch

**Embedding Pipeline:**
```
Text → SentenceTransformer → [512-dim vector]
Audio → Whisper → Text → SentenceTransformer → [512-dim vector]
Image → CLIP Encoder → [512-dim vector]
```

---

### 3. **Vector Storage Layer**

**Purpose:** Reliable, scalable storage and retrieval of embeddings

**Components:**
- `pinecone_db.py` - Pinecone client wrapper
- Namespace strategy for multi-tenancy
- Metadata schema for filtering

**Key Design Decisions:**
- ✅ **Namespace per meeting** - Enables per-meeting operations
- ✅ **Metadata-rich schema** - Supports filtering, debugging, auditing
- ✅ **Batch upserts** - Configurable batch size (default: 100)
- ✅ **Safe deletion** - Namespace-scoped for rollback

**Pinecone Schema:**
```json
{
  "id": "meeting_20260131_chunk_14",
  "values": [0.12, -0.45, ...],  // 512-dim
  "metadata": {
    "meeting_id": "meeting_20260131",
    "modality": "text",
    "type": "text_chunk",
    "chunk_index": 14,
    "position": 14,
    "timestamp_start": 120,  // For audio
    "timestamp_end": 145,
    "source": "transcript",
    "text": "Original text content..."
  }
}
```

---

### 4. **Retrieval Layer**

**Purpose:** Hybrid search strategies for high-quality context retrieval

**Components:**
- `multimodal_rag.py` - Retrieval logic
- `MultimodalRetriever` - Hybrid search orchestration

**Key Design Decisions:**
- ✅ **Hybrid search** - Semantic + keyword + query expansion
- ✅ **Deterministic ranking** - No stochastic behavior
- ✅ **Modality awareness** - Cross-modal context assembly
- ✅ **Deduplication** - Remove duplicate results
- ✅ **Score normalization** - Consistent ranking across strategies

**Retrieval Strategies:**

1. **Semantic Search** (Primary)
   - Vector similarity via Pinecone
   - Returns top-10 most similar chunks
   - Boost factor: 1.0

2. **Keyword Search** (Recall Enhancement)
   - Extract keywords from query
   - Search for each keyword
   - Boost factor: 0.8

3. **Query Expansion** (General Questions)
   - Detect general queries ("summary", "overview")
   - Expand to broader search terms
   - Boost factor: 0.6

**Hybrid Scoring:**
```python
hybrid_score = (
    original_score *
    search_type_boost *
    content_type_boost *
    position_boost *
    keyword_overlap_boost
)
```

---

### 5. **Inference Layer**

**Purpose:** Stateless LLM inference with context assembly

**Components:**
- `llm.py` - Ollama HTTP client
- `qa_agent.py` - Q&A orchestration
- Context formatting and prompt engineering

**Key Design Decisions:**
- ✅ **Local inference** - No external API dependencies
- ✅ **Stateless execution** - Each query is independent
- ✅ **Context constraints** - Token limit management
- ✅ **Source attribution** - Track which chunks contributed
- ✅ **Modality indicators** - LLM knows source modality

**Prompt Structure:**
```
System Instructions
   ↓
Available Modalities: [text, audio, image]
   ↓
Context by Modality:
  - TEXT TRANSCRIPTS: [chunks...]
  - AUDIO TRANSCRIPTION: [segments...]
  - IMAGE DESCRIPTIONS: [descriptions...]
   ↓
User Question
   ↓
Answer Guidelines
```

---

### 6. **API/UI Layer**

**Purpose:** Thin client for user interaction

**Components:**
- `app.py` - Gradio interface
- RESTful design (could be FastAPI in future)

**Key Design Decisions:**
- ✅ **Thin client** - Business logic stays in agents
- ✅ **Async-friendly** - Non-blocking operations
- ✅ **Simple deployment** - Single Python file
- ✅ **Production-ready** - Health checks, error handling

---

## 📊 Operational Characteristics

### Observability

**Implemented:**
- ✅ Structured logging (Python logging module)
- ✅ Processing success/failure tracking
- ✅ Embedding dimension validation
- ✅ Upsert count metrics
- ✅ Retrieval latency logging

**Planned:**
- 🔄 Prometheus metrics export
- 🔄 Distributed tracing (OpenTelemetry)
- 🔄 Dashboard (Grafana)

### Reliability

**Patterns:**
- ✅ **Idempotency** - Safe to retry operations
- ✅ **Graceful degradation** - Partial failures don't block entire pipeline
- ✅ **Validation gates** - Catch errors early
- ✅ **Namespace isolation** - Failures isolated per meeting

### Scalability

**Current State:**
- ✅ Batch processing (chunking, embedding, upsert)
- ✅ Pinecone handles vector scaling
- ✅ Stateless services (easy horizontal scaling)

**Future Improvements:**
- 🔄 Async ingestion workers
- 🔄 Queue-based processing (Celery/RQ)
- 🔄 Distributed inference

---

## 🔄 Data Flow (End-to-End)

```
User Upload (Gradio)
    ↓
File Saved to data/raw/
    ↓
Coordinator Receives File
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│   Text File     │   Audio File     │   Image File    │
└────────┬────────┴────────┬─────────┴────────┬────────┘
         ↓                 ↓                  ↓
  DocumentProcessor   AudioAgent        ImageAgent
         ↓                 ↓                  ↓
    Text Chunks      Transcription      Visual Embed
         ↓                 ↓                  ↓
  SentenceTransformer  → Text Chunks   CLIP Encoder
         ↓                 ↓                  ↓
     Embeddings       Embeddings        Embeddings
         └──────────────┬──────────────────┘
                        ↓
                PineconeDB.upsert_documents()
                        ↓
                Pinecone Index
               (namespace=meeting_id)
                        ↓
                  [QUERY PHASE]
                        ↓
                User asks question
                        ↓
              MultimodalRetriever
                 (Hybrid Search)
                        ↓
            Ranked Context Chunks
                        ↓
                Context Assembly
            (modality-aware formatting)
                        ↓
                  LLM (Ollama)
                        ↓
                Generated Answer
                        ↓
                 User receives
            Answer + Source Attribution
```

---

## 🛡️ Error Handling Strategy

### Validation Failures
- **Where:** Ingestion layer
- **Action:** Reject with clear error message
- **Impact:** No downstream processing

### Processing Failures
- **Where:** Agent layer (audio/image)
- **Action:** Log error, skip chunk, continue batch
- **Impact:** Partial meeting processing

### Embedding Failures
- **Where:** Embedding layer
- **Action:** Dimension check fails → log and skip
- **Impact:** Chunk excluded from index

### Retrieval Failures
- **Where:** Query time
- **Action:** Return empty context, inform user
- **Impact:** Degraded answer quality

### LLM Failures
- **Where:** Inference layer
- **Action:** Timeout or error → return fallback message
- **Impact:** User notified of failure

---

## 🔐 Security Considerations

- ✅ API keys via environment variables (not hardcoded)
- ✅ No external API calls (local Ollama)
- ✅ File validation before processing
- ✅ Container isolation (Docker)
- 🔄 **TODO:** Authentication/authorization for multi-user
- 🔄 **TODO:** Data encryption at rest
- 🔄 **TODO:** Rate limiting on API endpoints

---

## 📦 Deployment Patterns

### Local Development
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Docker (Single Container)
```bash
docker build -t minemeets:latest .
docker run -p 7860:7860 --env-file .env minemeets:latest
```

### Docker Compose (Production-like)
```bash
docker-compose up --build
```

### Future: Kubernetes
- 🔄 Deployment manifests
- 🔄 Service mesh (Istio)
- 🔄 Horizontal pod autoscaling
- 🔄 Persistent volume for data/

---

## 🧪 Testing Strategy

### Unit Tests
- ✅ `test_document_processor.py` - Chunking logic
- ✅ `test_pinecone_db.py` - Vector operations
- ✅ `test_llm.py` - LLM interface

### Integration Tests
- 🔄 End-to-end ingestion pipeline
- 🔄 Query → retrieval → inference flow
- 🔄 Multi-file processing

### Performance Tests
- 🔄 Large file processing time
- 🔄 Query latency under load
- 🔄 Concurrent user simulation

---

## 📈 Monitoring & Metrics

### Key Metrics to Track

**Ingestion:**
- Files processed per minute
- Processing failures (by type)
- Average chunk count per file
- Embedding generation time

**Storage:**
- Total vectors in index
- Namespace count (= meeting count)
- Upsert latency
- Index size

**Retrieval:**
- Query latency (p50, p95, p99)
- Context chunk count per query
- Hybrid search breakdown (semantic/keyword/expanded)
- Empty result rate

**Inference:**
- LLM generation time
- Token usage per query
- Error rate
- User satisfaction (if feedback collected)

---

## 🔄 Reprocessing & Maintenance

### Per-Meeting Reprocessing
```python
# Delete old meeting data
db.delete_vectors(namespace=meeting_id, delete_all=True)

# Reprocess meeting
coordinator.process_meeting(meeting_data)
```

### Selective Modality Reindexing
```python
# Delete only text chunks
db.delete_vectors(
    namespace=meeting_id,
    filter={"type": "text_chunk"}
)

# Reprocess only text
# ... (selective processing)
```

### Full Database Flush
```python
# ⚠️ DANGEROUS - deletes everything
db.flush_database()
```

---

## 🚀 Future Enhancements

### High Priority
- [ ] Add FastAPI for RESTful API
- [ ] Implement proper authentication
- [ ] Add Prometheus metrics
- [ ] Create Kubernetes manifests

### Medium Priority
- [ ] Add speaker diarization to audio
- [ ] Implement image OCR for text extraction
- [ ] Add meeting comparison feature
- [ ] Build analytics dashboard

### Research/Experimental
- [ ] Multi-turn conversation support
- [ ] Real-time streaming ingestion
- [ ] Automatic meeting summarization on upload
- [ ] Cross-meeting search

---

## 📚 References

**Technologies:**
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Gradio Docs](https://gradio.app/docs/)
- [Ollama Docs](https://ollama.com/docs/)

**MLOps Best Practices:**
- [Google MLOps Whitepaper](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [ML Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)

---

**Document Version:** 1.0  
**Last Updated:** January 31, 2026  
**Author:** MineMEETS Development Team
