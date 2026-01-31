# MineMEETS — Multimodal RAG Meeting Intelligence Platform (MLOps-Focused)

MineMEETS is a **production-style multimodal RAG system** for processing and retrieving meeting intelligence across **text, audio/video, and visual content**, with an emphasis on **reliable pipelines, vector infrastructure, and operational concerns**.

The project demonstrates **end-to-end ML system deployment practices** using **Pinecone**, Whisper, CLIP, and a locally hosted LLM runtime.

---

## 🎯 Project Intent (Very Important)

> Focus on **operational ML pipelines**, system reliability, and data flow —
> **not model innovation or research contributions**.

This is an **MLOps / ML Systems Engineering** portfolio project demonstrating:

* Production-grade ingestion pipelines
* Vector database operations and management
* Deterministic preprocessing with validation
* Stateless retrieval services
* Operational observability and monitoring
* Container-based deployment patterns

---

## 🧠 Core Capabilities

### Ingestion Pipelines (Operational Focus)

* Batch ingestion of:
  * Text transcripts (.txt, .pdf, .docx)
  * Audio/video files (.mp3, .wav, .m4a)
  * Images/screenshots (.png, .jpg, .jpeg)
* Deterministic preprocessing with validation and fallback paths
* Idempotent processing per `meeting_id`
* Dimension validation for embeddings
* Batch upsert with configurable sizes

---

### Feature Engineering & Embeddings

* **Text embeddings**: Sentence Transformers (CLIP ViT-B/32)
* **Audio → Text**: Whisper transcription, then embedded
* **Image embeddings**: CLIP ViT-B/32 visual encoder
* **Unified embedding interface** with strict dimensional checks (512-dim)
* Preprocessing includes chunking with configurable overlap

---

### Vector Infrastructure (Pinecone)

* **Namespace-per-meeting isolation** enables:
  * Per-meeting reprocessing without affecting others
  * Selective deletion and rollback
  * Cost-controlled operations
* **Metadata-first schema design** for:
  * Semantic similarity search
  * Modality-aware retrieval (text/audio/image)
  * Time-range filtering
  * Debugging and auditability

---

## 🏗️ System Architecture (MLOps View)

```
Raw Inputs (Text/Audio/Images)
   ↓
Ingestion Jobs (Validation & Routing)
   ↓
Preprocessing & Chunking (Deterministic)
   ↓
Embedding Workers (Whisper/CLIP/SentenceTransformer)
   ↓
Vector Store (Pinecone with Namespaces)
   ↓
Retrieval Service (Hybrid Search)
   ↓
LLM Inference (Ollama - Local)
   ↓
Gradio UI (Thin Client)
```

**Each stage is:**
* Independently testable
* Restartable without side effects
* Observable with logging and metrics

---

## 🔧 Technology Stack

| Layer              | Tool                   | MLOps Reasoning                     |
| ------------------ | ---------------------- | ----------------------------------- |
| Language           | Python 3.10+           | ML ecosystem standard               |
| Orchestration      | Explicit pipelines     | Predictable execution               |
| Vector DB          | Pinecone               | Managed scaling & reliability       |
| Audio Processing   | Whisper                | Deterministic transcription         |
| Vision Processing  | CLIP ViT-B/32          | Stable multimodal embeddings        |
| LLM Runtime        | Ollama                 | Local inference control             |
| UI                 | Gradio                 | Simple production-ready interface   |
| Containerization   | Docker                 | Reproducible deployments            |
| CI/CD              | GitHub Actions         | Automated testing and builds        |
| Code Quality       | Black, Pylint, Pytest  | Maintainable, tested codebase       |

---

## 📦 Pinecone Index Design (Operational)

### Namespace Strategy

* `meeting_id` = namespace
* **Enables:**
  * Per-meeting reprocessing
  * Safe rollback of bad data
  * Cost-controlled deletion
  * Isolation for multi-tenant scenarios

### Metadata Schema

```json
{
  "meeting_id": "meeting_20260131_143022",
  "modality": "text | audio | image",
  "type": "text_chunk | audio_segment | image_embed",
  "source": "transcript | whisper | screenshot",
  "chunk_id": "meeting_20260131_143022_chunk_14",
  "chunk_index": 14,
  "timestamp_start": 120,
  "timestamp_end": 145,
  "position": 14,
  "total_chunks": 47
}
```

**Used for:**
* Filtered retrieval by modality or time range
* Debugging incorrect answers
* Audit trails and compliance
* Performance monitoring

---

## 🔍 Retrieval Layer

* **Hybrid search strategies:**
  * Semantic similarity via vector embeddings
  * Keyword-based search for better recall
  * Query expansion for general questions
* **Metadata filtering** for modality and temporal constraints
* **Deterministic ranking logic** (no stochastic agent behavior)
* **Deduplication** and score normalization

> Retrieval is treated as a **service**, not an experiment.

---

## 💬 Inference & Serving

* **Context assembly** with:
  * Token limit constraints
  * Modality indicators for cross-modal reasoning
  * Source attribution
* **LLM served locally** via Ollama HTTP API
* **Stateless Q&A execution** (easy to containerize and scale)
* **No external API dependencies** (privacy-preserving)

---

## 📊 Observability & Reliability

Implemented operational hooks:

* ✅ **Ingestion logging**: Success/failure per meeting
* ✅ **Embedding validation**: Dimension checks before upsert
* ✅ **Pinecone upsert counts**: Per-job metrics
* ✅ **Retrieval latency**: Tracked per query
* ✅ **Graceful fallbacks**: On partial pipeline failures
* ✅ **Error logging**: Structured logs with context
* 🔄 **Metrics collection**: (Planned for monitoring dashboards)

---

## 🔁 Reprocessing & Maintenance

* ✅ **Full meeting re-ingestion supported**
* ✅ **Selective modality reindexing** (e.g., text-only, audio-only)
* ✅ **Safe deletion via namespace purge**
* ✅ **Idempotent operations** (running twice produces same result)

This is **classic MLOps hygiene**.

---

## 🚀 Deployment Model

Designed to run:

* ✅ **Locally** for development and testing
* ✅ **In Docker** for reproducible environments
* ✅ **As batch jobs** + API service for production
* ✅ **No hard dependency on UI** (can run headless)
* ✅ **LLM runtime isolated** from ingestion pipeline

---

## ❌ Explicit Non-Goals (MLOps-Correct)

* ❌ Model fine-tuning or training
* ❌ Novel architectures or research
* ❌ Research benchmarks or leaderboards
* ❌ Autonomous agents with complex planning
* ❌ Overlapping orchestration frameworks (e.g., Airflow, Prefect)

---

## 🚀 Quick Start

### Prerequisites

* Python 3.10+
* Docker (optional, for containerized deployment)
* Ollama installed and running locally
* Pinecone account and API key

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
# Pinecone
PINECONE_API_KEY=your-pinecone-api-key-here

# Whisper
WHISPER_MODEL=base
WHISPER_CACHE_DIR=.cache/whisper

# Ollama
OLLAMA_MODEL=llama3.1
OLLAMA_HOST=http://localhost:11434
```

### 3. Start Ollama

```bash
# Download and start Ollama from https://ollama.com/download
ollama pull llama3.1
ollama serve  # Runs on http://localhost:11434
```

### 4. Run Application

```bash
# Using Make (recommended)
make run

# Or directly with Python
python app.py
```

### 5. Use the Interface

* Open browser to `http://localhost:7860` (Gradio default)
* Upload meeting files (text, audio, images)
* Click "Process Meeting"
* Ask questions in the Q&A tab

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
make docker-build

# Run container
make docker-run

# Or use docker-compose
docker-compose up --build
```

### Environment Variables

Pass environment variables via `.env` file or docker-compose:

```yaml
environment:
  - PINECONE_API_KEY=${PINECONE_API_KEY}
  - OLLAMA_HOST=http://host.docker.internal:11434
```

---

## 🛠️ Development

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Run tests
make test

# Run all quality checks
make check
```

### Project Structure

```
MineMEETS/
├── agents/                  # Core pipeline modules
│   ├── audio_agent.py       # Whisper transcription
│   ├── image_agent.py       # CLIP image embeddings
│   ├── document_processor.py # Text chunking
│   ├── pinecone_db.py       # Vector operations
│   ├── multimodal_rag.py    # Retrieval logic
│   ├── qa_agent.py          # Q&A orchestration
│   ├── llm.py               # LLM interface
│   └── coordinator.py       # Pipeline coordinator
├── tests/                   # Unit and integration tests
├── data/                    # Data storage
│   └── raw/                 # Input files
├── app.py                   # Gradio UI application
├── requirements.txt         # Production dependencies
├── pyproject.toml           # Project metadata & dev deps
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-container orchestration
├── Makefile                 # Operational commands
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD pipeline
└── README.md                # This file
```

---

## 📈 CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

* ✅ **Lint**: Black, Pylint checks
* ✅ **Test**: Pytest with coverage
* ✅ **Build**: Docker image creation
* ✅ **Validate**: Type checking with MyPy

Runs on:
* Every push to `main`
* All pull requests

---

## 📌 Resume-Ready Description (MLOps Version)

**MineMEETS — Multimodal RAG Meeting Intelligence Platform**

* Built an end-to-end MLOps-oriented pipeline for ingesting, embedding, and retrieving meeting data across text, audio, and images
* Designed Pinecone-backed vector infrastructure with namespace isolation, metadata filtering, and safe reindexing workflows
* Integrated Whisper and CLIP into deterministic embedding pipelines with validation and fallback mechanisms
* Implemented stateless retrieval and LLM inference with latency monitoring and operational safeguards
* Containerized deployment with Docker, CI/CD with GitHub Actions, and production-grade code quality tools

---

## 🎤 Interview Explanation (30 Seconds)

> "MineMEETS is an MLOps-focused multimodal RAG system. I built ingestion pipelines for text, audio, and images, generated embeddings with Whisper and CLIP, and indexed everything in Pinecone using meeting-scoped namespaces. The emphasis was on operational reliability — reprocessing, metadata filtering, latency monitoring, and safe deletion — rather than model experimentation. It's containerized, tested, and has CI/CD integrated."

**This answer demonstrates production ML engineering skills.**

---

## 📚 Documentation

* [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed system design and data flow
* [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
* [CHANGELOG.md](CHANGELOG.md) - Version history

---

## 🤝 Contributing

This is a portfolio project, but contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests and linting (`make check`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* **Whisper** - OpenAI's speech recognition model
* **CLIP** - OpenAI's vision-language model
* **Pinecone** - Managed vector database
* **Ollama** - Local LLM runtime
* **Gradio** - ML interface framework

---

**Built with a focus on MLOps best practices, not research novelty.**
