# MineMEETS: Enhanced Multimodal RAG Meeting Assistant with LangChain Integration

MineMEETS is a sophisticated multimodal RAG application that processes text transcripts, transcribes audio/video, and embeds screenshots/images for intelligent retrieval. It features **LangChain-Pinecone integration** for seamless vector operations, hybrid search capabilities, and enhanced multimodal understanding across text, audio, and visual content.

---

## Enhanced Features

- **🔄 LangChain-Pinecone Integration**: Seamless vector database operations with advanced retrieval chains
- **🎯 Hybrid Search**: Combines semantic, keyword, and query expansion search strategies for better recall
- **🎨 True Multimodal RAG**: Cross-modal analysis synthesizing text, audio, and visual information
- **📝 Smart Document Processing**: LangChain document loaders with intelligent chunking and text splitting
- **🎙️ Audio Intelligence**: Whisper transcription with speaker diarization and temporal context
- **🖼️ Visual Understanding**: CLIP ViT-B/32 embeddings with multimodal context integration
- **💬 Enhanced Q&A**: Context-aware responses with source attribution and modality indicators
- **📊 Meeting Analytics**: Comprehensive summaries with modality breakdown and search strategy insights

### Core Capabilities

- **Text + Audio/Video**: Upload `.txt` transcripts or audio/video for Whisper transcription
- **Screenshots/Images**: Upload `.png/.jpg/.jpeg/.webp/.bmp`; embedded with CLIP ViT-B/32
- **Intelligent Chunking**: Recursive text splitting with semantic boundary detection
- **Meeting-Scoped Namespaces**: Each upload isolated per `meeting_id` with enhanced metadata
- **Multimodal Insights**: Cross-references information across text, audio, and visual modalities
- **Streamlit UI**: Intuitive upload → process → ask workflow

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

# Whisper
WHISPER_MODEL=base
WHISPER_CACHE_DIR=.cache/whisper

# Ollama (required)
OLLAMA_MODEL=llama3.1
OLLAMA_HOST=http://localhost:11434
```

3. **Launch Ollama locally:**

- Download and start from https://ollama.com/download
- `ollama pull llama3.1` (recommended for better multimodal understanding)
- Ollama will serve HTTP at `http://localhost:11434` by default

4. **Run Streamlit app:**

```bash
streamlit run app.py
```

5. **Use it**
- Upload `.txt`, audio/video, and screenshots
- Click "Process Meeting"; then ask questions in the Q&A tab
- Explore enhanced multimodal responses with source attribution

---

## Architecture and How It Works

### Enhanced Agent System
- **`multimodal_rag.py`**: LangChain-based RAG chain with hybrid retrieval and cross-modal analysis
- **`document_processor.py`**: LangChain document loaders and intelligent text splitters
- **`pinecone_db.py`**: Enhanced Pinecone integration with LangChain VectorStore support
- **`qa_agent.py`**: Multimodal Q&A with enhanced context understanding
- **`coordinator.py`**: Orchestrates multimodal ingestion with fallback mechanisms
- **`audio_agent.py`**: Whisper transcription with temporal segmentation
- **`image_agent.py`**: CLIP embeddings for visual content understanding

### Advanced Vector Operations
- **Hybrid Retrieval**: Combines semantic, keyword, and expanded query strategies
- **Multimodal Context**: Categorizes and processes content by modality (text, audio, visual)
- **Smart Chunking**: Uses LangChain's RecursiveCharacterTextSplitter for optimal text segmentation
- **Enhanced Metadata**: Rich metadata including search strategies, temporal information, and modality types

### Multimodal Processing Pipeline
1. **Document Ingestion**: LangChain loaders process various file formats
2. **Intelligent Chunking**: Semantic-aware text splitting with overlap management
3. **Hybrid Embedding**: Multiple search strategies for comprehensive retrieval
4. **Cross-Modal Analysis**: Synthesizes information across text, audio, and visual content
5. **Context-Aware Response**: Generates responses with modality-specific insights

---

## Enhanced Capabilities

### Hybrid Search Strategies
- **Semantic Search**: Vector similarity using CLIP embeddings
- **Keyword Search**: Targeted keyword matching for precision
- **Query Expansion**: Broadens search for general questions
- **Multimodal Ranking**: Combines relevance scores with modality-specific boosts

### Advanced Document Processing
- **PDF Support**: Enhanced PDF text extraction with page metadata
- **DOCX Support**: Word document processing with formatting preservation
- **Text Splitting**: Intelligent boundary detection (sentences, paragraphs, sections)
- **Metadata Enhancement**: Rich metadata including file stats, chunk positions, and content types

### Multimodal Understanding
- **Cross-Modal Analysis**: Identifies patterns and consistency across modalities
- **Temporal Context**: Audio timestamps and text positions for timeline understanding
- **Visual Context**: Image descriptions and their relation to discussion content
- **Source Attribution**: Clear indication of which modality provided each piece of information

---

## Troubleshooting

- **LangChain Installation**: Ensure all LangChain packages are installed: `langchain-pinecone`, `langchain-core`, `langchain-community`
- **Pinecone Dimension**: Index dimension must match `EMBEDDING_DIM` (512 for CLIP ViT-B/32)
- **Memory Issues**: Large files are automatically chunked; monitor memory usage for very large documents
- **Ollama Models**: Use `llama3.1` or similar for best multimodal understanding capabilities
- **Fallback Mechanisms**: System gracefully falls back to legacy methods if LangChain features fail

---

## Why MineMEETS Excels

- **🔧 Production-Ready**: Comprehensive error handling with graceful fallbacks
- **🎯 LangChain Integration**: Leverages the power of LangChain for advanced RAG capabilities
- **🔄 Hybrid Search**: Multiple retrieval strategies ensure comprehensive information discovery
- **🎨 True Multimodal**: Understands and synthesizes across text, audio, and visual content
- **📊 Rich Analytics**: Detailed insights into search strategies, modalities used, and response confidence
- **🛠️ Extensible Design**: Clean agent architecture allows easy addition of new modalities or features
- **💻 Windows Compatible**: Optimized for Windows development environments
- **📝 Well-Documented**: Comprehensive documentation and clear code structure

---

## License
MIT License
