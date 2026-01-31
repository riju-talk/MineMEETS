# MineMEETS - Quick Start Guide

**Production MLOps Multimodal RAG Platform — Get Running in 5 Minutes**

---

## 🎯 What You're Building

A production-ready ML system that:
- Ingests text, audio, and images from meetings
- Embeds content using Whisper & CLIP
- Stores vectors in Pinecone
- Retrieves context with hybrid search
- Answers questions using local LLM (Ollama)

**This demonstrates MLOps skills, not research.**

---

## ✅ Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Python 3.10 or higher** (`python --version`)
- [ ] **pip** package manager
- [ ] **Git** for version control
- [ ] **Pinecone account** (free tier works) - [Sign up here](https://www.pinecone.io/)
- [ ] **Ollama installed** - [Download here](https://ollama.com/download)
- [ ] **Docker** (optional, for containerized deployment)

---

## 🚀 Method 1: Local Development (Recommended for Testing)

### Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/MineMEETS.git
cd MineMEETS

# Create virtual environment
python -m venv venv

# Activate (choose your OS)
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows CMD
venv\Scripts\Activate.ps1         # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your Pinecone API key
# You can use any text editor:
nano .env      # Linux/Mac
notepad .env   # Windows
```

**Required `.env` contents:**
```env
PINECONE_API_KEY=your-actual-pinecone-api-key-here
WHISPER_MODEL=base
WHISPER_CACHE_DIR=.cache/whisper
OLLAMA_MODEL=llama3.1
OLLAMA_HOST=http://localhost:11434
```

### Step 3: Start Ollama

```bash
# Download the model (one-time setup)
ollama pull llama3.1

# Start Ollama server (keep this terminal open)
ollama serve
```

**Verify Ollama is running:**
- Open browser: http://localhost:11434
- You should see "Ollama is running"

### Step 4: Run Application

```bash
# In a new terminal (with venv activated)
python app.py
```

**Or use Makefile:**
```bash
make run
```

### Step 5: Open Browser

Open: http://localhost:7860

You should see the MineMEETS interface! 🎉

---

## 🐳 Method 2: Docker (Recommended for Production)

### Step 1: Prepare Environment

```bash
# Clone repository
git clone https://github.com/yourusername/MineMEETS.git
cd MineMEETS

# Copy and edit .env file
cp .env.example .env
# Edit .env with your PINECONE_API_KEY
```

### Step 2: Start Ollama on Host

```bash
# Ollama must run on host machine (not in container)
ollama pull llama3.1
ollama serve
```

### Step 3: Build and Run with Docker Compose

```bash
# Build and start
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

**Or use Makefile:**
```bash
make docker-run
```

### Step 4: Access Application

Open: http://localhost:7860

**To stop:**
```bash
docker-compose down
# Or
make docker-stop
```

---

## 📝 How to Use

### 1. Upload Files

**Tab: "Upload & Process"**

- **Text files**: Upload `.txt`, `.pdf`, or `.docx` transcripts
- **Audio files**: Upload `.mp3`, `.wav`, or `.m4a` recordings
- **Image files**: Upload `.png`, `.jpg`, or `.jpeg` screenshots

Click the appropriate "Process" button and wait for completion.

**Important:** Copy the Meeting ID from the success message!

Example: `meeting_20260131_143022`

### 2. Ask Questions

**Tab: "Q&A"**

1. Paste your Meeting ID
2. Enter your question
3. Click "Ask Question"

**Example questions:**
- "What were the main discussion points?"
- "What decisions were made?"
- "Summarize the key action items"
- "What did [person] say about [topic]?"

### 3. Get Summary

Click "Get Meeting Summary" for a comprehensive overview.

### 4. View Stats

**Tab: "Database Stats"**

Check how many meetings and vectors are stored.

---

## 🧪 Verify Installation

### Run Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
make test

# Or directly
pytest tests/ -v
```

### Check Code Quality

```bash
# Format code
make format

# Run linter
make lint

# Run all checks
make check
```

---

## 🔧 Troubleshooting

### Issue: "PINECONE_API_KEY not set"

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Check it contains your key
cat .env | grep PINECONE_API_KEY

# Make sure you activated venv before running
source venv/bin/activate
```

### Issue: "Connection refused to Ollama"

**Solution:**
```bash
# Verify Ollama is running
curl http://localhost:11434

# Restart Ollama
ollama serve

# Check firewall isn't blocking port 11434
```

### Issue: "Gradio not found"

**Solution:**
```bash
# Ensure you're in virtual environment
which python  # Should show venv path

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: "Import errors" in VSCode

**Solution:**
- Select the correct Python interpreter
- Press `Ctrl+Shift+P` → "Python: Select Interpreter"
- Choose the venv interpreter

### Issue: Docker can't connect to Ollama

**Solution:**

Edit `docker-compose.yml`:
```yaml
environment:
  - OLLAMA_HOST=http://host.docker.internal:11434  # Mac/Windows
  # OR
  - OLLAMA_HOST=http://172.17.0.1:11434  # Linux
```

---

## 📊 Next Steps

### For Development

1. **Explore the code:**
   - `agents/` - Core pipeline modules
   - `app.py` - Gradio interface
   - `tests/` - Unit tests

2. **Read documentation:**
   - `README.md` - Overview
   - `ARCHITECTURE.md` - System design
   - `CONTRIBUTING.md` - Development guide

3. **Make improvements:**
   - Add tests
   - Optimize performance
   - Enhance error handling

### For Deployment

1. **Set up CI/CD:**
   - GitHub Actions workflow included
   - Automatically runs on push

2. **Deploy to production:**
   - Use Docker Compose
   - Set up monitoring
   - Configure logging

3. **Scale:**
   - Add load balancing
   - Implement queue-based processing
   - Deploy to Kubernetes

---

## 🎓 Learning Resources

**MLOps Best Practices:**
- [Google MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Chip Huyen's ML Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)

**Technical Documentation:**
- [Pinecone Docs](https://docs.pinecone.io/)
- [Whisper GitHub](https://github.com/openai/whisper)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Ollama Docs](https://ollama.com/docs/)

---

## 💡 Tips

1. **Start small:** Test with small text files first
2. **Monitor logs:** Watch console for errors
3. **Check database:** Use "Database Stats" tab
4. **Save meeting IDs:** You'll need them for querying
5. **Experiment:** Try different question phrasings

---

## 🤝 Get Help

- **Issues:** [GitHub Issues](https://github.com/yourusername/MineMEETS/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/MineMEETS/discussions)
- **Documentation:** Check ARCHITECTURE.md for internals

---

## ✅ Success Checklist

After following this guide, you should have:

- [ ] Application running locally or in Docker
- [ ] Successfully uploaded and processed a file
- [ ] Received a Meeting ID
- [ ] Asked a question and got an answer
- [ ] Viewed database statistics
- [ ] Tests passing
- [ ] Understanding of system architecture

**If you can check all these boxes, congratulations! 🎉**

You now have a production-ready MLOps platform demonstrating:
- ML pipeline design
- Vector database operations
- Multimodal processing
- Containerization
- CI/CD automation
- Code quality practices

---

**Ready to showcase your MLOps skills!** 🚀
