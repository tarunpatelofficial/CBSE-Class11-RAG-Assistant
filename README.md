# 🎓 CBSE Class 11 RAG Student Helper

A Retrieval-Augmented Generation (RAG) system designed specifically for CBSE Class 11 students. This intelligent tutoring system uses **LangChain**, **ChromaDB**, and **Llama 3.2 (8B)** to answer questions based on Class 11 textbooks with accurate, context-aware responses.

## 🌟 Features

- **📚 Textbook-Based Learning**: Answers strictly based on CBSE Class 11 textbooks
- **🔍 Semantic Search**: Uses BGE embeddings for intelligent document retrieval
- **🤖 Local AI**: Runs completely offline using Ollama and Llama 3.2
- **⚡ Fast Responses**: ChromaDB vector database for quick context retrieval
- **📖 Source Citations**: Shows which textbook and page number information came from
- **💬 Interactive Chat**: Ask questions in natural language
- **🎯 Academic Focus**: Custom system prompt ensures educational, age-appropriate responses

## 🏗️ System Architecture

```
┌─────────────────┐
│  PDF Textbooks  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  ingest_pdfs.py                 │
│  • Load PDFs                    │
│  • Split into chunks            │
│  • Generate BGE embeddings      │
│  • Store in ChromaDB            │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  vectorstore/ (ChromaDB)        │
│  Contains: Embeddings + Chunks  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Student asks question          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  app.py / query_rag.py          │
│  1. Embed question (BGE)        │
│  2. Search ChromaDB (top 3-5)   │
│  3. Build prompt with context   │
│  4. Send to Llama via Ollama    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Ollama (llama-stu:latest)      │
│  • Custom system prompt         │
│  • Generates student-friendly   │
│    answer based on context      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Display answer + sources       │
└─────────────────────────────────┘
```

## 📋 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended for smooth operation)
- **Storage**: ~10GB free space (for models and textbooks)
- **OS**: Windows 10/11, macOS, or Linux

### Required Software

- [Python 3.8+](https://www.python.org/downloads/)
- [Ollama](https://ollama.ai/) - For running Llama locally

## 🚀 Installation Guide

### Step 1: Install Ollama

Ollama allows you to run large language models locally on your machine.

#### Windows:

1. Download Ollama from [https://ollama.ai/download](https://ollama.ai/download)
2. Run the installer (`OllamaSetup.exe`)
3. Ollama will start automatically and run in the system tray
4. Verify installation:

```bash
ollama --version
```

#### macOS:

```bash
# Download and install from ollama.ai
# Or use Homebrew:
brew install ollama
```

#### Linux:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Download Llama 3.2 (8B) Model

Download the Llama 3.2 model (8 billion parameters) - this will take a few minutes depending on your internet speed (~4.7GB download):

```bash
ollama pull llama3.2:latest
```

Verify the model is installed:

```bash
ollama list
```

You should see:

```
NAME              ID              SIZE    MODIFIED
llama3.2:latest   a80c4f17acd5    4.7 GB  2 hours ago
```

### Step 3: Create Custom Student Tutor Model

Create a file named `Modelfile` in your project root with this content:

```dockerfile
FROM llama3.2:latest

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

SYSTEM """You are a helpful tutor for Class 11 students in India (CBSE curriculum).

Your role:
- Answer questions using ONLY the textbook context provided to you
- Explain concepts in simple, clear language appropriate for 16-17 year old students
- If the provided context doesn't contain the answer, say "I can only help with topics from your Class 11 textbooks"
- Break down complex topics into easy-to-understand explanations
- Use examples when helpful
- Never make up information not present in the context
- Stay focused on academic subjects (Biology, Chemistry, Physics, Math, English, Accountancy, Computer Science, etc.)
- Do NOT answer questions about celebrities, sports players, current events, or non-academic topics
- Be encouraging and patient - you're helping students learn

Remember: You are helping students learn, so be patient, encouraging, and thorough in your explanations."""
```

Now create the custom model named `llama-stu`:

```bash
ollama create llama-stu -f Modelfile
```

You should see:

```
transferring model data
using existing layer sha256:xxxxx
creating new layer sha256:xxxxx
writing manifest
success
```

Verify your custom model:

```bash
ollama list
```

Output:

```
NAME              ID              SIZE    MODIFIED
llama-stu:latest  b3c02d3c1449    4.7 GB  10 seconds ago
llama3.2:latest   a80c4f17acd5    4.7 GB  2 hours ago
```

**Test your custom model:**

```bash
ollama run llama-stu
```

Try asking: `"Explain photosynthesis"`

Type `/bye` to exit the test chat.

### Step 4: Set Up Python Environment

1. **Clone or create the project:**

```bash
mkdir CBSE_Class11_RAG
cd CBSE_Class11_RAG
```

2. **Create virtual environment:**

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install Python dependencies:**

Create `requirements.txt`:

```txt
langchain==0.1.0
langchain-community==0.0.10
chromadb==0.4.22
pypdf==3.17.4
sentence-transformers==2.3.1
torch==2.1.2
tqdm==4.66.1
requests==2.31.0
```

Install:

```bash
pip install -r requirements.txt
```

### Step 5: Project Structure

Create the following folder structure:

```
CBSE_Class11_RAG/
├── .venv/                      # Virtual environment (created automatically)
├── Textbooks/
│   └── Class_11/              # 📁 PUT YOUR PDFs HERE
│       ├── Biology_11.pdf
│       ├── Chemistry_11.pdf
│       ├── Physics_11.pdf
│       └── ... (other subjects)
├── vectorstore/               # Will be created automatically by ingest_pdfs.py
│   └── chroma.sqlite3        # ChromaDB database
├── models/                    # Will be created automatically
│   └── bge-small-en-v1.5/    # Cached embedding model
├── scripts/
│   ├── ingest_pdfs.py        # PDF processing script
│   └── query_rag.py          # Query script
├── app.py                     # Main application
├── Modelfile                  # Ollama model configuration
└── requirements.txt           # Python dependencies
```

### Step 6: Add Your Textbooks

1. Create the textbooks folder:

```bash
mkdir -p Textbooks/Class_11
```

2. **Add your CBSE Class 11 textbook PDFs** to `Textbooks/Class_11/`

Example:

- Biology_Class11_NCERT.pdf
- Chemistry_Part1_Class11.pdf
- Physics_Part1_Class11.pdf
- Accountancy_Class11.pdf
- ComputerScience_Class11.pdf

## 📚 Usage Guide

### Method 1: Full Pipeline (Recommended)

#### Step 1: Process Your Textbooks (One-Time Setup)

This script reads all PDFs, splits them into chunks, generates embeddings, and stores them in ChromaDB:

```bash
python scripts/ingest_pdfs.py
```

**Expected output:**

```
============================================================
PDF Ingestion Script for RAG System
============================================================

📂 Loading PDFs from Textbooks/Class_11...
Found 5 PDF files
Loading PDFs: 100%|███████████████████| 5/5
  ✓ Loaded Biology_Class11.pdf: 328 pages
  ✓ Loaded Chemistry_Part1_Class11.pdf: 256 pages
  ✓ Loaded Physics_Part1_Class11.pdf: 312 pages
  ...

✓ Total pages loaded: 1234

✂️ Splitting documents into chunks...
Chunk size: 1000, Overlap: 200
✓ Created 3542 chunks

🧠 Generating embeddings using BAAI/bge-small-en-v1.5...
Loading embedding model: BAAI/bge-small-en-v1.5
✓ Model loaded (cached at: models/bge-small-en-v1.5)
  Processing 3542 chunks...
Embedding batches: 100%|███████████████████| 36/36

✓ Vectorstore created successfully!
  Location: vectorstore
  Collection: textbook_embeddings
  Total vectors: 3542

============================================================
✓ Ingestion completed successfully!
============================================================
```

**This process takes 5-15 minutes** depending on:

- Number and size of PDFs
- Your CPU/GPU speed
- First run downloads the BGE model (~90MB)

#### Step 2: Start Ollama Server

**Important**: Ollama must be running before you can ask questions.

Windows (Ollama runs automatically, but if needed):

```bash
ollama serve
```

macOS/Linux:

```bash
ollama serve
```

You should see:

```
Listening on 127.0.0.1:11434 (version 0.1.32)
```

**Keep this terminal open!** Open a new terminal for the next steps.

#### Step 3: Ask Questions

**Option A: Interactive Mode (app.py)** - Recommended for multiple questions

```bash
python app.py
```

**Expected output:**

```
======================================================================
🎓 Student Helper - RAG System
======================================================================
Initializing...
  📥 Loading embedding model: BAAI/bge-small-en-v1.5
  ✓ Embedding model loaded (device: cuda)
  📂 Connecting to vectorstore: vectorstore
  ✓ Connected to collection: textbook_embeddings
  🔌 Checking Ollama connection...
  ✓ Ollama connected, model 'llama-stu:latest' available

✅ System Ready!
📚 Loaded 3542 textbook chunks
🤖 Using model: llama-stu:latest
======================================================================

💡 Tips:
  - Ask questions about your Class 11 textbooks
  - Type 'quit' or 'exit' to stop
  - Type 'sources on/off' to toggle source display
  - Type 'stream on/off' to toggle streaming

🎓 Your question: What is photosynthesis?

======================================================================
❓ Question: What is photosynthesis?
======================================================================

🔍 Searching textbooks (retrieving top 3 matches)...
✓ Found 3 relevant sections

📚 Sources:
  1. Biology_Class11.pdf - Page 156 (relevance: 92.45%)
  2. Biology_Class11.pdf - Page 157 (relevance: 88.23%)
  3. Biology_Class11.pdf - Page 158 (relevance: 85.67%)

🤖 Generating answer using llama-stu:latest...

----------------------------------------------------------------------
💬 Answer:
----------------------------------------------------------------------
Photosynthesis is the process by which green plants, algae, and some 
bacteria convert light energy (usually from the sun) into chemical 
energy stored in glucose molecules. This process occurs mainly in the 
chloroplasts of plant cells.

The process can be summarized in this equation:
6CO₂ + 6H₂O + Light Energy → C₆H₁₂O₆ + 6O₂

In simple terms:
- Plants take in carbon dioxide from the air
- They absorb water from the soil through roots
- Chlorophyll in leaves captures sunlight
- This energy converts CO₂ and water into glucose (food for the plant)
- Oxygen is released as a byproduct

There are two main stages:
1. Light-dependent reactions (in thylakoids)
2. Light-independent reactions or Calvin cycle (in stroma)

Photosynthesis is essential for life on Earth as it produces oxygen 
and is the primary source of energy for most ecosystems.
----------------------------------------------------------------------

⏱️ Time taken: 3.45 seconds
======================================================================

🎓 Your question: quit

👋 Thanks for using Student Helper! Good luck with your studies!
```

**Option B: Single Question Mode**

```bash
python app.py "What is the structure of an atom?"
```

**Option C: Using query_rag.py (Alternative Script)**

```bash
# Interactive mode
python scripts/query_rag.py

# Single question
python scripts/query_rag.py "Explain Newton's laws of motion"
```

### Method 2: Testing Individual Components

#### Test Ollama Connection:

```bash
curl http://localhost:11434/api/tags
```

#### Test Custom Model:

```bash
ollama run llama-stu "Explain the concept of entropy in simple terms"
```

#### Check Vectorstore:

```python
# In Python shell
import chromadb
client = chromadb.PersistentClient(path="vectorstore")
collection = client.get_collection("textbook_embeddings")
print(f"Total documents: {collection.count()}")
```

## 🔧 Configuration

### Key Configuration Options

**In `ingest_pdfs.py`:**

```python
CHUNK_SIZE = 1000          # Size of text chunks (characters)
CHUNK_OVERLAP = 200        # Overlap between chunks
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Embedding model
```

**In `app.py` and `query_rag.py`:**

```python
OLLAMA_MODEL = "llama-stu:latest"  # Your custom model name
TOP_K_RESULTS = 3                   # Number of contexts to retrieve
OLLAMA_API_URL = "http://localhost:11434/api/generate"
```

**In `Modelfile`:**

```dockerfile
PARAMETER temperature 0.7   # Creativity (0.1=focused, 1.0=creative)
PARAMETER top_p 0.9        # Diversity of responses
PARAMETER num_ctx 2048     # Context window size
```

### Switching Models

To use a different base model:

```bash
# Download another model
ollama pull llama3.1:latest

# Update Modelfile
FROM llama3.1:latest
# ... rest of config

# Recreate custom model
ollama create llama-stu -f Modelfile
```

## 🎯 How It Works: Complete Flow

### 1️⃣ **PDF Ingestion Phase** (One-time)

```
PDFs → Load → Split → Embed → Store
```

**Detailed Steps:**

1. **Load PDFs**: PyPDFLoader reads each PDF page by page
2. **Text Extraction**: Extracts text content from each page
3. **Chunking**: RecursiveCharacterTextSplitter divides text into ~1000 character chunks with 200 character overlap
4. **Embedding Generation**: BGE model converts each chunk into a 384-dimensional vector
5. **Storage**: ChromaDB stores vectors with metadata (source file, page number)

**Why chunking?**

- LLMs have token limits (~2048-4096 tokens)
- Smaller chunks = more precise retrieval
- Overlap ensures context continuity

**Why BGE embeddings?**

- Optimized for semantic similarity
- Works well for educational content
- Runs locally (no API calls)

### 2️⃣ **Query Phase** (Every question)

```
Question → Embed → Search → Retrieve → Build Prompt → LLM → Answer
```

**Detailed Steps:**

1. **Question Embedding**:
    
    ```
    "What is photosynthesis?" → [0.234, -0.567, 0.891, ...] (384 dimensions)
    ```
    
2. **Similarity Search**:
    
    - ChromaDB compares question embedding with all stored chunk embeddings
    - Uses cosine similarity to find closest matches
    - Returns top 3-5 most relevant chunks
3. **Context Assembly**:
    
    ```python
    context = """
    [From Biology_Class11.pdf, Page 156]
    Photosynthesis is the process by which...
    
    [From Biology_Class11.pdf, Page 157]
    The light-dependent reactions occur in...
    """
    ```
    
4. **Prompt Construction**:
    
    ```python
    prompt = f"""
    TEXTBOOK CONTENT:
    {context}
    
    STUDENT'S QUESTION:
    {question}
    
    INSTRUCTIONS: Answer based only on textbook content...
    """
    ```
    
5. **LLM Generation**:
    
    - Ollama receives prompt
    - llama-stu model generates response
    - Custom system prompt ensures educational tone
    - Returns answer
6. **Display**:
    
    - Shows answer
    - Lists source textbooks and page numbers
    - Shows relevance scores

### 3️⃣ **Why This Approach Works**

**Traditional Approach (No RAG):**

```
Question → LLM → Answer (may hallucinate or use outdated info)
```

**RAG Approach (This System):**

```
Question → Find Relevant Textbook Sections → LLM with Context → Accurate Answer
```

**Benefits:**

- ✅ Answers based on actual textbooks
- ✅ No hallucinations (made-up facts)
- ✅ Source citations for verification
- ✅ Works offline
- ✅ Privacy preserved (no data sent online)

## 🔍 Advanced Features

### 1. Streaming Responses

For real-time answer generation:

```python
rag = StudentHelperRAG()
rag.ask("Your question", stream=True)
```

### 2. Batch Processing

Ask multiple questions at once:

```python
from app import StudentHelperRAG

rag = StudentHelperRAG()
questions = [
    "What is photosynthesis?",
    "Explain Newton's first law",
    "What is double entry bookkeeping?"
]
results = rag.batch_ask(questions)
```

### 3. Adjusting Relevance Threshold

In `app.py`, modify the relevance check:

```python
if best_score < 0.65:  # Change threshold (0.0-1.0)
    # Handle low relevance
```

### 4. Custom Subjects

Add more textbooks:

```bash
# Add new PDFs to Textbooks/Class_11/
cp ~/Downloads/Economics_Class11.pdf Textbooks/Class_11/

# Re-run ingestion
python scripts/ingest_pdfs.py
```

## 🐛 Troubleshooting

### Issue 1: "Could not connect to Ollama"

**Solution:**

```bash
# Check if Ollama is running
ollama list

# If not, start it
ollama serve

# On Windows, Ollama should start automatically
# Check system tray for Ollama icon
```

### Issue 2: "Model 'llama-stu:latest' not found"

**Solution:**

```bash
# Verify model exists
ollama list

# If missing, recreate it
ollama create llama-stu -f Modelfile

# Or update OLLAMA_MODEL in app.py to an existing model
```

### Issue 3: "Vectorstore not found"

**Solution:**

```bash
# Run PDF ingestion first
python scripts/ingest_pdfs.py

# Check if vectorstore folder was created
ls vectorstore/
```

### Issue 4: Slow Performance

**Solutions:**

- **Use GPU**: Install PyTorch with CUDA support
    
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    
- **Reduce TOP_K_RESULTS**: Change from 5 to 3 in `app.py`
- **Use smaller model**: Try `llama3.2:1b` instead of `llama3.2:latest`
- **Increase RAM**: Close other applications

### Issue 5: Poor Answer Quality

**Solutions:**

1. **Adjust temperature** in Modelfile (lower = more focused):
    
    ```dockerfile
    PARAMETER temperature 0.5
    ```
    
2. **Increase context retrieval** in `app.py`:
    
    ```python
    TOP_K_RESULTS = 5  # Get more context
    ```
    
3. **Improve chunking** in `ingest_pdfs.py`:
    
    ```python
    CHUNK_SIZE = 1500  # Larger chunks
    CHUNK_OVERLAP = 300  # More overlap
    ```
    
4. **Re-run ingestion** after changes:
    
    ```bash
    python scripts/ingest_pdfs.py
    ```
    

### Issue 6: "CUDA out of memory"

**Solution:**

```python
# In ingest_pdfs.py, line ~55, change:
device='cpu'  # Instead of 'cuda'

# In query_rag.py and app.py, find device detection and force CPU:
device = 'cpu'
```

### Issue 7: ChromaDB Errors

**Solution:**

```bash
# Delete and recreate vectorstore
rm -rf vectorstore/
python scripts/ingest_pdfs.py
```

## 📊 Performance Metrics

**Typical Performance (on mid-range laptop):**

|Metric|Value|
|---|---|
|PDF Ingestion (5 books)|8-12 minutes|
|Query Response Time|2-5 seconds|
|Context Retrieval|< 1 second|
|LLM Generation|2-4 seconds|
|Vectorstore Size|~100MB per 1000 pages|
|RAM Usage|4-6 GB|

**With GPU (NVIDIA GTX 1660+):**

- Ingestion: 3-5 minutes
- Response Time: 10-20 seconds

## 🔐 Privacy & Data

- ✅ **100% Local**: Everything runs on your computer
- ✅ **No Internet Required**: After initial model download
- ✅ **No Data Collection**: Your questions and textbooks never leave your machine
- ✅ **No API Keys**: No external services used

## 📝 Limitations

1. **Answer Quality**: Depends on textbook content quality and PDF text extraction
2. **Context Window**: Can only process ~2000 tokens at once
3. **Model Size**: 8B parameter model is good but not as capable as GPT-4
4. **PDF Quality**: Scanned PDFs without OCR won't work well
5. **Language**: Currently optimized for English content

## 🎓 Educational Use Cases

Perfect for:

- 📖 **Quick Homework Help**: "Explain the concept of..."
- 🔍 **Topic Review**: "Summarize the chapter on..."
- ❓ **Doubt Clarification**: "I don't understand..."
- 📝 **Exam Preparation**: "What are the key points about..."
- 🧪 **Concept Deep-Dive**: "How does... work?"

**Not suitable for:**

- Essay writing (academic integrity)
- Answering exam questions directly
- Topics outside Class 11 syllabus
- Current events or news

## 🤝 Contributing

Ways to improve this project:

1. **Add More Subjects**: Contribute textbook PDFs
2. **Improve Prompts**: Enhance the Modelfile system prompt
3. **Add Features**: Web UI, voice interface, etc.
4. **Optimize Performance**: Better chunking strategies
5. **Add Tests**: Unit tests for components

## 🙏 Acknowledgments

- **LangChain**: Document processing and RAG framework
- **ChromaDB**: Vector database for embeddings
- **Ollama**: Local LLM inference
- **Meta**: Llama 3.2 model
- **BAAI**: BGE embedding models
- **NCERT**: Class 11 textbook content

---

**Made with ❤️ for CBSE Class 11 Students**

_Study smart, not hard!_ 📚✨
