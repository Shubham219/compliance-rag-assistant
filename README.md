# 🏛️ Compliance RAG Assistant

An AI-powered **Retrieval Augmented Generation (RAG)** system for regulatory compliance queries. Get instant, accurate answers about GDPR, SOX, HIPAA, and other regulatory frameworks with source citations.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🌟 Features

- **💬 Intelligent Q&A**: Ask natural language questions about compliance requirements
- **📚 Multi-Regulation Support**: GDPR, SOX, HIPAA, PCI-DSS, ISO 27001, and more
- **🔍 Source Citations**: Every answer includes references to source documents
- **🌐 Beautiful Web UI**: Easy-to-use Gradio interface
- **📤 Document Upload**: Add your own regulatory documents
- **🔎 Similarity Search**: Find relevant document sections without generating answers
- **📊 Query History**: Track and export compliance queries for audit trails
- **🤖 Multiple LLM Support**: Works with OpenAI (GPT) or Ollama (local, free)

---

## 📋 Prerequisites

- **Python**: 3.11.8 (recommended) or 3.10.13+
- **pip**: 23.0 or higher
- **Ollama** (optional): For free local LLM - [Download here](https://ollama.ai)
- **OpenAI API Key** (optional): For cloud-based GPT models

---

## ⚡ Quick Start

### 1️⃣ Clone or Create Project Directory

```bash
mkdir compliance-rag-assistant
cd compliance-rag-assistant
```

### 2️⃣ Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 4️⃣ Setup Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings (optional)
# For Ollama (free): No changes needed
# For OpenAI: Add your API key
```

### 5️⃣ Create Sample Documents

```bash
# Generate sample regulatory documents
python scripts/create_sample_docs.py
```

### 6️⃣ Launch the Application

```bash
# Start the web UI
python -m src.ui.gradio_app
```

**Open your browser to:** `http://localhost:7860`

---

## 🎯 How to Use

### Using the Web Interface

#### **Step 1: Initialize the System**

1. Open the web interface at `http://localhost:7860`
2. Go to **"🚀 System Setup"** tab
3. Select LLM provider:
   - **Ollama** (free, runs locally) - No API key needed
   - **OpenAI** (paid, cloud) - Requires API key
4. Enter model name:
   - For Ollama: `llama2`, `mistral`, `mixtral`
   - For OpenAI: `gpt-3.5-turbo`, `gpt-4`
5. Click **"Initialize System"**
6. Wait for confirmation: "✅ System initialized successfully!"

#### **Step 2: Ask Questions**

1. Go to **"💬 Ask Questions"** tab
2. Type your compliance question in the text box
3. Optionally filter by regulation type (GDPR, SOX, HIPAA, etc.)
4. Enable **"Show Sources"** to see document references
5. Click **"Get Answer"**
6. View the answer with source citations!

**Example Questions:**
- "What are the GDPR requirements for data retention?"
- "What security controls does HIPAA require for PHI?"
- "How long do I have to notify authorities about a data breach?"
- "What are SOX 404 internal control requirements?"
- "What encryption is required for protected health information?"

#### **Step 3: Search Documents** (Optional)

1. Go to **"🔍 Search Documents"** tab
2. Enter a search query (e.g., "encryption requirements")
3. Adjust number of results
4. Click **"Search"**
5. View similar document sections with similarity scores

#### **Step 4: Upload Your Documents**

1. Go to **"📤 Upload Documents"** tab
2. Click **"Upload Documents"**
3. Select PDF, TXT, or DOCX files
4. Click **"Process Documents"**
5. Wait for confirmation
6. Your documents are now searchable!

#### **Step 5: View History**

1. Go to **"📊 History"** tab
2. Click **"Refresh"** to see recent queries
3. Click **"Export"** to save query history for audit purposes

---

## 🗂️ Project Structure

```
compliance-rag-assistant/
│
├── src/                          # Source code
│   ├── core/                     # Core RAG functionality
│   │   ├── document_loader.py   # Load PDF, TXT, DOCX files
│   │   ├── text_processor.py    # Intelligent text chunking
│   │   ├── embeddings.py        # Embedding model management
│   │   ├── vector_store.py      # FAISS vector database
│   │   └── rag_engine.py        # Main RAG orchestrator
│   │
│   ├── models/                   # LLM provider integrations
│   │   ├── llm_factory.py       # Factory pattern for LLMs
│   │   ├── openai_provider.py   # OpenAI GPT integration
│   │   └── ollama_provider.py   # Ollama local models
│   │
│   ├── ui/                       # User interface
│   │   ├── gradio_app.py        # Gradio web application
│   │   └── components.py        # UI components & logic
│   │
│   └── utils/                    # Utilities
│       ├── config.py            # Configuration management
│       └── logger.py            # Logging utilities
│
├── configs/                      # Configuration files
│   ├── default.yaml             # Default settings
│   ├── development.yaml         # Dev environment
│   └── production.yaml          # Production settings
│
├── data/                         # Data storage
│   ├── regulatory_documents/    # Your documents go here
│   └── vector_db/               # Vector database storage
│
├── scripts/                      # Utility scripts
│   ├── create_sample_docs.py   # Generate sample documents
│   └── rebuild_vector_db.py    # Rebuild vector database
│
├── logs/                         # Application logs
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
```

---

## 🔧 Configuration

### Using Ollama (Free, Local)

**Setup Ollama:**
```bash
# 1. Download Ollama from https://ollama.ai

# 2. Pull a model
ollama pull llama2

# 3. Start Ollama (usually auto-starts)
ollama serve

# 4. In the UI, select:
#    - Provider: ollama
#    - Model: llama2
```

**Supported Ollama Models:**
- `llama2` - Fast, good quality
- `mistral` - Better quality, slightly slower
- `mixtral` - Best quality, requires more resources
- `codellama` - Good for technical compliance

### Using OpenAI (Paid, Cloud)

**Setup OpenAI:**
```bash
# 1. Get API key from https://platform.openai.com

# 2. Add to .env file
OPENAI_API_KEY=sk-your-key-here

# 3. In the UI, select:
#    - Provider: openai
#    - Model: gpt-3.5-turbo or gpt-4
#    - Enter your API key
```

### Adjusting Settings

Edit `configs/default.yaml`:

```yaml
# Chunk size for document splitting
rag:
  chunk_size: 1000              # Increase for longer chunks
  chunk_overlap: 200            # Context overlap

# Retrieval settings
  top_k: 4                      # Number of sources to retrieve

# LLM temperature
  temperature: 0.1              # Lower = more deterministic
```

---

## 📚 Adding Your Own Documents

### Supported File Formats
- PDF (`.pdf`)
- Text files (`.txt`)
- Word documents (`.docx`)
- Markdown (`.md`)

### Method 1: Using the Web UI
1. Go to **"📤 Upload Documents"** tab
2. Upload your files
3. Click **"Process Documents"**

### Method 2: Direct File Copy
```bash
# Copy files to documents directory
cp your_regulation.pdf data/regulatory_documents/

# Rebuild vector database
python scripts/rebuild_vector_db.py
```

### Organizing Documents
```bash
data/regulatory_documents/
├── gdpr/
│   ├── gdpr_full_text.pdf
│   └── gdpr_guidelines.pdf
├── sox/
│   ├── sox_section_302.pdf
│   └── sox_section_404.pdf
└── hipaa/
    ├── hipaa_privacy_rule.pdf
    └── hipaa_security_rule.pdf
```

---

## 🎓 Understanding RAG

### What is RAG?

**RAG (Retrieval Augmented Generation)** combines document search with AI generation:

1. **📄 Load**: Import your regulatory documents
2. **✂️ Chunk**: Split into manageable pieces (chunks)
3. **🔢 Embed**: Convert text to numerical vectors (embeddings)
4. **💾 Store**: Save in vector database (FAISS)
5. **🔍 Retrieve**: Find relevant chunks for your query
6. **🤖 Generate**: LLM creates answer using retrieved context

### Why RAG for Compliance?

- ✅ **Accurate**: Answers based on actual documents, not memorization
- ✅ **Transparent**: Shows source citations
- ✅ **Up-to-date**: Add new regulations easily
- ✅ **Private**: Can run entirely locally with Ollama
- ✅ **Auditable**: Track what was asked and answered

---

## 🧪 Testing the System

### Run the Test Script

```python
# test_system.py
from src.core.rag_engine import RegulatoryComplianceRAG

# Initialize
rag = RegulatoryComplianceRAG(
    llm_provider="ollama",
    model_name="llama2"
)

# Test query
result = rag.query(
    question="What are GDPR data retention requirements?",
    return_sources=True
)

print("Answer:", result['answer'])
print(f"Sources: {result['num_sources']}")
```

```bash
python test_system.py
```

---

## 🐛 Troubleshooting

### Issue: "System not initialized"
**Solution:** Go to System Setup tab and initialize the system first.

### Issue: "Ollama connection error"
**Solution:** 
```bash
# Check if Ollama is running
ollama list

# Start Ollama
ollama serve

# Pull the model
ollama pull llama2
```

### Issue: "No documents found"
**Solution:**
```bash
# Create sample documents
python scripts/create_sample_docs.py

# Or upload your own via the UI
```

### Issue: "OpenAI API key error"
**Solution:** Add your API key to `.env` file:
```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### Issue: "Import errors"
**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: "Slow responses"
**Solution:**
- Use a smaller model (llama2 instead of mixtral)
- Reduce `top_k` in config (fewer sources retrieved)
- Use GPU if available (requires `faiss-gpu`)

---

## 📖 Learn More

### Understanding the Code

Start reading in this order:
1. `src/utils/config.py` - Configuration management
2. `src/core/document_loader.py` - How documents are loaded
3. `src/core/text_processor.py` - How text is chunked
4. `src/core/embeddings.py` - How embeddings work
5. `src/core/vector_store.py` - Vector database operations
6. `src/core/rag_engine.py` - Main orchestrator (brings it all together)
7. `src/ui/gradio_app.py` - Web interface

### Key Concepts

**Embeddings**: Converting text to numbers that represent meaning
```python
"data privacy" → [0.23, 0.56, 0.12, ...]
"personal information" → [0.24, 0.55, 0.13, ...]
# Similar meanings = similar vectors
```

**Vector Search**: Finding similar text using math
```python
query = "encryption requirements"
# Finds documents about: encryption, security, data protection
```

**Chunking**: Splitting documents while maintaining context
```python
Document (5000 words) →
    Chunk 1 (1000 chars) ─┐
    Chunk 2 (1000 chars) ─┼─ 200 char overlap
    Chunk 3 (1000 chars) ─┘
```

---

## 💡 Tips for Best Results

1. **Be Specific**: Ask detailed questions
   - ❌ "Tell me about GDPR"
   - ✅ "What are the GDPR requirements for data retention periods?"

2. **Use Filters**: Select regulation type when you know it
   - Faster and more accurate results

3. **Check Sources**: Always review source citations
   - Verify the information from original documents

4. **Add More Documents**: The more documents you add, the better the answers
   - Upload your company policies
   - Add regulatory updates

5. **Experiment with Models**:
   - Fast queries: `llama2`, `gpt-3.5-turbo`
   - Best quality: `mixtral`, `gpt-4`

---

## ⚠️ Important Notes

### Disclaimer
This tool is for **informational purposes only** and does not constitute legal advice. Always consult with qualified legal professionals for compliance matters.

### Data Privacy
- **With Ollama**: All data stays on your machine (100% private)
- **With OpenAI**: Queries are sent to OpenAI's servers (read their privacy policy)

### Limitations
- Answers are only as good as the documents you provide
- AI can make mistakes - always verify important information
- Not a replacement for compliance officers or legal counsel

---

## 📞 Support

Having issues? Check:
1. The troubleshooting section above
2. Application logs in `logs/app.log`
3. Status messages in the UI

---

## 📄 License

MIT License - See LICENSE file for details

---

**Made with ❤️ for the compliance community**

*Last Updated: Oct2025*