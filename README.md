# TULens - TU Wien Informatics Assistant

A RAG-powered chatbot for TU Wien master students in informatics programs. Ask questions about curricula, courses, admission requirements, and more.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F61?style=flat)
![Groq](https://img.shields.io/badge/Groq-00D4AA?style=flat)

## Features

- 🎓 **Chat Interface** - Streamlit-based conversational UI
- 📚 **Multi-source RAG** - Combines web pages and PDF curricula
- 📄 **Upload PDFs** - Add your own documents for context (session-based)
- 🔍 **Smart Retrieval** - Prioritizes user uploads with relevance filtering
- 🌐 **Rich Sources** - Shows program, page numbers, and direct PDF links

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   TUWienRAG      │────▶│   Groq LLM      │
│   (app.py)      │     │   (rag_system.py)│     │   (Llama 3.3)   │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
           ┌───────────────┐         ┌───────────────┐
           │ User Uploads  │         │ Pre-scraped   │
           │ (In-Memory)   │         │ (Docker)      │
           │ ChromaDB      │         │ ChromaDB      │
           └───────────────┘         └───────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key from [console.groq.com](https://console.groq.com)

### 3. Start ChromaDB

```bash
docker-compose up -d
```

### 4. Ingest Documents

Run the scraping pipeline to populate the vector database:

```bash
python document_scraping.py
```

This will:
- Scrape TU Wien informatics program pages
- Extract and process curriculum PDFs
- Store embeddings in ChromaDB

### 5. Launch the Chat UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

### Asking Questions

Simply type your question in the chat input:

- "What are the requirements for the Data Science master program?"
- "How many ECTS credits do I need to complete?"
- "What courses are available in Software Engineering?"

### Uploading Documents

1. Use the sidebar to upload additional PDF documents
2. Uploaded documents are stored in-memory for your session
3. Results from your uploads are prioritized in responses

## Project Structure

```
TULens/
├── app.py                  # Streamlit chat UI
├── rag_system.py           # RAG system with dual vector stores
├── document_scraping.py    # Web scraping & ingestion pipeline
├── config.py               # Configuration settings
├── docker-compose.yaml     # ChromaDB container setup
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (create this)
```

## Configuration

Edit `config.py` to customize:

```python
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
LLM_MODEL = "llama-3.3-70b-versatile"
CHROMADB_COLLECTION = "tu_wien_informatics"
```

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3.3 70B)
- **Embeddings**: HuggingFace (BGE-small)
- **Vector Store**: ChromaDB
- **Framework**: LangChain

## License

MIT