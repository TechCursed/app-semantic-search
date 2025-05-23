# Semantic Search App

A semantic search system that lets you query your documents using natural language. Built with Ollama, FAISS, and FastAPI.

## Features

- Document ingestion from `.txt` files
- Vector-based semantic search using FAISS
- AI-powered responses with context from your documents
- FastAPI web service for easy integration

## Prerequisites

1. **Install Ollama** from [ollama.ai](https://ollama.ai)
2. **Pull required models**:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3
   ```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create docs folder and add your `.txt` files**:
   ```bash
   mkdir docs
   # Add your .txt files to the docs/ folder
   ```

## Usage

### Method 1: FastAPI Web Service

1. **Start the server**:
   ```bash
   uvicorn main:app --reload
   ```
   Server runs at `http://localhost:8000`

2. **Ingest documents**:
   ```bash
   curl -X POST "http://localhost:8000/ingest/"
   ```

3. **Query documents**:
   ```bash
   curl -X POST "http://localhost:8000/query/" \
     -H "Content-Type: application/json" \
     -d '{"question": "Your question here"}'
   ```

4. **Interactive API docs**: Visit `http://localhost:8000/docs`

### Method 2: Command Line

1. **Ingest documents**:
   ```bash
   python ingest.py
   ```

2. **Query documents**:
   ```bash
   python query.py
   ```

## API Endpoints

### POST `/ingest/`
Processes all `.txt` files in the `docs/` folder and creates vector embeddings.

**Response**:
```json
{
  "status": "success",
  "message": "5 documents ingested and indexed"
}
```

### POST `/query/`
Search documents and get AI-generated answers.

**Request**:
```json
{
  "question": "What are the main topics discussed?"
}
```

**Response**:
```json
{
  "question": "What are the main topics discussed?",
  "answer": "Based on the documents, the main topics include..."
}
```

## File Structure

```
├── requirements.txt    # Dependencies
├── ingest.py          # Document processing
├── query.py           # CLI interface
├── main.py            # FastAPI service
├── docs/              # Your .txt files
├── vector.index       # Generated FAISS index
└── metadata.json      # Generated metadata
```
