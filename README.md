# AI-Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that allows users to query a custom document corpus. This system intelligently decides whether to answer queries using the uploaded document knowledge base or via web search, powered by OpenAI and LangChain.

## Features

- Vector Database: FAISS-based vector store for fast similarity search
- Smart Embeddings: OpenAI Embeddings for high-quality document representations
- RAG Pipeline: LangChain implementation with chunking, retrieval, and LLM-based answering
- Agentic AI: Automatic decision between RAG and web search based on retrieval confidence
- REST API: FastAPI endpoints for document upload and querying
- Dashboard: HTML dashboard to visualize document chunks and similarity scores
- Docker Support: Full containerization with Docker and docker-compose
- Python SDK: Simple client library for programmatic API access
- Evaluation Metrics: Retrieval confidence and similarity scores

## Requirements

Python 3.9+, OpenAI API Key, Docker & Docker Compose (optional for containerized deployment)

## Installation

Clone the repository: `git clone <your-repo-url> && cd ai-assistant`  
Create virtual environment: `python -m venv venv && source venv/bin/activate` (Windows: `venv\Scripts\activate`)  
Install dependencies: `pip install -r requirements.txt`  
Configure environment variables: `cp env .env` and add `OPENAI_API_KEY=your_openai_api_key_here`  
Run the application locally: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`  
For Docker: `cp env .env && docker-compose up --build`  
API available at `http://localhost:8000`

## API Endpoints

Health check: `GET /` returns `{"message": "API is running!"}`  
Upload documents: `POST /documents` with `files=@docs/university.txt` and `files=@docs/technology.txt` returns `{"uploaded_files": ["university.txt","technology.txt"], "total_chunks": 50, "new_chunks": 25}`  
Query (GET): `GET /query?q=What+is+AI?`  
Query (POST): `POST /query` with `q=What is AI?` returns `{"query": "What is AI?", "source": "rag", "results":[{"chunk": "AI is the simulation of human intelligence in machines...", "score": 0.125}], "answer": "AI refers to computer systems that can perform tasks that normally require human intelligence...", "retrieval_confidence": "High (avg_score: 0.125)", "agent_decision": "Used RAG due to high retrieval confidence"}`  
Dashboard: `GET /dashboard` shows all stored chunks and scores with filtering options for High/Medium/Low confidence

## Python SDK Usage

Initialize client: `from client import AIClient; client = AIClient(api_url="http://localhost:8000")`  
Upload documents: `resp = client.upload_documents("docs")`  
Query: `resp = client.query("What is AI?")`  
Interactive CLI: `python client_upload.py`

## Agentic AI Decision-Making

RAG used when retrieval confidence is HIGH (avg_score ≤ 0.8)  
Web Search used when retrieval confidence is LOW (avg_score > 0.8)  
Decision logic: `if avg_score > 0.8: answer = web_search_and_answer(query) else: answer = rag_pipeline(query)`

## Architecture

Client → POST /documents → Upload & Chunk → FAISS  
Client → GET /query → Similarity Search → Confidence Check → High: RAG Pipeline → OpenAI, Low: Web Search → OpenAI → Response

Components: Text Splitter RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200), Embeddings OpenAI Embeddings, Vector Store FAISS (persistent in data/texts.json), LLM GPT-4o-mini, Web Search DuckDuckGo (fallback)

## Evaluation & Metrics

Similarity Scores (L2 distance), Retrieval Confidence (High/Low), Top-k Retrieval (3 chunks), Source Attribution (RAG/Web Search)

## Docker Compose Services

app: FastAPI app (port 8000)

## Project Structure

`ai-assistant/`  
`├── main.py`  
`├── rag_langchain.py`  
`├── client.py`  
`├── client_upload.py`  
`├── requirements.txt`  
`├── Dockerfile`  
`├── Dockerfile.cpu`  
`├── docker-compose.yml`  
`├── env`  
`├── data/texts.json`  
`├── docs/university.txt`  
`├── docs/technology.txt`  
`└── templates/dashboard.html`

## Testing

Manual: `curl -X POST "http://localhost:8000/documents" -F "files=@docs/university.txt"`  
`curl "http://localhost:8000/query?q=What%20is%20AI?"`  
Python SDK: `python client_upload.py`

## Configuration

Adjust chunk size & overlap in `main.py`  
Adjust top-k retrieval in `main.py`  
Adjust confidence threshold in `main.py` (`if avg_score > 0.8`)

## License

MIT License


