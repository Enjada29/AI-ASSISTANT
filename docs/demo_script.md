# AI Assistant Platform Demo Script

## Overview

This demo script demonstrates the comprehensive AI Assistant Platform v2.0 capabilities, showcasing all major features including orchestration, monitoring, MCP integration, fine-tuning, and more.

## Prerequisites

Before running the demo, ensure:
- Python 3.9+ installed
- OpenAI API key configured in `.env` file
- All dependencies installed: `pip install -r requirements.txt`
- Platform running: `uvicorn main:app --reload`

## Demo Sections

### 1. Platform Health Check

Start by verifying the platform is running and all components are available:

```bash
# Check overall health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "components": {
    "orchestration": true,
    "monitoring": true,
    "fine_tuning": true,
    "vectorstore": true,
    "a2a": true
  }
}
```

### 2. Document Upload and RAG Setup

Upload sample documents to establish the knowledge base:

```bash
# Upload multiple documents
curl -X POST "http://localhost:8000/documents" \
  -F "files=@docs/university.txt" \
  -F "files=@docs/technology.txt" \
  -F "files=@docs/science.txt"

# Expected response:
{
  "message": "Documents processed successfully",
  "uploaded_files": ["university.txt", "technology.txt", "science.txt"],
  "total_chunks": 150,
  "new_chunks": 75
}
```

### 3. Basic RAG Query

Demonstrate basic document-based question answering:

```bash
# Simple query
curl -X POST "http://localhost:8000/query" \
  -F "q=What is artificial intelligence?"

# Expected features demonstrated:
# - Vector similarity search
# - Retrieval confidence scoring
# - Source attribution (RAG vs Web Search)
# - Intelligent agent decision-making
```

### 4. Advanced Orchestration Features

Showcase LangChain Expression Language orchestration:

#### A. Summarization Tool
```bash
# Text summarization
curl -X POST "http://localhost:8000/orchestration/summarize" \
  -F "text=Artificial intelligence is a broad field of computer science focused on creating systems that can perform tasks typically requiring human intelligence. This includes machine learning, natural language processing, computer vision, and robotics. AI systems can learn from data, recognize patterns, make predictions, and even generate creative content." \
  -F "summary_length=medium"

# Expected response includes:
# - Concise summary generation
# - Length-controlled output
# - Performance metrics
```

#### B. Translation Tool
```bash
# Text translation
curl -X POST "http://localhost:8000/orchestration/translate" \
  -F "text=Hello, how are you today?" \
  -F "target_language=Spanish"

# Expected response includes:
# - Accurate translation
# - Language detection
# - Translation confidence
```

#### C. Advanced Query with Enhancements
```bash
# Query with A2A enhancements
curl -X POST "http://localhost:8000/query/advanced" \
  -F "q=Explain the impact of AI on healthcare" \
  -F "enhancements=sentiment,translate,summarize"
```

### 5. Monitoring and Analytics

Demonstrate real-time monitoring capabilities:

```bash
# Get performance metrics
curl "http://localhost:8000/monitoring/metrics?days=7"

# Get Evidently AI reports
curl "http://localhost:8000/monitoring/report?days=7"

# List prompt performance
curl "http://localhost:8000/prompts"

# Check specific prompt performance
curl "http://localhost:8000/prompts/rag_query/performance"
```

#### Access Monitoring Dashboard
```bash
# Open monitoring dashboard in browser
# URL: http://localhost:8501
```

### 6. Prompt Engineering

Showcase the prompt registry system:

```bash
# List all available prompts
curl "http://localhost:8000/prompts"

# Get prompt details
curl "http://localhost:8000/prompts/rag_query"

# Query using specific prompt template
curl -X POST "http://localhost:8000/query" \
  -F "q=What are the benefits of renewable energy?" \
  -F "prompt_template=rag_query"
```

### 7. Fine-Tuning Demo

Demonstrate PEFT fine-tuning capabilities:

#### A. Setup Fine-tuning
```bash
# Initialize fine-tuning pipeline
curl -X POST "http://localhost:8000/fine-tuning/setup"

# Expected response:
{
  "message": "Fine-tuning pipeline initialized",
  "status": "setup_started"
}
```

#### B. Check Training Status
```bash
# Check if model is trained
curl "http://localhost:8000/fine-tuning/status"

# Expected response:
{
  "is_trained": true,
  "model_available": true
}
```

#### C. Query Fine-tuned Model
```bash
# Use fine-tuned model for inference
curl -X POST "http://localhost:8000/fine-tuning/query" \
  -F "query=What is machine learning?"
```

### 8. MCP Server Integration

Demonstrate Model Context Protocol capabilities:

```bash
# Start MCP server (in separate terminal)
python run_mcp_server.py

# Available MCP tools will be accessible via VS Code extension or CLI tools
```

#### MCP Tools Available:
- `query-docs`: Query uploaded documents
- `summarize`: Text summarization
- `translate`: Language translation
- `web-search`: Web search capabilities

### 9. Dashboard Visualization

Access the HTML dashboard for visual inspection:

```bash
# Open dashboard in browser
# URL: http://localhost:8000/dashboard?q=artificial intelligence

# Features to demonstrate:
# - Document chunk visualization
# - Similarity score filtering
# - Confidence-based highlighting
# - Interactive query interface
```

### 10. API-to-API Integration

Demonstrate external service integration:

```bash
# Query with external AI service enhancements
curl -X POST "http://localhost:8000/query/advanced" \
  -F "q=Analyze the sentiment of this statement: I love this new AI technology" \
  -F "enhancements=sentiment,translate"
```

## Demo Flow Summary

### Quick 5-Minute Demo
1. Health check → Document upload → RAG query
2. Show dashboard → Demonstrate orchestration (summarize)
3. Check monitoring metrics

### Full 15-Minute Demo
1. Platform overview and health check
2. Document upload and RAG setup
3. Basic and advanced querying
4. Orchestration tools (summarize, translate)
5. Monitoring dashboard walkthrough
6. Prompt engineering showcase
7. Fine-tuning demo (if time permits)

### Technical Deep Dive (30+ minutes)
1. Complete platform architecture overview
2. All API endpoints demonstration
3. Monitoring and analytics deep dive
4. MCP integration setup and usage
5. Fine-tuning pipeline walkthrough
6. A2A integration examples
7. Performance optimization discussion

## Troubleshooting Demo Issues

### Common Problems and Solutions

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Issues**
   ```bash
   # Check .env file has OPENAI_API_KEY
   cat .env | grep OPENAI_API_KEY
   ```

3. **Port Conflicts**
   ```bash
   # Kill existing processes on ports 8000, 8501
   lsof -ti:8000 | xargs kill -9
   lsof -ti:8501 | xargs kill -9
   ```

4. **Document Upload Issues**
   ```bash
   # Verify docs directory exists with sample files
   ls -la docs/
   ```

## Expected Demo Outcomes

### Key Features Demonstrated
- **Intelligent RAG**: Context-aware document answering
- **Orchestration**: Multi-tool pipeline chaining
- **Monitoring**: Real-time performance tracking
- **MCP Integration**: External tool accessibility
- **Fine-tuning**: Domain-specific model adaptation
- **A2A Services**: External AI service integration

### Performance Benchmarks
- Query response time: < 2 seconds
- Document processing: ~10 documents/minute
- Monitoring latency: Real-time updates
- Fine-tuning setup: < 30 seconds initialization

## Post-Demo Q&A Topics

### Technical Questions
- LoRA adapter efficiency vs. full fine-tuning
- Vector similarity scoring methodology
- LangChain Expression Language benefits
- MCP protocol advantages for extensibility

### Architecture Discussion
- Scalability considerations
- Production deployment strategies
- Monitoring and alerting setup
- Performance optimization techniques

This demo script provides a comprehensive walkthrough of all platform capabilities while maintaining practical, hands-on examples that showcase the production-ready nature of the AI Assistant Platform v2.0.
