# AI Assistant Platform v2.0 - Production-Ready AI Orchestrator

## Overview

This is a comprehensive, production-ready AI platform that extends beyond the original RAG system into a full AI orchestration and monitoring ecosystem. Built according to AI Engineering Part II requirements, it includes advanced orchestration, monitoring, MCP protocol support, API-to-API integrations, prompt engineering, and fine-tuning capabilities.

## New Features (AI Engineering Part II Implementation)

### Part 1: LLM Orchestration
- **LangChain Expression Language**: Advanced pipeline orchestration
- **Tool Chaining**: RAG, Summarization, Translation tools integrated
- **Smart Routing**: Automatic tool selection based on query analysis
- **Workflow Management**: Comprehensive orchestration workflow system

### Part 2: Model Evaluation & Monitoring
- **Evidently AI Integration**: Advanced data drift and quality monitoring
- **Real-time Metrics**: Latency, token usage, retrieval accuracy tracking
- **PostgreSQL Storage**: Persistent metrics storage (SQLite fallback)
- **Prometheus Integration**: Production-ready metrics exposure

### Part 3: Model Context Protocol (MCP)
- **MCP Server**: Full protocol implementation
- **External Integrations**: VS Code, CLI tool access
- **Command Support**: `query-docs`, `summarize`, `translate` commands
- **Resource Management**: Prompt and metrics resource access

### Part 4: API-to-API Integration
- **External AI Services**: Sentiment analysis, translation, knowledge base APIs
- **Service Orchestration**: Multi-service enhancement capabilities
- **Fallback Mechanisms**: Graceful degradation when external services fail
- **Configuration Management**: Flexible service configuration

### Part 5: Prompt Engineering
- **Prompt Registry**: YAML-based versioned prompt management
- **Performance Tracking**: Prompt effectiveness monitoring
- **Template Library**: 5+ optimized templates for different tasks
- **A/B Testing**: Prompt performance comparison capabilities

### Part 6: Fine-Tuning & PEFT
- **LoRA Adapters**: Parameter-efficient fine-tuning support
- **Model Integration**: Fine-tuned model as optional backend
- **Training Pipeline**: Automated fine-tuning from document corpus
- **Evaluation Framework**: Model performance assessment

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   MCP Clients    │    │  External APIs  │
│   (Web, CLI)    │    │  (VS Code, etc)  │    │  (Sentiment,    │
└─────────┬───────┘    └─────────┬────────┘    │   Translation)  │
          │                      │             └─────────┬───────┘
          │                      │                       │
          ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Main Application                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │  Orchestration  │ │   Monitoring    │ │   A2A Client    │  │
│  │    Workflow     │ │    Service      │ │  Orchestrator   │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │  Prompt         │ │   Fine-tuning   │ │   MCP Server    │  │
│  │  Registry       │ │   Manager       │ │                 │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │                      │                       │
          ▼                      ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Vector Store  │    │    PostgreSQL    │    │   File System   │
│     (FAISS)     │    │   (Metrics DB)   │    │  (Models, Data) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API Key
- Docker & Docker Compose (optional)
- Git

### Installation

1. **Clone and Setup**
```bash
git clone <repository-url>
cd ai-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Environment Configuration**
```bash
cp .env.template .env
# Edit .env and add your OPENAI_API_KEY
```

3. **Run the Platform**
```bash
# Full platform with orchestration
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or with Docker
docker-compose up --build
```

### Access Points
- **Main API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Monitoring Dashboard**: http://localhost:8501 (Streamlit)
- **Prometheus Metrics**: http://localhost:8001/metrics
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Core Endpoints
- `GET /` - Health check
- `POST /documents` - Upload documents for RAG
- `GET|POST /query` - Query with orchestration
- `POST /query/advanced` - Advanced queries with A2A enhancements

### Orchestration Endpoints
- `POST /orchestration/summarize` - Text summarization
- `POST /orchestration/translate` - Text translation
- `GET /orchestration/status` - Orchestration status

### Monitoring Endpoints
- `GET /monitoring/metrics?days=7` - Performance metrics
- `GET /monitoring/report?days=7` - Evidently AI reports
- `GET /prompts` - List available prompts
- `GET /prompts/{name}/performance` - Prompt performance stats

### Fine-tuning Endpoints
- `POST /fine-tuning/setup` - Initialize fine-tuning
- `GET /fine-tuning/status` - Training status
- `POST /fine-tuning/query` - Query fine-tuned model

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here
DATABASE_URL=sqlite:///./data/ai_assistant.db  # or PostgreSQL URL
PROMETHEUS_PORT=8001
STREAMLIT_PORT=8501
```

### Prompt Registry Configuration
Edit `prompts/registry.yaml` to customize prompts:
```yaml
prompts:
  - name: "rag_query"
    version: "1.0"
    description: "RAG query prompt"
    template: |
      Context: {context}
      Question: {question}
      Answer:
    variables: ["context", "question"]
```

## Development

### Running Individual Components

1. **Main Application**
```bash
uvicorn main:app --reload --port 8000
```

2. **MCP Server**
```bash
python run_mcp_server.py
```

3. **Monitoring Dashboard**
```bash
python run_dashboard.py
# or directly:
streamlit run src/dashboard/monitoring_app.py --server.port 8501
```

### Testing
```bash
# Test orchestration
curl -X POST "http://localhost:8000/orchestration/summarize" \
  -F "text=Your text here" \
  -F "summary_length=medium"

# Test advanced query with enhancements
curl -X POST "http://localhost:8000/query/advanced" \
  -F "q=What is AI?" \
  -F "enhancements=sentiment,translate"

# Check health
curl http://localhost:8000/health
```

## Monitoring & Analytics

### Real-time Metrics
- Request latency and throughput
- Token usage and costs
- Retrieval accuracy scores
- Error rates and patterns

### Data Drift Detection
- Automated data quality monitoring
- Performance degradation alerts
- Historical trend analysis

### Dashboard Features
- Interactive visualizations (Plotly)
- Real-time metrics display
- Export capabilities for reports
- Component health monitoring

## MCP Integration

### VS Code Extension
The MCP server enables VS Code integration:
```json
{
  "mcp.servers": {
    "ai-assistant": {
      "command": "python",
      "args": ["run_mcp_server.py"]
    }
  }
}
```

### Available MCP Tools
- `query-docs` - Query uploaded documents
- `summarize` - Summarize text content
- `translate` - Translate between languages
- `web-search` - Search web for current information

## Fine-tuning

### Setup Fine-tuning
```bash
# Upload documents first via /documents endpoint
# Then initialize fine-tuning
curl -X POST http://localhost:8000/fine-tuning/setup
```

### Supported Models
- DialoGPT-medium (default, lightweight)
- LLaMA-2 (if available)
- FLAN-T5 (for instruction following)

## Production Deployment

### Docker Deployment
```bash
docker-compose up -d
```

### Environment Considerations
- Set `DATABASE_URL` to PostgreSQL for production
- Configure proper CORS origins
- Set up log aggregation
- Monitor resource usage

### Scaling Considerations
- Use Redis for session management
- Load balance multiple instances
- Separate monitoring database
- Implement proper logging

## Troubleshooting

### Common Issues

1. **Import Errors**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt
```

2. **Database Issues**
```bash
# Check database permissions and connectivity
curl http://localhost:8000/health
```

3. **Fine-tuning Failures**
```bash
# Check GPU availability and memory
nvidia-smi  # For GPU debugging
```

### Logs
Check application logs for detailed error information:
```bash
# Docker logs
docker-compose logs -f app

# Local logs
tail -f logs/app.log
```

## Performance Optimization

### Recommendations
- Use GPU for fine-tuning and inference
- Optimize chunk sizes for your domain
- Monitor token usage and costs
- Cache frequently accessed vectors
- Use connection pooling for databases

### Benchmarking
The platform includes built-in performance monitoring and can be benchmarked using the `/monitoring/metrics` endpoint.

## Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

## License

MIT License - see LICENSE file for details.

---

## AI Engineering Part II Requirements Status

**Part 1: LLM Orchestration** - LangChain Expression Language with RAG, summarization, and translation chaining

**Part 2: Model Evaluation & Monitoring** - Evidently AI integration with latency, token usage, and retrieval accuracy tracking

**Part 3: MCP Server Protocol** - Complete Model Context Protocol implementation with VS Code/CLI access

**Part 4: A2A Protocol Integration** - API-to-API integration for external AI services (sentiment analysis, translation)

**Part 5: Prompt Engineering** - YAML-based prompt registry with versioning and 5+ optimized templates

**Part 6: Fine-Tuning & PEFT** - LoRA adapters on smaller LLMs with optional backend integration

