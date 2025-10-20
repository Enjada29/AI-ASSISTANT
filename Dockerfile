FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY rag_langchain.py .
COPY client.py .
COPY client_upload.py .
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY templates/ ./templates/
COPY run_mcp_server.py .
COPY run_dashboard.py .

RUN mkdir -p data models/peft_adapters

COPY .env* ./

EXPOSE 8000 8001 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
