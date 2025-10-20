import os
import json
import time
import uuid
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, Query, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import numpy as np

from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.prompts.registry import prompt_registry, PromptMetric
except ImportError as e:
    logger.warning(f"Prompt registry not available: {e}")
    prompt_registry = None

try:
    from src.orchestration.workflow import OrchestrationWorkflow
except ImportError as e:
    logger.warning(f"Orchestration workflow not available: {e}")
    OrchestrationWorkflow = None

try:
    from src.monitoring.metrics import MonitoringService, QueryMetrics
except ImportError as e:
    logger.warning(f"Monitoring service not available: {e}")
    MonitoringService = None
    QueryMetrics = None

try:
    from src.finetuning.peft_adapter import PEFTAdapterManager, FineTuningConfig
except ImportError as e:
    logger.warning(f"Fine-tuning components not available: {e}")
    PEFTAdapterManager = None
    FineTuningConfig = None

try:
    from src.integrations.a2a_client import A2AOrchestrator
except ImportError as e:
    logger.warning(f"A2A orchestrator not available: {e}")
    A2AOrchestrator = None

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env!")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="AI Assistant Platform",
    description="Production-ready AI platform with orchestration, monitoring, MCP, and fine-tuning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TEXTS_PATH = DATA_DIR / "texts.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"


texts: List[str] = []
vectorstore = None
qa_chain = None
orchestration_workflow = None
monitoring_service = None
a2a_orchestrator = None
fine_tuning_manager = None


def initialize_platform():
    """Initialize all platform components"""
    global orchestration_workflow, monitoring_service, a2a_orchestrator, fine_tuning_manager
    
    try:
        
        if MonitoringService:
            monitoring_service = MonitoringService()
            logger.info("Monitoring service initialized")
        else:
            logger.warning("Monitoring service not available")
        
        if A2AOrchestrator:
            a2a_orchestrator = A2AOrchestrator()
            logger.info("A2A orchestrator initialized")
        else:
            logger.warning("A2A orchestrator not available")
        
        if FineTuningConfig and PEFTAdapterManager:
            try:
                fine_tuning_config = FineTuningConfig()
                fine_tuning_manager = PEFTAdapterManager(fine_tuning_config)
                logger.info("Fine-tuning manager initialized")
            except Exception as e:
                logger.warning(f"Fine-tuning manager initialization failed: {e}")
                fine_tuning_manager = None
        else:
            logger.warning("Fine-tuning components not available")
            fine_tuning_manager = None
        
        logger.info("Platform components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize platform components: {e}")

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def load_texts():
    global texts
    if TEXTS_PATH.exists():
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            texts.extend(json.load(f))
    print(f"Loaded {len(texts)} chunks.")

def save_texts():
    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

def save_vectorstore():
    if vectorstore:
        vectorstore.save_local(str(FAISS_INDEX_PATH))
        print(f"Vectorstore saved to {FAISS_INDEX_PATH}")

def load_vectorstore():
    global vectorstore, qa_chain
    if FAISS_INDEX_PATH.exists():
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True  
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        print(f"Vectorstore loaded from {FAISS_INDEX_PATH}")

def update_vectorstore():
    global vectorstore, qa_chain, orchestration_workflow
    if texts:
        vectorstore = FAISS.from_texts(texts, embedding=embeddings)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        if orchestration_workflow:
            orchestration_workflow.vectorstore = vectorstore
        
        logger.info("Vectorstore and RAG pipeline updated.")

load_texts()
load_vectorstore()
initialize_platform()

if not vectorstore:
    update_vectorstore()

if not orchestration_workflow and OrchestrationWorkflow:
    orchestration_workflow = OrchestrationWorkflow(OPENAI_API_KEY, vectorstore)
    logger.info("Orchestration workflow initialized")
elif not OrchestrationWorkflow:
    logger.warning("Orchestration workflow not available")

def record_query_metrics(query: str, response: Dict[str, Any], start_time: float, tokens_used: int = 0):
    """Record metrics for monitoring"""
    try:
        if monitoring_service:
            query_id = str(uuid.uuid4())
            source = response.get("source", "unknown")
            confidence = response.get("confidence", None)
            
            if isinstance(confidence, str) and "(" in confidence:
                try:
                    confidence = float(confidence.split("(")[1].split(",")[0].split(":")[1].strip())
                except:
                    confidence = None
            
            metrics = QueryMetrics(
                query_id=query_id,
                timestamp=datetime.utcnow(),
                query_text=query,
                source=source,
                response_time=time.time() - start_time,
                tokens_used=tokens_used,
                retrieval_confidence=confidence,
                user_rating=None,
                error_occurred="error" in response or "Error" in str(response.get("answer", ""))
            )
            
            monitoring_service.record_query_metric(metrics)
    except Exception as e:
        logger.error(f"Failed to record metrics: {e}")

def web_search(query: str, max_results: int = 3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [{"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")} for r in results]
    except Exception as e:
        print(f"Web search error: {e}")
        return []

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.post("/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    global texts
    new_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for file in files:
        contents = await file.read()
        text = contents.decode("utf-8")
        chunks = text_splitter.split_text(text)
        texts.extend(chunks)
        new_chunks.extend(chunks)

    if new_chunks:
        save_texts()
        update_vectorstore()
        save_vectorstore()

    print(f"Uploaded {len(new_chunks)} new chunks.")
    return {"uploaded_files": [file.filename for file in files], "total_chunks": len(texts), "new_chunks": len(new_chunks)}

@app.post("/query")
async def query_post(q: str = Form(...)):
    return await query_handler(q)

@app.get("/query")
async def query_get(q: str = Query(..., description="Query string")):
    return await query_handler(q)

async def query_handler(q: str, enhancements: Optional[List[str]] = None):
    """Enhanced query handler using orchestration workflow"""
    start_time = time.time()
    
    if not orchestration_workflow:
        if not qa_chain or not texts:
            return JSONResponse({"error": "No documents uploaded and orchestration not available!"}, status_code=400)
        
        return await original_query_handler(q)
    
    try:
        result = orchestration_workflow.orchestrate_query(q)
        
        if enhancements and a2a_orchestrator:
            try:
                enhanced_result = await a2a_orchestrator.multi_service_enhancement(
                    q, result.get("answer", ""), enhancements
                )
                result.update(enhanced_result)
            except Exception as e:
                logger.error(f"A2A enhancement failed: {e}")
        
        tokens_used = len(str(result.get("answer", "")).split()) * 1.3  # Rough estimate
        record_query_metrics(q, result, start_time, int(tokens_used))
        
        return result
        
    except Exception as e:
        error_response = {
            "query": q,
            "source": "error",
            "results": [],
            "answer": f"Error: {e}",
            "agent_decision": "Error during query processing"
        }
        record_query_metrics(q, error_response, start_time)
        return error_response

async def original_query_handler(q: str):
    """Original query handler as fallback"""
    if not qa_chain or not texts:
        return JSONResponse({"error": "No documents uploaded!"}, status_code=400)

    try:
        docs_with_scores = vectorstore.similarity_search_with_score(q, k=3)
        results = [{"chunk": d.page_content, "score": float(s)} for d, s in docs_with_scores]

        avg_score = np.mean([r["score"] for r in results]) if results else 1.0
        logger.info(f"Average distance: {avg_score:.4f}")

        if avg_score > 0.8:
            logger.info("Agent Decision: Using Web Search (low retrieval confidence)")
            web_results = web_search(q, max_results=3)
            if web_results:
                web_context = "\n\n".join([f"Source: {r['title']}\n{r['snippet']}\nURL: {r['url']}" for r in web_results])
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
                answer = llm.predict(f"Based on the following web search results, answer the question.\n\nQuestion: {q}\n\nWeb Results:\n{web_context}\n\nAnswer:")
                return {
                    "query": q,
                    "source": "web_search",
                    "results": web_results,
                    "answer": answer,
                    "retrieval_confidence": f"Low (avg_score: {avg_score:.4f})",
                    "agent_decision": "Used web search due to low retrieval confidence"
                }
            else:
                answer = qa_chain.run(q)
                return {
                    "query": q,
                    "source": "rag_fallback",
                    "results": results,
                    "answer": answer,
                    "retrieval_confidence": f"Low (avg_score: {avg_score:.4f})",
                    "agent_decision": "Web search failed, used RAG fallback"
                }
        else:
            logger.info("Agent Decision: Using RAG (good retrieval confidence)")
            answer = qa_chain.run(q)
            return {
                "query": q,
                "source": "rag",
                "results": results,
                "answer": answer,
                "retrieval_confidence": f"High (avg_score: {avg_score:.4f})",
                "agent_decision": "Used RAG due to high retrieval confidence"
            }

    except Exception as e:
        return {
            "query": q,
            "source": "error",
            "results": [],
            "answer": f"Error: {e}",
            "agent_decision": "Error during query processing"
        }


@app.post("/query/advanced")
async def advanced_query(
    q: str = Form(...),
    enhancements: Optional[str] = Form(None)
):
    """Advanced query endpoint with optional A2A enhancements"""
    enhancement_list = []
    if enhancements:
        enhancement_list = [e.strip() for e in enhancements.split(",")]
    
    return await query_handler(q, enhancement_list)

@app.post("/orchestration/summarize")
async def summarize_text(
    text: str = Form(...),
    summary_length: str = Form("medium")
):
    """Summarization endpoint using orchestration"""
    if not orchestration_workflow:
        return JSONResponse({"error": "Orchestration not available"}, status_code=503)
    
    try:
        result = orchestration_workflow.summarization_tool({
            "text": text,
            "summary_length": summary_length
        })
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/orchestration/translate")
async def translate_text(
    text: str = Form(...),
    source_language: str = Form("auto"),
    target_language: str = Form("English")
):
    """Translation endpoint using orchestration"""
    if not orchestration_workflow:
        return JSONResponse({"error": "Orchestration not available"}, status_code=503)
    
    try:
        result = orchestration_workflow.translation_tool({
            "text": text,
            "source_language": source_language,
            "target_language": target_language
        })
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/monitoring/metrics")
async def get_monitoring_metrics(days: int = Query(7, ge=1, le=90)):
    """Get monitoring metrics"""
    if not monitoring_service:
        return JSONResponse({"error": "Monitoring not available"}, status_code=503)
    
    try:
        stats = monitoring_service.get_performance_statistics(days=days)
        return stats
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/monitoring/report")
async def get_monitoring_report(days: int = Query(7, ge=1, le=90)):
    """Get Evidently monitoring report"""
    if not monitoring_service:
        return JSONResponse({"error": "Monitoring not available"}, status_code=503)
    
    try:
        report = monitoring_service.generate_evidently_report(days=days)
        if report is None:
            return {"message": "No report data available", "days": days}
        
        return {
            "report_data": report.get_dict() if hasattr(report, 'get_dict') else str(report),
            "days": days
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/prompts")
async def list_prompts():
    """List available prompts from registry"""
    if not prompt_registry:
        return JSONResponse({"error": "Prompt registry not available"}, status_code=503)
    
    try:
        prompts = prompt_registry.list_prompts()
        return {"prompts": prompts}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/prompts/{prompt_name}/performance")
async def get_prompt_performance(prompt_name: str, days: int = Query(30, ge=1, le=90)):
    """Get performance statistics for a specific prompt"""
    if not prompt_registry:
        return JSONResponse({"error": "Prompt registry not available"}, status_code=503)
    
    try:
        stats = prompt_registry.get_performance_stats(prompt_name, days=days)
        return {"prompt_name": prompt_name, "stats": stats, "days": days}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/fine-tuning/setup")
async def setup_fine_tuning(background_tasks: BackgroundTasks):
    """Setup fine-tuning pipeline"""
    if not fine_tuning_manager:
        return JSONResponse({"error": "Fine-tuning not available"}, status_code=503)
    
    def run_fine_tuning():
        try:
            success = fine_tuning_manager.setup_fine_tuning_pipeline(texts)
            logger.info(f"Fine-tuning setup completed: {success}")
        except Exception as e:
            logger.error(f"Fine-tuning setup failed: {e}")
    
    background_tasks.add_task(run_fine_tuning)
    return {"message": "Fine-tuning setup initiated in background"}

@app.get("/fine-tuning/status")
async def get_fine_tuning_status():
    """Get fine-tuning status"""
    if not fine_tuning_manager:
        return JSONResponse({"error": "Fine-tuning not available"}, status_code=503)
    
    return {
        "is_trained": fine_tuning_manager.is_loaded,
        "model_available": fine_tuning_manager.is_loaded
    }

@app.post("/fine-tuning/query")
async def query_fine_tuned_model(query: str = Form(...)):
    """Query using fine-tuned model"""
    if not fine_tuning_manager or not fine_tuning_manager.is_loaded:
        return JSONResponse({"error": "Fine-tuned model not available"}, status_code=503)
    
    try:
        result = fine_tuning_manager.generate_response(query)
        return {"query": query, "response": result, "source": "fine_tuned_model"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    components = {
        "orchestration": orchestration_workflow is not None,
        "monitoring": monitoring_service is not None,
        "fine_tuning": fine_tuning_manager is not None,
        "vectorstore": vectorstore is not None,
        "a2a": a2a_orchestrator is not None
    }
    
    overall_health = all(components.values())
    
    return {
        "status": "healthy" if overall_health else "degraded",
        "components": components,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/dashboard")
def dashboard(request: Request):
    """Enhanced dashboard with new metrics"""
    chunks_display = []
    
    try:
        if vectorstore and texts:
            try:
                # For dashboard, we want to show all chunks with meaningful scores
                # Use a generic but meaningful query that represents general document content
                generic_query = "information content data knowledge document text"
                
                # Get all available chunks with similarity scores
                results_with_scores = vectorstore.similarity_search_with_score(generic_query, k=len(texts))
                
                # Convert results to display format with proper scores
                for doc, score in results_with_scores:
                    chunks_display.append({"text": doc.page_content, "score": float(score)})
                    
                logger.info(f"Dashboard: Retrieved {len(chunks_display)} chunks with similarity scores (query: '{generic_query}')")
                        
            except Exception as e:
                logger.warning(f"Could not get similarity scores from vectorstore: {e}")
                # Fallback: show chunks without proper scores but indicate they're available
                try:
                    # Try to get documents without similarity scores as a fallback
                    docs = vectorstore.similarity_search("content", k=min(len(texts), 50))
                    for doc in docs:
                        chunks_display.append({"text": doc.page_content, "score": 0.5})  # Use neutral score for fallback
                except Exception as fallback_error:
                    logger.warning(f"Could not get documents from vectorstore fallback: {fallback_error}")
                    # Ultimate fallback: show raw text chunks with neutral scores
                    for chunk in texts[:50]:  # Limit to 50 for performance
                        chunks_display.append({"text": chunk, "score": 0.5})
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        chunks_display = []
    
    # Final fallback if no chunks from vectorstore
    if not chunks_display and texts:
        for chunk in texts[:50]:  # Limit to 50 for performance
            chunks_display.append({"text": chunk, "score": 0.5})
    
    system_status = {
        "orchestration_available": orchestration_workflow is not None,
        "monitoring_available": monitoring_service is not None,
        "fine_tuning_available": fine_tuning_manager is not None and (hasattr(fine_tuning_manager, 'is_loaded') and fine_tuning_manager.is_loaded),
        "total_chunks": len(texts),
        "vectorstore_available": vectorstore is not None
    }
    
    return templates.TemplateResponse(
        "dashboard.html", 
        {
            "request": request, 
            "chunks": chunks_display,
            "system_status": system_status
        }
    )
