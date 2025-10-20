import time
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS
from src.prompts.registry import prompt_registry

class OrchestrationWorkflow:
    def __init__(self, openai_api_key: str, vectorstore: Optional[FAISS] = None):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.vectorstore = vectorstore
        
    def rag_tool(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """RAG tool for document-based question answering"""
        start_time = time.time()
        
        query = inputs["query"]
        
        if not self.vectorstore:
            return {
                "answer": "No documents available for RAG.",
                "confidence": 0.0,
                "source": "rag_no_docs",
                "response_time": time.time() - start_time
            }
        
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=3)
        
        if not docs_with_scores:
            return {
                "answer": "No relevant documents found.",
                "confidence": 0.0,
                "source": "rag_no_results",
                "response_time": time.time() - start_time
            }
        
        
        scores = [score for _, score in docs_with_scores]
        avg_score = sum(scores) / len(scores) if scores else 1.0
        confidence = max(0.0, 1.0 - avg_score)
        
        context = "\n\n".join([doc.page_content for doc, _ in docs_with_scores])
        
        prompt_template = prompt_registry.render_prompt(
            "rag_query",
            {"context": context, "question": query}
        )
        
        response = self.llm.invoke(prompt_template)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "confidence": confidence,
            "source": "rag",
            "context": context,
            "docs_found": len(docs_with_scores),
            "response_time": time.time() - start_time
        }
    
    def web_search_tool(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Web search tool for current information"""
        start_time = time.time()
        
        query = inputs["query"]
        max_results = inputs.get("max_results", 3)
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                search_results = [
                    {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")} 
                    for r in results
                ]
                
            if not search_results:
                return {
                    "answer": "No web search results found.",
                    "confidence": 0.0,
                    "source": "web_search_no_results",
                    "response_time": time.time() - start_time
                }
            
            formatted_results = "\n\n".join([
                f"Title: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}"
                for r in search_results
            ])
            
            prompt_template = prompt_registry.render_prompt(
                "web_search_qa",
                {"question": query, "search_results": formatted_results}
            )
            
            response = self.llm.invoke(prompt_template)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "answer": answer,
                "confidence": 0.7,  
                "source": "web_search",
                "search_results": search_results,
                "response_time": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "answer": f"Web search failed: {str(e)}",
                "confidence": 0.0,
                "source": "web_search_error",
                "response_time": time.time() - start_time
            }
    
    def summarization_tool(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Summarization tool"""
        start_time = time.time()
        
        text = inputs["text"]
        summary_length = inputs.get("summary_length", "medium")
        
        prompt_template = prompt_registry.render_prompt(
            "summarization",
            {"text": text, "summary_length": summary_length}
        )
        
        response = self.llm.invoke(prompt_template)
        summary = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "summary": summary,
            "source": "summarization",
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(text) if text else 0,
            "response_time": time.time() - start_time
        }
    
    def translation_tool(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Translation tool"""
        start_time = time.time()
        
        text = inputs["text"]
        source_lang = inputs.get("source_language", "auto")
        target_lang = inputs.get("target_language", "English")
        
        prompt_template = prompt_registry.render_prompt(
            "translation",
            {
                "text": text,
                "source_language": source_lang,
                "target_language": target_lang
            }
        )
        
        response = self.llm.invoke(prompt_template)
        translation = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "translation": translation,
            "source": "translation",
            "source_language": source_lang,
            "target_language": target_lang,
            "response_time": time.time() - start_time
        }
    
    def agent_decision_tool(self, inputs: Dict[str, Any]) -> str:
        """Tool to decide which approach to use"""
        query = inputs["query"]
        context_score = inputs.get("context_score", 1.0)
        
        prompt_template = prompt_registry.render_prompt(
            "agent_decision",
            {"query": query, "context_score": context_score}
        )
        
        response = self.llm.invoke(prompt_template)
        decision = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
        
        if decision not in ["RAG", "WEB", "TRANSLATE", "SUMMARIZE"]:
            if any(word in query.lower() for word in ["translate", "translation"]):
                decision = "TRANSLATE"
            elif any(word in query.lower() for word in ["summarize", "summary", "brief"]):
                decision = "SUMMARIZE"
            elif context_score < 0.8 and self.vectorstore:
                decision = "RAG"
            else:
                decision = "WEB"
        
        return decision
    
    def create_rag_chain(self):
        """Create RAG chain using LangChain Expression Language"""
        
        def format_rag_inputs(inputs):
            return {"query": inputs["query"]}
        
        rag_chain = RunnableLambda(format_rag_inputs) | RunnableLambda(self.rag_tool)
        return rag_chain
    
    def create_web_search_chain(self):
        """Create web search chain"""
        
        def format_web_inputs(inputs):
            return {
                "query": inputs["query"],
                "max_results": inputs.get("max_results", 3)
            }
        
        web_chain = RunnableLambda(format_web_inputs) | RunnableLambda(self.web_search_tool)
        return web_chain
    
    def create_summarization_chain(self):
        """Create summarization chain"""
        
        def format_summary_inputs(inputs):
            return {
                "text": inputs["text"],
                "summary_length": inputs.get("summary_length", "medium")
            }
        
        summary_chain = RunnableLambda(format_summary_inputs) | RunnableLambda(self.summarization_tool)
        return summary_chain
    
    def create_translation_chain(self):
        """Create translation chain"""
        
        def format_translation_inputs(inputs):
            return {
                "text": inputs["text"],
                "source_language": inputs.get("source_language", "auto"),
                "target_language": inputs.get("target_language", "English")
            }
        
        translation_chain = RunnableLambda(format_translation_inputs) | RunnableLambda(self.translation_tool)
        return translation_chain
    
    def orchestrate_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Main orchestration method to handle different types of queries"""
        start_time = time.time()
        
        context_score = 1.0
        if self.vectorstore:
            try:
                docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=1)
                if docs_with_scores:
                    context_score = 1.0 - docs_with_scores[0][1]  # Convert distance to similarity
            except:
                context_score = 1.0
        
        decision_inputs = {
            "query": query,
            "context_score": context_score
        }
        
        decision = self.agent_decision_tool(decision_inputs)
        
        result = {"query": query, "decision": decision, "context_score": context_score}
        
        if decision == "RAG" and self.vectorstore:
            rag_result = self.rag_tool({"query": query})
            result.update(rag_result)
            
        elif decision == "WEB":
            web_result = self.web_search_tool({"query": query, "max_results": kwargs.get("max_results", 3)})
            result.update(web_result)
            
        elif decision == "TRANSLATE":
            if "text" not in kwargs:
                return {"error": "Translation requires 'text' parameter"}
            translation_result = self.translation_tool(kwargs)
            result.update(translation_result)
            
        elif decision == "SUMMARIZE":
            if "text" not in kwargs:
                return {"error": "Summarization requires 'text' parameter"}
            summary_result = self.summarization_tool(kwargs)
            result.update(summary_result)
        
        else:
            web_result = self.web_search_tool({"query": query, "max_results": kwargs.get("max_results", 3)})
            result.update(web_result)
        
        result["total_response_time"] = time.time() - start_time
        return result
