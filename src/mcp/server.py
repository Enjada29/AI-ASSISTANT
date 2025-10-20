import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime
import uuid

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    import mcp.types as types
    from mcp.server.stdio import stdio_server
except ImportError:
    logging.warning("MCP library not available - MCP server features disabled")

from src.orchestration.workflow import OrchestrationWorkflow
from src.prompts.registry import prompt_registry

logger = logging.getLogger(__name__)

class MCPAssistantServer:
    def __init__(self, orchestration_workflow: OrchestrationWorkflow):
        self.server = Server("ai-assistant")
        self.workflow = orchestration_workflow
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup MCP server handlers for different tools"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List available tools"""
            return [
                types.Tool(
                    name="query-docs",
                    description="Query uploaded documents using RAG",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The question to ask about the documents"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 3
                            }
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="summarize",
                    description="Summarize a given text",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to summarize"
                            },
                            "summary_length": {
                                "type": "string",
                                "description": "Length of summary (short, medium, long)",
                                "default": "medium"
                            }
                        },
                        "required": ["text"]
                    }
                ),
                types.Tool(
                    name="translate",
                    description="Translate text between languages",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to translate"
                            },
                            "source_language": {
                                "type": "string",
                                "description": "Source language (auto-detect if not specified)",
                                "default": "auto"
                            },
                            "target_language": {
                                "type": "string",
                                "description": "Target language",
                                "default": "English"
                            }
                        },
                        "required": ["text"]
                    }
                ),
                types.Tool(
                    name="web-search",
                    description="Search the web for current information",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 3
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[types.TextContent]:
            """Handle tool calls"""
            try:
                if name == "query-docs":
                    query = arguments.get("query", "")
                    max_results = arguments.get("max_results", 3)
                    
                    result = await self._handle_query_docs(query, max_results)
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "summarize":
                    text = arguments.get("text", "")
                    summary_length = arguments.get("summary_length", "medium")
                    
                    result = await self._handle_summarize(text, summary_length)
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "translate":
                    text = arguments.get("text", "")
                    source_lang = arguments.get("source_language", "auto")
                    target_lang = arguments.get("target_language", "English")
                    
                    result = await self._handle_translate(text, source_lang, target_lang)
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "web-search":
                    query = arguments.get("query", "")
                    max_results = arguments.get("max_results", 3)
                    
                    result = await self._handle_web_search(query, max_results)
                    return [types.TextContent(type="text", text=result)]
                
                else:
                    error_msg = f"Unknown tool: {name}"
                    logger.error(error_msg)
                    return [types.TextContent(type="text", text=error_msg)]
                    
            except Exception as e:
                error_msg = f"Error executing tool {name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return [types.TextContent(type="text", text=error_msg)]
    
    async def _handle_query_docs(self, query: str, max_results: int) -> str:
        """Handle document querying"""
        try:
            result = self.workflow.orchestrate_query(query, max_results=max_results)
            
            if "error" in result:
                return f"Error: {result['error']}"
            
            response = {
                "query": result["query"],
                "source": result["source"],
                "answer": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "response_time": result.get("total_response_time", 0.0)
            }
            
            if result.get("docs_found"):
                response["docs_found"] = result["docs_found"]
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Error in query-docs: {e}", exc_info=True)
            return f"Error querying documents: {str(e)}"
    
    async def _handle_summarize(self, text: str, summary_length: str) -> str:
        """Handle text summarization"""
        try:
            result = self.workflow.summarization_tool({
                "text": text,
                "summary_length": summary_length
            })
            
            response = {
                "summary": result.get("summary", ""),
                "original_length": result.get("original_length", 0),
                "summary_length": result.get("summary_length", 0),
                "compression_ratio": result.get("compression_ratio", 0.0),
                "response_time": result.get("response_time", 0.0)
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Error in summarize: {e}", exc_info=True)
            return f"Error summarizing text: {str(e)}"
    
    async def _handle_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Handle text translation"""
        try:
            result = self.workflow.translation_tool({
                "text": text,
                "source_language": source_lang,
                "target_language": target_lang
            })
            
            response = {
                "translation": result.get("translation", ""),
                "source_language": result.get("source_language", source_lang),
                "target_language": result.get("target_language", target_lang),
                "response_time": result.get("response_time", 0.0)
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Error in translate: {e}", exc_info=True)
            return f"Error translating text: {str(e)}"
    
    async def _handle_web_search(self, query: str, max_results: int) -> str:
        """Handle web search"""
        try:
            result = self.workflow.web_search_tool({
                "query": query,
                "max_results": max_results
            })
            
            response = {
                "query": query,
                "answer": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "search_results": result.get("search_results", []),
                "response_time": result.get("response_time", 0.0)
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Error in web-search: {e}", exc_info=True)
            return f"Error searching web: {str(e)}"
    
    @self.server.list_resources()
    async def handle_list_resources() -> List[types.Resource]:
        """List available resources"""
        return [
            types.Resource(
                uri="ai-assistant://prompts",
                name="Available Prompts",
                description="List of all available prompt templates",
                mimeType="application/json"
            ),
            types.Resource(
                uri="ai-assistant://metrics",
                name="Performance Metrics",
                description="Current performance metrics and statistics",
                mimeType="application/json"
            )
        ]
    
    @self.server.read_resource()
    async def handle_read_resource(uri: str) -> str:
        """Read resource content"""
        if uri == "ai-assistant://prompts":
            prompts = prompt_registry.list_prompts()
            return json.dumps({
                "prompts": [
                    {
                        "name": p["name"],
                        "version": p["version"],
                        "description": p["description"],
                        "variables": p["variables"]
                    }
                    for p in prompts
                ]
            }, indent=2)
        
        elif uri == "ai-assistant://metrics":
            return json.dumps({
                "message": "Metrics resource not yet implemented",
                "uri": uri
            }, indent=2)
        
        else:
            raise ValueError(f"Unknown resource URI: {uri}")
    
    async def run_server(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="ai-assistant",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None,
                    ),
                ),
            )

def create_mcp_server(workflow: OrchestrationWorkflow) -> MCPAssistantServer:
    """Factory function to create MCP server"""
    return MCPAssistantServer(workflow)
