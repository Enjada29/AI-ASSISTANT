import asyncio
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.mcp.server import MCPAssistantServer
from src.orchestration.workflow import OrchestrationWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run MCP server"""
    try:
        workflow = OrchestrationWorkflow(openai_api_key="", vectorstore=None)
        
        mcp_server = MCPAssistantServer(workflow)
        
        logger.info("Starting MCP server...")
        await mcp_server.run_server()
        
    except Exception as e:
        logger.error(f"MCP server failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
