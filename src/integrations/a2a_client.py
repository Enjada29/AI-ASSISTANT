import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ExternalServiceConfig:
    name: str
    base_url: str
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30

class SentimentAnalysisService:
    """Integration with external sentiment analysis service"""
    
    def __init__(self, config: ExternalServiceConfig):
        self.config = config
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.config.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        if not self.session:
            raise RuntimeError("Service context manager not used")
        
        try:
            payload = {"text": text}
            
            async with self.session.post(
                f"{self.config.base_url}/sentiment",
                json=payload,
                headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "sentiment": result.get("sentiment", "unknown"),
                        "confidence": result.get("confidence", 0.0),
                        "scores": result.get("scores", {}),
                        "source": f"{self.config.name}_sentiment"
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Sentiment API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._fallback_sentiment_analysis(text)
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Simple fallback sentiment analysis using keywords"""
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "horrible", "hate", "dislike", "worst", "disappointed"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": {"positive": positive_count, "negative": negative_count},
            "source": "fallback_sentiment"
        }

class TranslationService:
    """Integration with external translation service"""
    
    def __init__(self, config: ExternalServiceConfig):
        self.config = config
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.config.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Translate text using external service"""
        if not self.session:
            raise RuntimeError("Service context manager not used")
        
        try:
            payload = {
                "text": text,
                "source_language": source_lang,
                "target_language": target_lang
            }
            
            async with self.session.post(
                f"{self.config.base_url}/translate",
                json=payload,
                headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "translation": result.get("translation", text),
                        "source_language": result.get("source_language", source_lang),
                        "target_language": result.get("target_language", target_lang),
                        "confidence": result.get("confidence", 1.0),
                        "source": f"{self.config.name}_translation"
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Translation API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                "translation": text,  
                "source_language": source_lang,
                "target_language": target_lang,
                "confidence": 0.0,
                "source": f"{self.config.name}_error",
                "error": str(e)
            }

class KnowledgeBaseService:
    """Integration with external knowledge base or Q&A service"""
    
    def __init__(self, config: ExternalServiceConfig):
        self.config = config
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.config.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def query_knowledge_base(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Query external knowledge base"""
        if not self.session:
            raise RuntimeError("Service context manager not used")
        
        try:
            payload = {"query": query}
            if context:
                payload["context"] = context
            
            async with self.session.post(
                f"{self.config.base_url}/query",
                json=payload,
                headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "answer": result.get("answer", ""),
                        "confidence": result.get("confidence", 0.0),
                        "sources": result.get("sources", []),
                        "query": query,
                        "source": f"{self.config.name}_kb"
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Knowledge base API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Knowledge base query failed: {e}")
            return {
                "answer": "Unable to query external knowledge base",
                "confidence": 0.0,
                "sources": [],
                "query": query,
                "source": f"{self.config.name}_error",
                "error": str(e)
            }

class A2AOrchestrator:
    """Orchestrator for API-to-API integrations"""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.default_configs = {
            "sentiment": ExternalServiceConfig(
                name="sentiment_service",
                base_url="https://api.huggingface.co/inference/default/",
                api_key=None  
            ),
            "translation": ExternalServiceConfig(
                name="translation_service", 
                base_url="https://api.deepl.com/v2",
                api_key=None  
            ),
            "knowledge": ExternalServiceConfig(
                name="knowledge_service",
                base_url="https://api.wikipedia.org/w/api.php",
                api_key=None
            )
        }
    
    def register_service(self, name: str, service: Any):
        """Register an external service"""
        self.services[name] = service
    
    async def enhance_query_with_sentiment(self, query: str, response: str) -> Dict[str, Any]:
        """Enhance query response with sentiment analysis"""
        try:
            async with SentimentAnalysisService(self.default_configs["sentiment"]) as sentiment_service:
                sentiment_result = await sentiment_service.analyze_sentiment(response)
                
                return {
                    "original_response": response,
                    "sentiment_analysis": sentiment_result,
                    "enhanced_response": f"{response}\n\n[Sentiment: {sentiment_result['sentiment']} (confidence: {sentiment_result['confidence']:.2f})]"
                }
        except Exception as e:
            logger.error(f"Sentiment enhancement failed: {e}")
            return {"original_response": response, "error": str(e)}
    
    async def translate_response(self, response: str, target_language: str = "Spanish") -> Dict[str, Any]:
        """Translate response using external service"""
        try:
            async with TranslationService(self.default_configs["translation"]) as translation_service:
                translation_result = await translation_service.translate_text(
                    response, "auto", target_language
                )
                
                return {
                    "original_response": response,
                    "translation": translation_result,
                    "enhanced_response": f"{response}\n\n[Translated to {target_language}: {translation_result['translation']}]"
                }
        except Exception as e:
            logger.error(f"Translation enhancement failed: {e}")
            return {"original_response": response, "error": str(e)}
    
    async def enhance_with_external_knowledge(self, query: str, local_response: str) -> Dict[str, Any]:
        """Enhance response with external knowledge base"""
        try:
            async with KnowledgeBaseService(self.default_configs["knowledge"]) as kb_service:
                kb_result = await kb_service.query_knowledge_base(query, local_response)
                
                if kb_result.get("confidence", 0.0) > 0.7:
                    return {
                        "local_response": local_response,
                        "external_knowledge": kb_result,
                        "enhanced_response": f"{local_response}\n\n[Additional context: {kb_result['answer']}]"
                    }
                else:
                    return {
                        "local_response": local_response,
                        "external_knowledge": kb_result,
                        "enhanced_response": local_response
                    }
        except Exception as e:
            logger.error(f"External knowledge enhancement failed: {e}")
            return {"local_response": local_response, "error": str(e)}
    
    async def multi_service_enhancement(self, query: str, response: str, enhancements: List[str]) -> Dict[str, Any]:
        """Apply multiple service enhancements"""
        result = {"original_response": response, "enhancements": {}}
        
        for enhancement in enhancements:
            try:
                if enhancement == "sentiment":
                    sentiment_result = await self.enhance_query_with_sentiment(query, response)
                    result["enhancements"]["sentiment"] = sentiment_result["sentiment_analysis"]
                
                elif enhancement == "translate":
                    translation_result = await self.translate_response(response)
                    result["enhancements"]["translation"] = translation_result["translation"]
                
                elif enhancement == "knowledge":
                    knowledge_result = await self.enhance_with_external_knowledge(query, response)
                    result["enhancements"]["knowledge"] = knowledge_result["external_knowledge"]
                    
            except Exception as e:
                logger.error(f"Enhancement {enhancement} failed: {e}")
                result["enhancements"][enhancement] = {"error": str(e)}
        
        return result
