import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataStabilityTestPreset
    from evidently.metrics import *
    from evidently.model_profile import Profile
    from evidently.model_profile.sections import DataProfileSection
except ImportError:
    logging.warning("Evidently AI not available - monitoring features will be limited")

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
except ImportError:
    logging.warning("SQLAlchemy not available - database features will be limited")

import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter('ai_assistant_requests_total', 'Total requests', ['source', 'status'])
REQUEST_DURATION = Histogram('ai_assistant_request_duration_seconds', 'Request duration', ['source'])
TOKEN_USAGE = Counter('ai_assistant_tokens_used_total', 'Total tokens used', ['model'])
RETRIEVAL_ACCURACY = Histogram('ai_assistant_retrieval_accuracy', 'Retrieval accuracy scores')

@dataclass
class QueryMetrics:
    query_id: str
    timestamp: datetime
    query_text: str
    source: str  
    response_time: float
    tokens_used: int
    retrieval_confidence: Optional[float]
    user_rating: Optional[int]
    error_occurred: bool = False
    context_score: Optional[float] = None

Base = declarative_base()

class QueryMetric(Base):
    __tablename__ = 'query_metrics'
    
    id = Column(Integer, primary_key=True)
    query_id = Column(String(255), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    query_text = Column(Text, nullable=False)
    source = Column(String(50), nullable=False)
    response_time = Column(Float, nullable=False)
    tokens_used = Column(Integer, nullable=False, default=0)
    retrieval_confidence = Column(Float)
    user_rating = Column(Integer)
    error_occurred = Column(Boolean, default=False)
    context_score = Column(Float)

class MonitoringService:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or "sqlite:///./data/ai_assistant.db"
        self.session_factory = None
        self.metrics_buffer: List[QueryMetrics] = []
        self.buffer_size = 100
        
        self._setup_database()
        self._start_prometheus_server()
    
    def _setup_database(self):
        """Setup database connection and create tables"""
        try:
            engine = create_engine(self.database_url, echo=False)
            Base.metadata.create_all(bind=engine)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            self.session_factory = SessionLocal
            logger.info(f"Database setup successful: {self.database_url}")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            
    def _start_prometheus_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(8001)
            logger.info("Prometheus metrics server started on port 8001")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")
    
    def record_query_metric(self, metrics: QueryMetrics):
        """Record a query metric"""
        self.metrics_buffer.append(metrics)
        
        REQUEST_COUNT.labels(source=metrics.source, status='success' if not metrics.error_occurred else 'error').inc()
        REQUEST_DURATION.labels(source=metrics.source).observe(metrics.response_time)
        TOKEN_USAGE.labels(model='gpt-4o-mini').inc(metrics.tokens_used)
        
        if metrics.retrieval_confidence is not None:
            RETRIEVAL_ACCURACY.observe(metrics.retrieval_confidence)
        
        if len(self.metrics_buffer) >= self.buffer_size:
            self._flush_metrics_buffer()
    
    def _flush_metrics_buffer(self):
        """Flush metrics buffer to database"""
        if not self.session_factory or not self.metrics_buffer:
            return
            
        try:
            session = self.session_factory()
            for metric in self.metrics_buffer:
                db_metric = QueryMetric(
                    query_id=metric.query_id,
                    timestamp=metric.timestamp,
                    query_text=metric.query_text,
                    source=metric.source,
                    response_time=metric.response_time,
                    tokens_used=metric.tokens_used,
                    retrieval_confidence=metric.retrieval_confidence,
                    user_rating=metric.user_rating,
                    error_occurred=metric.error_occurred,
                    context_score=metric.context_score
                )
                session.add(db_metric)
            
            session.commit()
            logger.info(f"Flushed {len(self.metrics_buffer)} metrics to database")
            self.metrics_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush metrics to database: {e}")
        finally:
            if session:
                session.close()
    
    def get_performance_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get performance statistics for the last N days"""
        if not self.session_factory:
            return {"error": "Database not available"}
        
        try:
            session = self.session_factory()
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            metrics = session.query(QueryMetric).filter(
                QueryMetric.timestamp >= cutoff_date
            ).all()
            
            if not metrics:
                return {"error": "No metrics found for the specified period"}
            
            data = []
            for metric in metrics:
                data.append({
                    'timestamp': metric.timestamp,
                    'source': metric.source,
                    'response_time': metric.response_time,
                    'tokens_used': metric.tokens_used,
                    'retrieval_confidence': metric.retrieval_confidence,
                    'user_rating': metric.user_rating,
                    'error_occurred': metric.error_occurred,
                    'context_score': metric.context_score
                })
            
            df = pd.DataFrame(data)
            
            
            stats = {
                "total_queries": len(df),
                "avg_response_time": df['response_time'].mean(),
                "total_tokens": df['tokens_used'].sum(),
                "avg_tokens_per_query": df['tokens_used'].mean(),
                "error_rate": df['error_occurred'].mean(),
                "avg_user_rating": df['user_rating'].dropna().mean() if not df['user_rating'].isna().all() else None,
                "avg_retrieval_confidence": df['retrieval_confidence'].dropna().mean(),
                "queries_by_source": df['source'].value_counts().to_dict()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get performance statistics: {e}")
            return {"error": str(e)}
        finally:
            if session:
                session.close()
    
    def generate_evidently_report(self, days: int = 7) -> Optional[Report]:
        """Generate Evidently AI report for data drift and quality"""
        try:
            if not self.session_factory:
                return None
            
            session = self.session_factory()
            cutoff_date = datetime.utcnow() - timedelta(days=days)
           
            recent_metrics = session.query(QueryMetric).filter(
                QueryMetric.timestamp >= cutoff_date
            ).all()
            
            historical_cutoff = cutoff_date - timedelta(days=days)
            historical_metrics = session.query(QueryMetric).filter(
                QueryMetric.timestamp >= historical_cutoff,
                QueryMetric.timestamp < cutoff_date
            ).all()
            
            if not recent_metrics or not historical_metrics:
                logger.warning("Insufficient data for Evidently report")
                return None
            
            recent_data = []
            for metric in recent_metrics:
                recent_data.append({
                    'response_time': metric.response_time,
                    'tokens_used': metric.tokens_used,
                    'retrieval_confidence': metric.retrieval_confidence or 0,
                    'context_score': metric.context_score or 0,
                    'error_occurred': int(metric.error_occurred)
                })
            
            historical_data = []
            for metric in historical_metrics:
                historical_data.append({
                    'response_time': metric.response_time,
                    'tokens_used': metric.tokens_used,
                    'retrieval_confidence': metric.retrieval_confidence or 0,
                    'context_score': metric.context_score or 0,
                    'error_occurred': int(metric.error_occurred)
                })
            
            recent_df = pd.DataFrame(recent_data)
            historical_df = pd.DataFrame(historical_data)
            
            column_mapping = ColumnMapping(
                numerical_features=['response_time', 'tokens_used', 'retrieval_confidence', 'context_score'],
                categorical_features=['error_occurred']
            )
            
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset()
            ])
            
            report.run(
                reference_data=historical_df,
                current_data=recent_df,
                column_mapping=column_mapping
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate Evidently report: {e}")
            return None
        finally:
            if session:
                session.close()
    
    def create_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Create data for monitoring dashboard"""
        stats = self.get_performance_statistics(days=7)
        
        if "error" in stats:
            return stats
        
        if self.session_factory:
            try:
                session = self.session_factory()
                cutoff_24h = datetime.utcnow() - timedelta(hours=24)
                
                metrics_24h = session.query(QueryMetric).filter(
                    QueryMetric.timestamp >= cutoff_24h
                ).all()

                hourly_data = {}
                for metric in metrics_24h:
                    hour = metric.timestamp.replace(minute=0, second=0, microsecond=0)
                    if hour not in hourly_data:
                        hourly_data[hour] = {
                            'count': 0,
                            'avg_response_time': 0,
                            'total_tokens': 0,
                            'errors': 0
                        }
                    
                    hourly_data[hour]['count'] += 1
                    hourly_data[hour]['avg_response_time'] += metric.response_time
                    hourly_data[hour]['total_tokens'] += metric.tokens_used
                    hourly_data[hour]['errors'] += int(metric.error_occurred)
                
            
                for hour_data in hourly_data.values():
                    if hour_data['count'] > 0:
                        hour_data['avg_response_time'] /= hour_data['count']
                
                stats['hourly_breakdown'] = {
                    str(hour): data for hour, data in sorted(hourly_data.items())
                }
                
            except Exception as e:
                logger.error(f"Failed to get hourly breakdown: {e}")
        
        return stats
    
    def cleanup_old_metrics(self, days_to_keep: int = 90):
        """Clean up old metrics to prevent database bloat"""
        if not self.session_factory:
            return
        
        try:
            session = self.session_factory()
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            deleted_count = session.query(QueryMetric).filter(
                QueryMetric.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            logger.info(f"Cleaned up {deleted_count} old metrics")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
        finally:
            if session:
                session.close()
