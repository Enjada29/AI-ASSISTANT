import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)

@dataclass
class PromptMetric:
    prompt_name: str
    version: str
    timestamp: datetime
    accuracy: float
    tokens_used: int
    response_time: float
    user_rating: Optional[int] = None

class PromptRegistry:
    def __init__(self, registry_path: str = "prompts/registry.yaml"):
        self.registry_path = Path(registry_path)
        self.prompts: Dict[str, Dict] = {}
        self.metrics: List[PromptMetric] = []
        self.load_registry()
        
    def load_registry(self):
        """Load prompt registry from YAML file"""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Prompt registry not found at {self.registry_path}")
            
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        self.prompts = {prompt["name"]: prompt for prompt in data["prompts"]}
        logger.info(f"Loaded {len(self.prompts)} prompts from registry")
        
    def get_prompt(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get prompt by name and optional version"""
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found in registry")
            
        prompt_data = self.prompts[name]
        
        if version and prompt_data["version"] != version:
            raise ValueError(f"Version {version} not found for prompt '{name}'")
            
        return prompt_data
    
    def render_prompt(self, name: str, variables: Dict[str, Any], version: Optional[str] = None) -> str:
        """Render prompt template with given variables"""
        prompt_data = self.get_prompt(name, version)
        template = Template(prompt_data["template"])
        
        required_vars = set(prompt_data["variables"])
        provided_vars = set(variables.keys())
        
        if not required_vars.issubset(provided_vars):
            missing = required_vars - provided_vars
            raise ValueError(f"Missing variables for prompt '{name}': {missing}")
            
        return template.render(**variables)
    
    def record_metric(self, metric: PromptMetric):
        """Record performance metric for a prompt"""
        self.metrics.append(metric)
        self._save_metrics()
        
    def _save_metrics(self):
        """Save metrics to JSON file"""
        metrics_path = Path("data/prompt_metrics.json")
        metrics_path.parent.mkdir(exist_ok=True)
        
        metrics_data = []
        for metric in self.metrics:
            metric_dict = asdict(metric)
            metric_dict["timestamp"] = metric.timestamp.isoformat()
            metrics_data.append(metric_dict)
            
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def load_metrics(self):
        """Load metrics from JSON file"""
        metrics_path = Path("data/prompt_metrics.json")
        if not metrics_path.exists():
            return
            
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
            
        self.metrics = []
        for metric_dict in metrics_data:
            metric_dict["timestamp"] = datetime.fromisoformat(metric_dict["timestamp"])
            self.metrics.append(PromptMetric(**metric_dict))
    
    def get_performance_stats(self, prompt_name: str, days: int = 30) -> Dict[str, float]:
        """Get performance statistics for a prompt"""
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
        
        prompt_metrics = [
            m for m in self.metrics 
            if m.prompt_name == prompt_name and m.timestamp >= cutoff_date
        ]
        
        if not prompt_metrics:
            return {"accuracy": 0.0, "avg_tokens": 0.0, "avg_response_time": 0.0}
            
        return {
            "accuracy": sum(m.accuracy for m in prompt_metrics) / len(prompt_metrics),
            "avg_tokens": sum(m.tokens_used for m in prompt_metrics) / len(prompt_metrics),
            "avg_response_time": sum(m.response_time for m in prompt_metrics) / len(prompt_metrics),
            "total_requests": len(prompt_metrics)
        }
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompts"""
        return list(self.prompts.values())
    
    def add_prompt(self, prompt_data: Dict[str, Any]):
        """Add new prompt to registry"""
        required_fields = ["name", "version", "description", "template", "variables"]
        for field in required_fields:
            if field not in prompt_data:
                raise ValueError(f"Missing required field: {field}")
        
        self.prompts[prompt_data["name"]] = prompt_data
        self.save_registry()
        
    def save_registry(self):
        """Save current registry to YAML file"""
        registry_data = {
            "version": "1.0",
            "prompts": list(self.prompts.values())
        }
        
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(registry_data, f, default_flow_style=False, allow_unicode=True)

prompt_registry = PromptRegistry()
