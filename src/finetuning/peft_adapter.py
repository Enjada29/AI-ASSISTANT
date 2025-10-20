import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch
from dataclasses import dataclass

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        DataCollatorForSeq2Seq
    )
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from accelerate import Accelerator
except ImportError:
    logging.warning("Fine-tuning dependencies not available - PEFT features disabled")

logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    model_name: str = "microsoft/DialoGPT-medium" 
    max_length: int = 512
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None

class PEFTAdapterManager:
    """Manager for PEFT (LoRA) adapters for fine-tuned LLMs"""
    
    def __init__(self, config: FineTuningConfig, save_dir: str = "models/peft_adapters"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.base_model = None
        self.peft_model = None
        self.is_loaded = False
        
    def setup_model(self) -> bool:
        """Setup the base model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if "t5" in self.config.model_name.lower() or "flan" in self.config.model_name.lower():
                self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            else:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            logger.info(f"Loaded model: {self.config.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            return False
    
    def setup_lora_config(self) -> Optional[LoraConfig]:
        """Setup LoRA configuration"""
        try:
            if "t5" in self.config.model_name.lower() or "flan" in self.config.model_name.lower():
                task_type = TaskType.SEQ_2_SEQ_LM
            else:
                task_type = TaskType.CAUSAL_LM
            
            target_modules = self.config.target_modules
            if target_modules is None:
                if "gpt" in self.config.model_name.lower():
                    target_modules = ["c_attn", "c_proj"]
                elif "llama" in self.config.model_name.lower():
                    target_modules = ["q_proj", "v_proj"]
                else:
                    target_modules = ["c_attn"]
            
            lora_config = LoraConfig(
                task_type=task_type,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
            )
            
            return lora_config
            
        except Exception as e:
            logger.error(f"Failed to setup LoRA config: {e}")
            return None
    
    def prepare_training_data(self, training_texts: List[str], max_length: Optional[int] = None) -> Optional[Dataset]:
        """Prepare training data for fine-tuning"""
        try:
            max_len = max_length or self.config.max_length
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=max_len,
                    return_tensors="pt"
                )
            
            dataset_dict = {"text": training_texts}
            dataset = Dataset.from_dict(dataset_dict)
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return None
    
    def fine_tune_model(self, training_texts: List[str], validation_texts: Optional[List[str]] = None) -> bool:
        """Fine-tune the model using PEFT/LoRA"""
        try:
            if not self.setup_model():
                return False
            
            lora_config = self.setup_lora_config()
            if lora_config is None:
                return False
            
            self.peft_model = get_peft_model(self.base_model, lora_config)
            
            train_dataset = self.prepare_training_data(training_texts)
            if train_dataset is None:
                return False
            
            eval_dataset = None
            if validation_texts:
                eval_dataset = self.prepare_training_data(validation_texts)
        
            if "t5" in self.config.model_name.lower() or "flan" in self.config.model_name.lower():
                data_collator = DataCollatorForSeq2Seq(
                    tokenizer=self.tokenizer,
                    model=self.peft_model,
                    padding=True
                )
            else:
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False,
                    pad_to_multiple_of=8
                )
            training_args = TrainingArguments(
                output_dir=str(self.save_dir),
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_train_epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                save_strategy="epoch",
                evaluation_strategy="epoch" if eval_dataset else "no",
                logging_steps=10,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to=None,  
                load_best_model_at_end=True if eval_dataset else False,
            )
            
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            
            logger.info("Starting fine-tuning...")
            trainer.train()
            
            adapter_path = self.save_dir / "final_adapter"
            trainer.save_model(str(adapter_path))
            self.tokenizer.save_pretrained(str(adapter_path))
            
            logger.info(f"Fine-tuning completed. Adapter saved to {adapter_path}")
            return True
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}", exc_info=True)
            return False
    
    def load_adapter(self, adapter_path: str) -> bool:
        """Load a pre-trained adapter"""
        try:
            if not self.setup_model():
                return False
            
            adapter_path = Path(adapter_path)
            if not adapter_path.exists():
                logger.error(f"Adapter path does not exist: {adapter_path}")
                return False
            
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                str(adapter_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            self.is_loaded = True
            logger.info(f"Loaded adapter from {adapter_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            return False
    
    def generate_response(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate response using fine-tuned model"""
        try:
            if not self.is_loaded:
                logger.error("No adapter loaded. Call load_adapter() first.")
                return ""
            
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                self.peft_model = self.peft_model.cuda()
            
            with torch.no_grad():
                outputs = self.peft_model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def evaluate_model(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate the fine-tuned model"""
        try:
            if not self.is_loaded:
                return {"error": "No adapter loaded"}
            
            total_tokens = 0
            valid_responses = 0
            
            for text in test_texts:
                response = self.generate_response(text, max_length=100)
                if response and not response.startswith("Error"):
                    valid_responses += 1
                    total_tokens += len(response.split())
            
            return {
                "valid_responses": valid_responses,
                "total_tests": len(test_texts),
                "success_rate": valid_responses / len(test_texts) if test_texts else 0,
                "avg_response_tokens": total_tokens / valid_responses if valid_responses else 0
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {"error": str(e)}
    
    def create_training_data_from_documents(self, documents: List[str]) -> List[str]:
        """Create training data formatted for conversation fine-tuning"""
        training_texts = []
        
        for doc in documents:
            training_texts.extend([
                f"Context: {doc[:200]}... Question: What is this about? Answer: This is about {doc[:100]}.",
                f"Context: {doc[:200]}... Question: Summarize this. Answer: {doc[:150]}.",
                f"Context: {doc[:200]}... Question: Explain this. Answer: {doc[:150]}."
            ])
        
        return training_texts[:100]  

class FineTuningOrchestrator:
    """Orchestrator for fine-tuning operations"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.adapter_manager = PEFTAdapterManager(config)
    
    def setup_fine_tuning_pipeline(self, training_documents: List[str]) -> bool:
        """Setup the complete fine-tuning pipeline"""
        try:
            training_texts = self.adapter_manager.create_training_data_from_documents(training_documents)
            
            if not training_texts:
                logger.error("No training data generated")
                return False
            
            split_idx = int(len(training_texts) * 0.8)
            train_texts = training_texts[:split_idx]
            val_texts = training_texts[split_idx:]
            
            success = self.adapter_manager.fine_tune_model(train_texts, val_texts)
            
            if success:
                adapter_path = self.adapter_manager.save_dir / "final_adapter"
                self.adapter_manager.load_adapter(str(adapter_path))
            
            return success
            
        except Exception as e:
            logger.error(f"Fine-tuning pipeline failed: {e}", exc_info=True)
            return False
    
    def get_fine_tuned_response(self, query: str) -> Dict[str, Any]:
        """Get response using fine-tuned model as fallback"""
        try:
            if not self.adapter_manager.is_loaded:
                return {
                    "response": "Fine-tuned model not available",
                    "source": "error",
                    "confidence": 0.0
                }
            
            response = self.adapter_manager.generate_response(query)
            
            return {
                "response": response,
                "source": "fine_tuned_model",
                "confidence": 0.8,  
                "model": self.config.model_name
            }
            
        except Exception as e:
            logger.error(f"Fine-tuned response failed: {e}")
            return {
                "response": f"Error: {str(e)}",
                "source": "fine_tuned_error",
                "confidence": 0.0
            }
