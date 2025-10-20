# Fine-Tuning Implementation Notes

## Overview

This document outlines the fine-tuning implementation for the AI Assistant Platform, which includes Parameter-Efficient Fine-Tuning (PEFT) using LoRA adapters on smaller language models.

## Architecture

The fine-tuning system is implemented in `src/finetuning/peft_adapter.py` and provides:

- **LoRA Adapters**: Parameter-efficient fine-tuning using Low-Rank Adaptation
- **Model Support**: DialoGPT-medium (default), LLaMA-2, FLAN-T5
- **Integration**: Optional backend integration with the main RAG system

## Configuration

### FineTuningConfig

The system uses a dataclass configuration with the following parameters:

```python
@dataclass
class FineTuningConfig:
    model_name: str = "microsoft/DialoGPT-medium"  # Base model
    max_length: int = 512                          # Maximum sequence length
    learning_rate: float = 2e-4                    # Learning rate
    num_epochs: int = 3                           # Training epochs
    batch_size: int = 4                           # Batch size
    gradient_accumulation_steps: int = 4          # Gradient accumulation
    lora_rank: int = 16                           # LoRA rank
    lora_alpha: int = 32                          # LoRA alpha
    lora_dropout: float = 0.1                     # LoRA dropout
    target_modules: Optional[List[str]] = None    # Target modules for LoRA
```

### Model Selection

Supported models for fine-tuning:
- **DialoGPT-medium**: Default lightweight conversational model
- **LLaMA-2**: More powerful instruction-following model (if available)
- **FLAN-T5**: Google's instruction-tuned model for task-specific fine-tuning

## Implementation Details

### PEFTAdapterManager

The main class handles:
- Model setup and tokenizer initialization
- LoRA adapter configuration
- Training pipeline setup
- Model inference and response generation

### Key Methods

1. **setup_model()**: Initializes base model and tokenizer
2. **setup_fine_tuning_pipeline()**: Prepares training data and LoRA configuration
3. **train_model()**: Executes the fine-tuning process
4. **generate_response()**: Inference with fine-tuned model

### LoRA Configuration

The system uses the following LoRA settings by default:
- Rank: 16 (balances efficiency and performance)
- Alpha: 32 (scaling factor)
- Dropout: 0.1 (regularization)
- Target modules: Automatically detected based on model architecture

## API Endpoints

### Setup Fine-tuning
```
POST /fine-tuning/setup
```
Initializes the fine-tuning pipeline using uploaded documents.

### Check Status
```
GET /fine-tuning/status
```
Returns whether the model has been trained and is available.

### Query Fine-tuned Model
```
POST /fine-tuning/query
Body: query: string
```
Uses the fine-tuned model to answer queries.

## Training Process

### Data Preparation

1. **Document Processing**: Uses existing uploaded documents from the RAG system
2. **Text Splitting**: Applies the same chunking strategy as the main system
3. **Format Conversion**: Converts chunks into training format for the specific model

### Training Pipeline

1. **Model Loading**: Loads the base model and tokenizer
2. **LoRA Setup**: Configures PEFT adapters
3. **Training Arguments**: Sets up training parameters
4. **Data Collation**: Prepares data for training
5. **Fine-tuning**: Executes the training process
6. **Model Persistence**: Saves the fine-tuned adapter

### Training Parameters

- Learning Rate: 2e-4 (conservative for stability)
- Epochs: 3 (prevents overfitting)
- Batch Size: 4 (memory-efficient)
- Gradient Accumulation: 4 (simulates larger batch size)
- Maximum Length: 512 tokens

## Integration with Main System

### Workflow Integration

The fine-tuned model can be used as an alternative backend:
- Standard queries use the main RAG + OpenAI pipeline
- Fine-tuned model provides domain-specific responses
- Supports cost reduction and improved performance for specific tasks

### Performance Monitoring

The system tracks:
- Training progress and loss
- Inference latency
- Response quality metrics
- Model availability status

## Hardware Requirements

### Minimum Requirements
- 8GB RAM
- CPU-only inference (with slower training)

### Recommended Requirements
- 16GB+ RAM
- GPU with 8GB+ VRAM (for faster training)
- CUDA-compatible GPU for optimal performance

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use gradient accumulation
2. **Training Failures**: Check document availability and format
3. **Model Loading**: Ensure all dependencies are installed

### Dependency Requirements

Key dependencies for fine-tuning:
```
transformers>=4.30.0
torch>=2.0.0
datasets>=2.14.0
peft>=0.6.0
accelerate>=0.20.0
bitsandbytes>=0.41.0
```

## Future Enhancements

### Planned Improvements
- Multiple adapter support for different tasks
- Automated hyperparameter tuning
- Model evaluation and comparison tools
- Support for additional model architectures

### Performance Optimizations
- Quantization support for smaller model sizes
- Faster inference with optimization techniques
- Batch inference capabilities

## Best Practices

### Training Guidelines
1. Use domain-specific documents for better performance
2. Monitor training loss to prevent overfitting
3. Validate on held-out data
4. Start with smaller models for initial testing

### Deployment Considerations
1. Test inference latency in production environment
2. Implement fallback to base models if needed
3. Monitor memory usage during inference
4. Consider model versioning for updates

## Examples

### Basic Usage

```python
from src.finetuning.peft_adapter import PEFTAdapterManager, FineTuningConfig

# Initialize configuration
config = FineTuningConfig(
    model_name="microsoft/DialoGPT-medium",
    num_epochs=3,
    learning_rate=2e-4
)

# Create manager
manager = PEFTAdapterManager(config)

# Setup and train
success = manager.setup_fine_tuning_pipeline(documents)
if success:
    manager.train_model()
    
# Query the fine-tuned model
response = manager.generate_response("What is artificial intelligence?")
```

### API Usage

```bash
# Setup fine-tuning
curl -X POST http://localhost:8000/fine-tuning/setup

# Check status
curl http://localhost:8000/fine-tuning/status

# Query fine-tuned model
curl -X POST http://localhost:8000/fine-tuning/query \
  -F "query=What is machine learning?"
```
