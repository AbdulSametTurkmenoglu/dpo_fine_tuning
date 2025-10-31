# DPO (Direct Preference Optimization)

A project for fine-tuning language models using Direct Preference Optimization with Hugging Face `trl`, `peft`, and `transformers` libraries. DPO is a simpler and more stable alternative to RLHF - it directly optimizes models using chosen/rejected response pairs instead of training a separate reward model.

##  Features

- **DPO Training**: Efficient training with `trl` library's `DPOTrainer` class
- **QLoRA Integration**: Memory-efficient 4-bit quantization for GPU training
- **Modular Architecture**: Clean separation of data processing, model initialization, and training logic
- **Automatic Model Management**: Handles both policy (trainable) and reference (frozen) models automatically
- **Flexible CLI**: Easy parameter configuration via `argparse`
- **Interactive Inference**: Test your trained model with a simple chat interface

##  Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for QLoRA)
- 8GB+ RAM (16GB+ recommended)

##  Installation
```bash
# Clone the repository
git clone https://github.com/AbdulSametTurkmenoglu/dpo.git
cd dpo

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

##  Usage

### Training

**Basic training with default settings (TinyLlama + UltraFeedback dataset):**
```bash
python train.py
```

**Customized training parameters:**
```bash
python train.py --num_samples 1000 --epochs 2 --beta 0.15 --output_dir dpo_model_v2
```

**Training on CPU/MPS (without quantization):**
```bash
python train.py --no_quantization
```

**Available training arguments:**
```bash
python train.py --help
```

Key parameters:
- `--model_name`: Base model to fine-tune (default: TinyLlama-1.1B-Chat-v1.0)
- `--dataset_name`: HuggingFace dataset (default: argilla/ultrafeedback-binarized-preferences-cleaned)
- `--num_samples`: Number of training samples (default: 500)
- `--epochs`: Training epochs (default: 1)
- `--beta`: DPO beta parameter for preference strength (default: 0.1)
- `--output_dir`: Directory to save the trained model (default: dpo_tinyllama_model)

### Inference

**Interactive chat with trained model:**
```bash
python inference.py
```

**Custom model path:**
```bash
python inference.py --base_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --adapter_path "dpo_model_v2"
```

The inference script loads the base model and applies your trained LoRA adapter on top of it.


##  How It Works

1. **Data Loading**: Loads preference pairs (chosen/rejected responses) from HuggingFace datasets
2. **Model Setup**: Initializes base model with optional QLoRA quantization
3. **DPO Training**: Trains the model to prefer chosen responses over rejected ones
4. **Adapter Saving**: Saves only the trained LoRA adapters (memory efficient)
5. **Inference**: Loads base model + adapters for text generation


