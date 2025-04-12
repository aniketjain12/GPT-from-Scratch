# GPT-from-Scratch

## Overview

In this project, we reproduce the GPT-2 model (124M parameters) from scratch using PyTorch. The process includes building the GPT-2 architecture, implementing an efficient training loop with mixed precision, FlashAttention, and torch.compile, and setting up training using hyperparameters inspired by the original GPT-2 and GPT-3 papers. Distributed Data Parallel (DDP) enables multi-GPU training, allowing for faster convergence with large batch sizes. We conclude by sampling generations from the trained model. This project serves as a deep dive into the core mechanics of transformer-based language models and replicates around 90% of nanoGPT’s capabilities.

## Project Introduction

This project is a full from-scratch reproduction of the GPT-2 model with 124 million parameters using PyTorch. The goal is not just to implement the architecture, but to deeply understand how a large language model (LLM) like GPT-2 is built, trained, and optimized for real-world performance. We begin by constructing the transformer-based GPT-2 neural network, implementing all core components such as multi-head self-attention, layer normalization, residual connections, positional embeddings, and causal masking.

After building the model, we focus on optimizing the training loop to make it as efficient as possible. The project leverages modern acceleration techniques including mixed precision training, FlashAttention for optimized attention mechanisms, and torch.compile() to fuse computation and minimize Python overhead. To scale the model training across multiple GPUs, we use PyTorch's DistributedDataParallel (DDP), which allows for parallelism in data processing and gradient synchronization across devices—enabling large batch sizes and faster convergence.

The training run is designed to mirror the setup described in the original GPT-2 and GPT-3 papers, including hyperparameter tuning strategies such as AdamW optimizer, learning rate scheduling (warmup + cosine decay), gradient clipping, and weight decay. We also employ gradient accumulation to handle larger effective batch sizes even on limited hardware.

After training, the model can generate samples from text prompts, showcasing its ability to "dream up" coherent sequences learned from the dataset. The results can be both insightful and entertaining, demonstrating the model's language understanding capabilities. This implementation reaches approximately 90% similarity with the well-known nanoGPT repository in design and performance—but is fully built step by step for educational clarity.

This project is perfect for those who want to go beyond using pretrained models and truly understand how LLMs work under the hood—from architecture to training optimization. Whether you're a machine learning enthusiast, a deep learning practitioner, or someone curious about building GPT models from scratch, this project offers a deep dive into every stage of that journey.

**Supplementary Research Papers:**

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [OpenAI GPT-3 paper](https://arxiv.org/abs/2005.14165)
- [OpenAI GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Flash Attention](https://arxiv.org/abs/2205.14135)


## Project Steps Overview

The project is structured around the following key steps:

1. **Model Implementation**  
   - **CasualSelfAttention:** Implements causal self-attention with two modes: standard attention (commented out) and efficient Flash Attention using `F.scaled_dot_product_attention`.
   - **MLP & Block:** Defines the feedforward neural network (MLP) and builds transformer blocks that combine layer normalization, self-attention, and the MLP with residual connections.
   - **GPT Model:** Stacks transformer blocks, applies token and positional embeddings, and shares weights between the input embedding and the language modeling head. Includes custom weight initialization based on the configuration.

2. **Data Preparation and Loading**  
   - **DataLoaderLite:** A lightweight data loader that:
     - Reads tokenized data shards from a specified directory.
     - Implements sharding logic for distributed training.
     - Provides batches of input (x) and target (y) tokens for training.

3. **Training Infrastructure**  
   - **Device Setup & DDP:**  
     - Auto-detects available devices (CPU, CUDA, or MPS) and sets up Distributed Data Parallel (DDP) if running in a multi-GPU environment.
     - Ensures synchronized gradient updates using gradient accumulation steps.
   - **Optimization and Learning Rate Scheduling:**  
     - The optimizer is configured with weight decay (AdamW, fused if available).
     - Implements a learning rate schedule with linear warmup followed by cosine decay.
   - **Training Loop:**  
     - Iterates over training batches with gradient accumulation.
     - Regularly evaluates on validation data, saves checkpoints, and logs performance.
     - Includes gradient clipping and mixed precision training with `torch.autocast`.

4. **Evaluation and Generation Utilities**  
   - **Validation and Checkpointing:**  
     - Periodically computes average validation loss over several batches.
     - Saves model checkpoints, including the state dict and training configuration.
   - **Helloswag Evaluation:**  
     - Evaluates model performance on a HellaSwag-inspired task using a helper function to find the most likely completion.
   - **Text Generation:**  
     - Uses top-k sampling to generate text sequences from a prompt.
     - Efficiently expands sequences while using autoregressive inference to produce readable model outputs.

This overview encapsulates the sequential steps taken in the project: from model design and data handling to training optimization and inference, providing a clear pathway through the code base.

## Setup & Training

Requires Python ≥3.10, PyTorch ≥2.1, and a CUDA-enabled GPU (optional but recommended)
```bash
# Clone the repo
git clone https://github.com/aniketjain12/GPT-from-Scratch.git
cd GPT-from-Scratch

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install Data Sets
python finewine.py
python helloswag.py

# Start training
python gpt_model.py
```
