# Day 4 Progress: First Fine-Tuned Model Deployed

## What I did today:
- Pivoted from API-based apps to actual model training
- Set up a complete fine-tuning pipeline
- Created a domain-specific dataset of 80 Q&A pairs
- Fine-tuned Qwen 2.5 0.5B, 1.5B, and 3B models using LoRA
- Compared model performance across sizes
- Improved accuracy by adding more data and increasing epochs
- Uploaded fine-tuned models to HuggingFace
- Built a Streamlit app to serve the fine-tuned model
- Hit deployment limitations with large models on free cloud tier
- Solved deployment issues by using a smaller model + HuggingFace token
- Successfully deployed the fine-tuned model to the public internet

## Live Demo:
https://nimbus-finetune-420yolomcswaggerpants.streamlit.app

## HuggingFace Models:
- 3B: https://huggingface.co/420yolomcswaggerpants/nimbus-coffee-assistant
- 0.5B: https://huggingface.co/420yolomcswaggerpants/nimbus-coffee-assistant-0.5b

## What I learned:
- Fine-tuning is fundamentally different from prompt engineering
- Loss curves show learning progress
- Bigger models learn better from the same data
- More data improves accuracy
- More epochs help with domain-specific memorization
- LoRA trains a small subset of parameters efficiently
- HuggingFace is the standard place to host fine-tuned models
- Free cloud deployment tiers have memory and timeout limits
- Large models (3B+) can be too heavy for free Streamlit Cloud
- Authentication tokens can speed up model downloads

## Results:

### Model: Qwen 2.5 0.5B (20 examples, 5 epochs)
- Loss: 10.93 → 1.32
- Answers were vague and generic

### Model: Qwen 2.5 1.5B (50 examples, 15 epochs)
- Loss: 3.06 → 0.24
- Answers were on-topic but slightly inaccurate

### Model: Qwen 2.5 3B (80 examples, 25 epochs)
- Loss: 4.14 → 0.04
- Answers matched training data exactly

### Model: Qwen 2.5 0.5B (80 examples, 25 epochs)
- Loss: 11.18 → 0.10
- Trained specifically for live deployment
- Successfully deployed to Streamlit Cloud

## Deployment Solution:
- 3B model hosted on HuggingFace as the main showcase
- 0.5B model deployed live on Streamlit Cloud for low-latency inference
- Added HuggingFace token to fix slow model downloads
- This shows both depth (3B quality) and practicality (0.5B deployment)

## Tech Stack:
- Python
- PyTorch
- Transformers
- PEFT (LoRA)
- HuggingFace Hub
- Streamlit
- Git & GitHub

## Key Realizations:

### 1. Training is iterative
I ran four training sessions. Each one taught me something. The progression from vague to exact answers was clear.

### 2. Data quality matters most
The biggest accuracy jump came from adding more specific Q&A pairs, not from model size alone.

### 3. Loss curves tell the story
Watching loss go down from 11.18 to 0.10 was proof the model was actually learning.

### 4. HuggingFace is essential
Hosting models on HuggingFace makes them accessible anywhere. That's the standard workflow.

### 5. CPU training is viable for small models
The 0.5B model trained in about 25 minutes on CPU. The 3B took about 2 hours. Both completely doable.

### 6. Deployment has its own constraints
Training a great model is only half the battle. Serving it requires fitting within platform limits. Smaller models deploy easier.

### 7. Authentication tokens matter
Adding a HuggingFace token fixed the slow download issue that was causing timeouts.

## Next Steps:
- Write READMEs for all projects
- Go deeper on RAG (semantic search, embeddings)
- Learn evaluation metrics
- Add more training data
- Try full fine-tuning vs LoRA
