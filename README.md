# Nimbus Coffee Fine-Tuned Assistant

A fine-tuned Qwen 2.5 model that knows everything about Nimbus Coffee Roasters.

## Live Demo
https://nimbus-finetune-420yolomcswaggerpants.streamlit.app

## HuggingFace Models
- 3B: https://huggingface.co/420yolomcswaggerpants/nimbus-coffee-assistant
- 0.5B: https://huggingface.co/420yolomcswaggerpants/nimbus-coffee-assistant-0.5b

## What This Is
A proof of concept demonstrating full fine-tuning of an open-source LLM using LoRA. The model was trained on a custom Q&A dataset and deployed as a working app.

## Training Details

| Metric | Value |
|---|---|
| Base Model | Qwen 2.5 3B |
| Method | LoRA |
| Dataset Size | 80 Q&A pairs |
| Epochs | 25 |
| Initial Loss | 4.14 |
| Final Loss | 0.04 |
| Trainable Params | 1.8M (0.06% of total) |

## Results
The fine-tuned model accurately answers domain-specific questions about products, pricing, shipping, and policies. Before fine-tuning, the base model gave generic or incorrect answers. After training, it matches the training data exactly.

## Model Progression
- **0.5B (20 examples, 5 epochs):** Vague, generic answers
- **1.5B (50 examples, 15 epochs):** On-topic but slightly inaccurate
- **3B (80 examples, 25 epochs):** Exact matches on test questions
- **0.5B (80 examples, 25 epochs):** Deployed live on Streamlit Cloud

## Deployment Strategy
- 3B model hosted on HuggingFace as the main showcase
- 0.5B model deployed live on Streamlit Cloud for low-latency inference
- HuggingFace token used to speed up model downloads

## Tech Stack
- Python
- PyTorch
- Transformers
- PEFT (LoRA)
- HuggingFace Hub
- Streamlit

## How to Run Locally
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Skills Demonstrated
- Model fine-tuning
- LoRA configuration
- Dataset creation
- Training pipeline development
- Model evaluation
- HuggingFace model hosting
- Deployment

## Future Improvements
- Increase dataset size
- Try full fine-tuning
- Add evaluation metrics
- Compare against base model systematically
