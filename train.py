import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# Load the dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# Load tokenizer and model
model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)

# Format the data
def format_example(example):
    text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
    return {"text": text}

dataset = dataset.map(format_example)

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./nimbus-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=25,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer
)

# Train
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./nimbus-finetuned")
tokenizer.save_pretrained("./nimbus-finetuned")

print("Fine-tuning complete! Model saved to ./nimbus-finetuned")