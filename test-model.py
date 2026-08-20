from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./nimbus-finetuned")
model = AutoModelForCausalLM.from_pretrained("./nimbus-finetuned")

def ask(question):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=None
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in answer:
        answer = answer.split("### Response:")[1].strip()
    return answer

questions = [
    "What coffees do you sell?",
    "How much is shipping?",
    "Do you sell decaf?",
    "What are your cafe hours?"
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {ask(q)}")
    print("-" * 40)