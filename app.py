import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

@st.cache_resource
def load_model():
    base_model_name = "Qwen/Qwen2.5-3B"
    adapter_name = "YOUR_USERNAME/nimbus-coffee-assistant"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("☕ Nimbus Coffee Assistant")
st.caption("A fine-tuned AI that knows everything about Nimbus Coffee Roasters.")

# FAQ Quick-Reply Buttons
st.markdown("**Quick Questions:**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    faq1 = st.button("☕ What do you sell?")
with col2:
    faq2 = st.button("🚚 Shipping cost?")
with col3:
    faq3 = st.button("☕ Do you sell decaf?")
with col4:
    faq4 = st.button("🕒 Cafe hours?")

user_input = st.text_input("Or ask your own question:")

if faq1:
    user_input = "What coffees do you sell?"
elif faq2:
    user_input = "How much is shipping?"
elif faq3:
    user_input = "Do you sell decaf?"
elif faq4:
    user_input = "What are your cafe hours?"

if user_input:
    prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        temperature=None,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in answer:
        answer = answer.split("### Response:")[1].strip()
    st.text(answer)