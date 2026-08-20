import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/nimbus-coffee-assistant")
    model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/nimbus-coffee-assistant")
    return tokenizer, model

tokenizer, model = load_model()

st.title("☕ Nimbus Coffee Assistant")
st.caption("A fine-tuned AI that knows everything about Nimbus Coffee Roasters.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# Determine what the user asked
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
    
    # Add to chat history
    st.session_state.chat_history.append({"user": user_input, "bot": answer})

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["user"])
    with st.chat_message("assistant"):
        st.text(chat["bot"])