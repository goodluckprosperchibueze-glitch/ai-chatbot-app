Import streamlit as st
from transformers import pipeline
import sys
import re

# --- ENCODING FIX ---
sys.stdout.reconfigure(encoding='utf-8')

# --- 1. LOAD THE MODEL ---
@st.cache_resource
def load_chatbot_model():
    """Loads a smart text2text model for cleaner English responses."""
    # Using a smarter, cleaner, and smaller model (FLAN-T5)
    return pipeline("text2text-generation", model="google/flan-t5-small")

chatbot = load_chatbot_model()

# --- 2. STREAMLIT UI SETUP ---
st.set_page_config(page_title="Specimen AI 🤖", layout="centered")
st.title("👑 Specimen Chatbot AI")
st.caption("Built by Specimen King using Streamlit + Hugging Face")

# --- 3. CHAT HISTORY SETUP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. MAIN CHAT LOGIC ---
if prompt := st.chat_input("Say something to Specimen AI..."):
    # Save user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Specimen AI is thinking... 💭"):
            raw_response = chatbot(prompt, max_length=150, temperature=0.7)
            ai_response = raw_response[0]['generated_text'].strip()

            # --- CLEAN RESPONSE ---
            ai_response = ai_response.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
            ai_response = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', ai_response)

            # Display nicely formatted text
            st.markdown(f"<p style='font-size:16px; line-height:1.6;'>{ai_response}</p>", unsafe_allow_html=True)

    # Save AI reply
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# --- 5. RESET BUTTON ---
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
