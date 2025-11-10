import streamlit as st
from transformers import pipeline
import sys
import re

# --- ENCODING FIX (Standard Python code, safe to run first) ---
sys.stdout.reconfigure(encoding='utf-8')

# --- 1. PAGE SETTINGS & THEME (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Specimen AI 🤖",
    page_icon="👑",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 👑 THEME DESIGN (Black + Gold) ---
# This uses st.markdown and is safe AFTER st.set_page_config
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;  /* Black background */
        color: #FFD700;  /* Gold text */
    }
    h1, h2, h3, h4, h5 {
        color: #FFD700 !important;
    }
    /* Style for the user input box */
    .stTextInput > div > div > input {
        color: #FFD700;
        border-color: #FFD700;
    }
    /* Style for the chat messages background */
    .stChatMessage {
        background: rgba(255, 215, 0, 0.1);
        border-radius: 10px;
        padding: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD THE MODEL (Now safe to run after page config) ---
@st.cache_resource
def load_chatbot_model():
    """Loads a smart text2text model for cleaner English responses."""
    # Using a smarter, cleaner, and smaller model (FLAN-T5)
    return pipeline("text2text-generation", model="google/flan-t5-small")

chatbot = load_chatbot_model()

# --- 3. TITLE & CAPTION ---
st.title("👑 Specimen Chatbot AI")
st.caption("Built by Specimen King using Streamlit + Hugging Face")

# --- 4. CHAT HISTORY SETUP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. MAIN CHAT LOGIC ---
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
            # Remove non-standard characters
            ai_response = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', ai_response)

            # Display nicely formatted text
            st.markdown(f"<p style='font-size:16px; line-height:1.6;'>{ai_response}</p>", unsafe_allow_html=True)

    # Save AI reply
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# --- 6. RESET BUTTON ---
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
