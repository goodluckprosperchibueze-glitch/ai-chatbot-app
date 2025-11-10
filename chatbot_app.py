import streamlit as st
from transformers import pipeline
import sys
import re

# --- ENCODING FIX ---
# This line helps ensure smooth text output on cloud environments
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass # Ignore if reconfigure is not available

# --- 1. LOAD THE MODEL (FLAN-T5-SMALL) ---
# This is a small, instruction-tuned model that gives clean, helpful responses.
@st.cache_resource
def load_chatbot_model():
    """Loads a smart text2text model for cleaner English responses."""
    return pipeline("text2text-generation", model="google/flan-t5-small")

chatbot = load_chatbot_model()

# --- 2. PAGE SETTINGS ---
st.set_page_config(
    page_title="Specimen AI 🤖",
    page_icon="👑",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 👑 THEME DESIGN (Black + Gold CSS) ---
# Inject custom CSS to create the dark theme
st.markdown("""
    <style>
    /* Main App Background and Text Color */
    .stApp {
        background-color: #000000;  /* Black background */
        color: #FFFFFF;  /* Fallback white text for general elements */
    }
    /* Title and Header Colors */
    h1, h2, h3, h4, h5 {
        color: #FFD700 !important; /* Gold */
    }
    /* Chat Input Box and Text Color */
    .stTextInput > div > div > input {
        color: #FFD700;
        background-color: #1a1a1a; /* Darker input box */
        border-color: #FFD700;
    }
    /* Chat Message Bubbles (Assistant/User) */
    .stChatMessage {
        background: rgba(255, 215, 0, 0.1); /* Light gold background */
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        border: 1px solid rgba(255, 215, 0, 0.3); /* Subtle gold border */
        color: #FFFFFF; /* Ensure chat text is white/readable */
    }
    /* Clear Chat Button Styling */
    .stButton>button {
        background-color: #FFD700; /* Gold background */
        color: #000000 !important; /* Black text */
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e6c200; /* Slightly darker gold on hover */
    }
    /* Spinner Color */
    .stSpinner > div > div {
        border-top-color: #FFD700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- TITLE & CAPTION ---
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
            raw_response = chatbot(
                prompt, 
                max_length=150, 
                temperature=0.7,
                do_sample=True 
            )
            ai_response = raw_response[0]['generated_text'].strip()

            # --- CLEAN RESPONSE ---
            # Remove any non-printable/garbled characters
            ai_response = ''.join(c for c in ai_response if c.isprintable())
            
            # Use regex to strip away any other potential junk characters
            ai_response = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', ai_response)

            # Display nicely formatted text
            st.markdown(f"<p style='font-size:16px; line-height:1.6;'>{ai_response}</p>", unsafe_allow_html=True)

    # Save AI reply
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# --- 5. RESET BUTTON ---
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
