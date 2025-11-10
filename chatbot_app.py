# specimen_ultra.py
import streamlit as st
import os
import re
import sys
import tempfile
import requests
import wikipedia
from transformers import pipeline

# Optional: text-to-speech and speech recognition
try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

# Ensure UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# -------------------------
# Config / Tokens
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))

ENABLE_VOICE = True
ENABLE_IMAGES = True
ENABLE_WIKI = True

# -------------------------
# UI Setup
# -------------------------
st.set_page_config(page_title="Specimen King AI", layout="wide")
st.title("👑 Specimen King AI (ChatGPT Edition)")
st.caption("Voice • Images • Memory • Knowledge — fully self-contained")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### Chat with Specimen King AI")
with col_right:
    st.markdown("### Controls / Settings")

# -------------------------
# Load AI Model
# -------------------------
@st.cache_resource
def load_model(model_name="google/flan-t5-large"):
    return pipeline("text2text-generation", model=model_name)

with st.spinner("Loading AI model..."):
    model = load_model()

# -------------------------
# Session state
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "persona" not in st.session_state:
    st.session_state.persona = (
        "You are Specimen King AI — confident, witty, friendly, and extremely helpful. "
        "Answer like a human friend would, with humor, clarity, and occasional emojis. "
        "You have vast knowledge, but always stay humble and polite."
    )

# -------------------------
# Right Panel: Settings
# -------------------------
with col_right:
    st.markdown("#### Settings")
    persona_text = st.text_area("AI Persona (edit to change tone)", value=st.session_state.persona, height=120)
    if st.button("Save Persona"):
        st.session_state.persona = persona_text
        st.success("Persona updated ✅")

    st.markdown("---")
    st.write("Optional API keys")
    hf_token_input = st.text_input("Hugging Face API Token (for image gen)", type="password", value=HF_TOKEN or "")
    if hf_token_input:
        HF_TOKEN = hf_token_input
        st.success("HF token set for this session.")

    st.markdown("---")
    st.write("Quick Utilities")
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.experimental_rerun()

# -------------------------
# Helper functions
# -------------------------
def clean_text(text):
    text = text.strip()
    text = re.sub(r"^(AI:|User:)\s*", "", text, flags=re.IGNORECASE)
    return text

def generate_response(user_message, max_length=200):
    """
    Generate AI response with enhanced 'ChatGPT style' personality
    """
    # Build chat history
    convo_text = ""
    for m in st.session_state.history[-6*2:]:  # last 6 exchanges
        role_label = "User" if m["role"] == "user" else "AI"
        convo_text += f"{role_label}: {m['content']}\n"

    prompt = (
        f"{st.session_state.persona}\n\n"
        f"Conversation so far:\n{convo_text}"
        f"User: {user_message}\nAI:"
    )

    out = model(prompt, max_length=max_length, temperature=0.7, top_p=0.9)
    reply = out[0].get("generated_text", "")
    reply = clean_text(reply)

    # Add personality touches
    if len(reply) < 20:
        reply += " 😊"

    return reply

# Wikipedia lookup
def wiki_lookup(query, sentences=2):
    if not ENABLE_WIKI:
        return None
    try:
        return wikipedia.summary(query, sentences=sentences)
    except Exception:
        return None

# Hugging Face image generator
def hf_generate_image(prompt_text, hf_token=HF_TOKEN):
    if not hf_token:
        raise RuntimeError("No HF token provided.")
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt_text}
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Image gen failed: {response.status_code} {response.text}")
    return response.content

# Text-to-Speech
def text_to_speech_bytes(text, lang="en"):
    if gTTS is None:
        return None
    tts = gTTS(text=text, lang=lang, slow=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    with open(tmp.name, "rb") as f:
        data = f.read()
    os.unlink(tmp.name)
    return data

# Speech Recognition
def recognize_speech_from_file(uploaded_file):
    if sr is None:
        return None
    r = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        tmp_path = tmp.name
    try:
        with sr.AudioFile(tmp_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        return text
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

# -------------------------
# Main Chat UI
# -------------------------
with col_left:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    st.markdown("---")
    st.markdown("**Send a message** (text or voice upload `.wav`):")

    cols = st.columns([4, 1, 1])
    user_text = cols[0].text_input("Type message...", key="user_input")
# specimen_ultra.py
import streamlit as st
import os
import re
import sys
import tempfile
import requests
import wikipedia
from transformers import pipeline

# Optional: text-to-speech and speech recognition
try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

# Ensure UTF-8 in all environments
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# -------------------------
# CONFIGURATION / SECRETS
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))
ENABLE_VOICE = True
ENABLE_IMAGES = True
ENABLE_WIKI = True

# -------------------------
# STREAMLIT UI SETUP
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI", layout="wide")
st.title("👑 Specimen King Ultra AI")
st.caption("Chat, voice, images, knowledge — powered by Transformers")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### Chat with Specimen King AI")
with col_right:
    st.markdown("### Controls / Settings")

# -------------------------
# SESSION STATE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # {"role": "user"/"assistant", "content": "..."}
if "persona" not in st.session_state:
    st.session_state.persona = (
        "You are Specimen King AI — confident, witty, insightful, and human-like. "
        "You answer politely, sometimes with humor, always clear, concise, and occasionally profound."
    )

# -------------------------
# RIGHT PANEL SETTINGS
# -------------------------
with col_right:
    st.markdown("#### Settings")
    persona_text = st.text_area("AI Persona", value=st.session_state.persona, height=120)
    if st.button("Save Persona"):
        st.session_state.persona = persona_text
        st.success("Persona updated ✅")
    
    st.markdown("---")
    st.write("Optional API Keys")
    hf_token_input = st.text_input("Hugging Face Token (for images)", type="password", value=HF_TOKEN or "")
    if hf_token_input:
        HF_TOKEN = hf_token_input
        st.success("HF Token set for this session")

    st.markdown("---")
    st.write("Utilities")
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.experimental_rerun()
    if st.button("💾 Download Chat"):
        chat_content = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.history])
        st.download_button("Download Chat History", chat_content, file_name="chat_history.txt")

# -------------------------
# LOAD AI MODEL
# -------------------------
@st.cache_resource
def load_model(model_name="google/flan-t5-large"):
    return pipeline("text2text-generation", model=model_name)

with st.spinner("Loading AI model..."):
    model = load_model()

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def clean_text(text):
    text = text.strip()
    text = re.sub(r"^(AI:|User:)\s*", "", text, flags=re.IGNORECASE)
    return text

def build_prompt(persona, history, user_message):
    max_exchanges = 6
    trimmed = history[-max_exchanges*2:]
    convo_text = ""
    for m in trimmed:
        role_label = "User" if m["role"]=="user" else "AI"
        convo_text += f"{role_label}: {m['content']}\n"
    prompt = f"{persona}\n\nConversation so far:\n{convo_text}\nUser: {user_message}\nAI:"
    return prompt

def generate_response(user_message, max_length=200):
    prompt = build_prompt(st.session_state.persona, st.session_state.history, user_message)
    out = model(prompt, max_length=max_length, temperature=0.7, top_p=0.9)
    raw = out[0].get("generated_text","")
    return clean_text(raw)

def wiki_lookup(query, sentences=2):
    if not ENABLE_WIKI:
        return None
    try:
        return wikipedia.summary(query, sentences=sentences)
    except Exception:
        return None

def hf_generate_image(prompt_text, hf_token=HF_TOKEN):
    if not hf_token:
        raise RuntimeError("No HF token provided")
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt_text}
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Image gen failed: {response.status_code} {response.text}")
    return response.content

def text_to_speech_bytes(text, lang="en"):
    if gTTS is None:
        return None
    tts = gTTS(text=text, lang=lang, slow=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    with open(tmp.name,"rb") as f:
        data = f.read()
    os.unlink(tmp.name)
    return data

def recognize_speech_from_file(uploaded_file):
    if sr is None:
        return None
    r = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        tmp_path = tmp.name
    try:
        with sr.AudioFile(tmp_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        return text
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

# -------------------------
# MAIN CHAT UI
# -------------------------
with col_left:
    # Show chat history
    for msg in st.session_state.history:
        if msg["role"]=="user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])
    
    st.markdown("---")
    st.markdown("**Send a message** (text or upload `.wav`/`.mp3`):")
    
    cols = st.columns([4,1,1])
    user_text = cols[0].text_input("Type message here...", key="user_input")
    voice_file = cols[1].file_uploader("🎤 Voice (wav/mp3)", type=["wav","mp3"], key="voice_upload")
    send_btn = cols[2].button("Send")
    
    # Image generator
    with st.expander("🎨 Image generation"):
        img_prompt = st.text_area("Describe image...", value="", height=80)
        if st.button("Generate Image"):
            if not HF_TOKEN:
                st.error("Please set Hugging Face API token")
            else:
                with st.spinner("Generating image..."):
                    try:
                        img_bytes = hf_generate_image(img_prompt, HF_TOKEN)
                        st.image(img_bytes)
                        st.success("Image generated ✅")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Wikipedia search
    with st.expander("📘 Knowledge Search"):
        search_q = st.text_input("Wikipedia search", value="", key="wiki_q")
        if st.button("Lookup Wikipedia"):
            if search_q.strip():
                with st.spinner("Searching Wikipedia..."):
                    summary = wiki_lookup(search_q)
                    if summary:
                        st.markdown(f"**Wikipedia summary:**\n\n{summary}")
                    else:
                        st.info("No summary found")
    
    # Handle input
    user_message_final = None
    if voice_file is not None:
