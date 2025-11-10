# popking_ultra_v5_1_plus.py
"""
Specimen King Ultra AI v5.1+ – Deluxe Edition

Streamlit app: Text generation (Flan-T5), Hugging Face image gen,
Wikipedia quick lookup, Google GenAI support, TTS/STT,
multiple AI modes, persona controls, memory, spicy modes
(Poetry, Advice, Meme Generator, Joke Mode, Story twists)
"""

import os
import re
import sys
import json
import base64
import tempfile
from typing import Optional, Tuple
import random

import streamlit as st
from transformers import pipeline
import wikipedia
import requests

# Optional audio libraries
try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

# Ensure UTF-8 output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -------------------------
# CONFIG / SECRETS
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
GENAI_API_KEY = "AIzaSyDTkx-2k4ESTJRTMvwnP5W_HDrksfNfyWw"  # your Google DC key
DEFAULT_MODEL_NAME = "google/flan-t5-large"
FALLBACK_MODEL_NAME = "google/flan-t5-small"

ENABLE_VOICE_DEFAULT = True
ENABLE_IMAGES_DEFAULT = True
ENABLE_WIKI_DEFAULT = True

# -------------------------
# PAGE / THEME
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI v5.1+", layout="wide", page_icon="👑")
st.markdown("""
<style>
.stApp { background: linear-gradient(#0f0f0f, #1a1a1a); color: #EDE0C8; }
.msg_user { background: #1f2937; padding: 10px; border-radius: 10px; color: #fff; margin-bottom: 10px; }
.msg_assistant { background: #111827; padding: 10px; border-radius: 10px; color: #ffd700; border-left: 3px solid #ffd700; margin-bottom: 10px; }
h3, h4 { color: #EDE0C8; }
</style>
""", unsafe_allow_html=True)

st.title("👑 Specimen King Ultra AI v5.1+")
st.caption("Voice • Images • Google GenAI • Memory • Spicy Modes — Streamlit + Transformers")

# -------------------------
# LAYOUT
# -------------------------
col_left, col_right = st.columns([2.6, 1])

# -------------------------
# HELPERS
# -------------------------
def safe_clean(text: str) -> str:
    if not text: return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = "".join(ch for ch in text if ord(ch) < 0x10000)
    text = re.sub(r"^(AI:|User:)\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

def b64_image_from_bytes(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# -------------------------
# MODEL LOADING
# -------------------------
@st.cache_resource
def load_text_model(model_name: str) -> Tuple:
    try:
        pipe = pipeline("text2text-generation", model=model_name)
        return pipe, model_name
    except Exception:
        try:
            pipe = pipeline("text2text-generation", model=FALLBACK_MODEL_NAME)
            return pipe, FALLBACK_MODEL_NAME
        except Exception as e2:
            raise RuntimeError(f"Failed to load both models: {e2}")

# -------------------------
# Sidebar / Right Column
# -------------------------
with col_right:
    st.markdown("### ⚙️ Controls & Settings")
    mode = st.selectbox(
        "Mode",
        ["Chat", "Story Mode", "Deep Search", "Poetry Master", "Advice Guru", "Meme Generator", "Joke Mode"],
        index=0,
    )
    model_choice = st.selectbox(
        "Model (text generation)",
        options=[DEFAULT_MODEL_NAME, FALLBACK_MODEL_NAME],
        index=0,
    )
    temp = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    max_len = st.slider("Max tokens", 32, 1024, 256, step=16)

    enable_voice = st.checkbox("Enable Voice (TTS/STT)", value=ENABLE_VOICE_DEFAULT)
    enable_images = st.checkbox("Enable Image Gen", value=ENABLE_IMAGES_DEFAULT)
    enable_wiki = st.checkbox("Enable Wikipedia lookup", value=ENABLE_WIKI_DEFAULT)

    st.markdown("---")
    st.markdown("#### Persona & Memory")
    if "persona" not in st.session_state:
        st.session_state.persona = "You are PopKing AI, friendly, concise, playful. Built by Osemeke Goodluck 👑."
    persona_text = st.text_area("AI Persona", value=st.session_state.persona, height=140)
    if st.button("Save Persona"):
        st.session_state.persona = persona_text
        st.success("Persona saved.")

    st.markdown("---")
    if st.button("Clear Chat & Memory"):
        st.session_state.history = []
        st.success("Chat history cleared.")
        st.experimental_rerun()

# -------------------------
# Load model
# -------------------------
with st.spinner("Loading text model..."):
    try:
        model_pipeline, loaded_model_name = load_text_model(model_choice)
        st.sidebar.success(f"Model loaded: {loaded_model_name}")
    except Exception as e:
        st.sidebar.error(f"Model load failed: {e}")
        st.error("Model failed to load — check internet.")
        st.stop()

# -------------------------
# Session state
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.settings = {
    "temp": temp, "top_p": top_p, "max_len": max_len,
    "enable_voice": enable_voice, "enable_images": enable_images,
    "enable_wiki": enable_wiki, "mode": mode,
}

# -------------------------
# Prompt / Response
# -------------------------
def build_prompt(persona: str, history, user_message: str) -> str:
    trimmed = history[-12:] if history else []
    convo_lines = [f"AI Persona Instructions: {persona}", "Conversation so far:"]
    for m in trimmed:
        label = "User" if m.get("role") == "user" else "AI"
        convo_lines.append(f"{label}: {safe_clean(m.get('content',''))}")
    convo_lines.append(f"\nUser: {safe_clean(user_message)}\nAI:")
    return "\n".join(convo_lines)

def generate_response_with_transformers(prompt: str, temperature: float, top_p_val: float, max_length_val: int) -> str:
    out = model_pipeline(prompt, max_length=max_length_val, do_sample=True, temperature=temperature, top_p=top_p_val, num_return_sequences=1)
    raw = out[0].get("generated_text", "") if out else ""
    return safe_clean(raw)

def wiki_lookup(query: str, sentences: int = 2) -> Optional[str]:
    if not st.session_state.settings.get("enable_wiki", True):
        return None
    try:
        return wikipedia.summary(query, sentences=sentences)
    except Exception:
        return None

def hf_generate_image_bytes(prompt_text: str, hf_token: str) -> bytes:
    if not hf_token: raise RuntimeError("HF token missing")
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt_text}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 200: return resp.content
    raise RuntimeError(f"Image gen failed: {resp.status_code} {resp.text}")

def text_to_speech_bytes(text: str, lang: str = "en") -> Optional[bytes]:
    if gTTS is None: return None
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        data = open(tmp.name,"rb").read()
        try: os.unlink(tmp.name)
        except Exception: pass
        return data
    except Exception:
        return None

# -------------------------
# Generate response dispatch
# -------------------------
def generate_response(user_message: str) -> str:
    persona = st.session_state.get("persona", "")
    mode = st.session_state.settings.get("mode","Chat")
    
    if mode=="Story Mode":
        prompt = f"{persona} - Creative story with plot twists: {user_message}"
        return generate_response_with_transformers(prompt, temp, top_p, max_len*2)
    elif mode=="Poetry Master":
        prompt = f"{persona} - Poetic emotional content: {user_message}"
        return generate_response_with_transformers(prompt, temp, top_p, max_len*2)
    elif mode=="Advice Guru":
        prompt = f"{persona} - Give advice or motivational guidance: {user_message}"
        return generate_response_with_transformers(prompt, temp, top_p, max_len)
    elif mode=="Meme Generator":
        prompt = f"{persona} - Generate funny meme caption: {user_message}"
        return generate_response_with_transformers(prompt, temp, top_p, 64)
    elif mode=="Joke Mode":
        jokes = ["Why did the computer go to therapy? It had too many bytes!", 
                 "Why was the math book sad? Too many problems.", 
                 "I told my AI a joke, it laughed in binary: 101010!"]
        return random.choice(jokes)
    elif mode=="Deep Search":
        prompt = f"{persona} - Factual answer with steps if possible: {user_message}"
        return generate_response_with_transformers(prompt, temp, top_p, max_len)
    else:  # default Chat
        prompt = build_prompt(persona, st.session_state.history, user_message)
        return generate_response_with_transformers(prompt, temp, top_p, max_len)

# -------------------------
# MAIN CHAT UI
# -------------------------
with col_left:
    st.markdown("### Chat with PopKing AI")
    for msg in st.session_state.history:
        label = "You" if msg.get("role")=="user" else msg.get("role_label","PopKing AI")
        style_class = "msg_user" if msg.get("role")=="user" else "msg_assistant"
        st.markdown(f"<div class='{style_class}'>**{label}:** {msg.get('content','')}</div>", unsafe_allow_html=True)

    st.markdown("---")
    cols = st.columns([4,1,1])
    user_text = cols[0].text_input("Type message here...", key="user_input", label_visibility="collapsed")
    voice_file = cols[1].file_uploader("Upload voice", type=["wav","mp3"], key="voice_upload", label_visibility="collapsed")
    send_btn = cols[2].button("Send")

    user_message_final = None
    if voice_file and sr:
        with st.spinner("Recognizing speech..."):
            try:
                r = sr.Recognizer()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(voice_file.getbuffer())
                    tmp.flush()
                    with sr.AudioFile(tmp.name) as source:
                        audio = r.record(source)
                        recognized = r.recognize_google(audio)
                        user_message_final = recognized
                        st.success(f"Recognized: {recognized}")
            except Exception:
                st.error("Could not recognize speech.")
    elif send_btn and user_text.strip():
        user_message_final = user_text.strip()

    if user_message_final:
        st.session_state.history.append({"role":"user","content":user_message_final})
        quick_fact = wiki_lookup(user_message_final) if enable_wiki else None
        with st.spinner("PopKing AI is thinking..."):
            try:
                if quick_fact: ai_reply = quick_fact + "\n\n(Quick Wikipedia summary.)"
                else: ai_reply = generate_response(user_message_final)
                ai_reply = safe_clean(ai_reply) or "Sorry, I couldn't produce an answer."
                st.session_state.history.append({"role":"assistant","role_label":"PopKing AI","content":ai_reply})
                if enable_voice and gTTS:
                    audio_bytes = text_to_speech_bytes(ai_reply)
                    if audio_bytes: st.audio(audio_bytes, format="audio/mp3")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"AI generation error: {e}")

st.markdown("---")
st.markdown("Built by Osemeke Goodluck 👑 — powered by Hugging Face, Google GenAI & Streamlit. Customize persona & modes on the right.")
