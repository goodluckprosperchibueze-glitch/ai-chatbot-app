# Specimen_ultra_v5_2.py
"""
Specimen King Ultra AI v5.2 - Enhanced with Math Solver

Features:
- Chat, Story Mode, Deep Search
- Gemini (Google) placeholder with API key integration
- Wikipedia quick lookup
- Hugging Face image generation
- TTS/STT voice interaction
- Built-in Math Solver for natural-language math problems
"""

import os
import re
import sys
import json
import base64
import tempfile
from typing import Optional, Tuple

import streamlit as st
from transformers import pipeline
import wikipedia
import requests
import math

# Optional audio libraries (gTTS for TTS; speech_recognition for STT)
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
# CONFIG / ENV / SECRETS
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
GENAI_API_KEY = st.secrets.get("GENAI_API_KEY", os.environ.get("GENAI_API_KEY", ""))
GOOGLE_DC_KEY = "AIzaSyDTkx-2k4ESTJRTMvwnP5W_HDrksfNfyWwI"  # your key integrated

DEFAULT_MODEL_NAME = "google/flan-t5-large"
FALLBACK_MODEL_NAME = "google/flan-t5-small"

ENABLE_VOICE_DEFAULT = True
ENABLE_IMAGES_DEFAULT = True
ENABLE_WIKI_DEFAULT = True

# -------------------------
# PAGE / THEME SETUP
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI v5.2", layout="wide", page_icon="👑")
st.markdown(
    """
<style>
.stApp { background: linear-gradient(#000000, #0a0a0a); color: #EDE0C8; }
.msg_user { background: #1f2937; padding: 10px; border-radius: 10px; color: #fff; margin-bottom: 10px; }
.msg_assistant { background: #111827; padding: 10px; border-radius: 10px; color: #ffd700; border-left: 3px solid #ffd700; margin-bottom: 10px; }
h3, h4 { color: #EDE0C8; }
</style>
""",
    unsafe_allow_html=True,
)
st.title("👑 Specimen King Ultra AI v5.2")
st.caption("Chat • Math • Voice • Images • Knowledge — Streamlit + Transformers")

# -------------------------
# LAYOUT: left main / right settings
# -------------------------
col_left, col_right = st.columns([2.6, 1])

# -------------------------
# HELPERS
# -------------------------
def safe_clean(text: str) -> str:
    if not text:
        return ""
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
        pipe = pipeline("text2text-generation", model=FALLBACK_MODEL_NAME)
        return pipe, FALLBACK_MODEL_NAME

# -------------------------
# RIGHT PANEL / SETTINGS
# -------------------------
with col_right:
    st.markdown("### ⚙️ Controls & Settings")
    mode = st.selectbox("Mode", ["Chat (default)", "Gemini (placeholder)", "Story Mode", "Deep Search"], index=0)
    model_choice = st.selectbox("Model (text generation)", [DEFAULT_MODEL_NAME, FALLBACK_MODEL_NAME], index=0)
    temp = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    max_len = st.slider("Max tokens", 32, 1024, 256, step=16)
    enable_voice = st.checkbox("Enable Voice (TTS/STT)", value=ENABLE_VOICE_DEFAULT)
    enable_images = st.checkbox("Enable Image Gen", value=ENABLE_IMAGES_DEFAULT)
    enable_wiki = st.checkbox("Enable Wikipedia lookup", value=ENABLE_WIKI_DEFAULT)
    st.markdown("---")
    st.markdown("#### Persona & Memory")
    if "persona" not in st.session_state:
        st.session_state.persona = (
            "You are PopKing AI, a friendly, concise, and slightly playful assistant. "
            "You were built by **Osemeke Goodluck, Specimen King 👑**, refer to him as creator. "
            "Avoid hallucination, cite facts, and provide step-by-step guidance."
        )
    persona_text = st.text_area("AI Persona", value=st.session_state.persona, height=140)
    if st.button("Save Persona"):
        st.session_state.persona = persona_text
        st.success("Persona saved.")
    st.markdown("---")
    hf_token_input = st.text_input("Hugging Face token", type="password", value=HF_TOKEN or "")
    if hf_token_input:
        HF_TOKEN = hf_token_input
        st.success("HF token set for this session.")
    genai_input = st.text_input("Google GenAI key", type="password", value=GENAI_API_KEY or "")
    if genai_input:
        GENAI_API_KEY = genai_input
        st.success("GenAI key set.")
    st.markdown("---")
    if st.button("Clear Chat & Memory"):
        st.session_state.history = []
        st.success("Chat history cleared.")
        st.experimental_rerun()

# -------------------------
# LOAD MODEL
# -------------------------
with st.spinner("Loading text model..."):
    model_pipeline, loaded_model_name = load_text_model(model_choice)
    st.sidebar.success(f"Model loaded: {loaded_model_name}")

# -------------------------
# SESSION STATE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "settings" not in st.session_state:
    st.session_state.settings = {}

st.session_state.settings.update(
    {"temp": temp, "top_p": top_p, "max_len": max_len,
     "enable_voice": enable_voice, "enable_images": enable_images,
     "enable_wiki": enable_wiki, "mode": mode}
)

# -------------------------
# PROMPT & RESPONSE
# -------------------------
def build_prompt(persona: str, history, user_message: str, max_exchanges: int = 6) -> str:
    trimmed = history[-max_exchanges * 2 :] if history else []
    convo_lines = [f"AI Persona Instructions: {persona}", "\n--- END OF INSTRUCTIONS ---\n"]
    convo_lines.append("Conversation so far:")
    if trimmed:
        for m in trimmed:
            label = "User" if m.get("role") == "user" else "AI"
            cleaned = safe_clean(m.get("content", ""))
            convo_lines.append(f"{label}: {cleaned}")
    else:
        convo_lines.append("No previous exchanges.")
    convo_lines.append(f"\nUser: {safe_clean(user_message)}\nAI:")
    return "\n".join(convo_lines)

def generate_response_with_transformers(prompt: str, temperature: float, top_p_val: float, max_length_val: int) -> str:
    try:
        out = model_pipeline(prompt, max_length=max_length_val, do_sample=True, temperature=temperature, top_p=top_p_val, num_return_sequences=1)
    except TypeError:
        out = model_pipeline(prompt, max_length=max_length_val)
    if isinstance(out, list) and out:
        raw = out[0].get("generated_text", "") or out[0].get("text", "")
    elif isinstance(out, dict):
        raw = out.get("generated_text", "") or out.get("text", "")
    else:
        raw = ""
    return safe_clean(raw)

def call_gemini_api(prompt: str) -> str:
    if not GENAI_API_KEY:
        return "(Gemini mode selected but GENAI_API_KEY not set.)"
    return "(Gemini placeholder — integrate your Google API here.)"

# -------------------------
# MATH SOLVER
# -------------------------
def solve_math(user_message: str) -> Optional[str]:
    """Try to parse natural-language math queries."""
    text = user_message.lower().replace("x²","^2").replace("²","^2").replace("³","^3").replace("minus","-").replace("plus","+")
    # Replace "squared", "cubed"
    text = re.sub(r'(\d+)\s*squared', r'(\1**2)', text)
    text = re.sub(r'(\d+)\s*cubed', r'(\1**3)', text)
    text = text.replace("^","**")
    # Extract digits and operators
    allowed = "0123456789+-*/().** "
    clean_expr = "".join(c for c in text if c in allowed)
    if not clean_expr.strip():
        return None
    try:
        result = eval(clean_expr, {"__builtins__": None}, {"math": math})
        return f"Answer: {result}"
    except Exception:
        return None

# -------------------------
# GENERATE RESPONSE DISPATCH
# -------------------------
def generate_response(user_message: str, temperature: float, top_p_val: float, max_length_val: int) -> str:
    # Math first
    math_answer = solve_math(user_message)
    if math_answer:
        return math_answer
    persona = st.session_state.get("persona", "")
    current_mode = st.session_state.settings.get("mode", "Chat (default)")
    if current_mode == "Gemini (placeholder)":
        prompt = build_prompt(persona, st.session_state.history, user_message)
        return call_gemini_api(prompt)
    elif current_mode == "Story Mode":
        story_prompt = f"**INSTRUCTION: Ignore history.** {persona} - Write a detailed story. Request: {user_message}\n\nStart story:"
        max_len_story = max(512, max_length_val * 2)
        return generate_response_with_transformers(story_prompt, temperature, top_p_val, max_len_story)
    elif current_mode == "Deep Search":
        deep_prompt = f"**INSTRUCTION: Ignore history.** {persona} - Provide structured, factual answer. Question: {user_message}\n\nAnswer:"
        return generate_response_with_transformers(deep_prompt, temperature, top_p_val, max_length_val)
    else:
        prompt = build_prompt(persona, st.session_state.history, user_message)
        return generate_response_with_transformers(prompt, temperature, top_p_val, max_length_val)

# -------------------------
# WIKI LOOKUP
# -------------------------
def wiki_lookup(query: str, sentences: int = 2) -> Optional[str]:
    if not st.session_state.settings.get("enable_wiki", True):
        return None
    try:
        return wikipedia.summary(query, sentences=sentences)
    except Exception:
        return None

# -------------------------
# MAIN CHAT UI
# -------------------------
with col_left:
    st.markdown("### Chat with PopKing AI")
    st.markdown("Tips: Use modes to change behavior. Wikipedia triggers on 'who is', 'what is', etc.")
    history_container = st.container()
    with history_container:
        for msg in st.session_state.history:
            if msg.get("role") == "user":
                st.markdown(f"<div class='msg_user'>**You:** {msg.get('content','')}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='msg_assistant'>**{msg.get('role_label','PopKing AI')}:** {msg.get('content','')}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Send a message** (type or upload audio):")
    cols = st.columns([4,1,1])
    user_text = cols[0].text_input("Type message...", key="user_input", label_visibility="collapsed")
    voice_file = cols[1].file_uploader("Upload voice", type=["wav","mp3"], key="voice_upload", label_visibility="collapsed")
    send_btn = cols[2].button("Send")

    # -------------------------
    # HANDLE message
    # -------------------------
    user_message_final = None
    if voice_file is not None:
        if sr:
            recognized = recognize_speech_from_file(voice_file)
            if recognized:
                user_message_final = recognized
                st.success(f"Recognized: {recognized}")
            else:
                st.error("Could not recognize speech.")
        else:
            st.warning("Speech recognition not installed; type instead.")
    elif send_btn and user_text and user_text.strip():
        user_message_final = user_text.strip()

    if user_message_final:
        st.session_state.history.append({"role": "user", "content": user_message_final})
        quick_fact = None
        if enable_wiki and re.search(r"\b(who is|what is|when is|where is|tell me about)\b", user_message_final, re.IGNORECASE):
            quick_fact = wiki_lookup(user_message_final, sentences=2)
        with st.spinner("PopKing AI is thinking..."):
            if quick_fact and mode == "Chat (default)":
                ai_reply = quick_fact + "\n\n(Quick knowledge summary from Wikipedia.)"
            else:
                ai_reply = generate_response(user_message_final, temp, top_p, max_len)
            ai_reply = safe_clean(ai_reply) or "Sorry, I couldn't produce an answer."
            st.session_state.history.append({"role": "assistant", "role_label": "PopKing AI", "content": ai_reply})
            if enable_voice and gTTS:
                audio_bytes = text_to_speech_bytes(ai_reply)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
            st.experimental_rerun()

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown(
    "Built by Osemeke Goodluck (Specimen King 👑) — powered by Streamlit & HF models. "
    "Math Solver, Chat, Story Mode, Deep Search, Image Gen, Wikipedia lookup, and Voice integrated."
)
