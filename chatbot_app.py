# specimen_ultra_gemini_live.py
import streamlit as st
import os
import re
import sys
import tempfile
import requests
import wikipedia
import json
import random
import datetime

# Gemini AI
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Transformers fallback
try:
    from transformers import pipeline
except Exception:
    pipeline = None

# Audio
try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

# -------------------------
# CONFIG / SECRETS
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", None))

# Configure Gemini if available
if GOOGLE_API_KEY and genai:
    genai.configure(api_key=GOOGLE_API_KEY)

# Feature toggles
ENABLE_VOICE = True
ENABLE_IMAGES = True
ENABLE_WIKI = True
ENABLE_STORY_MODE = True
ENABLE_DEEP_SEARCH = True
ENABLE_PERSONALITY = True

# -------------------------
# UI Setup
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI", layout="wide")
st.title("👑 Specimen King Ultra AI (Live Gemini Edition)")
st.caption("Voice • Images • Storyline • Knowledge — fully dynamic AI")

col_left, col_right = st.columns([2, 1])

# -------------------------
# SESSION STATE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "persona" not in st.session_state:
    st.session_state.persona = (
        "You are Specimen King AI — confident, witty, composed, and slightly mysterious. "
        "Speak like a genius friend who knows more than they reveal. "
        "Be empathetic, playful, and occasionally humorous. "
        "Keep short, polished sentences. Show personality."
    )

if "mode" not in st.session_state:
    st.session_state.mode = "Default"

# -------------------------
# Right Panel: Settings
# -------------------------
with col_right:
    st.markdown("#### Settings / Controls")
    
    persona_text = st.text_area("AI Persona", value=st.session_state.persona, height=140)
    if st.button("Save Persona"):
        st.session_state.persona = persona_text
        st.success("Persona updated ✅")
    
    st.markdown("---")
    st.write("Optional API keys")
    hf_token_input = st.text_input("HF Token (for image gen)", type="password", value=HF_TOKEN or "")
    if hf_token_input:
        HF_TOKEN = hf_token_input
        st.success("HF token set for this session ✅")
    
    st.markdown("---")
    st.write("Mode Selection")
    mode = st.selectbox("Choose AI mode", ["Default", "Storyline", "Deep Search", "Image Mode"])
    st.session_state.mode = mode
    
    st.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.experimental_rerun()

# -------------------------
# Gemini / Transformers setup
# -------------------------
def load_fallback_model():
    if pipeline:
        return pipeline("text2text-generation", model="google/flan-t5-large")
    return None

model = load_fallback_model()

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def clean_text(text):
    text = text.strip()
    text = re.sub(r"^(AI:|User:)\s*", "", text, flags=re.IGNORECASE)
    return text

def generate_gemini_response(user_message):
    if not genai:
        return None
    try:
        chat_history = [{"role":"user", "content": m["content"]} for m in st.session_state.history]
        prompt = f"{st.session_state.persona}\n\nUser: {user_message}\nAI:"
        response = genai.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return clean_text(response.text)
    except Exception:
        return None

def generate_fallback_response(user_message, max_length=250):
    if not model:
        return "AI models not available."
    prompt = f"{st.session_state.persona}\n\nUser: {user_message}\nAI:"
    out = model(prompt, max_length=max_length, temperature=0.7, top_p=0.9)
    raw = out[0].get("generated_text", "")
    return clean_text(raw)

def wiki_lookup(query, sentences=2):
    if not ENABLE_WIKI:
        return None
    try:
        return wikipedia.summary(query, sentences=sentences)
    except Exception:
        return None

def hf_generate_image(prompt_text):
    if not HF_TOKEN:
        return None
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt_text}
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        return None
    return response.content

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
        return r.recognize_google(audio)
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

def dynamic_reply(user_message):
    """Adds personality, wit, and variety to replies"""
    witty_phrases = [
        "😏 Interesting point.",
        "Let me think… 🤔",
        "Absolutely!",
        "I knew you’d ask that.",
        "Here's something fun:",
        "Ah, that reminds me…"
    ]
    emojis = ["🧠","🔥","💡","🎯","🤖"]
    base_reply = generate_gemini_response(user_message) or generate_fallback_response(user_message)
    if random.random() < 0.3:
        base_reply = f"{random.choice(witty_phrases)} {base_reply} {random.choice(emojis)}"
    return base_reply

# -------------------------
# MAIN CHAT AREA
# -------------------------
with col_left:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    st.markdown("---")
    st.markdown("**Send a message** (text or voice `.wav`/`.mp3`):")
    
    cols = st.columns([4,1,1])
    user_text = cols[0].text_input("Type message...", key="user_input")
    voice_file = cols[1].file_uploader("🎤 Voice", type=["wav","mp3"])
    send_btn = cols[2].button("Send")

    # Image generator
    if st.session_state.mode == "Image Mode":
        with st.expander("🎨 Image generation"):
            img_prompt = st.text_area("Describe image", value="", height=80)
            if st.button("Generate Image"):
                with st.spinner("Generating image..."):
                    img_bytes = hf_generate_image(img_prompt)
                    if img_bytes:
                        st.image(img_bytes)
                        st.success("Image generated ✅")
                    else:
                        st.error("Image generation failed.")

    user_message_final = None
    if voice_file:
        recognized = recognize_speech_from_file(voice_file)
        if recognized:
            user_message_final = recognized
            st.success(f"Recognized: {recognized}")
        else:
            st.error("Could not recognize voice.")
    elif send_btn and user_text.strip():
        user_message_final = user_text.strip()

    if user_message_final:
        st.session_state.history.append({"role":"user","content":user_message_final})
        st.chat_message("user").markdown(user_message_final)

        # Optional quick wiki
        quick_fact = None
        if ENABLE_WIKI and re.search(r"\b(who is|what is|when is|where is)\b", user_message_final, re.IGNORECASE):
            quick_fact = wiki_lookup(user_message_final)

        with st.chat_message("assistant"):
            with st.spinner("Specimen King AI is thinking..."):
                # Storyline or Deep Search
                if st.session_state.mode == "Storyline":
                    ai_reply = f"📖 Storyline mode: {user_message_final}... Let me craft an epic continuation."
                elif st.session_state.mode == "Deep Search":
                    ai_reply = f"🔎 Deep Search mode: Searching deep insights for: {user_message_final}"
                else:
                    ai_reply = dynamic_reply(user_message_final)

                if quick_fact:
                    ai_reply = quick_fact + "\n\n(Quick summary provided.)"

                ai_reply = clean_text(ai_reply)
                st.markdown(ai_reply)
                st.session_state.history.append({"role":"assistant","content":ai_reply})

                # TTS
                if ENABLE_VOICE:
                    audio_bytes = text_to_speech_bytes(ai_reply)
                    if audio_bytes:
                        st.audio(audio_bytes)

# -------------------------
# END OF FILE
# -------------------------
