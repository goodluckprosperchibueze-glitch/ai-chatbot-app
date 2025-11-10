# pop_ultra_v5_1.py
"""
pop King Ultra AI v5.1 – Mega Edition

Streamlit app: Text generation (Flan-T5), optional Hugging Face image gen,
Wikipedia quick lookup, TTS/STT, multiple AI modes, persona controls,
memory, spicy modes (Poetry, Advice, Meme Generator), safe prompt handling.
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

# Optional audio libraries
try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

# Ensure UTF-8 output in all environments
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -------------------------
# CONFIG / SECRETS
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
GENAI_API_KEY = st.secrets.get("GENAI_API_KEY", os.environ.get("GENAI_API_KEY", ""))

# Default models
DEFAULT_MODEL_NAME = "google/flan-t5-large"
FALLBACK_MODEL_NAME = "google/flan-t5-small"

# Feature toggles defaults
ENABLE_VOICE_DEFAULT = True
ENABLE_IMAGES_DEFAULT = True
ENABLE_WIKI_DEFAULT = True

# -------------------------
# PAGE / THEME SETUP
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI v5.1", layout="wide", page_icon="👑")
st.markdown("""
<style>
.stApp { background: linear-gradient(#0f0f0f, #1a1a1a); color: #EDE0C8; }
.msg_user { background: #1f2937; padding: 10px; border-radius: 10px; color: #fff; margin-bottom: 10px; }
.msg_assistant { background: #111827; padding: 10px; border-radius: 10px; color: #ffd700; border-left: 3px solid #ffd700; margin-bottom: 10px; }
h3, h4 { color: #EDE0C8; }
</style>
""", unsafe_allow_html=True)

st.title("👑 Specimen King Ultra AI v5.1")
st.caption("Voice • Images • Memory • Knowledge • Spicy Modes — Streamlit + Transformers")

# -------------------------
# LAYOUT: left main / right settings
# -------------------------
col_left, col_right = st.columns([2.6, 1])

# -------------------------
# HELPERS / UTILITIES
# -------------------------
def safe_clean(text: str) -> str:
    """Basic cleaning: strip, remove odd control chars, trim repeated labels."""
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
        try:
            pipe = pipeline("text2text-generation", model=FALLBACK_MODEL_NAME)
            return pipe, FALLBACK_MODEL_NAME
        except Exception as e2:
            raise RuntimeError(f"Failed to load both models: {e2}")

# -------------------------
# Sidebar / Right Column: controls & settings
# -------------------------
with col_right:
    st.markdown("### ⚙️ Controls & Settings")

    mode = st.selectbox(
        "Mode",
        ["Chat", "Story Mode", "Deep Search", "Gemini Placeholder", "Poetry Master", "Advice Guru", "Meme Generator"],
        index=0,
    )

    model_choice = st.selectbox(
        "Model (text generation)",
        options=[DEFAULT_MODEL_NAME, FALLBACK_MODEL_NAME],
        index=0,
    )

    temp = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9, step=0.05)
    max_len = st.slider("Max tokens", 32, 1024, 256, step=16)

    enable_voice = st.checkbox("Enable Voice (TTS/STT)", value=ENABLE_VOICE_DEFAULT)
    enable_images = st.checkbox("Enable Image Gen (Hugging Face)", value=ENABLE_IMAGES_DEFAULT)
    enable_wiki = st.checkbox("Enable Wikipedia lookup", value=ENABLE_WIKI_DEFAULT)
    st.markdown("---")
    st.markdown("#### Persona & Memory")

    if "persona" not in st.session_state:
        st.session_state.persona = (
            "You are PopKing AI, a friendly, concise, and playful assistant. "
            "You were built by **Osemeke Goodluck, Specimen King 👑**."
        )

    persona_text = st.text_area("AI Persona", value=st.session_state.persona, height=140)
    if st.button("Save Persona"):
        st.session_state.persona = persona_text
        st.success("Persona saved.")

    st.markdown("---")
    st.markdown("#### API Tokens")
    hf_token_input = st.text_input("Hugging Face token (for images)", type="password", value=HF_TOKEN or "")
    if hf_token_input:
        HF_TOKEN = hf_token_input
        st.success("HF token set for this session (not persisted).")

    genai_input = st.text_input("Google GenAI key (optional)", type="password", value=GENAI_API_KEY or "")
    if genai_input:
        GENAI_API_KEY = genai_input
        st.success("GenAI key set for this session (not persisted).")

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
        st.error("Model failed to load — check internet connection.")
        st.stop()

# -------------------------
# Session state
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "settings" not in st.session_state:
    st.session_state.settings = {}

st.session_state.settings.update(
    {
        "temp": temp,
        "top_p": top_p,
        "max_len": max_len,
        "enable_voice": enable_voice,
        "enable_images": enable_images,
        "enable_wiki": enable_wiki,
        "mode": mode,
    }
)

# -------------------------
# Prompt building & generation
# -------------------------
def build_prompt(persona: str, history, user_message: str, max_exchanges: int = 6) -> str:
    trimmed = history[-max_exchanges * 2 :] if history else []
    convo_lines = [f"AI Persona Instructions: {persona}", "\n--- END OF INSTRUCTIONS ---\n", "Conversation so far:"]
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

def wiki_lookup(query: str, sentences: int = 2) -> Optional[str]:
    if not st.session_state.settings.get("enable_wiki", True):
        return None
    try:
        return wikipedia.summary(query, sentences=sentences)
    except Exception:
        return None

def hf_generate_image_bytes(prompt_text: str, hf_token: str) -> bytes:
    if not hf_token:
        raise RuntimeError("HF token missing.")
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt_text}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 200:
        return resp.content
    raise RuntimeError(f"Image generation failed: {resp.status_code} {resp.text}")

def text_to_speech_bytes(text: str, lang: str = "en") -> Optional[bytes]:
    if gTTS is None:
        return None
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        with open(tmp.name, "rb") as f:
            data = f.read()
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        return data
    except Exception:
        return None

def recognize_speech_from_file(uploaded_file) -> Optional[str]:
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
        except Exception:
            pass

# -------------------------
# Generate response dispatch
# -------------------------
def generate_response(user_message: str, temperature: float, top_p_val: float, max_length_val: int) -> str:
    persona = st.session_state.get("persona", "")
    current_mode = st.session_state.settings.get("mode", "Chat")
    
    if current_mode == "Gemini Placeholder":
        return "(Gemini mode placeholder — not implemented.)"
    elif current_mode == "Story Mode":
        prompt = f"{persona} - Write a long, creative story based on: {user_message}"
        return generate_response_with_transformers(prompt, temperature, top_p_val, max(max_length_val, 512))
    elif current_mode == "Deep Search":
        prompt = f"{persona} - Provide factual answer, steps or bullets if possible. Question: {user_message}"
        return generate_response_with_transformers(prompt, temperature, top_p_val, max_length_val)
    elif current_mode == "Poetry Master":
        prompt = f"{persona} - Write a poetic, rhythmic, and emotional poem based on: {user_message}"
        return generate_response_with_transformers(prompt, temperature, top_p_val, max_length_val*2)
    elif current_mode == "Advice Guru":
        prompt = f"{persona} - Give life advice, tips, or motivational guidance for: {user_message}"
        return generate_response_with_transformers(prompt, temperature, top_p_val, max_length_val)
    elif current_mode == "Meme Generator":
        prompt = f"{persona} - Generate a funny meme caption for: {user_message}"
        return generate_response_with_transformers(prompt, temperature, top_p_val, 64)
    else:  # default Chat
        prompt = build_prompt(persona, st.session_state.history, user_message)
        return generate_response_with_transformers(prompt, temperature, top_p_val, max_length_val)

# -------------------------
# MAIN CHAT UI (left column)
# -------------------------
with col_left:
    st.markdown("### Chat with PopKing AI")
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
    user_text = cols[0].text_input("Type message here...", key="user_input", label_visibility="collapsed")
    voice_file = cols[1].file_uploader("Upload voice", type=["wav","mp3"], key="voice_upload", label_visibility="collapsed")
    send_btn = cols[2].button("Send")

    # Image generation expander
    with st.expander("🖼️ Image generation"):
        img_prompt = st.text_area("Describe the image", value="", height=80, key="img_prompt")
        img_style = st.selectbox("Style", ["photorealistic","digital art","anime","cartoon","fantasy"], index=0)
        if st.button("Generate Image", key="gen_img_btn"):
            if not HF_TOKEN:
                st.error("HF token required")
            else:
                with st.spinner("Generating image..."):
                    try:
                        img_bytes = hf_generate_image_bytes(f"{img_prompt} -- style: {img_style}", HF_TOKEN)
                        st.image(img_bytes)
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")

    # Wikipedia quick search
    with st.expander("🔎 Knowledge / Quick search"):
        search_q = st.text_input("Ask something to lookup", value="", key="wiki_q")
        if st.button("Lookup Wikipedia", key="wiki_lookup_btn") and search_q.strip():
            with st.spinner("Searching Wikipedia..."):
                summ = wiki_lookup(search_q, sentences=3)
                if summ:
                    st.markdown(f"**Wikipedia summary:**\n\n{safe_clean(summ)}")
                else:
                    st.info("No summary found.")

    # -------------------------
    # HANDLE sending message
    # -------------------------
    user_message_final = None
    if voice_file:
        if sr is None:
            st.warning("Speech recognition not installed; type instead.")
        else:
            with st.spinner("Recognizing speech..."):
                recognized = recognize_speech_from_file(voice_file)
                if recognized:
                    user_message_final = recognized
                    st.success(f"Recognized: {recognized}")
                else:
                    st.error("Could not recognize speech.")
    elif send_btn and user_text.strip():
        user_message_final = user_text.strip()

    if user_message_final:
        st.session_state.history.append({"role":"user","content":user_message_final})
        quick_fact = None
        if enable_wiki and re.search(r"\b(who is|what is|when is|where is|tell me about)\b", user_message_final, re.IGNORECASE):
            quick_fact = wiki_lookup(user_message_final, sentences=2)

        with st.spinner("PopKing AI is thinking..."):
            try:
                if quick_fact and st.session_state.settings.get("mode") == "Chat":
                    ai_reply = quick_fact + "\n\n(Quick Wikipedia summary.)"
                else:
                    ai_reply = generate_response(user_message_final, temp, top_p, max_len)
                ai_reply = safe_clean(ai_reply) or "Sorry, I couldn't produce an answer."

                st.session_state.history.append({"role":"assistant","role_label":"PopKing AI","content":ai_reply})

                if enable_voice and gTTS:
                    audio_bytes = text_to_speech_bytes(ai_reply)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

                st.experimental_rerun()
            except Exception as e:
                st.error(f"AI generation error: {e}")

# -------------------------
# FOOTER / ABOUT
# -------------------------
st.markdown("---")
st.markdown("Built by Osemeke Goodluck (Specimen King 👑) — powered by Hugging Face models and Streamlit. Customize persona and model settings on the right.")
