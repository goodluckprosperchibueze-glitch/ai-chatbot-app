# specimen_ultra_v2.py
"""
Specimen King Ultra AI (v5.1)
- Streamlit app: text generation (Flan-T5), optional Hugging Face image gen,
  optional Wikipedia quick lookup, optional gTTS TTS and speech_recognition STT,
  persona controls, safer prompt building, and multiple modes.
"""

import os
import re
import sys
import json
import base64
import tempfile
from typing import Optional

import streamlit as st
from transformers import pipeline
import wikipedia
import requests

# Optional audio libraries (gTTS for TTS; speech_recognition for STT)
try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

# Ensure UTF-8 output in some environments
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -------------------------
# CONFIG / ENV / SECRETS
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
GENAI_API_KEY = st.secrets.get("GENAI_API_KEY", os.environ.get("GENAI_API_KEY", ""))

# Default model settings
DEFAULT_MODEL_NAME = "google/flan-t5-large"
FALLBACK_MODEL_NAME = "google/flan-t5-small"

# Feature toggles defaults
ENABLE_VOICE_DEFAULT = True
ENABLE_IMAGES_DEFAULT = True
ENABLE_WIKI_DEFAULT = True

# -------------------------
# PAGE / THEME SETUP
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI", layout="wide", page_icon="👑")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(#000000, #0a0a0a); color: #EDE0C8; }
    .msg_user { background: #1f2937; padding: 8px; border-radius: 8px; color: #fff; }
    .msg_assistant { background: #111827; padding: 8px; border-radius: 8px; color: #ffd700; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("👑 Specimen King Ultra AI (v5.1)")
st.caption("Voice • Images • Memory • Knowledge — Streamlit + Transformers")

# -------------------------
# LAYOUT: left main / right settings
# -------------------------
col_left, col_right = st.columns([2.6, 1])

# -------------------------
# HELPERS / UTILITIES
# -------------------------
def safe_clean(text: str) -> str:
    """Basic cleaning: strip, remove odd control chars and huge codepoints, trim repeated labels."""
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
# MODEL LOADING (with fallback)
# -------------------------
@st.cache_resource
def load_text_model(model_name: str):
    """Load text2text-generation pipeline safely. If heavy model fails, try fallback."""
    try:
        pipe = pipeline("text2text-generation", model=model_name)
        return pipe, model_name
    except Exception as e:
        try:
            pipe = pipeline("text2text-generation", model=FALLBACK_MODEL_NAME)
            return pipe, FALLBACK_MODEL_NAME
        except Exception as e2:
            raise RuntimeError(f"Failed to load both '{model_name}' and fallback '{FALLBACK_MODEL_NAME}'. Errors: {e} | {e2}")


# -------------------------
# Sidebar / Right Column: controls & settings
# -------------------------
with col_right:
    st.markdown("### ⚙️ Controls & Settings")

    # Mode selection
    mode = st.selectbox("Mode", ["Chat (default)", "Gemini (placeholder)", "Story Mode", "Deep Search"], index=0)

    # Model choice
    model_choice = st.selectbox(
        "Model (text generation)", options=[DEFAULT_MODEL_NAME, FALLBACK_MODEL_NAME], index=0
    )

    # Runtime parameter controls
    temp = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9, step=0.05)
    max_len = st.slider("Max tokens (approx)", 32, 512, 256, step=16)

    # Feature toggles
    enable_voice = st.checkbox("Enable Voice (TTS/STT)", value=ENABLE_VOICE_DEFAULT)
    enable_images = st.checkbox("Enable Image Gen (Hugging Face)", value=ENABLE_IMAGES_DEFAULT)
    enable_wiki = st.checkbox("Enable Wikipedia lookup", value=ENABLE_WIKI_DEFAULT)

    st.markdown("---")
    st.markdown("#### Persona & Memory")

    if "persona" not in st.session_state:
        st.session_state.persona = (
            "You are PopKing AI (friendly, concise, slightly playful). Avoid hallucination, cite facts, and provide clear steps."
        )
    persona_text = st.text_area("AI Persona", value=st.session_state.persona, height=120)
    if st.button("Save Persona"):
        st.session_state.persona = persona_text
        st.success("Persona saved.")

    st.markdown("---")
    st.markdown("#### API Tokens")
    hf_token_input = st.text_input("Hugging Face token (for images)", type="password", value=HF_TOKEN or "")
    if hf_token_input:
        HF_TOKEN = hf_token_input
        st.success("HF token set for this session (not persisted).")

    genai_input = st.text_input("Google GenAI key (optional, for Gemini mode)", type="password", value=GENAI_API_KEY or "")
    if genai_input:
        GENAI_API_KEY = genai_input
        st.success("GenAI key set for this session (not persisted).")

    st.markdown("---")
    if st.button("Clear Chat & Memory"):
        st.session_state.history = []
        st.success("Chat history cleared.")
        st.experimental_rerun()


# -------------------------
# Load model using selected name (cached)
# -------------------------
with st.spinner("Loading text model..."):
    try:
        model_pipeline, loaded_model_name = load_text_model(model_choice)
        st.sidebar.success(f"Model loaded: {loaded_model_name}")
    except Exception as e:
        st.sidebar.error(f"Model load failed: {e}")
        st.error("Model failed to load — check logs and internet connection.")
        st.stop()

# -------------------------
# Session state: chat memory
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts {"role": "user"/"assistant", "content": "..."}

if "settings" not in st.session_state:
    st.session_state.settings = {
        "temp": temp,
        "top_p": top_p,
        "max_len": max_len,
        "enable_voice": enable_voice,
        "enable_images": enable_images,
        "enable_wiki": enable_wiki,
        "mode": mode,
    }

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
# PROMPT BUILDING
# -------------------------
def build_prompt(persona: str, history, user_message: str, mode: str, max_exchanges: int = 6) -> str:
    """Build prompt including persona, conversation history, and mode instructions."""
    trimmed = history[-max_exchanges*2:] if history else []
    convo_lines = []
    for m in trimmed:
        label = "User" if m["role"] == "user" else "AI"
        cleaned = safe_clean(m["content"])
        convo_lines.append(f"{label}: {cleaned}")
    convo_text = "\n".join(convo_lines)

    base_prompt = f"{persona}\n\nConversation so far:\n{convo_text}\nUser: {safe_clean(user_message)}\nAI:"

    # Mode-specific instructions
    if mode == "Story Mode":
        base_prompt = f"{persona}\n\nWrite a long, imaginative story based on the conversation so far and user input:\n{convo_text}\nUser: {user_message}\nStart the story now."
    elif mode == "Deep Search":
        base_prompt = f"{persona}\n\nProvide a careful, stepwise, in-depth answer based on conversation history and user input:\n{convo_text}\nUser: {user_message}\nInclude verification steps and suggested search keywords."
    elif mode == "Gemini (placeholder)":
        base_prompt = user_message  # raw pass-through

    return base_prompt


def generate_response_with_transformers(prompt: str, temperature: float, top_p_val: float, max_length_val: int) -> str:
    """Call HF transformers pipeline safely."""
    try:
        out = model_pipeline(
            prompt, max_length=max_length_val, do_sample=True, temperature=temperature, top_p=top_p_val, num_return_sequences=1
        )
    except TypeError:
        out = model_pipeline(prompt, max_length=max_length_val)
    raw = out[0].get("generated_text", "") if isinstance(out, list) else out.get("generated_text", "")
    return safe_clean(raw)


def call_gemini_api(prompt: str) -> str:
    """Placeholder for Gemini."""
    if not GENAI_API_KEY:
        return "(Gemini mode selected but GENAI_API_KEY not set.)"
    return "(Gemini mode placeholder — configure GENAI_API_KEY and implement call_gemini_api to use actual Gemini responses.)"


def generate_response(user_message: str, temperature: float, top_p_val: float, max_length_val: int) -> str:
    """Dispatch by mode and build appropriate prompt/behavior."""
    persona = st.session_state.get("persona", "")
    current_mode = st.session_state.settings.get("mode", "Chat (default)")
    prompt = build_prompt(persona, st.session_state.history, user_message, current_mode)

    if current_mode == "Gemini (placeholder)":
        return call_gemini_api(prompt)
    else:
        return generate_response_with_transformers(prompt, temperature, top_p_val, max_length_val)


# -------------------------
# Wikipedia quick lookup
# -------------------------
def wiki_lookup(query: str, sentences: int = 2) -> Optional[str]:
    if not st.session_state.settings.get("enable_wiki", True):
        return None
    try:
        return wikipedia.summary(query, sentences=sentences)
    except Exception:
        return None


# -------------------------
# Hugging Face image generation helper
# -------------------------
def hf_generate_image_bytes(prompt_text: str, hf_token: str) -> bytes:
    if not hf_token:
        raise RuntimeError("Hugging Face token missing. Set it in the right panel.")
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt_text}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 200:
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                data = resp.json()
                if isinstance(data, dict) and "images" in data:
                    b64 = data["images"][0]
                    return base64.b64decode(b64)
            except Exception as e:
                raise RuntimeError(f"Unexpected JSON image response: {e}")
        return resp.content
    raise RuntimeError(f"Image generation failed: {resp.status_code} {resp.text}")


# -------------------------
# TTS & STT wrappers
# -------------------------
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
        except:
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
        except:
            pass


# -------------------------
# MAIN CHAT UI (left column)
# -------------------------
with col_left:
    st.markdown("### Chat with PopKing AI")
    st.markdown("**Tips:** Use modes to change behavior. Wikipedia lookup triggers on 'who is', 'what is', etc.")

    # Display history
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"<div class='msg_user'>**You:** {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='msg_assistant'>**{msg.get('role_label','PopKing AI')}:** {msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Send a message** (type or upload short audio):")

    cols = st.columns([4, 1, 1])
    user_text = cols[0].text_input("Type message here...", key="user_input")
    voice_file = cols[1].file_uploader("Upload voice (wav/mp3, optional)", type=["wav", "mp3"], key="voice_upload")
    send_btn = cols[2].button("Send")

    # Image generation expander
    with st.expander("🖼️ Image generation (optional)"):
        img_prompt = st.text_area("Describe the image you want", value="", height=80, key="img_prompt")
        img_style = st.selectbox("Style (suggestion)", ["photorealistic", "digital art", "anime", "cartoon", "fantasy"], index=0)
        if st.button("Generate Image", key="gen_img_btn"):
            if not HF_TOKEN:
                st.error("Image generation needs a Hugging Face token. Set it in the right panel.")
            else:
                with st.spinner("Generating image..."):
                    try:
                        full_prompt = f"{img_prompt} -- style: {img_style}"
                        img_bytes = hf_generate_image_bytes(full_prompt, HF_TOKEN)
                        st.image(img_bytes)
                        st.success("Image generated.")
                    except Exception as e:
                        st.error(f"Image gen failed: {e}")

    # Wikipedia quick lookup
    with st.expander("🔎 Knowledge / Quick search (Wikipedia)"):
        search_q = st.text_input("Ask something to lookup (Wikipedia)", value="", key="wiki_q")
        if st.button("Lookup Wikipedia", key="wiki_lookup_btn"):
            if search_q.strip():
                with st.spinner("Searching Wikipedia..."):
                    summ = wiki_lookup(search_q, sentences=3)
                    if summ:
                        st.markdown(f"**Wikipedia summary:**\n\n{safe_clean(summ)}")
                    else:
                        st.info("No summary found. Try rephrasing or use fewer words.")

    # -------------------------
    # Handle sending message
    # -------------------------
    user_message_final = None
    if voice_file is not None:
        if sr is None:
            st.warning("Speech recognition not installed; please type or install `speechrecognition`.")
        else:
            with st.spinner("Recognizing speech..."):
                recognized = recognize_speech_from_file(voice_file)
                if recognized:
                    user_message_final = recognized
                    st.success(f"Recognized: {recognized}")
                else:
                    st.error("Could not recognize speech. Try clearer audio or type the message.")
    elif send_btn and user_text and user_text.strip():
        user_message_final = user_text.strip()

    if user_message_final:
        st.session_state.history.append({"role": "user", "content": user_message_final})
        st.markdown(f"<div class='msg_user'>**You:** {user_message_final}</div>", unsafe_allow_html=True)

        quick_fact = None
        if enable_wiki and re.search(r"\b(who is|what is|when is|where is|tell me about)\b", user_message_final, re.IGNORECASE):
            quick_fact = wiki_lookup(user_message_final, sentences=2)

        with st.spinner("PopKing AI is thinking..."):
            try:
                if quick_fact:
                    ai_reply = quick_fact + "\n\n(Quick knowledge summary — verify with source.)"
                else:
                    ai_reply = generate_response(user_message_final, temperature=temp, top_p_val=top_p, max_length_val=max_len)

                ai_reply = safe_clean(ai_reply) or "Sorry, I couldn't produce an answer. Try rephrasing."
                st.markdown(f"<div class='msg_assistant'>**PopKing AI:** {ai_reply}</div>", unsafe_allow_html=True)
                st.session_state.history.append({"role": "assistant", "role_label": "PopKing AI", "content": ai_reply})

                if enable_voice and gTTS is not None:
                    audio_bytes = text_to_speech_bytes(ai_reply)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                st.error(f"AI generation error: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(f"<div class='msg_assistant'>**PopKing AI:** {ai_reply or 'Sorry, I could not generate a reply.'}</div>", unsafe_allow_html=True)
