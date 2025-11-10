# specimen_ultra.py
"""
Specimen King Ultra AI (Gemini-first) — Streamlit app

Features:
- Primary engine: Google Gemini via google-generativeai (if configured)
- Fallback engine: Hugging Face text2text (flan-t5-large) via transformers
- Persona/modes editing, memory with summarization, TTS (gTTS), STT (SpeechRecognition)
- Wikipedia quick lookup and Hugging Face image generation (optional)
- Safe secret handling: read GOOGLE_API_KEY / HF_TOKEN from Streamlit secrets or environment
- Polished assistant persona (Specimen King = "boss" tone)
"""

import os
import re
import sys
import time
import json
import tempfile
import datetime
import random
import requests
from typing import Optional, Tuple

import streamlit as st
import wikipedia

# transformers fallback
from transformers import pipeline

# Optional audio libs
try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

# Gemini SDK (optional)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Ensure UTF-8 output where possible
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -------------------------
# CONFIG / SECRETS (SET THESE SECURELY)
# -------------------------
# Recommended: add to .streamlit/secrets.toml:
# GOOGLE_API_KEY = "your_key_here"
# HF_TOKEN = "hf_xxx"

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", None))
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))

# Feature toggles
ENABLE_VOICE = True      # TTS (gTTS) - optional
ENABLE_STT = True        # Speech-to-text - optional
ENABLE_IMAGES = True     # HF image gen - optional (HF_TOKEN required)
ENABLE_WIKI = True       # Wikipedia lookup
MAX_HISTORY_EXCHANGES = 12  # number of recent exchanges to keep (user+assistant pairs)

# -------------------------
# Configure Gemini if available and key present
# -------------------------
GEMINI_AVAILABLE = False
if genai is not None and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        GEMINI_AVAILABLE = True
    except Exception:
        GEMINI_AVAILABLE = False

# -------------------------
# Page UI Setup
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI (Gemini)", layout="wide")
st.title("👑 Specimen King Ultra AI — (Gemini-first)")
st.caption("Specimen King is your bossy, confident, helpful AI — Voice • Images • Memory • Knowledge")

left_col, right_col = st.columns([2, 1])

# -------------------------
# Session state (initialize)
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts {"role":"user"/"assistant"/"system","content":str,"ts":...}

if "persona" not in st.session_state:
    # The default 'boss' persona (you can edit via UI)
    st.session_state.persona = (
        "You are Specimen King AI — a confident, composed, and slightly playful leader. "
        "Speak with authority and clarity. Use short, polished sentences. "
        "Be helpful, direct, and occasionally witty; show empathy when appropriate. "
        "Keep a slight mysterious edge."
    )

if "mode" not in st.session_state:
    st.session_state.mode = "Default"

if "summary_cache" not in st.session_state:
    st.session_state.summary_cache = ""  # summarized older history when trimmed

if "hf_token_session" not in st.session_state:
    st.session_state.hf_token_session = HF_TOKEN

# -------------------------
# Right column: Controls, Persona, Utilities
# -------------------------
with right_col:
    st.markdown("### Controls & Settings")

    st.markdown("**Persona (edit, then Save Persona)**")
    persona_input = st.text_area("AI Persona (tone & system instructions)",
                                 value=st.session_state.persona, height=160)
    if st.button("Save Persona"):
        st.session_state.persona = persona_input
        st.success("Persona saved ✅")

    st.markdown("---")
    st.markdown("**AI Mode** (changes flavor of replies)")
    mode_choice = st.selectbox("Mode", ["Default", "Dark/Psycho", "Flirty/Casual", "Mentor", "Storyteller"])
    st.session_state.mode = mode_choice

    st.markdown("---")
    st.markdown("**Optional Keys** (enter below to set for this session only)")
    input_hf = st.text_input("Hugging Face Token (for images)", type="password", value=st.session_state.hf_token_session or "")
    if input_hf:
        st.session_state.hf_token_session = input_hf
        st.success("HF token set for this session (not persisted).")

    st.markdown("---")
    st.markdown("**Utilities**")
    if st.button("🧹 Clear chat history"):
        st.session_state.history = []
        st.session_state.summary_cache = ""
        st.success("Chat cleared.")
        st.experimental_rerun()

    # Download chat
    chat_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.history])
    st.download_button("💾 Download Chat", data=chat_text, file_name="specimen_king_chat.txt", mime="text/plain")

    st.markdown("---")
    st.markdown("**Backend status**")
    if GEMINI_AVAILABLE:
        st.success("Gemini: available")
    else:
        if genai is None:
            st.warning("Gemini SDK not installed (google-generativeai). HF fallback will be used.")
        elif not GOOGLE_API_KEY:
            st.warning("Gemini key not set (add GOOGLE_API_KEY in Streamlit secrets or env). HF fallback will be used.")
        else:
            st.warning("Gemini configured but unavailable; HF fallback will be used.")

# -------------------------
# Loading HF fallback model (cached)
# -------------------------
@st.cache_resource
def load_hf_model(model_name="google/flan-t5-large"):
    """Load HF text2text pipeline. If GPU available, transformers will try to use it."""
    return pipeline("text2text-generation", model=model_name)

hf_model = None
try:
    hf_model = load_hf_model()
except Exception:
    hf_model = None  # we'll handle unavailability later

# -------------------------
# Utility functions
# -------------------------
def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def clean_ai_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^(AI:|Assistant:|User:)\s*", "", t, flags=re.IGNORECASE)
    return t

def summarize_old_history():
    """
    If history gets large, summarize older messages into summary_cache.
    Keep only last MAX_HISTORY_EXCHANGES exchanges (user+assistant pairs).
    """
    # Count exchanges (user + assistant pairs)
    msgs = st.session_state.history
    # Each exchange roughly = 2 messages. We'll keep last MAX_HISTORY_EXCHANGES*2 messages
    keep_msg_count = MAX_HISTORY_EXCHANGES * 2
    if len(msgs) <= keep_msg_count:
        return
    older = msgs[:-keep_msg_count]
    recent = msgs[-keep_msg_count:]
    # Create a digest string to summarize
    older_text = "\n".join([f"{m['role']}: {m['content']}" for m in older])
    summary = None
    # Try to use hf_model to summarize, else naive truncate
    try:
        if hf_model is not None:
            prompt = "Summarize the following conversation briefly in 3-4 sentences:\n\n" + older_text + "\n\nSummary:"
            out = hf_model(prompt, max_length=200)
            summary = out[0].get("generated_text","").strip() if isinstance(out, list) else str(out)
        else:
            summary = (older_text[:400] + "...") if len(older_text) > 400 else older_text
    except Exception:
        summary = (older_text[:400] + "...") if len(older_text) > 400 else older_text
    # store summary and replace older messages with a single system note
    st.session_state.summary_cache = summary
    st.session_state.history = [{"role":"system","content": f"(SUMMARY) {summary}"}] + recent

# -------------------------
# Gemini helpers
# -------------------------
def to_gemini_history():
    hist = []
    for m in st.session_state.history:
        # Convert to gemini roles: user / model / system
        if m["role"] == "user":
            hist.append({"role":"user","parts":[m["content"]]})
        elif m["role"] == "assistant":
            hist.append({"role":"model","parts":[m["content"]]})
        else:
            hist.append({"role":"system","parts":[m["content"]]})
    return hist

def generate_with_gemini(user_message: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (reply, error) using Gemini; error is None if successful"""
    if not GEMINI_AVAILABLE:
        return None, "Gemini unavailable"
    try:
        history = to_gemini_history()
        model_obj = genai.GenerativeModel("gemini-1.5-flash")
        chat = model_obj.start_chat(history=history)
        # Attach persona & mode as system-level context by sending them in the prompt
        system_prefix = st.session_state.persona + f"\nMode: {st.session_state.mode}\n"
        prompt = f"{system_prefix}\nUser: {user_message}\nAI:"
        response = chat.send_message(prompt)
        text = response.text if hasattr(response, "text") else str(response)
        return clean_ai_text(text), None
    except Exception as e:
        return None, str(e)

# -------------------------
# HF fallback helper
# -------------------------
def generate_with_hf(user_message: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (reply, error) using HF Flan-T5 fallback"""
    if hf_model is None:
        return None, "HF model unavailable"
    try:
        # Build prompt: include summary_cache, persona, last exchanges
        prompt_parts = []
        if st.session_state.summary_cache:
            prompt_parts.append(f"(Earlier summary) {st.session_state.summary_cache}")
        prompt_parts.append(st.session_state.persona)
        prompt_parts.append(f"Mode: {st.session_state.mode}")
        # append last few messages
        last_msgs = st.session_state.history[-(MAX_HISTORY_EXCHANGES*2):]
        for m in last_msgs:
            role_label = "User" if m["role"]=="user" else ("Assistant" if m["role"]=="assistant" else "System")
            prompt_parts.append(f"{role_label}: {m['content']}")
        prompt_parts.append(f"User: {user_message}\nAssistant:")
        full_prompt = "\n".join(prompt_parts)
        out = hf_model(full_prompt, max_length=256)
        # out could be list of dicts
        if isinstance(out, list) and out and isinstance(out[0], dict):
            raw = out[0].get("generated_text","")
        else:
            raw = str(out)
        return clean_ai_text(raw), None
    except Exception as e:
        return None, str(e)

# -------------------------
# Top-level generate function
# -------------------------
def generate_response(user_message: str) -> Tuple[str, Optional[str]]:
    # Trim & summarize if needed
    summarize_old_history()
    # Try Gemini first
    if GEMINI_AVAILABLE:
        reply, err = generate_with_gemini(user_message)
        if reply:
            return reply, None
        # else log and try HF
        gem_err = err
    else:
        gem_err = "Gemini not enabled"
    # HF fallback
    reply, hf_err = generate_with_hf(user_message)
    if reply:
        return reply, None
    # Ultimate fallback
    fallback = "I cannot generate a reply right now — backend unavailable. Try again later."
    return fallback, f"Gemini: {gem_err}; HF: {hf_err}"

# -------------------------
# Wikipedia helper
# -------------------------
def wiki_lookup_safe(q: str, sentences:int=2) -> Optional[str]:
    if not ENABLE_WIKI:
        return None
    try:
        return wikipedia.summary(q, sentences=sentences)
    except Exception:
        return None

# -------------------------
# Image generation helper
# -------------------------
def hf_generate_image(prompt_text: str, hf_token: Optional[str] = None) -> bytes:
    token = hf_token or st.session_state.hf_token_session or HF_TOKEN
    if not token:
        raise RuntimeError("No HF token provided")
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt_text}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HF image error: {resp.status_code} {resp.text}")
    return resp.content

# -------------------------
# TTS / STT helpers
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
# Fun small features
# -------------------------
JOKES = [
    "Why did the AI cross the road? To optimize the chicken's path.",
    "I told my laptop a joke — it responded with a solid state laugh.",
    "Why do programmers prefer dark mode? Because light attracts bugs."
]
STORIES = [
    "Once, a tiny process learned to whistle while it compiled — and everyone listened.",
    "A dataset wrote a poem. An engineer cried. The model learned empathy.",
    "In the server room, a log file became a song and humans called it music."
]

def random_joke():
    return random.choice(JOKES)

def random_story():
    return random.choice(STORIES)

# -------------------------
# UI: Main conversation panel (left)
# -------------------------
with left_col:
    st.markdown("### Conversation")
    # render history
    for m in st.session_state.history:
        if m["role"] == "user":
            st.chat_message("user").markdown(m["content"])
        elif m["role"] == "assistant":
            st.chat_message("assistant").markdown(m["content"])
        else:
            # system messages as info
            st.info(m["content"])

    st.markdown("---")
    st.markdown("**Send a message** — type or upload a short voice file (.wav / .mp3)")

    cols = st.columns([4, 1, 1])
    user_text = cols[0].text_input("Type your message here...", key="user_input")
    voice_file = cols[1].file_uploader("Upload voice (.wav/.mp3)", type=["wav","mp3"], key="voice_upload")
    send_btn = cols[2].button("Send")

    # Image generation expander
    with st.expander("🎨 Image generation (Hugging Face)"):
        img_prompt = st.text_area("Describe image...", height=100)
        if st.button("Generate Image"):
            try:
                if not ENABLE_IMAGES:
                    st.error("Image generation disabled in config.")
                elif not (st.session_state.hf_token_session or HF_TOKEN):
                    st.error("No HF token set — enter one in the right panel.")
                else:
                    with st.spinner("Generating image..."):
                        img_bytes = hf_generate_image(img_prompt)
                        st.image(img_bytes)
                        st.success("Image generated ✅")
            except Exception as e:
                st.error(f"Image generation failed: {e}")

    # Wiki quick search
    with st.expander("📘 Quick knowledge (Wikipedia)"):
        wiki_q = st.text_input("Ask Wikipedia about...", key="wiki_q")
        if st.button("Lookup Wikipedia", key="wiki_lookup_button"):
            if wiki_q.strip():
                with st.spinner("Searching Wikipedia..."):
                    summary = wiki_lookup_safe(wiki_q)
                    if summary:
                        st.markdown(f"**Wikipedia summary:**\n\n{summary}")
                    else:
                        st.info("No result found.")

    # Determine user message (voice has priority)
    user_message_final = None
    if voice_file is not None:
        if not ENABLE_STT or sr is None:
            st.warning("Speech recognition unavailable. Install SpeechRecognition or disable STT.")
        else:
            with st.spinner("Recognizing speech..."):
                recognized = recognize_speech_from_file(voice_file)
                if recognized:
                    user_message_final = recognized
                    st.success(f"Recognized: {recognized}")
                else:
                    st.error("Could not recognize speech from audio. Try a clearer recording.")
    elif send_btn and user_text and user_text.strip():
        user_message_final = user_text.strip()

    # If user sent message => process
    if user_message_final:
        # add user message to history
        st.session_state.history.append({"role":"user","content": user_message_final, "ts": now_iso()})
        st.chat_message("user").markdown(user_message_final)

        # quick wiki heuristic
        quick_fact = None
        if ENABLE_WIKI and re.search(r"\b(who is|what is|define|when is|where is)\b", user_message_final, re.IGNORECASE):
            quick_fact = wiki_lookup_safe(user_message_final)

        # triggers for jokes/stories
        low = user_message_final.lower()
        if "joke" in low or "tell me a joke" in low:
            reply_text = random_joke()
            err = None
        elif "story" in low or "tell me a story" in low:
            reply_text = random_story()
            err = None
        elif quick_fact:
            reply_text = quick_fact + "\n\n(Quick summary provided.)"
            err = None
        else:
            reply_text, err = generate_response(user_message_final)

        reply_text = clean_ai_text(reply_text) or "I have nothing to say right now."

        # modify reply based on mode to feel "alive"
        if st.session_state.mode == "Default":
            final_reply = reply_text
        elif st.session_state.mode == "Dark/Psycho":
            final_reply = "🖤 " + reply_text
        elif st.session_state.mode == "Flirty/Casual":
            final_reply = "😉 " + reply_text
        elif st.session_state.mode == "Mentor":
            final_reply = "🧭 " + reply_text
        elif st.session_state.mode == "Storyteller":
            final_reply = "📖 " + reply_text
        else:
            final_reply = reply_text

        # append assistant reply
        st.session_state.history.append({"role":"assistant","content": final_reply, "ts": now_iso()})

        with st.chat_message("assistant"):
            st.markdown(final_reply)

            # TTS playback
            if ENABLE_VOICE and gTTS is not None:
                audio_bytes = text_to_speech_bytes(final_reply)
                if audio_bytes:
                    st.audio(audio_bytes)

            # show debug if generation had error (helpful when configuring)
            if err:
                st.caption(f"DEBUG: generation error -> {err}")

# -------------------------
# END
# -------------------------
