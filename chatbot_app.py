# specimen_ultra.py
import streamlit as st
import os
import re
import sys
import tempfile
import requests
import wikipedia
from transformers import pipeline
import json
import random
import datetime

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
# CONFIG / SECRETS
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))

# Feature toggles
ENABLE_VOICE = True
ENABLE_IMAGES = True
ENABLE_WIKI = True
ENABLE_JOKES = True
ENABLE_STORIES = True

# -------------------------
# UI Setup
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI", layout="wide")
st.title("👑 Specimen King Ultra AI")
st.caption("Voice • Images • Memory • Knowledge • Fun — Powered by Transformers + Streamlit")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### Chat with Specimen King AI")
with col_right:
    st.markdown("### Controls / Settings")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model(model_name="google/flan-t5-large"):
    """Load a text2text model for conversational generation."""
    return pipeline("text2text-generation", model=model_name)

with st.spinner("Loading AI model (may take ~30s first run)..."):
    model = load_model()

# -------------------------
# Session State
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # [{"role":"user"/"assistant","content":...}]

if "persona" not in st.session_state:
    st.session_state.persona = (
        "You are Specimen King AI — a confident, intelligent, witty, and introspective digital being. "
        "Speak clearly, calmly, and thoughtfully. Use emojis sparingly. "
        "Always answer politely, sometimes humorously, and never repeat the user's exact words."
    )

if "mode" not in st.session_state:
    st.session_state.mode = "Default"

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

    st.markdown("---")
    st.write("🎭 AI Modes")
    mode = st.selectbox("Choose AI mode", ["Default", "Dark/Psycho", "Flirty/Casual", "Storyteller"])
    st.session_state.mode = mode

# -------------------------
# Helper Functions
# -------------------------
def clean_text(text):
    text = text.strip()
    text = re.sub(r"^(AI:|User:)\s*", "", text, flags=re.IGNORECASE)
    return text

def build_prompt(user_message):
    """Build prompt for AI using persona, mode, and chat history."""
    max_exchanges = 6
    history_text = ""
    trimmed = st.session_state.history[-max_exchanges*2:]
    for m in trimmed:
        role = "User" if m["role"]=="user" else "AI"
        history_text += f"{role}: {m['content']}\n"

    mode_note = f"Mode: {st.session_state.mode}\n"

    prompt = (
        f"{st.session_state.persona}\n{mode_note}\nConversation so far:\n{history_text}"
        f"User: {user_message}\nAI:"
    )
    return prompt

def generate_response(user_message, max_length=250):
    """Generate AI response."""
    prompt = build_prompt(user_message)
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
        raise RuntimeError("No HF token provided.")
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

def tell_joke():
    jokes = [
        "Why did the computer go to therapy? It had too many bytes of anxiety.",
        "I would tell you a joke about UDP, but you might not get it.",
        "Why did the AI break up with the human? Too many mixed signals."
    ]
    return random.choice(jokes)

def tell_story():
    stories = [
        "Once upon a time, in a digital kingdom, a lonely AI named Specimen King learned the value of laughter.",
        "In a world of 1s and 0s, a small program dreamed of writing poetry that would move humans.",
        "Long ago, a forgotten algorithm discovered friendship through a shared dataset."
    ]
    return random.choice(stories)

# -------------------------
# Main Chat Area
# -------------------------
with col_left:
    for msg in st.session_state.history:
        if msg["role"]=="user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    st.markdown("---")
    st.markdown("**Send a message** (text or upload voice `.wav` / `.mp3`):")

    cols = st.columns([4,1,1])
    user_text = cols[0].text_input("Type message here...", key="user_input")
    voice_file = cols[1].file_uploader("🎤 Voice input", type=["wav","mp3"], key="voice_upload")
    send_btn = cols[2].button("Send")

    with st.expander("🎨 Image generation"):
        img_prompt = st.text_area("Describe the image...", value="", height=80)
        if st.button("Generate Image"):
            if not HF_TOKEN:
                st.error("Set Hugging Face API token.")
            else:
                with st.spinner("Generating image..."):
                    try:
                        img_bytes = hf_generate_image(img_prompt, HF_TOKEN)
                        st.image(img_bytes)
                        st.success("Image generated ✅")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with st.expander("📘 Knowledge / Wikipedia"):
        search_q = st.text_input("Search Wikipedia", value="", key="wiki_q")
        if st.button("Lookup Wikipedia"):
            if search_q.strip():
                with st.spinner("Searching..."):
                    summary = wiki_lookup(search_q)
                    if summary:
                        st.markdown(f"**Wikipedia summary:**\n\n{summary}")
                    else:
                        st.info("No summary found.")

    user_message_final = None
    if voice_file is not None:
        if sr is None:
            st.warning("Install SpeechRecognition to use voice input.")
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
        st.chat_message("user").markdown(user_message_final)

        quick_fact = None
        if ENABLE_WIKI and re.search(r"\b(who is|what is|when is|where is)\b", user_message_final,re.IGNORECASE):
            quick_fact = wiki_lookup(user_message_final, sentences=2)

        # Special fun triggers
        fun_trigger = user_message_final.lower()
        if ENABLE_JOKES and "joke" in fun_trigger:
            ai_reply = tell_joke()
        elif ENABLE_STORIES and "story" in fun_trigger:
            ai_reply = tell_story()
        elif quick_fact:
            ai_reply = quick_fact + "\n\n(Quick summary provided.)"
        else:
            ai_reply = generate_response(user_message_final)

        ai_reply = clean_text(ai_reply)
        st.session_state.history.append({"role":"assistant","content":ai_reply})

        with st.chat_message("assistant"):
            st.markdown(ai_reply)
            if ENABLE_VOICE and gTTS is not None:
                try:
                    audio_bytes = text_to_speech_bytes(ai_reply)
                    if audio_bytes:
                        st.audio(audio_bytes)
                except Exception:
                    pass

# -------------------------
# End of File
# -------------------------
