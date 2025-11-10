# specimen_ultra.py
import streamlit as st
from transformers import pipeline
import wikipedia
import tempfile
import os
import re
import sys
import json
import requests

# Audio libraries (gTTS for TTS; speech_recognition for STT)
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
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# -------------------------
# CONFIG / Secrets (optional)
# -------------------------
# If you want image generation via Hugging Face Inference API, put your token in Streamlit secrets or env:
# st.secrets["HF_TOKEN"] = "hf_xxx"  (or set environment variable HF_TOKEN)
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))

# Toggle features
ENABLE_VOICE = True     # set False to disable voice controls
ENABLE_IMAGES = True    # set False if you don't have HF token
ENABLE_WIKI = True      # quick knowledge mode via wikipedia library

# -------------------------
# UI Setup
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI", layout="wide")
st.title("👑 Specimen King Ultra AI (v5)")
st.caption("Voice • Images • Memory • Knowledge — Streamlit + Transformers")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### Chat with Specimen King AI")
with col_right:
    st.markdown("### Controls / Settings")

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model(model_name="google/flan-t5-large"):
    """Load a text2text model for conversational generation."""
    # Using text2text-generation pipeline for prompts
    return pipeline("text2text-generation", model=model_name)

# Loading might take time
with st.spinner("Loading AI model (may take ~30s first run)..."):
    model = load_model()

# -------------------------
# Session state: chat memory
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {"role": "user"/"assistant", "content": "..."}

if "persona" not in st.session_state:
    # Default persona (you can edit in settings)
    st.session_state.persona = (
        "You are Specimen King AI — confident, friendly, and clear. "
        "Answer politely, be concise but thorough, use emojis sparingly, and avoid repeating the user's exact words."
    )

# -------------------------
# Right column: settings & tools
# -------------------------
with col_right:
    st.markdown("#### Settings")
    persona_text = st.text_area("AI Persona (edit to change tone)", value=st.session_state.persona, height=120)
    if st.button("Save Persona"):
        st.session_state.persona = persona_text
        st.success("Persona updated.")

    st.markdown("---")
    st.write("Optional API keys")
    hf_token_input = st.text_input("Hugging Face Inference API Token (for image gen)", type="password", value=HF_TOKEN or "")
    if hf_token_input:
        HF_TOKEN = hf_token_input
        st.success("HF token set for this session (not persisted).")

    st.markdown("---")
    st.write("Quick Utilities")
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.experimental_rerun()

# -------------------------
# Helper functions
# -------------------------
def build_prompt(persona, history, user_message):
    """
    Build a single prompt that provides persona + limited chat history + the user question.
    Keep history trimmed to reduce input size.
    """
    # keep last N exchanges
    max_exchanges = 6
    trimmed = history[-max_exchanges * 2 :]  # because role entries
    convo_text = ""
    for m in trimmed:
        role_label = "User" if m["role"] == "user" else "AI"
        convo_text += f"{role_label}: {m['content']}\n"
    prompt = (
        f"{persona}\n\n"
        f"Conversation so far:\n{convo_text}\n"
        f"User: {user_message}\nAI:"
    )
    return prompt

def clean_text(text):
    text = text.strip()
    # remove any residual "AI:" labels or "User:" lines that might be echoed
    text = re.sub(r"^(AI:|User:)\s*", "", text, flags=re.IGNORECASE)
    # keep basic punctuation and characters
    return text

def generate_response(user_message, max_length=200):
    prompt = build_prompt(st.session_state.persona, st.session_state.history, user_message)
    # Use the text2text pipeline
    out = model(prompt, max_length=max_length, temperature=0.7, top_p=0.9)
    raw = out[0].get("generated_text", "")
    raw = clean_text(raw)
    return raw

# Optional: Wikipedia quick lookup
def wiki_lookup(query, sentences=2):
    if not ENABLE_WIKI:
        return None
    try:
        summary = wikipedia.summary(query, sentences=sentences)
        return summary
    except Exception:
        return None

# Optional: Hugging Face Inference for image generation (text-to-image)
def hf_generate_image(prompt_text, hf_token=HF_TOKEN):
    if not hf_token:
        raise RuntimeError("No HF token provided.")
    # Using the Hugging Face Inference API (text-to-image stable diffusion)
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt_text}
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Image gen failed: {response.status_code} {response.text}")
    # The HF image API returns binary content if successful, or JSON error
    return response.content

# TTS: convert text to audio bytes using gTTS if available
def text_to_speech_bytes(text, lang="en"):
    if gTTS is None:
        return None
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

# STT: record from user audio (optional; uses speech_recognition if installed)
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
# Main Chat UI (left column)
# -------------------------
with col_left:
    # show history
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # Input options: text, or upload voice
    st.markdown("---")
    st.markdown("**Send a message** (text or upload a short voice `.wav`):")

    cols = st.columns([4, 1, 1])
    user_text = cols[0].text_input("Type message here...", key="user_input")
    voice_file = cols[1].file_uploader("Upload voice (wav/mp3)", type=["wav", "mp3"], key="voice_upload")
    send_btn = cols[2].button("Send")

    # Image generation area
    with st.expander("Image generation (optional)"):
        img_prompt = st.text_area("Describe the image you want", value="", height=80)
        if st.button("Generate Image"):
            if not HF_TOKEN:
                st.error("Image generation requires a Hugging Face API token (set it in the right panel).")
            else:
                with st.spinner("Generating image... this can take a while"):
                    try:
                        img_bytes = hf_generate_image(img_prompt, HF_TOKEN)
                        st.image(img_bytes)
                        st.success("Image generated (download from above).")
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")

    # Knowledge mode quick-search
    with st.expander("Knowledge / Quick search"):
        search_q = st.text_input("Ask something to lookup (Wikipedia)", value="", key="wiki_q")
        if st.button("Lookup Wikipedia"):
            if search_q.strip():
                with st.spinner("Searching Wikipedia..."):
                    summary = wiki_lookup(search_q)
                    if summary:
                        st.markdown(f"**Wikipedia summary:**\n\n{summary}")
                    else:
                        st.info("No summary found. Try a different query.")

    # HANDLE sending message (priority: voice upload -> text input)
    user_message_final = None
    if voice_file is not None:
        if sr is None:
            st.warning("Speech recognition package not installed; please type your message or install `speechrecognition`.")
        else:
            with st.spinner("Recognizing speech..."):
                recognized = recognize_speech_from_file(voice_file)
                if recognized:
                    user_message_final = recognized
                    st.success(f"Recognized: {recognized}")
                else:
                    st.error("Could not recognize speech. Try a clearer audio or type your message.")
    elif send_btn and user_text.strip():
        user_message_final = user_text.strip()

    # If we have a message to send:
    if user_message_final:
        # Save user message
        st.session_state.history.append({"role": "user", "content": user_message_final})
        st.chat_message("user").markdown(user_message_final)

        # Optionally perform a quick wiki check if user asked for facts
        # (This is a simple heuristic: if message contains "who is" or "what is")
        quick_fact = None
        if ENABLE_WIKI and re.search(r"\b(who is|what is|when is|where is)\b", user_message_final, re.IGNORECASE):
            quick_fact = wiki_lookup(user_message_final, sentences=2)

        with st.chat_message("assistant"):
            with st.spinner("Specimen King AI is thinking..."):
                try:
                    if quick_fact:
                        ai_reply = quick_fact + "\n\n(Quick knowledge summary provided.)"
                    else:
                        ai_reply = generate_response(user_message_final, max_length=250)

                    ai_reply = clean_text(ai_reply)

                    # Display reply
                    st.markdown(ai_reply)

                    # Save reply to history
                    st.session_state.history.append({"role": "assistant", "content": ai_reply})

                    # Optionally create TTS and offer an audio player
                    if ENABLE_VOICE and gTTS is not None:
                        try:
                            audio_bytes = text_to_speech_bytes(ai_reply)
                            if audio_bytes:
                                st.audio(audio_bytes)
                        except Exception:
                            # silently continue if TTS fails
                            pass

                except Exception as e:
                    st.error(f"AI generation error: {e}")

# -------------------------
# End of file
# -------------------------
