# specimen_ultra.py
import streamlit as st
import os
import re
import sys
import tempfile
import requests
import wikipedia

# Optional: text-to-speech and speech recognition
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
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# -------------------------
# CONFIG / API KEYS
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))

# Google Gemini safe import
try:
    import google.generativeai as genai
    GOOGLE_API_KEY = "AIzaSyDjJgrg8j9UZ0yNUqGqNUGavyKfKvXKf_M"  # Your API key
    genai.configure(api_key=GOOGLE_API_KEY)
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ Google Gemini not available. Install `google-generativeai` to enable AI chat.")

# Feature toggles
ENABLE_VOICE = True
ENABLE_IMAGES = True
ENABLE_WIKI = True

# -------------------------
# UI SETUP
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI", layout="wide")
st.title("👑 Specimen King Ultra AI")
st.caption("Voice • Images • Knowledge • Memory — Powered by Google Gemini")

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
        "You are Specimen King AI — confident, intelligent, and calm. "
        "You answer politely, sound slightly mysterious, and speak like a quiet genius. "
        "Use clear, concise, and insightful replies, with minimal emojis."
    )

# -------------------------
# RIGHT PANEL SETTINGS
# -------------------------
with col_right:
    st.markdown("#### AI Persona")
    persona_text = st.text_area("Edit persona:", value=st.session_state.persona, height=120)
    if st.button("Save Persona"):
        st.session_state.persona = persona_text
        st.success("Persona updated ✅")

    st.markdown("---")
    st.write("Optional API keys")
    hf_token_input = st.text_input("Hugging Face API Token (for images)", type="password", value=HF_TOKEN or "")
    if hf_token_input:
        HF_TOKEN = hf_token_input
        st.success("HF token set for this session.")

    st.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.experimental_rerun()

# -------------------------
# GEMINI MODEL SETUP
# -------------------------
if GEMINI_AVAILABLE:
    @st.cache_resource
    def load_gemini_model():
        return genai.GenerativeModel("gemini-1.5-flash")
    model = load_gemini_model()

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def clean_text(text):
    text = text.strip()
    text = re.sub(r"^(AI:|User:)\s*", "", text, flags=re.IGNORECASE)
    return text

def generate_response(user_message):
    if not GEMINI_AVAILABLE:
        return "Google Gemini AI is not available. Please install `google-generativeai` and restart."

    chat_history = []
    for m in st.session_state.history:
        role = "user" if m["role"] == "user" else "model"
        chat_history.append({"role": role, "parts": [m["content"]]})

    chat = model.start_chat(history=chat_history)
    prompt = f"{st.session_state.persona}\n\nUser: {user_message}\nAI:"
    response = chat.send_message(prompt)
    return clean_text(response.text)

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
    for msg in st.session_state.history:
        role = "user" if msg["role"] == "user" else "assistant"
        st.chat_message(role).markdown(msg["content"])

    st.markdown("---")
    st.markdown("**Send a message:** (text or voice `.wav`/`.mp3`)")

    cols = st.columns([4, 1, 1])
    user_text = cols[0].text_input("Type your message...", key="user_input")
    voice_file = cols[1].file_uploader("🎤 Voice", type=["wav", "mp3"])
    send_btn = cols[2].button("Send")

    # Image generation
    with st.expander("🎨 Generate Image"):
        img_prompt = st.text_area("Describe image...", value="", height=80)
        if st.button("Generate Image"):
            if not HF_TOKEN:
                st.error("Set Hugging Face token first.")
            else:
                with st.spinner("Generating image..."):
                    try:
                        img_bytes = hf_generate_image(img_prompt, HF_TOKEN)
                        st.image(img_bytes)
                        st.success("Image generated ✅")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Wikipedia lookup
    with st.expander("📘 Knowledge Search"):
        search_q = st.text_input("Wikipedia search", value="", key="wiki_q")
        if st.button("Lookup Wikipedia") and search_q.strip():
            with st.spinner("Searching..."):
                summary = wiki_lookup(search_q)
                if summary:
                    st.markdown(f"**Wikipedia summary:**\n\n{summary}")
                else:
                    st.info("No summary found.")

    # Handle input
    user_message_final = None
    if voice_file:
        if sr is None:
            st.warning("SpeechRecognition not installed.")
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

    # Send message
    if user_message_final:
        st.session_state.history.append({"role": "user", "content": user_message_final})
        st.chat_message("user").markdown(user_message_final)

        quick_fact = None
        if ENABLE_WIKI and re.search(r"\b(who is|what is|when is|where is)\b", user_message_final, re.IGNORECASE):
            quick_fact = wiki_lookup(user_message_final, sentences=2)

        with st.chat_message("assistant"):
            with st.spinner("Specimen King AI is thinking..."):
                try:
                    if quick_fact:
                        ai_reply = quick_fact + "\n\n(Quick summary provided.)"
                    else:
                        ai_reply = generate_response(user_message_final)

                    ai_reply = clean_text(ai_reply)
                    st.markdown(ai_reply)
                    st.session_state.history.append({"role": "assistant", "content": ai_reply})

                    # TTS
                    if ENABLE_VOICE and gTTS:
                        audio_bytes = text_to_speech_bytes(ai_reply)
                        if audio_bytes:
                            st.audio(audio_bytes)
                except Exception as e:
                    st.error(f"AI generation error: {e}")

# -------------------------
# END OF FILE
# -------------------------
