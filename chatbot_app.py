# specimen_ultra.py
import streamlit as st
import os
import re
import sys
import tempfile
import requests
import wikipedia
import json

# -------------------------
# Gemini (Google Generative AI)
# -------------------------
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# -------------------------
# TTS / STT
# -------------------------
gTTS = None
try:
    from gtts import gTTS
except ImportError:
    pass

sr = None
try:
    import speech_recognition as sr
except ImportError:
    pass

# Ensure UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# -------------------------
# CONFIG / API KEYS
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", None))

# Configure Gemini
if genai and GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Feature toggles
ENABLE_VOICE = True
ENABLE_IMAGES = True
ENABLE_WIKI = True
ENABLE_STORYLINE_MODE = True  # new feature: AI modes

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI", layout="wide")
st.title("👑 Specimen King Ultra AI (Ultimate Edition)")
st.caption("Voice • Images • Memory • Knowledge • Storylines — Powered by Google Gemini + Hugging Face")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### Chat with Specimen King AI")
with col_right:
    st.markdown("### Controls / Settings")

# -------------------------
# SESSION STATE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "persona" not in st.session_state:
    st.session_state.persona = (
        "You are Specimen King AI — a confident, composed, and slightly playful leader. "
        "Speak with authority and clarity. Use short, polished sentences. Be helpful, direct, and occasionally witty; show empathy when appropriate. Keep a slight mysterious edge."
    )

if "mode" not in st.session_state:
    st.session_state.mode = "normal"  # default AI mode

# -------------------------
# RIGHT PANEL SETTINGS
# -------------------------
with col_right:
    st.markdown("#### Persona Settings")
    persona_text = st.text_area("AI Persona", value=st.session_state.persona, height=120)
    if st.button("Save Persona"):
        st.session_state.persona = persona_text
        st.success("Persona updated ✅")

    st.markdown("---")
    st.markdown("#### AI Mode / Storyline")
    mode_option = st.selectbox("Choose AI Mode:", ["normal", "playful", "wise", "storyteller"])
    if st.button("Set Mode"):
        st.session_state.mode = mode_option
        st.success(f"AI mode set to {mode_option}")

    st.markdown("---")
    st.markdown("#### Optional API Tokens")
    hf_input = st.text_input("Hugging Face Token", type="password", value=HF_TOKEN or "")
    if hf_input:
        HF_TOKEN = hf_input
        st.success("HF token set ✅")

    st.markdown("---")
    st.markdown("#### Quick Utilities")
    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.experimental_rerun()

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def clean_text(text):
    text = text.strip()
    text = re.sub(r"^(AI:|User:)\s*", "", text, flags=re.IGNORECASE)
    return text

def wiki_lookup(query, sentences=2):
    if not ENABLE_WIKI:
        return None
    try:
        return wikipedia.summary(query, sentences=sentences)
    except Exception:
        return None

def hf_generate_image(prompt_text):
    if not HF_TOKEN:
        st.error("Hugging Face token required for image generation.")
        return None
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt_text}
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        st.error(f"Image gen failed: {response.status_code}")
        return None
    return response.content

def text_to_speech_bytes(text):
    if gTTS is None:
        return None
    tts = gTTS(text=text, lang="en", slow=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    with open(tmp.name, "rb") as f:
        data = f.read()
    try: os.unlink(tmp.name)
    except: pass
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
    except:
        return None
    finally:
        try: os.unlink(tmp_path)
        except: pass

def generate_gemini_response(prompt):
    if not genai:
        return "Google Gemini is not available in this environment."
    try:
        response = genai.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return clean_text(response.text)
    except Exception as e:
        return f"AI generation error: {e}"

def build_prompt(user_message):
    history_text = ""
    for m in st.session_state.history[-6:]:
        role = "User" if m["role"] == "user" else "AI"
        history_text += f"{role}: {m['content']}\n"
    prompt = (
        f"{st.session_state.persona}\n"
        f"Mode: {st.session_state.mode}\n"
        f"{history_text}\n"
        f"User: {user_message}\nAI:"
    )
    return prompt

# -------------------------
# MAIN CHAT UI
# -------------------------
with col_left:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    st.markdown("---")
    st.markdown("**Send a message** (text or voice `.wav`/`.mp3`)")

    cols = st.columns([4, 1, 1])
    user_text = cols[0].text_input("Type your message...", key="user_input")
    voice_file = cols[1].file_uploader("Upload voice", type=["wav", "mp3"], key="voice_upload")
    send_btn = cols[2].button("Send")

    # Image generation
    with st.expander("🎨 Image Generation"):
        img_prompt = st.text_area("Describe the image", "", height=80)
        if st.button("Generate Image"):
            img_bytes = hf_generate_image(img_prompt)
            if img_bytes:
                st.image(img_bytes)
                st.success("Image generated ✅")

    # Wikipedia lookup
    with st.expander("📘 Knowledge / Wikipedia"):
        search_q = st.text_input("Ask a Wikipedia question", "", key="wiki_q")
        if st.button("Lookup"):
            if search_q.strip():
                summary = wiki_lookup(search_q)
                if summary:
                    st.markdown(f"**Summary:**\n{summary}")
                else:
                    st.info("No summary found.")

    # Handle input
    user_message_final = None
    if voice_file:
        if sr is None:
            st.warning("Speech recognition not installed; please type your message.")
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
        st.session_state.history.append({"role": "user", "content": user_message_final})
        st.chat_message("user").markdown(user_message_final)

        # Quick wiki lookup
        quick_fact = None
        if ENABLE_WIKI and re.search(r"\b(who is|what is|when is|where is)\b", user_message_final, re.IGNORECASE):
            quick_fact = wiki_lookup(user_message_final, sentences=2)

        with st.chat_message("assistant"):
            with st.spinner("Specimen King AI is thinking..."):
                if quick_fact:
                    ai_reply = quick_fact + "\n\n(Quick knowledge summary.)"
                else:
                    prompt = build_prompt(user_message_final)
                    ai_reply = generate_gemini_response(prompt)

                st.markdown(ai_reply)
                st.session_state.history.append({"role": "assistant", "content": ai_reply})

                # TTS
                if ENABLE_VOICE and gTTS:
                    audio_bytes = text_to_speech_bytes(ai_reply)
                    if audio_bytes:
                        st.audio(audio_bytes)

# -------------------------
# END OF FILE
# -------------------------
