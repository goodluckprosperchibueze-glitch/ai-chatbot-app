# specimen_ultra.py
import streamlit as st
import os
import re
import sys
import tempfile
import requests
import wikipedia 
import google.generativeai as genai
# Audio libraries
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
# CONFIG / Secrets
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))
GOOGLE_API_KEY = "AIzaSyDjJgrg8j9UZ0yNUqGqNUGavyKfKvXKf_M"

# Initialize Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Feature toggles
ENABLE_VOICE = True
ENABLE_IMAGES = True
ENABLE_WIKI = True
MAX_HISTORY = 20

# -------------------------
# UI Setup
# -------------------------
st.set_page_config(page_title="Specimen King Ultra AI", layout="wide")
st.title("👑 Specimen King Ultra AI (Gemini Edition)")
st.caption("Voice • Images • Memory • Knowledge • Multi-Mode AI")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### Chat with Specimen King AI")
with col_right:
    st.markdown("### Controls / Settings")

# -------------------------
# Session State
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "persona" not in st.session_state:
    st.session_state.persona = (
        "You are Specimen King AI — confident, composed, and slightly playful. "
        "Speak with authority and clarity. Short polished sentences. Be helpful, witty, and occasionally mysterious."
    )

if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = []

if "mode" not in st.session_state:
    st.session_state.mode = "Quick Facts"  # default mode

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
    st.write("Choose AI Mode")
    mode_selected = st.radio(
        "AI Mode",
        options=["Quick Facts", "Deep Search", "Storyline"],
        index=["Quick Facts", "Deep Search", "Storyline"].index(st.session_state.mode)
    )
    st.session_state.mode = mode_selected

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
        for file in st.session_state.audio_cache:
            try: os.unlink(file)
            except: pass
        st.session_state.audio_cache = []
        st.experimental_rerun()

# -------------------------
# Helper Functions
# -------------------------
def clean_text(text):
    text = text.strip()
    text = re.sub(r"^(AI:|User:)\s*", "", text, flags=re.IGNORECASE)
    return text

def generate_response(user_message):
    """
    Use Gemini to generate response with persona, chat history, and AI mode
    """
    # Keep last MAX_HISTORY messages
    history_trimmed = st.session_state.history[-MAX_HISTORY:]
    chat_history = "\n".join(
        [f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" for m in history_trimmed]
    )

    # Customize prompt based on mode
    if st.session_state.mode == "Quick Facts":
        mode_instruction = "Answer in short, concise sentences. Provide facts quickly."
    elif st.session_state.mode == "Deep Search":
        mode_instruction = "Provide a detailed and well-researched answer. Use examples and context."
    elif st.session_state.mode == "Storyline":
        mode_instruction = "Respond creatively, continuing a story or roleplay. Add imagination."

    prompt = (
        f"{st.session_state.persona}\nMode: {st.session_state.mode}\n{mode_instruction}\n\n"
        f"Conversation so far:\n{chat_history}\nUser: {user_message}\nAI:"
    )

    # Gemini API call
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        temperature=0.7
    )
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
    st.session_state.audio_cache.append(tmp.name)
    with open(tmp.name, "rb") as f:
        data = f.read()
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
# Main Chat Area
# -------------------------
with col_left:
    # Display history
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    st.markdown("---")
    st.markdown("**Send a message** (text or upload a short voice `.wav`):")

    cols = st.columns([4,1,1])
    user_text = cols[0].text_input("Type message here...", key="user_input")
    voice_file = cols[1].file_uploader("🎤 Voice (wav/mp3)", type=["wav","mp3"], key="voice_upload")
    send_btn = cols[2].button("Send")

    # Image generator
    with st.expander("🎨 Image generation"):
        img_prompt = st.text_area("Describe image...", value="", height=80)
        if st.button("Generate Image"):
            if not HF_TOKEN:
                st.error("HF token missing.")
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
        if st.button("Lookup Wikipedia"):
            if search_q.strip():
                with st.spinner("Searching Wikipedia..."):
                    summary = wiki_lookup(search_q)
                    if summary:
                        st.markdown(f"**Wikipedia summary:**\n\n{summary}")
                    else:
                        st.info("No summary found.")

    # Handle input
    user_message_final = None
    if voice_file is not None:
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

    if user_message_final:
        st.session_state.history.append({"role":"user","content":user_message_final})
        st.chat_message("user").markdown(user_message_final)

        quick_fact = None
        if ENABLE_WIKI and re.search(r"\b(who is|what is|when is|where is)\b", user_message_final, re.IGNORECASE):
            quick_fact = wiki_lookup(user_message_final, sentences=2)

        with st.chat_message("assistant"):
            with st.spinner("Specimen King AI is thinking..."):
                try:
                    if quick_fact and st.session_state.mode=="Quick Facts":
                        ai_reply = quick_fact + "\n\n(Quick knowledge summary)"
                    else:
                        ai_reply = generate_response(user_message_final)

                    st.markdown(ai_reply)
                    st.session_state.history.append({"role":"assistant","content":ai_reply})

                    if ENABLE_VOICE and gTTS:
                        try:
                            audio_bytes = text_to_speech_bytes(ai_reply)
                            if audio_bytes:
                                st.audio(audio_bytes)
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"AI generation error: {e}")
