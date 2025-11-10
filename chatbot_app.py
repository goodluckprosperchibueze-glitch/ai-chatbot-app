import streamlit as st
import os
import base64
import tempfile
import re
import requests
import """

# ------------------------
# PAGE SETUP
# ------------------------
st.set_page_config(page_title="Specimen King Ultra AI v6", layout="wide")
st.title("👑 Specimen King Ultra AI — Full Energy Mode")
st.caption("Chat • Story • Deep Search • Images • Voice — Gemini + HF + TTS")

# ------------------------
# API KEYS
# ------------------------
GENAI_API_KEY = st.secrets.get("GENAI_API_KEY", os.environ.get("GENAI_API_KEY", ""))
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

client = genai.Client(api_key=GENAI_API_KEY) if GENAI_API_KEY else None

# ------------------------
# SESSION STATE
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "persona" not in st.session_state:
    st.session_state.persona = (
        "You are PopKing AI: energetic, encouraging, witty, super study-focused, "
        "friendly, remembers past conversations, gives stepwise answers, and can generate "
        "images and voice. Avoid hallucinations."
    )

# ------------------------
# SIDEBAR CONTROLS
# ------------------------
st.sidebar.header("⚙️ Controls")
mode = st.sidebar.selectbox("Mode", ["Chat", "Story", "Deep Search", "Gemini", "All Energy"])
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.8)
max_tokens = st.sidebar.slider("Max reply tokens", 50, 512, 200)
enable_voice = st.sidebar.checkbox("Enable Voice TTS", value=True)
enable_images = st.sidebar.checkbox("Enable Image Generation", value=True)

persona_text = st.sidebar.text_area("Persona", value=st.session_state.persona, height=120)
if st.sidebar.button("Save Persona"):
    st.session_state.persona = persona_text
    st.sidebar.success("Persona saved!")

# ------------------------
# HELPERS
# ------------------------
def safe_clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
    text = "".join(ch for ch in text if ord(ch) < 0x10000)
    return text.strip()

def build_prompt(user_input: str) -> str:
    """Build prompt with memory + persona + mode"""
    trimmed_history = st.session_state.history[-6:]  # last 6 exchanges
    convo_text = ""
    for msg in trimmed_history:
        role = "User" if msg["role"] == "user" else "AI"
        convo_text += f"{role}: {msg['content']}\n"
    prompt = f"{st.session_state.persona}\nMode: {mode}\n{convo_text}User: {user_input}\nAI:"
    return prompt

def call_gemini(prompt: str) -> str:
    if not client:
        return "(Gemini key not set; set GENAI_API_KEY)"
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return getattr(response, "text", "Hmm, I couldn't respond.")
    except Exception as e:
        return f"(Gemini error: {e})"

def hf_generate_image_bytes(prompt: str) -> bytes:
    if not HF_TOKEN:
        raise RuntimeError("Hugging Face token missing")
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 200:
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            data = resp.json()
            b64 = data.get("images", [None])[0]
            if b64:
                return base64.b64decode(b64)
        return resp.content
    raise RuntimeError(f"HF image error: {resp.status_code}")

# Optional: TTS
try:
    from gtts import gTTS
except Exception:
    gTTS = None

def speak_text(text: str) -> bytes:
    if not gTTS:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(tmp.name)
    with open(tmp.name, "rb") as f:
        data = f.read()
    return data

# ------------------------
# USER INPUT
# ------------------------
user_input = st.chat_input("Talk to PopKing AI...")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    prompt = build_prompt(user_input)
    
    # Determine response based on mode
    if mode in ["Gemini", "All Energy"]:
        reply = call_gemini(prompt)
    elif mode == "Story":
        reply = call_gemini(prompt + "\nWrite a long, imaginative story now.")
    elif mode == "Deep Search":
        reply = call_gemini(prompt + "\nProvide stepwise reasoning and suggested references.")
    else:
        reply = call_gemini(prompt)
    
    reply = safe_clean(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})

# ------------------------
# DISPLAY CHAT
# ------------------------
for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if enable_voice:
                audio_bytes = speak_text(msg["content"])
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

# ------------------------
# IMAGE GENERATION PANEL
# ------------------------
with st.expander("🖼️ Generate Image"):
    img_prompt = st.text_area("Describe image...", "")
    img_style = st.selectbox("Style", ["photorealistic", "digital art", "anime", "cartoon", "fantasy"])
    if st.button("Generate Image"):
        try:
            full_prompt = f"{img_prompt} -- style: {img_style}"
            img_bytes = hf_generate_image_bytes(full_prompt)
            st.image(img_bytes)
            st.success("Image generated!")
        except Exception as e:
            st.error(f"Image failed: {e}")

# ------------------------
# FOOTER
# ------------------------
st.markdown("---")
st.caption("Specimen King Ultra AI v6 — Chat naturally, create freely, learn deeply 👑")
