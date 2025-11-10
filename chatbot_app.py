# popoking_streamlit.py
# Specimen King -> PopoKing: Phone-friendly Streamlit chat using Gemini (preferred) or fallback.
# Modes: AI Chat, Story Mode, Deep Search, Laughing.
# Text + Voice (gTTS). Designed to run in Pydroid3 or Termux (phone) or PC.

import streamlit as st
import os, time, re, tempfile, sys
from typing import Optional

# Optional dependencies
try:
    from google import genai
except Exception:
    genai = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import wikipedia
except Exception:
    wikipedia = None

# Ensure UTF-8 output
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ---------- UI setup ----------
st.set_page_config(page_title="PopoKing AI", layout="centered")
st.markdown("<center><h1>👑 PopoKing</h1><p style='margin-top:-12px'>Your Gemini-powered assistant (text + voice)</p></center>", unsafe_allow_html=True)
st.write("---")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    persona = st.text_area("Persona (how PopoKing sounds)", value="You are PopoKing — confident, kind, playful, and short when asked. Use emojis lightly.", height=100)
    st.markdown("**Gemini API key (optional)**")
    GEMINI_API_KEY = st.text_input("GEMINI_API_KEY", type="password", value=os.environ.get("GEMINI_API_KEY",""))
    enable_tts = st.checkbox("Enable voice replies (gTTS)", value=True)
    st.markdown("---")
    st.write("Tips: Paste your Gemini key above for best responses. Use Clear Chat to reset.")
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.experimental_rerun()

# Modes and session init
modes = ["AI Chat", "Story Mode", "Deep Search 🔎", "Laughing (Jokes)"]
mode = st.selectbox("Choose mode", modes)

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts {role, text}

# Initialize Gemini client if key present
gemini_client = None
if GEMINI_API_KEY and genai is not None:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        gemini_client = None

# Helpers
def build_prompt(user_msg: str):
    # include persona and last few exchanges for context
    recent = st.session_state.history[-6:]
    convo = "\n".join([f"User: {m['text']}" if m['role']=="user" else f"AI: {m['text']}" for m in recent])
    prompt = f"{persona}\n\nConversation so far:\n{convo}\nUser: {user_msg}\nAI:"
    return prompt

def call_gemini(prompt: str) -> str:
    if not gemini_client:
        return "[No Gemini client available — set your GEMINI_API_KEY in sidebar]"
    try:
        resp = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        # gemini response object may expose .text or .response.text()
        text = getattr(resp, "text", None) or (resp.response.text() if hasattr(resp, "response") else None)
        if not text:
            # try string conversion
            text = str(resp)
        return text
    except Exception as e:
        return f"[Gemini error: {e}]"

def wiki_lookup(q: str) -> Optional[str]:
    if wikipedia is None:
        return None
    try:
        return wikipedia.summary(q, sentences=2)
    except Exception:
        return None

def speak_bytes(text: str) -> Optional[bytes]:
    if gTTS is None:
        return None
    try:
        t = gTTS(text=text, lang="en", slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        t.save(tmp.name)
        with open(tmp.name, "rb") as f:
            b = f.read()
        try:
            os.unlink(tmp.name)
        except:
            pass
        return b
    except Exception:
        return None

# Display history
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["text"])
    else:
        st.chat_message("assistant").markdown(msg["text"])

# Input box and send
user_input = st.text_input("Type your message here...", key="input_box")
send = st.button("Send")

# If Send pressed, create response
if send and user_input.strip():
    user_text = user_input.strip()
    st.session_state.history.append({"role":"user","text":user_text})
    st.chat_message("user").markdown(user_text)

    # Mode-specific prompt
    if mode == "AI Chat":
        prompt = build_prompt(user_text)
    elif mode == "Story Mode":
        prompt = f"You are a brilliant storyteller. Write a creative short story about: {user_text}. Keep it vivid and original."
    elif mode == "Deep Search 🔎":
        # try Wikipedia first
        summary = wiki_lookup(user_text)
        if summary:
            ai_reply = f"🔎 Quick summary (Wikipedia):\n\n{summary}"
            st.chat_message("assistant").markdown(ai_reply)
            st.session_state.history.append({"role":"assistant","text":ai_reply})
            # TTS
            if enable_tts:
                audio = speak_bytes(ai_reply)
                if audio:
                    st.audio(audio)
            # done
            st.experimental_rerun()
        else:
            prompt = f"Provide a clear, factual summary about: {user_text}."
    else:  # Laughing (Jokes)
        prompt = f"Tell a clean, short joke or playful reply about: {user_text}. Make it funny and light."

    # Show typing animation (spinner + slight delay for realism)
    with st.spinner("PopoKing is thinking..."):
        # small random delay to feel natural
        time.sleep(0.7 + random.random()*0.6)

        # Prefer Gemini if available
        if gemini_client:
            ai_reply = call_gemini(prompt)
        else:
            # Minimal fallback: gentle canned replies for phone (fast)
            ai_reply = "[No Gemini key] PopoKing says: I need your GEMINI_API_KEY in the sidebar to answer fully. Try: explain " + user_text

    # Clean reply (remove repeated prompt echoes)
    ai_reply = re.sub(r"^\s*User:.*", "", ai_reply, flags=re.IGNORECASE).strip()
    if not ai_reply:
        ai_reply = "Hmm — I couldn't make a good reply. Try again."

    # Display assistant message and save history
    st.chat_message("assistant").markdown(ai_reply)
    st.session_state.history.append({"role":"assistant","text":ai_reply})

    # Play TTS if enabled
    if enable_tts:
        audio_b = speak_bytes(ai_reply)
        if audio_b:
            st.audio(audio_b)
