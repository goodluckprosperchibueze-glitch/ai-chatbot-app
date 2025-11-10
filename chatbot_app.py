# popking_ultra_final.py
"""
PopKing Ultra Final
- Gemini-first (google-genai) conversational assistant
- Controlled local model fallback (load manually)
- Long-term memory saved to popking_memory.json
- Modes: AI Chat, Story, Deep Search, Laugh (Jokes)
- Text + optional voice (gTTS) replies
- Designed to avoid blocking the UI; safe for phone (Pydroid/Termux)
"""

import streamlit as st
import os, sys, time, json, re, tempfile
from typing import Optional

# Optional libs (guarded)
try:
    from google import genai
except Exception:
    genai = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import wikipedia
except Exception:
    wikipedia = None

# ensure utf-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -----------------------
# Config & constants
# -----------------------
MEMORY_FILE = "popking_memory.json"
LOCAL_MODEL_NAME = "google/flan-t5-small"  # small & lighter
LOCAL_MODEL_MAX = 200

# -----------------------
# Helpers: memory handling
# -----------------------
def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"user_name": "", "notes": [], "chats": []}
    return {"user_name": "", "notes": [], "chats": []}

def save_memory(mem: dict):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

memory = load_memory()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="PopKing Ultra", layout="wide")
st.markdown("<h1 style='text-align:center'>👑 PopKing Ultra</h1>", unsafe_allow_html=True)
st.write("<div style='text-align:center; color:gray'>Gemini-powered — Phone friendly</div>", unsafe_allow_html=True)
st.write("---")

# Sidebar: settings + keys
with st.sidebar:
    st.header("PopKing Settings")
    st.markdown("**Assistant persona (edit)**")
    persona = st.text_area(
        "Persona",
        value=st.session_state.get(
            "persona",
            "You are PopKing — confident, kind, playful, and clear. Use short answers, be helpful, avoid repeating the user's words."
        ),
        height=120,
    )
    if st.button("Save Persona"):
        st.session_state["persona"] = persona
        st.success("Persona saved to session.")

    st.markdown("---")
    st.markdown("**Gemini (Google) API Key**")
    gem = st.text_input("GEMINI_API_KEY (optional, paste here)", type="password", value=os.environ.get("GEMINI_API_KEY",""))
    if st.button("Store Gemini key for session"):
        st.session_state["GEMINI_API_KEY"] = gem
        st.success("Gemini key stored for this session (not permanent).")

    st.markdown("---")
    st.markdown("Model / Voice")
    st.radio("Backend", ["Gemini (recommended)", "Local model (manual load)"], index=0, key="backend")
    if pipeline is None:
        st.info("Local model libs not installed. Use Gemini or install transformers.")
    else:
        if st.button("Load local model (manual)"):
            st.session_state["load_local"] = True

    st.checkbox("Enable voice replies (gTTS)", value=True, key="enable_tts")
    st.markdown("---")
    st.write("Long-term memory")
    if memory.get("user_name"):
        st.markdown(f"**User name (saved):** {memory.get('user_name')}")
    name_input = st.text_input("Set / update your name (will be saved)", value=memory.get("user_name",""))
    if st.button("Save name"):
        memory["user_name"] = name_input.strip()
        save_memory(memory)
        st.success("Name saved in PopKing memory.")

    if st.button("Clear memory (delete saved file)"):
        try:
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
            memory.clear()
            memory.update({"user_name":"", "notes":[], "chats":[]})
            st.success("Memory cleared.")
        except Exception:
            st.error("Could not clear memory.")

# -----------------------
# Session flags & local model load
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"role","text"}
if "busy" not in st.session_state:
    st.session_state.busy = False
if "local_model" not in st.session_state:
    st.session_state.local_model = None
if "load_local" not in st.session_state:
    st.session_state.load_local = False
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state.GEMINI_API_KEY = st.session_state.get("GEMINI_API_KEY","") or os.environ.get("GEMINI_API_KEY","")

# If user requested local model load -> load now (safe controlled)
if st.session_state.get("load_local", False) and pipeline is not None and st.session_state.local_model is None:
    try:
        with st.spinner("Loading local model (may take ~20-40s)..."):
            st.session_state.local_model = pipeline("text2text-generation", model=LOCAL_MODEL_NAME)
        st.success("Local model ready.")
    except Exception as e:
        st.error(f"Local model load failed: {e}")
    finally:
        st.session_state.load_local = False

# -----------------------
# Top controls
# -----------------------
colA, colB = st.columns([3,1])
with colA:
    mode = st.selectbox("Mode", ["AI Chat", "Story Mode", "Deep Search 🔎", "Laughing (Jokes)"])
with colB:
    if st.button("Download chat"):
        txt = "\n\n".join([f"{m['role'].upper()}: {m['text']}" for m in st.session_state.history])
        st.download_button("Download .txt", txt, file_name="popking_chat.txt")

# show memory quick
if memory.get("user_name"):
    st.caption(f"Saved name: {memory['user_name']}")

# Chat display
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["text"])
    else:
        st.chat_message("assistant").markdown(msg["text"])

st.markdown("---")
st.info("Type your message below and press Send. If the app says 'PopKing is thinking...' wait a few seconds; it won't block typing.")

# Input area
input_col, btn_col = st.columns([5,1])
with input_col:
    user_input = st.text_area("Your message", height=90, key="user_input")
with btn_col:
    send = st.button("Send")

# Helper: build prompt for model
def build_prompt(user_text: str) -> str:
    persona_text = st.session_state.get("persona", persona)
    recent = st.session_state.history[-8:]
    convo = "\n".join([f"User: {m['text']}" if m['role']=="user" else f"AI: {m['text']}" for m in recent])
    user_name = memory.get("user_name", "")
    name_line = f"User name: {user_name}\n" if user_name else ""
    prompt = f"{persona_text}\n{name_line}\nConversation so far:\n{convo}\nUser: {user_text}\nAI:"
    return prompt

# Helper: call Gemini
def call_gemini(prompt: str) -> str:
    api_key = st.session_state.get("GEMINI_API_KEY","")
    if not api_key:
        return "[No Gemini key set. Paste in sidebar.]"
    if genai is None:
        return "[google-genai not installed on this device.]"
    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = getattr(resp, "text", None) or (resp.response.text() if hasattr(resp, "response") else None)
        if not text:
            text = str(resp)
        return text
    except Exception as e:
        return f"[Gemini error: {e}]"

# Helper: local generation
def call_local(prompt: str) -> str:
    lm = st.session_state.local_model
    if lm is None:
        return "[Local model not loaded]"
    try:
        out = lm(prompt, max_length=LOCAL_MODEL_MAX, temperature=0.7, top_p=0.9)
        text = out[0].get("generated_text","")
        # remove prompt echo
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text
    except Exception as e:
        return f"[Local generation error: {e}]"

# Helper: wiki summary
def wiki_summary(q: str) -> Optional[str]:
    if wikipedia is None:
        return None
    try:
        return wikipedia.summary(q, sentences=2)
    except Exception:
        return None

# Helper: tts
def tts_bytes(text: str) -> Optional[bytes]:
    if gTTS is None or not st.session_state.get("enable_tts", True):
        return None
    try:
        t = gTTS(text=text, lang="en", slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        t.save(tmp.name)
        with open(tmp.name, "rb") as f:
            b = f.read()
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        return b
    except Exception:
        return None

# Non-blocking send guard:
if send and user_input and not st.session_state.busy:
    st.session_state.busy = True
    # append user message
    st.session_state.history.append({"role":"user","text":user_input})
    # save to memory chats
    memory.setdefault("chats", []).append({"role":"user","text":user_input, "time":time.time()})
    save_memory(memory)
    # reset input
    st.session_state.user_input = ""
    # force a rerun so the UI shows the user's message immediately and generation can run
    st.experimental_rerun()

# Generation step: if last message is user and no assistant reply yet, generate now
if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    # ensure we only generate once per user message
    if not (len(st.session_state.history) >= 2 and st.session_state.history[-2]["role"]=="assistant"):
        # show typing placeholder
        placeholder = st.empty()
        with placeholder.container():
            st.chat_message("assistant").markdown("_PopKing is thinking..._")

        last_user = st.session_state.history[-1]["text"]
        # prepare prompt per mode
        if mode == "AI Chat":
            prompt_text = build_prompt(last_user)
        elif mode == "Story Mode":
            prompt_text = f"You are a brilliant storyteller. Write a vivid, original short story about: {last_user}"
        elif mode == "Deep Search 🔎":
            # attempt wikipedia first
            summary = wiki_summary(last_user)
            if summary:
                ai_answer = f"🔎 Wikipedia summary:\n\n{summary}"
                st.session_state.history.append({"role":"assistant","text":ai_answer})
                memory.setdefault("chats", []).append({"role":"assistant","text":ai_answer, "time":time.time()})
                save_memory(memory)
                placeholder.empty()
                st.session_state.busy = False
                st.experimental_rerun()
            else:
                prompt_text = f"Provide a clear, factual, and well-structured explanation about: {last_user}"
        else:  # Laughing (Jokes)
            prompt_text = f"Reply playfully and short with a clean joke about: {last_user}"

        # choose backend
        ai_answer = None
        try:
            # prefer Gemini if key present
            if st.session_state.get("GEMINI_API_KEY"):
                ai_answer = call_gemini(prompt_text)
                # if gemini returned an obvious error, fallback to local if loaded
                if ai_answer and ai_answer.startswith("[Gemini error") and st.session_state.local_model:
                    ai_answer = call_local(prompt_text)
            else:
                # no gemini: try local model
                if st.session_state.local_model:
                    ai_answer = call_local(prompt_text)
                else:
                    ai_answer = "[No backend available. Set GEMINI_API_KEY in the sidebar or click Load Local Model.]"
        except Exception as e:
            ai_answer = f"[Generation exception: {e}]"

        if not ai_answer:
            ai_answer = "[PopKing couldn't create a reply — check keys/models.]"

        # clean answer
        ai_answer = re.sub(r"^\s*User:.*", "", ai_answer, flags=re.IGNORECASE).strip()
        if not ai_answer:
            ai_answer = "[Empty reply from backend]"

        # save and show
        st.session_state.history.append({"role":"assistant","text":ai_answer})
        memory.setdefault("chats", []).append({"role":"assistant","text":ai_answer, "time":time.time()})
        save_memory(memory)

        # replace placeholder and show result
        placeholder.empty()
        st.chat_message("assistant").markdown(ai_answer)

        # TTS (if enabled)
        audio = tts_bytes(ai_answer)
        if audio:
            st.audio(audio)

        st.session_state.busy = False
        st.experimental_rerun()

# end file
