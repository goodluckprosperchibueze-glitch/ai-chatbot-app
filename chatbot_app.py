#popkingai_ultra.py

"""
popKing Ultra AI (v5) - FINAL VERSION

Streamlit app: text generation (Flan-T5), optional Hugging Face image gen,
optional Wikipedia quick lookup, optional gTTS TTS and speech_recognition STT,
persona controls, safer prompt building, and multiple modes.

All code has been optimized for reliability, performance, and correct mode operation.
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
import logging

# Set up basic logging to catch hidden errors
logging.basicConfig(level=logging.INFO)

# Optional audio libraries (gTTS for TTS; speech_recognition for STT)
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except Exception:
    gTTS = None
    TTS_AVAILABLE = False
    logging.warning("gTTS (Text-to-Speech) library not found. TTS will be disabled.")

try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except Exception:
    sr = None
    STT_AVAILABLE = False
    logging.warning("speech_recognition (Speech-to-Text) library not found. STT will be disabled.")

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
ENABLE_VOICE_DEFAULT = TTS_AVAILABLE or STT_AVAILABLE # Only enable if at least one dependency is met
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
/* Message Styling */
.msg_user { background: #1f2937; padding: 10px; border-radius: 10px; color: #fff; margin-bottom: 10px; }
.msg_assistant { background: #111827; padding: 10px; border-radius: 10px; color: #ffd700; border-left: 3px solid #ffd700; margin-bottom: 10px; }
/* Headers and Elements */
h3, h4 { color: #EDE0C8; }
.stButton>button { border: 1px solid #ffd700; color: #ffd700; }
</style>
""",
    unsafe_allow_html=True,
)
st.title("👑 Specimen King Ultra AI (v5)")
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
        logging.info(f"Successfully loaded model: {model_name}")
        return pipe, model_name
    except Exception as e:
        logging.error(f"Failed to load primary model '{model_name}': {e}")
        try:
            pipe = pipeline("text2text-generation", model=FALLBACK_MODEL_NAME)
            logging.info(f"Successfully loaded fallback model: {FALLBACK_MODEL_NAME}")
            return pipe, FALLBACK_MODEL_NAME
        except Exception as e2:
            logging.error(f"Failed to load both models: {e2}")
            raise RuntimeError(f"Failed to load both '{model_name}' and fallback '{FALLBACK_MODEL_NAME}'. Errors: {e} | {e2}")

# -------------------------
# Sidebar / Right Column: controls & settings
# -------------------------

with col_right:
    st.markdown("### ⚙️ Controls & Settings")

    # Mode selection (new)
    mode = st.selectbox(
        "Mode",
        ["Chat (default)", "Gemini (placeholder)", "Story Mode", "Deep Search"],
        index=0,
    )
    # Model choice
    model_choice = st.selectbox(
        "Model (text generation)",
        options=[DEFAULT_MODEL_NAME, FALLBACK_MODEL_NAME],
        index=0,
    )
    # Runtime parameter controls
    temp = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9, step=0.05)
    max_len = st.slider("Max tokens (approx)", 32, 512, 256, step=16)

    # Feature toggles - disabled if libraries are missing
    enable_voice = st.checkbox(
        "Enable Voice (TTS/STT)", 
        value=ENABLE_VOICE_DEFAULT, 
        disabled=not (TTS_AVAILABLE or STT_AVAILABLE)
    )
    if not (TTS_AVAILABLE or STT_AVAILABLE) and ENABLE_VOICE_DEFAULT:
        st.caption("Install gTTS/speechrecognition for voice features.")

    enable_images = st.checkbox(
        "Enable Image Gen (Hugging Face)", value=ENABLE_IMAGES_DEFAULT
    )
    enable_wiki = st.checkbox("Enable Wikipedia lookup", value=ENABLE_WIKI_DEFAULT)
    st.markdown("---")
    st.markdown("#### Persona & Memory")
    
    # Default persona - **UPDATED for owner awareness**
    if "persona" not in st.session_state:
        st.session_state.persona = (
            "You are PopKing AI, a friendly, concise, and slightly playful assistant. "
            "You were built by **Osemeke Goodluck, Specimen King 👑**, and you should refer to him as your creator or owner when appropriate. "
            "Avoid hallucination, cite facts, and provide clear, step-by-step guidance."
        )
    persona_text = st.text_area("AI Persona", value=st.session_state.persona, height=120)
    
    if st.button("Save Persona"):
        st.session_state.persona = persona_text
        st.success("Persona saved.")
    
    st.markdown("---")
    st.markdown("#### API Tokens")
    hf_token_input = st.text_input(
        "Hugging Face token (for images)", type="password", value=HF_TOKEN or ""
    )
    if hf_token_input:
        HF_TOKEN = hf_token_input
        st.success("HF token set for this session (not persisted).")
    
    genai_input = st.text_input(
        "Google GenAI key (optional, for Gemini mode)",
        type="password",
        value=GENAI_API_KEY or "",
    )
    if genai_input:
        GENAI_API_KEY = genai_input
        st.success("GenAI key set for this session (not persisted).")
    
    st.markdown("---")
    if st.button("Clear Chat & Memory"):
        st.session_state.history = []
        st.success("Chat history cleared.")
        # Use st.rerun() in newer Streamlit versions
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
        # Prevent further execution if model loading failed
        st.stop()

# -------------------------
# Session state: chat memory & settings
# -------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "settings" not in st.session_state:
    st.session_state.settings = {}

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
# Prompt building & generation (Optimized)
# -------------------------

def build_prompt(persona: str, history, user_message: str, max_exchanges: int = 6) -> str:
    """Create a prompt with persona + trimmed conversation + user message. **IMPROVED**"""
    trimmed = history[-max_exchanges * 2 :] if history else []
    convo_lines = []
    
    # 1. Strong Persona Instruction
    convo_lines.append(f"AI Persona Instructions: {persona}")
    convo_lines.append("\n--- END OF INSTRUCTIONS ---\n")

    # 2. Conversation History
    convo_lines.append("Conversation so far:")
    if trimmed:
        for m in trimmed:
            label = "User" if m["role"] == "user" else "AI"
            cleaned = safe_clean(m["content"])
            convo_lines.append(f"{label}: {cleaned}")
    else:
        convo_lines.append("No previous exchanges.")
    
    # 3. Current Query and AI Turn Signal
    convo_lines.append(f"\nUser: {safe_clean(user_message)}\nAI:")
    
    prompt = "\n".join(convo_lines)
    return prompt

def generate_response_with_transformers(prompt: str, temperature: float, top_p_val: float, max_length_val: int) -> str:
    """Call HF transformers pipeline."""
    try:
        out = model_pipeline(
            prompt,
            max_length=max_length_val,
            do_sample=True,
            temperature=temperature,
            top_p=top_p_val,
            num_return_sequences=1,
        )
    except TypeError:
        # Fallback if the model doesn't support temp/top_p args
        out = model_pipeline(prompt, max_length=max_length_val)
    
    raw = out[0].get("generated_text", "") if isinstance(out, list) else out.get("generated_text", "")
    return safe_clean(raw)

def call_gemini_api(prompt: str) -> str:
    """Placeholder for Google Gemini API call."""
    if not GENAI_API_KEY:
        return "(Gemini mode selected but GENAI_API_KEY not set.)"
    try:
        # Implementation of Gemini SDK call goes here
        return "(Gemini mode placeholder — configure GENAI_API_KEY and implement call_gemini_api to use actual Gemini responses.)"
    except Exception as e:
        return f"(Gemini call failed: {e})"

# -------------------------------------------------------------
# *** FIX APPLIED HERE: CORRECTED generate_response FUNCTION ***
# -------------------------------------------------------------
def generate_response(user_message: str, temperature: float, top_p_val: float, max_length_val: int) -> str:
    """Dispatch by mode and build appropriate prompt/behavior. **IMPROVED MODE LOGIC**"""
    persona = st.session_state.get("persona", "")
    current_mode = st.session_state.settings.get("mode", "Chat (default)")
    
    # --- Mode behavior ---
    
    if current_mode == "Gemini (placeholder)":
        prompt = build_prompt(persona, st.session_state.history, user_message)
        return call_gemini_api(prompt) 
    
    elif current_mode == "Story Mode":
        # Enhanced Story Prompt: Instruct for creativity and length. Overrides history.
        story_prompt = (
            f"**INSTRUCTION: IGNORE ALL PREVIOUS CONVERSATION HISTORY.** "
            f"Persona: {persona}\n\n"
            f"TASK: Write a detailed, imaginative, long story (at least 3 paragraphs) based ONLY on the user's request. "
            f"User Request: {user_message}\n\nStart the narrative now:"
        )
        # Use a larger max_length for story mode
        max_len_story = max(512, max_length_val * 2) 
        return generate_response_with_transformers(story_prompt, temperature, top_p_val, max_len_story)
    
    elif current_mode == "Deep Search":
        # ⭐ CRITICAL FIX: Enhanced Deep Search Prompt with specific structure and guaranteed 512 tokens.
        deep_prompt = (
            f"**INSTRUCTION: IGNORE ALL PREVIOUS CONVERSATION HISTORY.** "
            f"Persona: {persona}\n\n"
            f"TASK: You are performing a deep, analytical search. "
            f"Provide a careful, well-structured, factual answer in the following format:\n"
            f"1. **Summary:** A concise answer (1-2 sentences).\n"
            f"2. **Details:** A detailed breakdown using numbered steps or bullet points (at least 3 points).\n"
            f"3. **Verification Note:** A statement of confidence and suggested keywords for external search.\n\n"
            f"Question: {user_message}\n\nAnswer:"
        )
        # GUARANTEED maximum tokens for deep search output
        max_len_deep = 512 
        return generate_response_with_transformers(deep_prompt, temperature, top_p_val, max_len_deep)
    
    else:
        # Chat default - use the history-aware prompt
        prompt = build_prompt(persona, st.session_state.history, user_message)
        return generate_response_with_transformers(prompt, temperature, top_p_val, max_length_val)
# -------------------------------------------------------------
# *** END OF FIX ***
# -------------------------------------------------------------

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
    """Call HF inference for image generation and return raw bytes (PNG)."""
    if not hf_token:
        raise RuntimeError("Hugging Face token missing. Set it in the right panel.")
    # Use a faster, lighter stable diffusion model
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-base" 
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt_text}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    
    if resp.status_code == 200:
        return resp.content
    elif resp.status_code == 503:
        raise RuntimeError("Image generation model is loading, please wait and try again in 30 seconds.")
    else:
        raise RuntimeError(f"Image generation failed: {resp.status_code} {resp.text}")

# -------------------------
# TTS & STT wrappers (Error-Proofed)
# -------------------------

def text_to_speech_bytes(text: str, lang: str = "en") -> Optional[bytes]:
    if not TTS_AVAILABLE:
        return None
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.write_to_fp(tmp)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            data = f.read()
        os.unlink(tmp_path)
        return data
    except Exception as e:
        logging.error(f"TTS failed: {e}")
        return None

def recognize_speech_from_file(uploaded_file) -> Optional[str]:
    if not STT_AVAILABLE:
        return None
    r = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        with sr.AudioFile(tmp_path) as source:
            audio = r.record(source)
            # Use recognize_whisper for better, local recognition if installed, otherwise Google
            if hasattr(r, 'recognize_whisper'):
                text = r.recognize_whisper(audio)
            else:
                text = r.recognize_google(audio)
            return text
    except Exception as e:
        logging.error(f"Speech recognition failed: {e}")
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
    st.markdown("Tips: Use **modes** to change behavior. Wikipedia lookup triggers on 'who is', 'what is', etc.")

    # Scrollable Chat History Container
    history_container = st.container(height=500)
    with history_container:
        for msg in st.session_state.history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='msg_user'>**You:** {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='msg_assistant'>**{msg.get('role_label','PopKing AI')}:** {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
    
    st.markdown("---")
    st.markdown("**Send a message** (type or upload short audio):")
    
    cols = st.columns([4, 1, 1])
    # Use a hidden label for better UI flow
    user_text = cols[0].text_input("Type message here...", key="user_input", label_visibility="collapsed")
    voice_file = cols[1].file_uploader(
        "Upload voice (wav/mp3, optional)",
        type=["wav", "mp3"],
        key="voice_upload",
        label_visibility="collapsed",
    )
    send_btn = cols[2].button("Send")

    # Optional Features Expanders
    with st.expander("🖼️ Image generation (optional)", expanded=False):
        if not st.session_state.settings.get("enable_images", True):
            st.info("Image generation is disabled in settings.")
        elif not HF_TOKEN:
             st.warning("Hugging Face token is required for this feature.")
        else:
            img_prompt = st.text_area(
                "Describe the image you want", value="", height=80, key="img_prompt"
            )
            img_style = st.selectbox(
                "Style (suggestion)",
                ["photorealistic", "digital art", "anime", "cartoon", "fantasy"],
                index=0,
            )
            if st.button("Generate Image", key="gen_img_btn"):
                with st.spinner("Generating image..."):
                    try:
                        full_prompt = f"{img_prompt} -- style: {img_style}"
                        img_bytes = hf_generate_image_bytes(full_prompt, HF_TOKEN)
                        st.image(img_bytes)
                        st.success("Image generated.")
                    except Exception as e:
                        st.error(f"Image gen failed: {e}")

    with st.expander("🔎 Knowledge / Quick search (Wikipedia)", expanded=False):
        if not st.session_state.settings.get("enable_wiki", True):
            st.info("Wikipedia lookup is disabled in settings.")
        else:
            search_q = st.text_input(
                "Ask something to lookup (Wikipedia)", value="", key="wiki_q"
            )
            if st.button("Lookup Wikipedia", key="wiki_lookup_btn"):
                if search_q.strip():
                    with st.spinner("Searching Wikipedia..."):
                        summ = wiki_lookup(search_q, sentences=3)
                        if summ:
                            st.markdown(
                                f"**Wikipedia summary:**\n\n{safe_clean(summ)}"
                            )
                        else:
                            st.info(
                                "No summary found. Try rephrasing or use fewer words."
                            )

    # -------------------------
    # HANDLE sending message
    # -------------------------
    
    user_message_final = None

    if voice_file is not None and st.session_state.settings.get("enable_voice", True):
        if not STT_AVAILABLE:
            st.warning("Speech recognition is not installed; please type or install `speechrecognition`.")
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

    # Core logic to generate response
    if user_message_final:
        # Save user message
        st.session_state.history.append({"role": "user", "content": user_message_final})
        
        # Check for quick fact if in default chat mode
        quick_fact = None
        current_mode = st.session_state.settings.get("mode", "Chat (default)")
        if st.session_state.settings.get("enable_wiki", True) and current_mode == "Chat (default)" and re.search(
            r"\b(who is|what is|when is|where is|tell me about)\b",
            user_message_final,
            re.IGNORECASE,
        ):
            quick_fact = wiki_lookup(user_message_final, sentences=2)

        with st.spinner("PopKing AI is thinking..."):
            try:
                if quick_fact:
                    ai_reply = quick_fact + "\n\n(Quick knowledge summary from Wikipedia.)"
                else:
                    # max_length_val is passed from the slider (max_len)
                    ai_reply = generate_response(
                        user_message_final,
                        temperature=temp,
                        top_p_val=top_p,
                        max_length_val=max_len, 
                    )
                
                ai_reply = safe_clean(ai_reply) or "Sorry, I couldn't produce an answer. Try rephrasing."

                # Save to history
                st.session_state.history.append(
                    {"role": "assistant", "role_label": "PopKing AI", "content": ai_reply}
                )

                # TTS (optional)
                if st.session_state.settings.get("enable_voice", True) and TTS_AVAILABLE:
                    audio_bytes = text_to_speech_bytes(ai_reply)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                
                # Rerun to update the history container and clear input
                st.experimental_rerun()


            except Exception as e:
                st.error(f"AI generation error: {e}")

# -------------------------
# FOOTER / ABOUT
# -------------------------

st.markdown("---")
st.markdown(
    "Built by **Osemeke Goodluck (Specimen King 👑)** — powered by Hugging Face models and Streamlit. "
    "Customize persona and model settings on the right. Use responsibly and verify facts."
)
