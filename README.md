#!/usr/bin/env python3
"""
PopKing Ultra (Pydroid-ready)
- Simple terminal chat client that calls Google Gemini (via google-genai) and Hugging Face image API.
- Modes: chat, story, deep, gemini
- Optional TTS (gTTS) and STT (speech_recognition)
- Usage:
    1. Set GENAI_API_KEY and HF_TOKEN as environment variables, or paste them below.
    2. Install optional libs if you want voice/images.
    3. Run: python popking_pydroid.py
- Commands (in chat):
    /mode chat      -> normal chat
    /mode story     -> story mode (long creative story)
    /mode deep      -> deep search style (stepwise answer)
    /mode gemini    -> direct Gemini prompt (no persona wrapper)
    /img <prompt>   -> generate image with HF (saves as last_image.png)
    /tts on/off     -> enable/disable TTS playback
    /help           -> show commands
    /exit           -> quit
"""

import os
import sys
import json
import base64
import requests
import tempfile
import time

# API keys: prefer environment variables; you may paste as strings for quick testing (not recommended)
GENAI_API_KEY = os.environ.get("GENAI_API_KEY", "")  # or paste your key: "AIza..."
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Hugging Face token

# Optional libraries
try:
    from google import genai
    HAVE_GENAI = True
except Exception:
    HAVE_GENAI = False

try:
    from gtts import gTTS
    HAVE_GTTS = True
except Exception:
    HAVE_GTTS = False

try:
    import speech_recognition as sr
    HAVE_SR = True
except Exception:
    HAVE_SR = False

# playsound for playback (may work on Pydroid)
try:
    from playsound import playsound
    HAVE_PLAYSOUND = True
except Exception:
    HAVE_PLAYSOUND = False

# Basic persona — tweak this to tune tone
PERSONA = (
    "You are PopKing AI (friendly, encouraging, study-focused, concise). "
    "Teach step-by-step when needed. Offer short practice questions. Avoid hallucination; say 'I don't know' if unsure."
)

# Runtime flags / settings
MODE = "chat"  # chat | story | deep | gemini
ENABLE_TTS = True
LAST_IMAGE_PATH = None

# Gemini client setup (if available)
genai_client = None
if HAVE_GENAI and GENAI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GENAI_API_KEY)
    except Exception as e:
        genai_client = None
        print("[warn] google-genai import worked but client init failed:", e)
elif HAVE_GENAI and not GENAI_API_KEY:
    print("[info] google-genai available but GENAI_API_KEY not set. Gemini mode will not work until set.")

# Simple helper: call Gemini (via google-genai client) if available
def call_gemini(contents: str, system_instruction: str = None) -> str:
    """
    Returns generated text from Gemini (gemini-2.5-flash) using google-genai client.
    If client missing, returns an explanatory message.
    """
    if genai_client is None:
        return "(Gemini not configured. Set GENAI_API_KEY and install google-genai.)"
    try:
        payload = {"model": "gemini-2.5-flash", "contents": contents}
        # google-genai client supports system_instruction param in many setups; include it if provided
        if system_instruction:
            # some client versions accept system_instruction; include as a first content string
            payload["contents"] = f"[SYSTEM]\n{system_instruction}\n\n{contents}"
        resp = genai_client.models.generate_content(**payload)
        # .text is usually the convenience property
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"(Gemini call error: {e})"

# Hugging Face image generation (Inference API)
def hf_generate_image(prompt_text: str, hf_token: str = None) -> str:
    """
    Calls Hugging Face inference API for an image model and saves PNG to a temp file.
    Returns path to saved image or raises RuntimeError.
    """
    token = hf_token or HF_TOKEN
    if not token:
        raise RuntimeError("Hugging Face token missing. Set HF_TOKEN env var or edit script.")
    # Recommended model: stabilityai/stable-diffusion-2 or sd-2-1
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt_text}
    print("[info] Sending image request to Hugging Face... (this can take a while)")
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        # try to show helpful message
        raise RuntimeError(f"Hugging Face returned {resp.status_code}: {resp.text}")
    ct = resp.headers.get("content-type", "")
    # HF sometimes returns raw image bytes or JSON with base64
    if "application/json" in ct:
        data = resp.json()
        # typical form: {"images": ["<b64>"]} OR {"error": "..."}
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HF error: {data['error']}")
        if isinstance(data, dict) and "images" in data:
            b64 = data["images"][0]
            img_bytes = base64.b64decode(b64)
        elif isinstance(data, list) and data and isinstance(data[0], str):
            img_bytes = base64.b64decode(data[0])
        else:
            raise RuntimeError("Unexpected JSON response structure from HF.")
    else:
        img_bytes = resp.content
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(img_bytes)
    tmp.flush()
    tmp.close()
    print("[info] Image saved to", tmp.name)
    return tmp.name

# TTS helper
def speak_text(text: str) -> str:
    """
    Convert text to speech with gTTS. Returns path to mp3 or error message.
    """
    if not HAVE_GTTS:
        return "(gTTS not installed; enable TTS by installing gTTS.)"
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        tmp.close()
        if HAVE_PLAYSOUND:
            try:
                playsound(tmp.name)
            except Exception as e:
                print("[warn] playsound failed:", e)
        else:
            print("[info] playsound not available; mp3 saved to:", tmp.name)
        return tmp.name
    except Exception as e:
        return f"(TTS error: {e})"

# STT helper (recognize from a wav/mp3 file path)
def transcribe_file(path: str) -> str:
    if not HAVE_SR:
        return "(SpeechRecognition not installed.)"
    r = sr.Recognizer()
    try:
        with sr.AudioFile(path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        return text
    except Exception as e:
        return f"(STT error: {e})"

# Small utility to build prompts based on current mode
def build_prompt(user_input: str, mode: str, persona: str) -> str:
    if mode == "story":
        return f"{persona}\n\nWrite a long, imaginative story based on: {user_input}\n\nStart now."
    elif mode == "deep":
        return (
            f"{persona}\n\nProvide a careful, stepwise, in-depth answer to: {user_input}\n"
            "Include suggested search keywords and verification steps."
        )
    elif mode == "gemini":
        # raw pass-through to Gemini (no persona wrapper) — user wants direct Gemini style
        return user_input
    else:  # chat
        return f"{persona}\n\nConversation:\nUser: {user_input}\nAI:"

# Simple local loop
def repl():
    global MODE, ENABLE_TTS, LAST_IMAGE_PATH
    print("=== PopKing Ultra (Pydroid) ===")
    print("Type /help for commands. Modes: chat, story, deep, gemini")
    print("If Gemini mode is used you must set GENAI_API_KEY and install google-genai.")
    print()

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user:
            continue

        # Commands
        if user.startswith("/"):
            cmd = user[1:].strip().lower()
            if cmd.startswith("mode"):
                # /mode <name>
                parts = cmd.split()
                if len(parts) >= 2 and parts[1] in ("chat", "story", "deep", "gemini"):
                    MODE = parts[1]
                    print(f"[mode] Switched to {MODE}")
                else:
                    print("[usage] /mode <chat|story|deep|gemini>")
                continue
            if cmd == "help":
                print(
                    "Commands:\n"
                    "  /mode <chat|story|deep|gemini>   - switch mode\n"
                    "  /img <prompt>                    - generate image via Hugging Face\n"
                    "  /tts on | /tts off               - enable/disable TTS playback\n"
                    "  /stt <path-to-audio>            - transcribe local audio file\n"
                    "  /exit                            - quit\n"
                    "  /help                            - show this message\n"
                )
                continue
            if cmd.startswith("img"):
                # /img your prompt
                rest = cmd[3:].strip()
                if not rest:
                    print("[usage] /img <prompt>")
                    continue
                try:
                    path = hf_generate_image(rest)
                    LAST_IMAGE_PATH = path
                    print(f"[img] Saved image to: {path}")
                except Exception as e:
                    print("[img error]", e)
                continue
            if cmd.startswith("tts"):
                # /tts on/off
                parts = cmd.split()
                if len(parts) >= 2 and parts[1] in ("on", "off"):
                    ENABLE_TTS = parts[1] == "on"
                    print(f"[tts] ENABLE_TTS = {ENABLE_TTS}")
                else:
                    print("[usage] /tts on | /tts off")
                continue
            if cmd.startswith("stt"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    print("[usage] /stt <audio-file-path>")
                    continue
                path = parts[1].strip()
                if not os.path.exists(path):
                    print("[stt] File not found:", path)
                    continue
                print("[stt] transcribing...")
                txt = transcribe_file(path)
                print("[stt] result:", txt)
                continue
            if cmd == "exit":
                print("Goodbye.")
                break
            print("[unknown command] type /help for commands.")
            continue

        # Non-command: user message to AI
        prompt = build_prompt(user, MODE, PERSONA)
        # Gemini mode prefers direct call; other modes use persona-wrapped prompts but may still call Gemini
        if HAVE_GENAI and genai_client:
            # Use Gemini
            if MODE == "gemini":
                resp = call_gemini(prompt, system_instruction=None)
            else:
                resp = call_gemini(prompt, system_instruction=PERSONA)
        else:
            # No gemini: fallback to a simple local echo or instruct user to install
            resp = "(No Gemini client available. Install google-genai and set GENAI_API_KEY to use Gemini.)"

        # Print reply
        print("\nPopKing AI:", resp, "\n")

        # TTS if enabled
        if ENABLE_TTS:
            if HAVE_GTTS:
                print("[tts] generating speech...")
                tpath = speak_text(resp)
                if isinstance(tpath, str) and tpath.startswith("(TTS"):
                    print(tpath)
                else:
                    print("[tts] audio saved to", tpath)
            else:
                print("[tts] gTTS not available; install gtts to enable.")

# Entry point
if __name__ == "__main__":
    repl()
