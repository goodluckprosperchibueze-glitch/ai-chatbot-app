#!/usr/bin/env python3
"""
PopKing Ultra (Pydroid-ready)
Modes: chat, story, deep, gemini
Simple terminal AI using Google Gemini
"""

import os
import sys
import tempfile

# Optional libraries
try:
    from google import genai
    HAVE_GENAI = True
except:
    HAVE_GENAI = False

try:
    from gtts import gTTS
    HAVE_GTTS = True
except:
    HAVE_GTTS = False

try:
    from playsound import playsound
    HAVE_PLAYSOUND = True
except:
    HAVE_PLAYSOUND = False

# -----------------------
# CONFIG
# -----------------------

GENAI_API_KEY = os.environ.get("GENAI_API_KEY", "")  # Paste your key if needed
PERSONA = ("You are PopKing AI, friendly, encouraging, concise. "
           "Answer clearly, step by step if needed. Admit if you don't know.")

MODE = "chat"  # chat, story, deep, gemini
ENABLE_TTS = False

# -----------------------
# Initialize Gemini client
# -----------------------
genai_client = None
if HAVE_GENAI and GENAI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GENAI_API_KEY)
        print("[info] Gemini client initialized.")
    except Exception as e:
        print("[warn] Failed to init Gemini:", e)

# -----------------------
# Helpers
# -----------------------
def call_gemini(prompt: str, system_instruction: str = None) -> str:
    if genai_client is None:
        return "(Gemini not configured. Set GENAI_API_KEY and install google-genai.)"
    try:
        payload = {"model": "gemini-2.5-flash", "contents": prompt}
        if system_instruction:
            payload["contents"] = f"[SYSTEM]\n{system_instruction}\n\n{prompt}"
        resp = genai_client.models.generate_content(**payload)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"(Gemini call error: {e})"

def speak_text(text: str):
    if not HAVE_GTTS:
        print("[tts] gTTS not installed.")
        return
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        tmp.close()
        if HAVE_PLAYSOUND:
            playsound(tmp.name)
        else:
            print("[tts] playsound not available. File saved:", tmp.name)
    except Exception as e:
        print("[tts error]", e)

def build_prompt(user_input: str, mode: str) -> str:
    if mode == "story":
        return f"{PERSONA}\n\nWrite a long, creative story based on: {user_input}"
    elif mode == "deep":
        return f"{PERSONA}\n\nAnswer carefully and stepwise: {user_input}"
    elif mode == "gemini":
        return user_input
    else:  # chat
        return f"{PERSONA}\nUser: {user_input}\nAI:"

# -----------------------
# Main REPL
# -----------------------
def repl():
    global MODE, ENABLE_TTS
    print("=== PopKing Ultra (Pydroid) ===")
    print("Type /help for commands. Modes: chat, story, deep, gemini")
    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user:
            continue

        # Commands
        if user.startswith("/"):
            parts = user[1:].strip().split()
            cmd = parts[0].lower()
            if cmd == "help":
                print("Commands:\n"
                      " /mode chat/story/deep/gemini\n"
                      " /tts on/off\n"
                      " /exit")
                continue
            elif cmd == "mode" and len(parts) > 1:
                if parts[1] in ("chat", "story", "deep", "gemini"):
                    MODE = parts[1]
                    print(f"[mode] Switched to {MODE}")
                else:
                    print("[usage] /mode chat|story|deep|gemini")
                continue
            elif cmd == "tts" and len(parts) > 1:
                ENABLE_TTS = parts[1] == "on"
                print(f"[tts] ENABLE_TTS = {ENABLE_TTS}")
                continue
            elif cmd == "exit":
                print("Goodbye!")
                break
            else:
                print("[unknown command]")
                continue

        # Build prompt
        prompt = build_prompt(user, MODE)

        # Generate response
        if HAVE_GENAI and genai_client:
            if MODE == "gemini":
                resp = call_gemini(prompt)
            else:
                resp = call_gemini(prompt, system_instruction=PERSONA)
        else:
            resp = "(No Gemini client available. Install google-genai and set GENAI_API_KEY.)"

        # Show response
        print("PopKing AI:", resp)
        if ENABLE_TTS:
            speak_text(resp)
        print()

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    repl()
