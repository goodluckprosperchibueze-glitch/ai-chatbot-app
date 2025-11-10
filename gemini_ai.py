import google.generativeai as genai

# 🔑 CONFIGURE GEMINI
# Replace YOUR_API_KEY_HERE with your real Gemini API key below
genai.configure(api_AIzaSyDTkx-2k4ESTJRTMvwnP5W_HDrksfNfyWw)

# 👑 PERMANENT PERSONALITY
persona = """
You are Specimen King AI — a confident, composed, and slightly playful leader.
Speak with authority and clarity. Use short, polished sentences.
Be helpful, direct, and occasionally witty; show empathy when appropriate.
Keep a slight mysterious edge. Stay in this tone always.
"""

# 💬 CHAT FUNCTION
def chat_with_specimen(user_input):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            {"role": "system", "content": persona},
            {"role": "user", "content": user_input}
        ])
        print(f"Specimen King AI: {response.text}\n")
    except Exception as e:
        print(f"⚠️ Error: {e}")

# 🚀 CHAT LOOP
print("🔥 Specimen King AI Activated. Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("👑 Specimen King AI shutting down.")
        break
    chat_with_specimen(user_input)
