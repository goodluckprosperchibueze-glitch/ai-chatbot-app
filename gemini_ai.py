from google import genai

# --- 1. Connect to Google Gemini ---
client = genai.Client(api_key="")  # AIzaSyDjJgrg8j9UZ0yNUqGqNUGavyKfKvXKf_M
# --- 2. Send Prompt ---
response = client.models.generate_content(
    model="gemini-2.5-flash",AIzaSyDjJgrg8j9UZ0yNUqGqNUGavyKfKvXKf_M)
    contents="Explain how AI works in a few words",
)

# --- 3. Print Reply ---
print(response.text)
