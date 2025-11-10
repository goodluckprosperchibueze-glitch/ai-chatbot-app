import requests
import json

GOOGLE_API_KEY = AIzaSyDjJgrg8j9UZ0yNUqGqNUGavyKfKvXKf_M

def generate_gemini(prompt):
    url = "https://generativeai.googleapis.com/v1beta2/models/text-bison-001:generate"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GOOGLE_API_KEY}"}
    data = {
        "prompt": {"text": prompt},
        "temperature": 0.7,
        "maxOutputTokens": 300
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["candidates"][0]["output"]
    else:
        return f"Error {response.status_code}: {response.text}"

print(generate_gemini("Explain AI in simple words"))
