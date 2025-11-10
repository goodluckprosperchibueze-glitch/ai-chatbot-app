from google import genai

client = genai.Client(api_key=AIzaSyDjJgrg8j9UZ0yNUqGqNUGavyKfKvXKf_M)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Write a short motivational quote for Osemeke Goodluck.",
)

print(response.text)
