# ai_content_generator.py

from transformers import pipeline

# Load AI model
generator = pipeline("text-generation", model="gpt2")

print("🤖 AI Content Generator — by Goodluck")
print("Type a topic and let the AI write something smart for you!\n")

while True:
    topic = input("Enter your topic (or 'exit' to quit): ")

    if topic.lower() == "exit":
        print("Goodbye 👋")
        break

    # Generate AI content
    result = generator(
        f"Write a creative short paragraph about {topic}:",
        max_length=120,
        do_sample=True,
        top_p=0.95,
        top_k=60
    )

    print("\n📝 AI Content:\n")
    print(result[0]['generated_text'])
    print("\n" + "-" * 60 + "\n")
