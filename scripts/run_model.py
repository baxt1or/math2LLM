from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY=os.getenv("API_KEY")

client = OpenAI(api_key=API_KEY)

print("You can start from here\n")

messages = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )

    reply = response.choices[0].message.content 
    print("Model:", reply)

    messages.append({"role": "assistant", "content": reply})