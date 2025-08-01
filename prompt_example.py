import openai
import os

# Ensure you have your OPENAI_API_KEY environment variable set
openai.api_key = os.getenv('OPENAI_API_KEY')

prompt = "Write an engaging product description for a solar-powered flashlight"

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=150,
    temperature=0.7
)

print(response.choices[0].text.strip())
