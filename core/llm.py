from google import genai
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def gemini_llm(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text
