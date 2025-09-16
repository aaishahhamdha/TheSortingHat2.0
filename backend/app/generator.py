import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key= KEY)
model=genai.GenerativeModel("gemini-2.5-flash")
def generate_message(prompt: str):
    try:
        response=model.generate_content(prompt)
        message_text = response.candidates[0].content.parts[0].text
        return message_text
    except Exception as e:
        raise RuntimeError(f"Message Generation failed: {e}")