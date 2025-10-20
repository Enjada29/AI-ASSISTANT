# test_openai.py
import os
from dotenv import load_dotenv
import openai

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("API key is not set!")
    exit()

openai.api_key = api_key
print("OPENAI_API_KEY successfully loaded:", api_key[:10] + "...")  

try:
    print("Simple test: API key is readable by Python.")
except Exception as e:
    print("Unexpected error:", e)
