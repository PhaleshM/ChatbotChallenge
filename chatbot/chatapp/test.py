import google.generativeai as genai
from dotenv import load_dotenv
import google.generativeai as genai
import os
load_dotenv()

gemini_api_key = os.environ["GEMINI_API_KEY"]
genai.configure(api_key = gemini_api_key)
import pprint
for model in genai.list_models():
    pprint.pprint(model)