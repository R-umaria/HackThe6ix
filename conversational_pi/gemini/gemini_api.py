# gemini_api.py

import requests
from config import GEMINI_API_KEY

def ask_gemini(prompt):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    res = requests.post(url, headers=headers, params=params, json=data)
    if res.status_code == 200:
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        print("Gemini error:", res.text)
        return "Sorry, I had a problem thinking."
