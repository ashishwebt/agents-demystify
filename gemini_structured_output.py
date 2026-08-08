import os, json
from dataclasses import dataclass, fields
import requests

@dataclass
class ContactInfo:
    name: str
    email: str
    plan_interest: str
    demo_requested: bool

def json_type(t):
    return "NUMBER" if t in (int, float) else "STRING" if t is str else "BOOLEAN"

def schema_from(cls):
    props = {f.name: {"type": json_type(f.type)} for f in fields(cls)}
    return {"type": "OBJECT", "properties": props, "required": list(props)}

API_KEY = os.environ["GEMINI_API_KEY"]
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

def run(query):
    body = {
        "contents": [{"role": "user", "parts": [{"text": query}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema_from(ContactInfo)},
    }
    resp = requests.post(URL, json=body).json()
    text = resp["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)

TEXT = "John Smith (john@acme.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."

if __name__ == "__main__":
    print("Gemini:", run(TEXT))

# API reference: https://ai.google.dev/gemini-api/docs/structured-output
