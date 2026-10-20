import json
from dataclasses import dataclass, fields

import requests

from env_utils import require_env


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

API_KEY = require_env("GEMINI_API_KEY")
URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
HEADERS = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}


def run(query):
    body = {
        "model": "gemini-3.8-flash",
        "input": query,
    }
    resp = requests.post(URL, headers=HEADERS, json=body).json()
    text = resp.get("output") or resp.get("response") or resp.get("text") or resp
    if isinstance(text, str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


TEXT = "John Smith (john@acme.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."

if __name__ == "__main__":
    print("Gemini:", run(TEXT))

# API reference: https://ai.google.dev/gemini-api/docs/api-overview
