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
    return "number" if t in (int, float) else "string" if t is str else "boolean"

def schema_from(cls):
    props = {f.name: {"type": json_type(f.type)} for f in fields(cls)}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

URL = "https://api.anthropic.com/v1/messages"
HEADERS = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"}

def run(query):
    body = {
        "model": "claude-opus-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": query}],
        "output_config": {"format": {"type": "json_schema", "schema": schema_from(ContactInfo)}},
    }
    resp = requests.post(URL, headers=HEADERS, json=body).json()
    text = next(b["text"] for b in resp["content"] if b["type"] == "text")
    return json.loads(text)

TEXT = "John Smith (john@acme.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."

if __name__ == "__main__":
    print("Claude:", run(TEXT))

# API reference: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
