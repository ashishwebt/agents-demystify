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
    return "number" if t in (int, float) else "string" if t is str else "boolean"

def schema_from(cls):
    props = {f.name: {"type": json_type(f.type)} for f in fields(cls)}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

ENDPOINT = require_env("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = require_env("AZURE_OPENAI_DEPLOYMENT")
URL = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version=2024-10-21"
HEADERS = {"api-key": require_env("AZURE_OPENAI_API_KEY"), "content-type": "application/json"}

def run(query):
    body = {
        "messages": [{"role": "user", "content": query}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "ContactInfo", "schema": schema_from(ContactInfo), "strict": True}},
    }
    resp = requests.post(URL, headers=HEADERS, json=body).json()
    return json.loads(resp["choices"][0]["message"]["content"])

TEXT = "John Smith (john@acme.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."

if __name__ == "__main__":
    print("Azure OpenAI:", run(TEXT))

# API reference: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/structured-outputs
