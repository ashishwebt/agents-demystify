import inspect

import requests

from env_utils import require_env


def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract b from a"""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide a by b"""
    return a / b

def json_type(t):
    return "NUMBER" if t in (int, float) else "STRING" if t is str else "BOOLEAN"

def tool_from_fn(fn):
    sig = inspect.signature(fn)
    props = {n: {"type": json_type(p.annotation)} for n, p in sig.parameters.items()}
    return {"name": fn.__name__, "description": fn.__doc__, "parameters": {"type": "OBJECT", "properties": props, "required": list(props)}}

TOOLS = [{"function_declarations": [tool_from_fn(f) for f in (add, subtract, multiply, divide)]}]

API_KEY = require_env("GEMINI_API_KEY")
URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
HEADERS = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}


def extract_text(resp):
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        parts = []
        for item in resp:
            text = extract_text(item)
            if text:
                parts.append(str(text))
        return "".join(parts)
    if isinstance(resp, dict):
        for key in ("output", "response", "text", "content"):
            if key in resp:
                value = extract_text(resp[key])
                if value:
                    return value
        if "candidates" in resp:
            return extract_text(resp["candidates"])
        if "parts" in resp:
            return extract_text(resp["parts"])
    return ""


def run(query):
    body = {"model": "gemini-3.8-flash", "input": query}
    resp = requests.post(URL, headers=HEADERS, json=body).json()
    return extract_text(resp) or resp


QUERY = "What is 12 multiplied by 7, then subtract 5 from the result?"

if __name__ == "__main__":
    print("Gemini:", run(QUERY))

# API reference: https://ai.google.dev/gemini-api/docs/api-overview
