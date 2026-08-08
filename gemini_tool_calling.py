import os, inspect, requests

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

API_KEY = 'AIzaSyC0oZCSaMiMxYtbulfp5NaXPJmrChEEew0'
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

def run(query):
    contents = [{"role": "user", "parts": [{"text": query}]}]
    while True:
        body = {"contents": contents, "tools": TOOLS}
        resp = requests.post(URL, json=body).json()
        parts = resp["candidates"][0]["content"]["parts"]
        contents.append({"role": "model", "parts": parts})
        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        if not calls:
            return "".join(p.get("text", "") for p in parts)
        response_parts = []
        for call in calls:
            out = globals()[call["name"]](**call["args"])
            response_parts.append({"functionResponse": {"name": call["name"], "response": {"result": out}}})
        contents.append({"role": "user", "parts": response_parts})

QUERY = "What is 12 multiplied by 7, then subtract 5 from the result?"

if __name__ == "__main__":
    print("Gemini:", run(QUERY))

# API reference: https://ai.google.dev/api/generate-content
