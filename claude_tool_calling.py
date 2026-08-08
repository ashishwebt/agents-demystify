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
    return "number" if t in (int, float) else "string" if t is str else "boolean"

def tool_from_fn(fn):
    sig = inspect.signature(fn)
    props = {n: {"type": json_type(p.annotation)} for n, p in sig.parameters.items()}
    return {"name": fn.__name__, "description": fn.__doc__, "input_schema": {"type": "object", "properties": props, "required": list(props)}}

TOOLS = [tool_from_fn(f) for f in (add, subtract, multiply, divide)]

URL = "https://api.anthropic.com/v1/messages"
HEADERS = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"}

def run(query):
    messages = [{"role": "user", "content": query}]
    while True:
        body = {"model": "claude-sonnet-4-6", "max_tokens": 1024, "tools": TOOLS, "messages": messages}
        resp = requests.post(URL, headers=HEADERS, json=body).json()
        messages.append({"role": "assistant", "content": resp["content"]})
        if resp["stop_reason"] != "tool_use":
            return "".join(b["text"] for b in resp["content"] if b["type"] == "text")
        results = []
        for block in resp["content"]:
            if block["type"] == "tool_use":
                out = globals()[block["name"]](**block["input"])
                results.append({"type": "tool_result", "tool_use_id": block["id"], "content": str(out)})
        messages.append({"role": "user", "content": results})

QUERY = "What is 12 multiplied by 7, then subtract 5 from the result?"

if __name__ == "__main__":
    print("Claude:", run(QUERY))

# API reference: https://platform.claude.com/docs/en/api/messages
