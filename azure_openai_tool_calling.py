import os, json, inspect, requests

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
    return {"type": "function", "function": {"name": fn.__name__, "description": fn.__doc__, "parameters": {"type": "object", "properties": props, "required": list(props)}}}

TOOLS = [tool_from_fn(f) for f in (add, subtract, multiply, divide)]

ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
URL = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version=2024-10-21"
HEADERS = {"api-key": os.environ["AZURE_OPENAI_API_KEY"], "content-type": "application/json"}

def run(query):
    messages = [{"role": "user", "content": query}]
    while True:
        body = {"messages": messages, "tools": TOOLS}
        resp = requests.post(URL, headers=HEADERS, json=body).json()
        msg = resp["choices"][0]["message"]
        messages.append(msg)
        if not msg.get("tool_calls"):
            return msg["content"]
        for tc in msg["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            out = globals()[tc["function"]["name"]](**args)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(out)})

QUERY = "What is 12 multiplied by 7, then subtract 5 from the result?"

if __name__ == "__main__":
    print("Azure OpenAI:", run(QUERY))

# API reference: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/function-calling
