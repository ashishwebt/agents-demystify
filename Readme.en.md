# LLM Tool Calling & Structured Outputs — Raw HTTP Demos

## Intent

Frameworks like the Claude Agent SDK, OpenAI's Agents SDK, and LangChain make tool calling
and structured output feel like magic: you decorate a function, hand it to an `Agent`, and
answers come back typed and correct. That magic is genuinely useful in production, but it
also hides the actual HTTP contract — which makes debugging, cost reasoning, and porting
between providers harder than it needs to be.

This repo strips all of that away. Six scripts, one per (provider × pattern), make **plain
`requests.post()` calls** with no `anthropic`, `openai`, `google-genai`, or agent-framework
package involved. The goal is to make two things concrete:

1. **Tool calling** — the model doesn't run code; it asks *you* to, in a structured way, and
   the "agent loop" is just you feeding results back in a `while` loop.
2. **Structured output** — the model's final answer is constrained to a JSON schema via
   constrained decoding; there's no loop and no local execution at all.

Once you've seen the raw shape of both, what agent frameworks add on top becomes much
easier to reason about (see §7).

| File | Provider | Pattern |
|---|---|---|
| `claude_tool_calling.py` | Anthropic Claude | Tool calling (agentic loop) |
| `azure_openai_tool_calling.py` | Azure OpenAI | Tool calling (agentic loop) |
| `gemini_tool_calling.py` | Google Gemini | Tool calling (agentic loop) |
| `claude_structured_output.py` | Anthropic Claude | Structured output (JSON extraction) |
| `azure_openai_structured_output.py` | Azure OpenAI | Structured output (JSON extraction) |
| `gemini_structured_output.py` | Google Gemini | Structured output (JSON extraction) |

---

## 1. Prerequisites

- Python 3.9+
- `pip install -r requirements.txt`
- API credentials for whichever provider(s) you want to run (see below)

No other packages are required — that's the point.

---

## 2. Environment variables

| Provider | Variable | Notes |
|---|---|---|
| Claude | `ANTHROPIC_API_KEY` | From the Claude Platform console |
| Azure OpenAI | `AZURE_OPENAI_ENDPOINT` | e.g. `https://your-resource.openai.azure.com` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | Resource key from Azure Portal → Keys and Endpoint |
| Azure OpenAI | `AZURE_OPENAI_DEPLOYMENT` | Your **deployment name** (not the base model name) |
| Gemini | `GEMINI_API_KEY` | From Google AI Studio |

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=AIza...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

---

## 3. Running

Each script is self-contained and runs once against a hardcoded example:

```bash
python claude_tool_calling.py
python azure_openai_tool_calling.py
python gemini_tool_calling.py

python claude_structured_output.py
python azure_openai_structured_output.py
python gemini_structured_output.py
```

---

## 4. What Is an "Agent Loop"?

A single LLM API call is stateless and can only return text. Tool calling turns that into
a *loop* because the model can't act on its own — it can only ask your code to act, one
response at a time. Stripped to its essence, an agent loop is:

```
messages = [user's request]
while True:
    response = call_model(messages, available_tools)
    append response to messages
    if response contains no tool call:
        return response as the final answer
    for each tool call in response:
        result = run_the_corresponding_local_function(tool_call.arguments)
        append result to messages, tagged so the model can match it to its request
```

That's the whole pattern — every "AI agent" you've used is this loop with more tools,
more guardrails, and usually a cap on iterations. Section 5 shows this loop implemented
by hand, once per provider, with no framework involved.

Three things make a loop implementation non-trivial once you go past a toy example:

- **State management** — `messages` grows every turn and must be replayed in full on every
  call (no server-side memory), so context window and cost both grow with loop length.
- **Termination** — you need a stop condition beyond "no tool call": max iterations, a
  token budget, a timeout, or a human-in-the-loop checkpoint before risky actions.
- **Parallel/concurrent tool calls** — a single response can request multiple tool calls at
  once; a correct loop must execute and return results for all of them before the next call.

---

## 5. Pattern A — Tool Calling (hand-rolled loop)

### 5.1 The four "tools"

```python
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b
# ...subtract, multiply, divide follow the same shape
```

The docstring becomes the tool's `description`; the type-hinted parameters become the
JSON schema `properties`.

### 5.2 Schema generation (no hand-written JSON)

```python
def json_type(t):
    return "number" if t in (int, float) else "string" if t is str else "boolean"

def tool_from_fn(fn):
    sig = inspect.signature(fn)
    props = {n: {"type": json_type(p.annotation)} for n, p in sig.parameters.items()}
    return {"name": fn.__name__, "description": fn.__doc__, ...}
```

`inspect.signature()` reads parameter names and type hints; `fn.__doc__` supplies the
description. Add a parameter to `add()` and the schema updates itself.

### 5.3 Dispatching the model's tool call

The tool's `name` is generated from `fn.__name__`, so the model's returned name is already
a valid key into the module's global namespace — no manual lookup table needed:

```python
out = globals()[block["name"]](**block["input"])
```

### 5.4 The loop, per provider

| Step | Claude | Azure OpenAI | Gemini |
|---|---|---|---|
| Endpoint | `POST /v1/messages` | `POST /openai/deployments/{deployment}/chat/completions` | `POST /models/{model}:generateContent` |
| Auth header | `x-api-key` | `api-key` | `?key=` query param |
| Tool call signal | `resp["stop_reason"] == "tool_use"` | `msg["tool_calls"]` present | `functionCall` parts present |
| Tool schema field | `input_schema` | `parameters` (nested in `function`) | `parameters` (nested in `function_declarations`) |
| Tool result role | `{"role": "user", "content": [{"type": "tool_result", ...}]}` | `{"role": "tool", "tool_call_id": ..., "content": ...}` | `{"role": "user", "parts": [{"functionResponse": {...}}]}` |
| JSON type casing | lowercase (`"number"`) | lowercase (`"number"`) | UPPERCASE (`"NUMBER"`) |

### 5.5 Sample input

```python
QUERY = "What is 12 multiplied by 7, then subtract 5 from the result?"
```

Forces two sequential tool calls (`multiply` then `subtract`) so the loop visibly iterates
more than once.

---

## 6. Pattern B — Structured Output (no loop)

Structured output is **not** tool calling: there's no function execution and no loop. You
give the model a JSON schema for its *answer*, and the API guarantees — via constrained
decoding / grammar compilation — that the response text is valid JSON matching that schema.

### 6.1 Shared schema source

```python
@dataclass
class ContactInfo:
    name: str
    email: str
    plan_interest: str
    demo_requested: bool

def schema_from(cls):
    props = {f.name: {"type": json_type(f.type)} for f in fields(cls)}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
```

### 6.2 Per-provider request shape

| Provider | Config field | Where the JSON lands in the response |
|---|---|---|
| Claude | `output_config: {"format": {"type": "json_schema", "schema": ...}}` on `/v1/messages` | `response["content"]` — a text block whose `text` is the JSON string |
| Azure OpenAI | `response_format: {"type": "json_schema", "json_schema": {"schema": ..., "strict": true}}` on chat completions | `choices[0]["message"]["content"]` — a JSON string |
| Gemini | `generationConfig: {"responseMimeType": "application/json", "responseSchema": ...}` | `candidates[0]["content"]["parts"][0]["text"]` — a JSON string |

Every script does `json.loads(...)` on that text to get a native Python dict.

> Claude's structured outputs previously required an `anthropic-beta:
> structured-outputs-2025-11-13` header and an `output_format` field. That's been promoted
> to the stable `output_config.format` used here (the beta header still works during a
> transition period but isn't required).

### 6.3 Sample input

```python
TEXT = "John Smith (john@acme.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."
```

Expected output shape from all three:

```python
{
    "name": "John Smith",
    "email": "john@acme.com",
    "plan_interest": "Enterprise",
    "demo_requested": True,
}
```

### 6.4 Tool calling vs. structured output

| | Tool Calling | Structured Output |
|---|---|---|
| Purpose | Let the model *act* — invoke functions with arguments | Constrain the model's *final answer* to a schema |
| Round trips | Multiple (loop until no more tool calls) | One |
| Who executes code | Your app, locally, per tool call | Nobody — the model just writes JSON text |
| Schema describes | Function inputs | The response body |
| Typical use case | Calculators, DB lookups, API calls, agents | Data extraction, classification, formatting |

---

## 7. What Agent Frameworks Add On Top

Given the loop in §4, here's what LangChain/LangGraph, the Claude Agent SDK, and the
OpenAI Agents SDK are actually doing for you — none of it is new capability, it's ergonomics
and guardrails around the same request/response cycle:

| Concern | Hand-rolled (this repo) | What a framework adds |
|---|---|---|
| Tool schema | `inspect.signature()` on your function, once | Same idea, plus support for Pydantic/Zod models, nested types, and auto-registering a whole class of tools |
| Dispatch | `globals()[name](**args)` | A typed tool registry with validation, error wrapping, and per-tool timeouts |
| The `while` loop | Written by hand per script | Provided as `agent.run()` / a graph executor, with iteration caps and configurable stop conditions built in |
| Message/state growth | You append to a list manually | Automatic history management, often with summarization or truncation once context limits approach |
| Parallel tool calls | Not handled in these demos (loop assumes one call at a time is fine to reason about) | Automatic concurrent execution + result aggregation |
| Retries / malformed calls | None | Automatic retry on invalid tool arguments or transient API errors |
| Human-in-the-loop | None | Interrupt/approval hooks before a tool actually executes (e.g. LangGraph checkpoints, Claude Agent SDK permission callbacks) |
| Multi-provider portability | You rewrite per-provider request shapes (§5.4 table) | A single tool/agent definition that maps to whichever provider's wire format underneath |
| Structured output | Hand-built JSON schema + `json.loads` | Typed parsing directly into Pydantic/Zod models, with validation errors surfaced instead of silent `json.loads` failures |
| Sub-agents / delegation | Not shown here | Spawning child agents with their own tools/context (e.g. LangGraph subgraphs, Claude Agent SDK subagents) |

The trade-off is the usual one: frameworks buy you less boilerplate and more safety nets,
at the cost of an abstraction layer you have to learn and that can obscure exactly which
API call is failing when something breaks. Reading the raw request/response once (this
repo) makes that debugging much faster even if you use a framework day-to-day.

---

## 8. Settings Worth Knowing (per provider)

These control the loop and output-shape behavior directly and are easy to miss when a
framework sets sensible defaults for you:

| Setting | Claude | Azure OpenAI | Gemini |
|---|---|---|---|
| Force / restrict tool use | `tool_choice: {"type": "tool", "name": "..."}` or `{"type": "any"}` / `{"type": "none"}` | `tool_choice: "required"` / `"none"` / `{"type": "function", "function": {"name": "..."}}` | `tool_config.function_calling_config.mode: "ANY" / "NONE" / "AUTO"` |
| Guarantee valid tool arguments | `strict: true` on a tool definition | `strict: true` inside `function` | Not a separate flag — enforced whenever `function_declarations` schema is well-formed |
| Parallel tool calls | Multiple `tool_use` blocks can appear in one response by default | `parallel_tool_calls: true/false` | Multiple `functionCall` parts can appear in one response by default |
| Cap response length | `max_tokens` (required) | `max_tokens` / `max_completion_tokens` | `generationConfig.maxOutputTokens` |
| Randomness | `temperature` | `temperature` | `generationConfig.temperature` |
| Structured output strictness | `output_config.format.schema` + implicit `additionalProperties: false` | `response_format.json_schema.strict: true` | `responseSchema` (schema itself is the only strictness control) |
| Automatic function calling (SDK convenience, not raw API) | N/A at HTTP level — SDK-only helper | N/A at HTTP level | `automatic_function_calling.disable` (Python SDK only; disabled by using the raw HTTP calls in this repo) |
| Extended "keep going" turns | `stop_reason: "pause_turn"` signals you should continue the loop without new input | N/A | N/A |

These map directly to the loop-control and termination concerns from §4 — e.g. `tool_choice`
and `parallel_tool_calls` change how many iterations a loop needs, and `strict`/schema
settings are what make structured output reliable enough to skip your own validation layer.

---

## 9. Troubleshooting

- **`KeyError` on env var** — you haven't exported the required variable for that provider (see §2).
- **401 / 403 from the API** — check the key is valid and, for Azure, that `AZURE_OPENAI_DEPLOYMENT` matches an actual deployment name in your resource, not the base model name.
- **Gemini `400` on schema** — Gemini's `responseSchema`/tool parameter schema doesn't support `additionalProperties`; the scripts already omit it for Gemini.
- **Loop never terminates (tool calling)** — a buggy tool schema or ambiguous query can cause the model to keep calling tools; these demos have no iteration cap since they're meant to be read, not hardened for production.

---

## 10. API References

- Claude tool use: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- Claude Messages API: https://platform.claude.com/docs/en/api/messages
- Claude structured outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Azure OpenAI function calling: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/function-calling
- Azure OpenAI structured outputs: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/structured-outputs
- Gemini function calling: https://ai.google.dev/gemini-api/docs/function-calling
- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Gemini generateContent reference: https://ai.google.dev/api/generate-content