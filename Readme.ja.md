# LLM のツール呼び出しと構造化出力 — 生HTTPデモ

## 目的（Intent）

Claude Agent SDK、OpenAI の Agents SDK、LangChain のようなフレームワークは、ツール呼び出し
（tool calling）や構造化出力（structured output）をまるで魔法のように感じさせます。関数に
デコレータを付けて `Agent` に渡すだけで、型付けされた正しい答えが返ってきます。この「魔法」
は本番環境では確かに便利ですが、その裏で実際に何が HTTP 上でやり取りされているかを隠して
しまい、デバッグ・コストの見積もり・プロバイダー間の移植を難しくしています。

このリポジトリはその抽象化を全て取り除きます。プロバイダー × パターンごとに6本のスクリプトが
あり、いずれも `anthropic`、`openai`、`google-genai` といった SDK やエージェントフレームワーク
を一切使わず、**素の `requests.post()` 呼び出し**だけで実装されています。目的は次の2点を
具体的に示すことです。

1. **ツール呼び出し** — モデル自身はコードを実行しません。モデルは構造化された形で「あなたに」
   実行を依頼するだけであり、「エージェントループ」とは、あなたがその結果を `while` ループで
   フィードバックし続けることに過ぎません。
2. **構造化出力** — モデルの最終回答は、制約付きデコーディング（constrained decoding）に
   よって JSON スキーマに強制されます。ここにはループも、ローカルでの関数実行も一切ありません。

この2つの生の形を一度理解してしまえば、エージェントフレームワークがその上に何を追加している
のかがずっと理解しやすくなります（7章を参照）。

| ファイル | プロバイダー | パターン |
|---|---|---|
| `claude_tool_calling.py` | Anthropic Claude | ツール呼び出し（エージェントループ） |
| `azure_openai_tool_calling.py` | Azure OpenAI | ツール呼び出し（エージェントループ） |
| `gemini_tool_calling.py` | Google Gemini | ツール呼び出し（エージェントループ） |
| `claude_structured_output.py` | Anthropic Claude | 構造化出力（JSON抽出） |
| `azure_openai_structured_output.py` | Azure OpenAI | 構造化出力（JSON抽出） |
| `gemini_structured_output.py` | Google Gemini | 構造化出力（JSON抽出） |

---

## 1. 前提条件

- Python 3.9 以上
- `pip install -r requirements.txt`
- 使用するプロバイダーの API 認証情報（下記参照）

これ以外のパッケージは一切不要です。それがこのリポジトリの主眼です。

---

## 2. 環境変数

サンプルファイルをコピーして、認証情報を埋めてください。

```bash
cp .env.example .env
```

その後、`.env` に値を設定するか、スクリプト実行前にシェルへ `export` してください。

```bash
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

すべてのスクリプトは `python-dotenv` を使ってプロジェクト直下の `.env` を自動読み込みし、必須キーが不足している場合は明確なメッセージで失敗します。

| プロバイダー | 変数名 | 備考 |
|---|---|---|
| Claude | `ANTHROPIC_API_KEY` | Claude Platform コンソールから取得 |
| Azure OpenAI | `AZURE_OPENAI_ENDPOINT` | 例: `https://your-resource.openai.azure.com` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | Azure Portal → Keys and Endpoint のリソースキー |
| Azure OpenAI | `AZURE_OPENAI_DEPLOYMENT` | **デプロイ名**（ベースモデル名ではない点に注意） |
| Gemini | `GEMINI_API_KEY` | Google AI Studio から取得 |

---

## 3. 実行方法

各スクリプトは単体で完結しており、ハードコードされたサンプル入力に対して一度だけ実行されます。

```bash
python claude_tool_calling.py
python azure_openai_tool_calling.py
python gemini_tool_calling.py

python claude_structured_output.py
python azure_openai_structured_output.py
python gemini_structured_output.py
```

---

## 4. 「エージェントループ」とは何か

単発の LLM API 呼び出しはステートレスで、テキストしか返せません。ツール呼び出しがこれを
「ループ」に変えるのは、モデル自身は行動できないからです。モデルはあなたのコードに行動を
「依頼」することしかできず、それも1レスポンスにつき1回ずつです。本質だけを取り出すと、
エージェントループは次のような形になります。

```
messages = [ユーザーのリクエスト]
while True:
    response = call_model(messages, available_tools)
    append response to messages
    if response にツール呼び出しが含まれない:
        return response を最終回答として返す
    response 内の各ツール呼び出しについて:
        result = 対応するローカル関数を引数付きで実行する
        result を messages に追加する（モデルが自分のリクエストと対応付けられる形式で）
```

これで全体の構造は尽きています。あなたがこれまで使ってきた「AI エージェント」は、すべて
このループにツールとガードレールと反復回数の上限を足したものに過ぎません。5章では、この
ループをフレームワークを一切使わずにプロバイダーごとに手書きで実装したものを示します。

トイ例を超えたところで、このループの実装を非自明にする要素は主に3つあります。

- **状態管理** — `messages` は1ターンごとに増え続け、毎回の呼び出しで全体を送り直す必要が
  あります（サーバー側にメモリはありません）。したがってコンテキストウィンドウとコストは、
  ループが長引くほど増えていきます。
- **終了条件** — 「ツール呼び出しがない」だけでは不十分で、反復回数の上限、トークン予算、
  タイムアウト、あるいはリスクのある操作を実行する前の人間による承認（human-in-the-loop）
  などの停止条件が必要になります。
- **並列／同時ツール呼び出し** — 1回のレスポンスで複数のツール呼び出しが同時に要求される
  ことがあります。正しいループは、次の呼び出しに進む前にそれら全てを実行し、結果を返す
  必要があります。

---

## 5. パターン A — ツール呼び出し（手書きループ)

### 5.1 4つの「ツール」

```python
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b
# ...subtract、multiply、divide も同じ形で続く
```

docstring がそのままツールの `description` になり、型ヒント付きの引数が JSON スキーマの
`properties` になります。

### 5.2 スキーマ生成（JSON を手書きしない）

```python
def json_type(t):
    return "number" if t in (int, float) else "string" if t is str else "boolean"

def tool_from_fn(fn):
    sig = inspect.signature(fn)
    props = {n: {"type": json_type(p.annotation)} for n, p in sig.parameters.items()}
    return {"name": fn.__name__, "description": fn.__doc__, ...}
```

`inspect.signature()` が引数名と型ヒントを読み取り、`fn.__doc__` が description を供給します。
`add()` に引数を1つ追加すれば、スキーマも自動的に更新されます。

### 5.3 モデルのツール呼び出しをディスパッチする

ツールの `name` は `fn.__name__` から生成されているため、モデルが返してくる名前はそのまま
モジュールのグローバル名前空間へのキーとして有効です。手動の対応表は不要です。

```python
out = globals()[block["name"]](**block["input"])
```

### 5.4 プロバイダーごとのループ

| ステップ | Claude | Azure OpenAI | Gemini |
|---|---|---|---|
| エンドポイント | `POST /v1/messages` | `POST /openai/deployments/{deployment}/chat/completions` | `POST /models/{model}:generateContent` |
| 認証ヘッダー | `x-api-key` | `api-key` | クエリパラメータ `?key=` |
| ツール呼び出しの合図 | `resp["stop_reason"] == "tool_use"` | `msg["tool_calls"]` が存在する | `functionCall` パートが存在する |
| ツールスキーマのフィールド名 | `input_schema` | `parameters`（`function` の中にネスト） | `parameters`（`function_declarations` の中にネスト） |
| ツール結果の role | `{"role": "user", "content": [{"type": "tool_result", ...}]}` | `{"role": "tool", "tool_call_id": ..., "content": ...}` | `{"role": "user", "parts": [{"functionResponse": {...}}]}` |
| JSON型の大文字小文字 | 小文字（`"number"`） | 小文字（`"number"`） | 大文字（`"NUMBER"`） |

### 5.5 サンプル入力

```python
QUERY = "What is 12 multiplied by 7, then subtract 5 from the result?"
```

（「12 に 7 を掛け、その結果から 5 を引くと？」）これにより `multiply` → `subtract` の2回の
連続したツール呼び出しが発生し、ループが実際に複数回反復する様子を確認できます。

---

## 6. パターン B — 構造化出力（ループなし）

構造化出力は**ツール呼び出しではありません**。関数の実行もループも一切発生しません。モデルの
「回答」に対して JSON スキーマを与えると、API は制約付きデコーディング（グラマーのコンパイル）
によって、レスポンステキストがそのスキーマに一致する有効な JSON であることを保証します。

### 6.1 共通のスキーマ定義元

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

### 6.2 プロバイダーごとのリクエスト形式

| プロバイダー | 設定フィールド | レスポンス中のJSONの位置 |
|---|---|---|
| Claude | `output_config: {"format": {"type": "json_schema", "schema": ...}}`（`/v1/messages` に指定） | `response["content"]` — テキストブロックの `text` が JSON 文字列 |
| Azure OpenAI | `response_format: {"type": "json_schema", "json_schema": {"schema": ..., "strict": true}}`（chat completions に指定） | `choices[0]["message"]["content"]` — JSON 文字列 |
| Gemini | `generationConfig: {"responseMimeType": "application/json", "responseSchema": ...}` | `candidates[0]["content"]["parts"][0]["text"]` — JSON 文字列 |

いずれのスクリプトも、そのテキストに対して `json.loads(...)` を実行し、Python の dict に
変換しています。

> Claude の構造化出力は以前、`anthropic-beta: structured-outputs-2025-11-13` ヘッダーと
> `output_format` フィールドが必須でした。現在はここで使っている安定版の
> `output_config.format` に昇格しています（旧ベータヘッダーも移行期間中は引き続き動作しますが
> 必須ではありません）。

### 6.3 サンプル入力

```python
TEXT = "John Smith (john@acme.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."
```

3つとも次のような出力形式が期待されます。

```python
{
    "name": "John Smith",
    "email": "john@acme.com",
    "plan_interest": "Enterprise",
    "demo_requested": True,
}
```

### 6.4 ツール呼び出し vs. 構造化出力

| | ツール呼び出し | 構造化出力 |
|---|---|---|
| 目的 | モデルに「行動」させる — 引数付きで関数を呼び出させる | モデルの「最終回答」をスキーマに制約する |
| ラウンドトリップ数 | 複数回（ツール呼び出しがなくなるまでループ） | 1回のみ |
| コードを実行する主体 | あなたのアプリ（ツール呼び出しごとにローカルで実行） | 誰も実行しない — モデルは JSON テキストを書くだけ |
| スキーマが記述する対象 | 関数の入力 | レスポンス本体 |
| 典型的な用途 | 電卓、DB検索、API呼び出し、エージェント | データ抽出、分類、フォーマット整形 |

---

## 7. エージェントフレームワークがその上に追加するもの

4章のループを踏まえると、LangChain/LangGraph、Claude Agent SDK、OpenAI Agents SDK が実際に
やっていることは、以下の表の通りです。新しい能力を追加しているわけではなく、同じ
リクエスト/レスポンスのサイクルに対して使い勝手とガードレールを足しているだけです。

| 関心事 | 手書き（このリポジトリ） | フレームワークが追加するもの |
|---|---|---|
| ツールスキーマ | 関数に対して1回 `inspect.signature()` を実行 | 同じ発想に加え、Pydantic/Zod モデル、ネストした型、ツール群の一括登録に対応 |
| ディスパッチ | `globals()[name](**args)` | 型付きのツールレジストリ、バリデーション、エラーラップ、ツールごとのタイムアウト |
| `while` ループ | スクリプトごとに手書き | `agent.run()` やグラフエグゼキュータとして提供され、反復上限や停止条件を設定可能 |
| メッセージ／状態の増加 | 手動でリストに追加 | 履歴の自動管理。コンテキスト上限が近づくと要約や切り詰めが行われることも多い |
| 並列ツール呼び出し | このデモでは扱っていない（1回に1呼び出しという前提で単純化） | 自動的な並行実行と結果の集約 |
| リトライ／不正な呼び出し | なし | 不正なツール引数や一時的な API エラーに対する自動リトライ |
| 人間の承認（human-in-the-loop） | なし | ツールが実際に実行される前の割り込み／承認フック（例: LangGraph のチェックポイント、Claude Agent SDK の permission callback） |
| 複数プロバイダー間の移植性 | プロバイダーごとにリクエスト形式を書き換える必要あり（5.4章の表） | 単一のツール／エージェント定義が、内部で各プロバイダーの通信形式にマッピングされる |
| 構造化出力 | JSON スキーマを手書きし `json.loads` する | Pydantic/Zod モデルへ直接型付きでパースし、バリデーションエラーを（無言の `json.loads` 失敗ではなく）表面化させる |
| サブエージェント／委譲 | このリポジトリでは扱っていない | 独自のツール／コンテキストを持つ子エージェントの生成（例: LangGraph のサブグラフ、Claude Agent SDK のサブエージェント） |

トレードオフはいつも通りです。フレームワークはボイラープレートの削減と安全網を提供する
代わりに、学習が必要な抽象化レイヤーを挟み込み、何か問題が起きたときにどの API 呼び出しが
失敗しているのかを見えにくくします。生のリクエスト/レスポンスを一度理解しておく（本
リポジトリのように）ことは、日常的にフレームワークを使う場合でも、デバッグを大幅に速くします。

---

## 8. 覚えておくべき設定項目（プロバイダー別）

これらはループの挙動や出力形式のふるまいを直接制御するもので、フレームワークが妥当な
デフォルト値を設定してくれるがゆえに見落としやすい項目です。

| 設定項目 | Claude | Azure OpenAI | Gemini |
|---|---|---|---|
| ツール使用の強制／制限 | `tool_choice: {"type": "tool", "name": "..."}` または `{"type": "any"}` / `{"type": "none"}` | `tool_choice: "required"` / `"none"` / `{"type": "function", "function": {"name": "..."}}` | `tool_config.function_calling_config.mode: "ANY" / "NONE" / "AUTO"` |
| ツール引数の妥当性を保証 | ツール定義に `strict: true` | `function` 内に `strict: true` | 独立したフラグはなし — `function_declarations` のスキーマが正しく定義されていれば強制される |
| 並列ツール呼び出し | デフォルトで1レスポンスに複数の `tool_use` ブロックが含まれ得る | `parallel_tool_calls: true/false` | デフォルトで1レスポンスに複数の `functionCall` パートが含まれ得る |
| レスポンス長の上限 | `max_tokens`（必須） | `max_tokens` / `max_completion_tokens` | `generationConfig.maxOutputTokens` |
| ランダム性 | `temperature` | `temperature` | `generationConfig.temperature` |
| 構造化出力の厳密さ | `output_config.format.schema` ＋暗黙の `additionalProperties: false` | `response_format.json_schema.strict: true` | `responseSchema`（スキーマ自体が唯一の厳密さの制御手段） |
| 自動関数呼び出し（SDKの利便機能。生APIの機能ではない） | HTTPレベルでは該当なし — SDK専用のヘルパー | HTTPレベルでは該当なし | `automatic_function_calling.disable`（Python SDK限定。本リポジトリのように生HTTPを使う場合は無効化されている状態と同じ） |
| ターンの継続 | `stop_reason: "pause_turn"` は、新たな入力なしにループを継続すべきことを示す | 該当なし | 該当なし |

これらは4章で述べたループ制御・終了条件の話と直接結び付いています。たとえば `tool_choice` や
`parallel_tool_calls` はループに必要な反復回数を変化させますし、`strict` やスキーマ関連の
設定は、独自のバリデーション層を省略できるほど構造化出力を信頼できるものにします。

---

## 9. トラブルシューティング

- **環境変数の `KeyError`** — 該当プロバイダーに必要な環境変数を export し忘れています（2章参照）。
- **API から 401 / 403 が返る** — キーが有効か、また Azure の場合は `AZURE_OPENAI_DEPLOYMENT`
  がベースモデル名ではなく、リソース内の実際のデプロイ名と一致しているかを確認してください。
- **Gemini のスキーマで `400` が返る** — Gemini の `responseSchema` / ツールのパラメータ
  スキーマは `additionalProperties` に対応していません。各スクリプトは Gemini 向けに
  すでにこれを省いています。
- **ツール呼び出しのループが終わらない** — スキーマの不備や曖昧なクエリにより、モデルが
  ツール呼び出しを繰り返し続けることがあります。これらのデモは「読んで理解する」ことを
  目的としており、反復回数の上限は設けていないため、本番運用向けにはハードニングが必要です。

---

## 10. API リファレンス

- Claude tool use: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- Claude Messages API: https://platform.claude.com/docs/en/api/messages
- Claude structured outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Azure OpenAI function calling: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/function-calling
- Azure OpenAI structured outputs: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/structured-outputs
- Gemini function calling: https://ai.google.dev/gemini-api/docs/function-calling
- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Gemini generateContent reference: https://ai.google.dev/api/generate-content