"""Small OpenAI-compatible client; credentials never enter logs or config files."""
import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = "http://aigw.dlut.edu.cn/v1"


def api_key():
    value = os.environ.get("DLUT_API_KEY")
    if not value and os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, "DLUT_API_KEY")
        except FileNotFoundError:
            pass
    if not value:
        raise RuntimeError("Configure the local user environment variable DLUT_API_KEY.")
    return value


class Gateway:
    def __init__(self, model, temperature=0.7, max_tokens=4096, token_limit=200000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_limit = token_limit
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.calls = 0
        self.transport_events = []

    def request(self, method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            BASE_URL + path, data=data, method=method,
            headers={"Authorization": "Bearer " + api_key(), "Content-Type": "application/json"},
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace").replace(api_key(), "[REDACTED]")
                error = f"Gateway HTTP {exc.code}: {body[:800]}"
                retryable = exc.code in (429, 500, 502, 503, 504)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                error = "Gateway network/timeout failure: " + type(exc).__name__
                retryable = True
            self.transport_events.append({"path": path, "attempt": attempt + 1, "error": error, "time": time.time()})
            if not retryable or attempt == 3:
                raise RuntimeError(error + " (attempt recorded)") from None
            time.sleep(2 ** (attempt + 1))

    def chat(self, messages, tools=None, **extra):
        if self.usage["total_tokens"] >= self.token_limit:
            raise RuntimeError("Run token stop threshold reached.")
        payload = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        payload.update(extra)
        started = time.monotonic()
        retry_start = len(self.transport_events)
        result = self.request("POST", "/chat/completions", payload)
        self.calls += 1
        usage = result.get("usage", {})
        for field in self.usage:
            self.usage[field] += usage.get(field, 0) or 0
        choice = result["choices"][0]
        return choice["message"], {
            "model_requested": self.model, "model_returned": result.get("model"),
            "system_fingerprint": result.get("system_fingerprint"),
            "finish_reason": choice.get("finish_reason"), "usage": usage,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "transport_events": self.transport_events[retry_start:],
        }

    def embed(self, texts):
        result = self.request("POST", "/embeddings", {
            "model": "Qwen3-Embedding-8B", "input": texts, "encoding_format": "float",
        })
        self.calls += 1
        for key in self.usage:
            self.usage[key] += result.get("usage", {}).get(key, 0) or 0
        return [row["embedding"] for row in sorted(result["data"], key=lambda row: row["index"])]


def probe(model):
    client = Gateway(model, max_tokens=2048, token_limit=16000)
    tools = [{"type": "function", "function": {
        "name": "lookup_stock", "description": "Read the inventory count for a product.",
        "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]},
    }}]
    messages = [
        {"role": "system", "content": "Use the inventory tool before answering. Do not invent stock counts."},
        {"role": "user", "content": "How many units of SKU EXP-742 are in stock?"},
    ]
    first, meta1 = client.chat(messages, tools=tools)
    calls = first.get("tool_calls", [])
    ok = len(calls) == 1 and calls[0]["function"]["name"] == "lookup_stock"
    if ok:
        ok = json.loads(calls[0]["function"]["arguments"]) == {"sku": "EXP-742"}
    report = {"model": model, "tool_call_valid": ok, "first_response": first, "first_meta": meta1}
    if ok:
        messages.extend([first, {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"sku":"EXP-742","units":17}'}])
        last, meta2 = client.chat(messages, tools=tools)
        report.update(final_response=last, final_meta=meta2, tool_result_used="17" in (last.get("content") or ""))
    report.update(usage=client.usage, calls=client.calls)
    return report


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    report = probe(args.model)
    out = Path(__file__).parent / "results" / "gateway"
    out.mkdir(parents=True, exist_ok=True)
    (out / (args.model + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model", "tool_call_valid", "tool_result_used", "usage", "calls") if k in report}))
