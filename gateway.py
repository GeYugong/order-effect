"""Small OpenAI-compatible client; credentials never enter logs or config files."""
import json
import os
from pathlib import Path
import sqlite3
import time
import urllib.error
import urllib.request

BASE_URL = "http://aigw.dlut.edu.cn/v1"


def rate_limit():
    """Cross-process pacing: 32 requests/minute against the observed 40/min limit."""
    directory = Path(__file__).resolve().parent / "results"
    directory.mkdir(exist_ok=True)
    with sqlite3.connect(directory / "gateway_rate.sqlite", timeout=30) as db:
        db.execute("CREATE TABLE IF NOT EXISTS pace (id INTEGER PRIMARY KEY, next_at REAL NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT next_at FROM pace WHERE id=1").fetchone()
        at = max(time.time(), row[0] if row else 0)
        db.execute("INSERT OR REPLACE INTO pace VALUES (1, ?)", (at + 60/32,))
        db.commit()
    delay = at - time.time()
    if delay > 0:
        time.sleep(delay)


def api_key(name="DLUT_API_KEY"):
    value = os.environ.get(name)
    if not value and os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, name)
        except FileNotFoundError:
            pass
    if not value:
        raise RuntimeError("Configure the local user environment variable " + name)
    return value


class Gateway:
    def __init__(self, model, temperature=0.7, max_tokens=4096, token_limit=200000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = "low"
        self.token_limit = token_limit
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.calls = 0
        self.transport_events = []
        self.ledger_path = None

    def set_ledger(self, path):
        self.ledger_path = Path(path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event["kind"] == "success":
                    self.calls += 1
                    for key in self.usage:
                        self.usage[key] += event.get("usage", {}).get(key, 0) or 0
                else:
                    self.transport_events.append(event)

    def _record(self, event):
        if self.ledger_path:
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

    def request(self, method, path, payload=None):
        deepseek = self.model in {"deepseek-flash", "deepseek-v4-pro"} and path != "/embeddings"
        base_url = "https://api.deepseek.com" if deepseek else BASE_URL
        key = api_key("DEEPSEEK_API_KEY" if deepseek else "DLUT_API_KEY")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            base_url + path, data=data, method=method,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        for attempt in range(4):
            if not deepseek:
                rate_limit()
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.load(response)
                self._record({"kind": "success", "path": path, "usage": result.get("usage", {}), "model": result.get("model"), "time": time.time()})
                return result
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace").replace(key, "[REDACTED]")
                error = f"Gateway HTTP {exc.code}: {body[:800]}"
                retryable = exc.code in (429, 500, 502, 503, 504)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                error = "Gateway network/timeout failure: " + type(exc).__name__
                retryable = True
            self.transport_events.append({"path": path, "attempt": attempt + 1, "error": error, "time": time.time()})
            self._record({"kind": "transport_error", **self.transport_events[-1]})
            if not retryable or attempt == 3:
                raise RuntimeError(error + " (attempt recorded)") from None
            time.sleep(2 ** (attempt + 1))

    def chat(self, messages, tools=None, **extra):
        if self.usage["total_tokens"] >= self.token_limit:
            raise RuntimeError("Run token stop threshold reached.")
        payload = {
            "model": self.model, "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.model in {"deepseek-flash", "deepseek-v4-pro"}:
            payload.update(thinking={"type": "enabled"}, reasoning_effort=self.reasoning_effort)
        elif self.temperature is not None:
            payload["temperature"] = self.temperature
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
            "reasoning_effort": self.reasoning_effort if self.model.startswith("deepseek-") else None,
            "system_fingerprint": result.get("system_fingerprint"),
            "finish_reason": choice.get("finish_reason"), "usage": usage,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "transport_events": self.transport_events[retry_start:],
        }

    def embed(self, texts, query=False):
        from embedding import embed
        return embed(texts, query=query)


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
