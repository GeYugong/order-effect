"""Isolated task execution, reasoning-memory adaptation, and frozen-pool replay."""
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from datetime import datetime, timezone

from benchmark import Runtime, evaluate, function_call, tools_for
from gateway import Gateway

ACTOR_PROMPT = """You are a tool-using assistant in a simulated environment. Complete each user request using the provided tools. Discover necessary state through tools; do not invent tool results or resource identifiers. Respect tool parameter constraints. Calls with dependencies must execute in order. A user turn ends when you respond without calling a tool. Only the current and previous user turns are known; do not anticipate future requests. Past-task memory, if present, is fallible guidance, not a replacement for current tool documentation or observations."""

EXTRACT_PROMPT = """Extract reusable reasoning memories from one completed tool-use task. You receive the actual conversation trajectory, tool documentation, and a binary execution-evaluator outcome. No reference answer is supplied. From success, identify the specific strategy that worked. From failure, identify observable mistakes and cautiously distinguish hypotheses from established facts. Produce at most three generalizable items, each with title, description, content. Preserve tool constraints and actionable recovery strategies. Do not retain task-specific names, exact answers, IDs, passwords, tokens or other credentials. Do not infer future user requests. Output only a JSON object: {\"items\":[{\"title\":\"...\",\"description\":\"...\",\"content\":\"...\"}]}. Each content must be at most 120 words. Return an empty list if nothing reliable can be learned."""

MERGE_PROMPT = """Maintain a reusable tool-use playbook from frozen experience items. Revise the existing playbook using the next experience. Merge duplicates, preserve important constraints and counterexamples, and discard unsupported or redundant advice. Do not invent evidence. Each retained rule must cite one or more source task IDs present in the input. Return only JSON: {\"rules\":[{\"content\":\"...\",\"sources\":[\"task_id\"]}]}. Use at most 12 rules and at most 100 words per rule. This is a bounded sequential-consolidation research baseline."""


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def parse_object(text):
    text = (text or "").strip()
    # This gateway may put MiniMax's thinking wrapper in content instead of a separate field.
    text = re.sub(r"^<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def clean_message(message):
    # Keep reasoning_content when supplied, as some gateways require it during tool turns.
    return {k: v for k, v in message.items() if k in {"role", "content", "tool_calls", "reasoning_content"} and v is not None}


def run_task(task, gold, client, memory_text="", max_steps=12, on_event=None):
    runtime = Runtime(task)
    tools = tools_for(task)
    allowed = {f["function"]["name"] for f in tools}
    system = ACTOR_PROMPT
    if memory_text:
        system += "\n\nPAST-TASK MEMORY (fallible):\n" + memory_text
    messages = [{"role": "system", "content": system}]
    decoded, calls_meta = [], []
    caps = []
    for turn_idx, user_turn in enumerate(task["question"]):
        messages.extend(copy.deepcopy(user_turn))
        steps = []
        for step in range(max_steps):
            message, meta = client.chat(messages, tools=tools)
            calls_meta.append(meta)
            messages.append(clean_message(message))
            if on_event:
                on_event({"turn": turn_idx, "step": step, "meta": meta, "message": message})
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                if meta["finish_reason"] == "length":
                    caps.append({"turn": turn_idx, "reason": "output_token_limit"})
                break
            batch = []
            for call in tool_calls:
                try:
                    fn = call["function"]
                    expression = function_call(fn["name"], json.loads(fn["arguments"]), allowed)
                    output = runtime.execute(expression)
                    batch.append(expression)
                except (KeyError, ValueError, TypeError) as exc:
                    output = "Invalid tool call: " + str(exc)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
            steps.append(batch)
        else:
            caps.append({"turn": turn_idx, "reason": "step_limit"})
        decoded.append(steps)
    check = evaluate(task, decoded, gold)
    return {
        "task_id": task["id"], "success": bool(check["valid"]),
        "evaluation": check, "decoded": decoded, "messages": messages,
        "api": calls_meta, "caps": caps, "memory_hash": digest(memory_text),
    }


def extract(task, result, client):
    # Only actual observations + binary reward. Never send reference calls/checker diagnostics.
    observed = [m for m in result["messages"] if m["role"] != "system"]
    request = {"trajectory": observed, "success": result["success"], "tools": tools_for(task)}
    message, meta = client.chat([
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
    ])
    if meta["finish_reason"] == "length":
        raise ValueError("Memory extraction exceeded output limit")
    items = parse_object(message.get("content"))["items"]
    if not isinstance(items, list) or len(items) > 3:
        raise ValueError("Memory extraction returned invalid item count")
    memories = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in ("title", "description", "content")):
            raise ValueError("Invalid memory schema")
        memories.append({**item, "id": task["id"] + ":" + str(idx), "source": task["id"]})
    return memories, {"message": message, "meta": meta}


def memory_text(item):
    return "\n".join(item.get(k, "") for k in ("title", "description", "content"))


class ReasoningMemory:
    """ReasoningBank-style adaptation: up to 3 items/task, top-3 embedding retrieval.

    Deliberate differences: BFCL tools, MiniMax writer, Qwen embeddings, binary official
    reward. No MaTTS. This is not a claim to reproduce original ReasoningBank scores.
    """
    def __init__(self, client):
        self.client = client
        self.items = []
        self.vectors = []

    def add(self, items):
        if items:
            self.vectors.extend(self.client.embed([memory_text(x) for x in items]))
            self.items.extend(copy.deepcopy(items))

    def retrieve(self, question):
        if not self.items:
            return "", []
        query = self.client.embed([question])[0]
        def cosine(vector):
            denom = math.sqrt(sum(v*v for v in vector) * sum(v*v for v in query))
            return sum(a*b for a, b in zip(vector, query)) / denom if denom else 0
        # Stable content-ID tie-breaking rather than arrival-order tie-breaking.
        ranked = sorted(range(len(self.items)), key=lambda i: (-cosine(self.vectors[i]), self.items[i]["id"]))[:3]
        return "\n\n".join(memory_text(self.items[i]) for i in ranked), [self.items[i]["id"] for i in ranked]


def merge(playbook, incoming, client):
    message, meta = client.chat([
        {"role": "system", "content": MERGE_PROMPT},
        {"role": "user", "content": json.dumps({"playbook": playbook, "experience": incoming}, ensure_ascii=False)},
    ])
    output = parse_object(message.get("content"))
    rules = output.get("rules")
    if not isinstance(rules, list) or len(rules) > 12:
        raise ValueError("Invalid playbook rule count")
    known = {s for r in playbook for s in r["sources"]} | {x["source"] for x in incoming}
    for rule in rules:
        if not isinstance(rule.get("content"), str) or not isinstance(rule.get("sources"), list) or not rule["sources"] or not set(rule["sources"]) <= known:
            raise ValueError("Invalid playbook provenance")
        if len(rule["content"].split()) > 100:
            raise ValueError("Playbook rule exceeds word budget")
    return rules, {"message": message, "meta": meta}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=repr), encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def event_sink(path, task_id):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def append(event):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"task_id": task_id, "time": now(), **event}, ensure_ascii=False) + "\n")
    return append
