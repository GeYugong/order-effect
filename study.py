"""Create the preregistered manifest and execute independent evolution runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import traceback

from benchmark import ROOT, dataset
from experiment import ReasoningMemory, digest, event_sink, extract, memory_text, merge, now, run_task, save
from gateway import Gateway

MODEL = "deepseek-flash"


def build_manifest():
    path = ROOT / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    tasks, _ = dataset()
    audit = json.loads((ROOT / "reports/oracle_audit.json").read_text(encoding="utf-8"))
    rng = random.Random(20260922)
    splits = {"development": [], "evolution": [], "evaluation": [], "reserve": []}
    used_groups = set()
    for domain, n_evolve in enumerate([8, 8, 7, 7]):
        ids = [k for k in tasks if int(k.rsplit("_", 1)[1]) // 50 == domain and k in audit["valid"]]
        rng.shuffle(ids)
        clean = []
        for key in ids:
            group = tuple(sorted(tasks[key]["path"]))
            if group not in used_groups:
                clean.append(key)
                used_groups.add(group)
        splits["development"].extend(clean[:4])
        splits["evolution"].extend(clean[4:4+n_evolve])
        splits["evaluation"].extend(clean[4+n_evolve:9+n_evolve])
        splits["reserve"].extend(clean[9+n_evolve:])
    for values in splits.values():
        values.sort(key=lambda k: int(k.rsplit("_", 1)[1]))
    orders = {"ordinal": splits["evolution"][:], "reverse": splits["evolution"][::-1]}
    for i, seed in enumerate([1729, 2718], 1):
        order = splits["evolution"][:]
        random.Random(seed).shuffle(order)
        orders[f"shuffle{i}"] = order
    manifest = {
        "created_utc": now(), "model": MODEL, "embedding_model": "BAAI/bge-base-en-v1.5",
        "temperature": None, "thinking": "enabled", "reasoning_effort": "low", "max_tokens": 8192, "max_steps_per_turn": 12,
        "repetitions": 5, "split_seed": 20260922, "order_seeds": [1729, 2718],
        "schedule_seed": 8675309, "splits": splits, "orders": orders,
        "methods": ["no_memory", "reasoning_memory"], "replay_methods": ["merge", "canonical_all"],
        "replay_evaluation_repetitions": 3,
        "primary_outcome": "frozen-memory heldout task success rate per complete run",
        "primary_contrast": "mean(shuffle1,shuffle2) minus ordinal; reasoning_memory minus no_memory",
        "generation_seed": "not supplied: endpoint seed determinism is not assumed",
        "source": json.loads((ROOT / "source.lock.json").read_text(encoding="utf-8"))["commit"],
        "split_note": "One example per identical sorted path group; stratified by four official 50-task domain blocks. Not an official BFCL split.",
    }
    save(path, manifest)
    return manifest


def client_for(manifest, model=None):
    client = Gateway(model or manifest["model"], manifest["temperature"], manifest["max_tokens"], token_limit=50000000)
    client.reasoning_effort = manifest.get("reasoning_effort", "low")
    return client


def run_sequence(job, manifest, study="A", model=None):
    name = f"{job['method']}_{job['order']}_r{job['rep']}"
    folder = ROOT / "results" / study / name
    if (folder / "summary.json").exists():
        return json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    tasks, gold = dataset()
    client = client_for(manifest, model)
    client.set_ledger(folder / "usage.jsonl")
    bank = ReasoningMemory(client)
    rows = []
    if (folder / "started.json").exists():
        previous = json.loads((folder / "started.json").read_text(encoding="utf-8"))
        if previous["manifest_hash"] != digest(manifest) or previous["model"] != client.model:
            raise RuntimeError("Run checkpoint configuration mismatch")
    else:
        save(folder / "started.json", {"job": job, "manifest_hash": digest(manifest), "started": now(), "model": client.model})
    try:
        for phase, ids in [("evolution", manifest["orders"][job["order"]]), ("evaluation", manifest["splits"]["evaluation"])]:
            for position, task_id in enumerate(ids):
                task = tasks[task_id]
                result_path = folder / f"{phase}_{position:03d}.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    context_path = folder / f"context_{phase}_{position:03d}.json"
                    if context_path.exists():
                        cached = json.loads(context_path.read_text(encoding="utf-8"))
                        context, retrieved = cached["text"], cached["retrieved"]
                    else:
                        context, retrieved = bank.retrieve(task["question"][0][0]["content"]) if job["method"] == "reasoning_memory" else ("", [])
                        save(context_path, {"text": context, "retrieved": retrieved})
                    result = run_task(task, gold[task_id], client, context, manifest["max_steps_per_turn"], event_sink(folder / "api_trace.jsonl", task_id), folder / f"checkpoint_{phase}_{position:03d}.json")
                    result.update(phase=phase, position=position, retrieved=retrieved)
                    save(result_path, result)
                rows.append({"task_id": task_id, "phase": phase, "success": result["success"], "caps": len(result["caps"])})
                if phase == "evolution" and job["method"] == "reasoning_memory":
                    memory_path = folder / f"memory_{position:03d}.json"
                    if memory_path.exists():
                        record = json.loads(memory_path.read_text(encoding="utf-8"))
                    else:
                        items, generation = extract(task, result, client)
                        record = {"items": items, "generation": generation}
                        save(memory_path, record)
                    if "vectors" not in record:
                        record["vectors"] = client.embed([memory_text(x) for x in record["items"]]) if record["items"] else []
                        save(memory_path, record)
                    bank.items.extend(record["items"])
                    bank.vectors.extend(record["vectors"])
                save(folder / "progress.json", {"completed": len(rows), "total": sum(len(x) for x in [manifest["orders"][job["order"]], manifest["splits"]["evaluation"]]), "usage": client.usage, "updated": now()})
        summary = {**job, "study": study, "model": client.model, "rows": rows, "usage": client.usage, "api_calls": client.calls, "transport_events": client.transport_events, "memory_items": len(bank.items), "manifest_hash": digest(manifest), "finished": now()}
        for phase in ("evolution", "evaluation"):
            selected = [r for r in rows if r["phase"] == phase]
            summary[phase + "_success"] = sum(r["success"] for r in selected) / len(selected)
        save(folder / "summary.json", summary)
        return summary
    except Exception as exc:
        save(folder / "failure.json", {"error": str(exc), "type": type(exc).__name__, "completed": len(rows), "usage": client.usage, "transport_events": client.transport_events, "when": now()})
        raise


def run_matrix(manifest, workers, study="A", model=None, reps=None):
    jobs = [{"method": method, "order": order, "rep": rep}
            for rep in range(reps or manifest["repetitions"])
            for order in manifest["orders"] for method in manifest["methods"]]
    random.Random(manifest["schedule_seed"]).shuffle(jobs)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_sequence, job, manifest, study, model): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                print(json.dumps({"complete": job, "evolution": result["evolution_success"], "evaluation": result["evaluation_success"], "tokens": result["usage"]["total_tokens"]}), flush=True)
            except Exception as exc:
                failures.append({"job": job, "error": str(exc)})
                print(json.dumps({"failed": job, "error": str(exc)}), flush=True)
    save(ROOT / "results" / study / "execution.json", {"jobs": len(jobs), "failures": failures, "finished": now()})
    if failures:
        raise RuntimeError(f"{len(failures)} incomplete runs; see execution.json")


def smoke(manifest, model=None):
    tasks, gold = dataset()
    client = client_for(manifest, model)
    bank = ReasoningMemory(client)
    folder = ROOT / "results" / ("smoke_" + client.model + "_" + client.reasoning_effort)
    rows = []
    # Development-only tasks from separate domains; no evolution/heldout outcomes used to tune.
    for idx in [0, 4, 8, 12]:
        key = manifest["splits"]["development"][idx]
        context, retrieved = bank.retrieve(tasks[key]["question"][0][0]["content"])
        result_path = folder / (key + ".json")
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = run_task(tasks[key], gold[key], client, context, manifest["max_steps_per_turn"], event_sink(folder / "api_trace.jsonl", key))
            save(result_path, result)
        items, generation = extract(tasks[key], result, client)
        bank.add(items)
        save(folder / (key + "_memory.json"), {"items": items, "generation": generation})
        rows.append({"id": key, "success": result["success"], "caps": result["caps"], "memories": len(items)})
        print(json.dumps(rows[-1]), flush=True)
    summary = {"rows": rows, "usage": client.usage, "api_calls": client.calls, "model": client.model, "transport_events": client.transport_events}
    save(ROOT / "reports" / ("smoke_" + client.model + "_" + client.reasoning_effort + ".json"), summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "smoke", "run"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--study", default="A")
    parser.add_argument("--model")
    parser.add_argument("--reps", type=int)
    args = parser.parse_args()
    manifest = build_manifest()
    if args.command == "prepare":
        print(json.dumps({k: len(v) for k, v in manifest["splits"].items()}))
    elif args.command == "smoke":
        smoke(manifest, args.model)
    else:
        run_matrix(manifest, args.workers, args.study, args.model, args.reps)
