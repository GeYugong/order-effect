"""Fixed-experience replay isolates consolidation order from experience generation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random

from benchmark import ROOT, dataset
from experiment import ReasoningMemory, digest, event_sink, extract, memory_text, merge, now, run_task, save
from study import build_manifest, client_for


def prepare_pool(manifest):
    dest = ROOT / "results/B/frozen_pool.json"
    if dest.exists():
        pool = json.loads(dest.read_text(encoding="utf-8"))
        if "canonical_contexts" in pool:
            return pool
    source = ROOT / "results/A/no_memory_ordinal_r0"
    if not (source / "summary.json").exists():
        raise RuntimeError("The prespecified no-memory ordinal r0 source run must finish first")
    tasks, _ = dataset()
    client = client_for(manifest)
    client.set_ledger(ROOT / "results/B/pool_usage.jsonl")
    experiences = {}
    for position, task_id in enumerate(manifest["orders"]["ordinal"]):
        intermediate = ROOT / "results/B/pool_items" / (task_id + ".json")
        if intermediate.exists():
            record = json.loads(intermediate.read_text(encoding="utf-8"))
        else:
            observed = json.loads((source / f"evolution_{position:03d}.json").read_text(encoding="utf-8"))
            items, generation = extract(tasks[task_id], observed, client)
            record = {"items": items, "generation": generation, "source_trajectory_hash": digest(observed)}
            save(intermediate, record)
        experiences[task_id] = record["items"]
    pool = {"experiences": experiences, "pool_hash": digest(experiences), "source": "A/no_memory_ordinal_r0", "created": now()}
    canonical = sorted([x for values in experiences.values() for x in values], key=lambda x: x["id"])
    bank = ReasoningMemory(client)
    bank.add(canonical)
    pool["canonical_memory"] = canonical
    pool["canonical_contexts"] = {key: bank.retrieve(tasks[key]["question"][0][0]["content"])[0] for key in manifest["splits"]["evaluation"]}
    pool.update(usage=client.usage, transport_events=client.transport_events)
    save(dest, pool)
    return pool


def replay_job(job, manifest, pool):
    folder = ROOT / "results/B" / f"{job['method']}_{job['order']}_r{job['rep']}"
    if (folder / "summary.json").exists():
        return json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    tasks, gold = dataset()
    client = client_for(manifest)
    client.set_ledger(folder / "usage.jsonl")
    if (folder / "started.json").exists():
        previous = json.loads((folder / "started.json").read_text(encoding="utf-8"))
        if previous["pool_hash"] != pool["pool_hash"] or previous["manifest_hash"] != digest(manifest):
            raise RuntimeError("Replay checkpoint configuration mismatch")
    else:
        save(folder / "started.json", {"job": job, "pool_hash": pool["pool_hash"], "manifest_hash": digest(manifest), "started": now()})
    rows = []
    try:
        rules = []
        if job["method"] == "merge":
            for position, task_id in enumerate(manifest["orders"][job["order"]]):
                record_path = folder / f"merge_{position:03d}.json"
                if record_path.exists():
                    rules = json.loads(record_path.read_text(encoding="utf-8"))["after"]
                else:
                    before = rules
                    rules, generation = merge(rules, pool["experiences"][task_id], client)
                    save(record_path, {"task_id": task_id, "before": before, "after": rules, "generation": generation})
                save(folder / "progress.json", {"stage": "merge", "completed": position + 1, "usage": client.usage, "updated": now()})
            final_memory = rules
            text = "\n".join(rule["content"] for rule in rules)
            contexts = {key: text for key in manifest["splits"]["evaluation"]}
        else:
            # Canonical reference sees precisely the same pool, independent of arrival order.
            final_memory = pool["canonical_memory"]
            contexts = pool["canonical_contexts"]
        save(folder / "final_memory.json", {"memory": final_memory, "hash": digest(final_memory), "contexts_hash": digest(contexts)})
        for eval_rep in range(manifest["replay_evaluation_repetitions"]):
            for position, key in enumerate(manifest["splits"]["evaluation"]):
                result_path = folder / f"eval{eval_rep}_{position:03d}.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    result = run_task(tasks[key], gold[key], client, contexts[key], manifest["max_steps_per_turn"], event_sink(folder / "api_trace.jsonl", key), folder / f"checkpoint_eval{eval_rep}_{position:03d}.json")
                    save(result_path, result)
                rows.append({"task_id": key, "eval_rep": eval_rep, "success": result["success"], "caps": len(result["caps"])})
                save(folder / "progress.json", {"stage": "evaluation", "completed": len(rows), "total": 60, "usage": client.usage, "updated": now()})
        summary = {**job, "study": "B", "model": client.model, "rows": rows, "evaluation_success": sum(x["success"] for x in rows) / len(rows), "usage": client.usage, "api_calls": client.calls, "transport_events": client.transport_events, "pool_hash": pool["pool_hash"], "memory_hash": digest(final_memory), "contexts_hash": digest(contexts), "finished": now()}
        save(folder / "summary.json", summary)
        return summary
    except Exception as exc:
        save(folder / "failure.json", {"error": str(exc), "usage": client.usage, "transport_events": client.transport_events, "completed": len(rows), "when": now()})
        raise


def run(manifest, workers):
    pool = prepare_pool(manifest)
    jobs = [{"method": "merge", "order": order, "rep": rep} for rep in range(manifest["repetitions"]) for order in manifest["orders"]]
    jobs.extend({"method": "canonical_all", "order": "canonical", "rep": rep} for rep in range(manifest["repetitions"]))
    random.Random(1234567).shuffle(jobs)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(replay_job, job, manifest, pool): job for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
                print(json.dumps({"complete": futures[future], "evaluation": result["evaluation_success"], "tokens": result["usage"]["total_tokens"]}), flush=True)
            except Exception as exc:
                failures.append({"job": futures[future], "error": str(exc)})
                print(json.dumps({"failed": futures[future], "error": str(exc)}), flush=True)
    save(ROOT / "results/B/execution.json", {"jobs": len(jobs), "failures": failures, "finished": now()})
    if failures:
        raise RuntimeError(f"{len(failures)} incomplete replay runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(build_manifest(), args.workers)
