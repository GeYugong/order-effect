"""Durable A -> B -> analysis -> private Git sync, with visible progress and checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock

from benchmark import ROOT
from experiment import digest, now, save
from study import build_manifest

RUNNER_FILES = ["benchmark.py", "embedding.py", "experiment.py", "gateway.py", "study.py", "replay.py"]


def seal():
    path = ROOT / "run.lock.json"
    current = {
        "manifest_hash": digest(build_manifest()),
        "code_sha256": {name: hashlib.sha256((ROOT/name).read_text(encoding="utf-8").encode("utf-8")).hexdigest() for name in RUNNER_FILES},
        "embedding_lock_sha256": hashlib.sha256((ROOT/"embedding.lock.json").read_bytes()).hexdigest(),
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != current:
            raise RuntimeError("Frozen experiment code/configuration changed; do not mix runs")
    else:
        save(path, current)
    return current


def progress(state="running", stage="A", error=None):
    result = {"state": state, "stage": stage, "updated": now(), "pid": os.getpid(), "model": build_manifest()["model"], "studies": {}}
    for label, expected_runs, expected_tasks in [("A", 40, 2000), ("B", 25, 1500)]:
        folder = ROOT / "results" / label
        summaries = list(folder.glob("*/summary.json"))
        task_files = list(folder.glob("*/evolution_[0-9]*.json")) + list(folder.glob("*/evaluation_[0-9]*.json")) if label == "A" else list(folder.glob("*/eval[0-9]_[0-9]*.json"))
        errors = [p for p in folder.glob("*/failure.json") if not (p.parent/"summary.json").exists()]
        result["studies"][label] = {"complete_runs": len(summaries), "expected_runs": expected_runs, "completed_tasks": len(task_files), "expected_tasks": expected_tasks, "runs_with_prior_failure": len(errors)}
    if error:
        result["error"] = error
    save(ROOT / "reports/progress.json", result)
    lines = ["# 实验运行状态", "", f"状态：**{state}**；阶段：{stage}；模型：`{result['model']}`。", "", f"更新时间（UTC）：{result['updated']}。", "", "| 实验 | 完整运行 | 任务执行 |", "|---|---:|---:|"]
    for label, row in result["studies"].items():
        lines.append(f"| {label} | {row['complete_runs']}/{row['expected_runs']} | {row['completed_tasks']}/{row['expected_tasks']} |")
    lines.extend(["", "A：任务顺序；B：固定经验融合顺序。只有完整运行进入统计。详细轨迹保留在本地 results/，汇总与图表同步到私有仓库。"])
    if error:
        lines.extend(["", "需要处理的错误：", "", "```text", error[:1500], "```"])
    (ROOT/"reports/STATUS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def sync(message):
    view = subprocess.run(["gh", "repo", "view", "GeYugong/order-effect", "--json", "isPrivate"], cwd=ROOT, check=True, capture_output=True, text=True)
    if not json.loads(view.stdout)["isPrivate"]:
        raise RuntimeError("Repository is not private; synchronization refused")
    subprocess.run(["git", "add", "reports"], cwd=ROOT, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode
    if changed:
        subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def run_stage(stage, args):
    seal()
    logfile = ROOT / "results" / ("stage_" + stage + ".log")
    for attempt in range(1, 4):
        if (ROOT/"results/STOP").exists():
            raise InterruptedError("STOP marker found")
        with logfile.open("a", encoding="utf-8") as log:
            log.write(f"\n{now()} stage={stage} invocation={attempt}\n")
            log.flush()
            child = subprocess.Popen([sys.executable, "-X", "utf8", *args], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                progress(stage=stage)
                if (ROOT/"results/STOP").exists():
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
                    else:
                        child.terminate()
                    child.wait(timeout=20)
                    raise InterruptedError("STOP marker; completed checkpoints retained")
                time.sleep(15)
        if child.returncode == 0:
            return
        if attempt < 3:
            time.sleep(60)
    raise RuntimeError(f"Stage {stage} failed after 3 checkpoint resumes. Inspect {logfile.name}.")


def await_preflight(pid):
    """Wait for the already-running development validation; never run paid fallbacks."""
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    try:
        while handle and kernel.WaitForSingleObject(handle, 0) == 258:
            progress(stage="development validation")
            if (ROOT / "results/STOP").exists():
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
                raise InterruptedError("STOP marker during development validation")
            time.sleep(15)
    finally:
        if handle:
            kernel.CloseHandle(handle)
    manifest = build_manifest()
    report_path = ROOT / "reports" / f"smoke_{manifest['model']}_{manifest['reasoning_effort']}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = [manifest["splits"]["development"][i] for i in (0, 4, 8, 12)]
    if report["model"] != manifest["model"] or [r["id"] for r in report["rows"]] != expected:
        raise RuntimeError("Development validation report does not match campus configuration")
    # Success rates do not gate selection; only the interface and valid memory generation do.
    if not any(r["memories"] for r in report["rows"]):
        raise RuntimeError("No valid development memory generated; inspect compatibility")
    from experiment import merge
    from study import client_for
    development = ROOT / "results" / f"smoke_{manifest['model']}_{manifest['reasoning_effort']}"
    items = []
    for task_id in expected:
        record = json.loads((development / f"{task_id}_memory.json").read_text(encoding="utf-8"))
        items.extend(record["items"])
    client = client_for(manifest)
    client.set_ledger(ROOT / "results/development_merge_usage.jsonl")
    rules, generation = merge([], items[:3], client)
    save(ROOT / "reports/campus_merge_validation.json", {"model": client.model, "rules": rules, "generation": generation})
    if not generation.get("update_valid") or not rules:
        raise RuntimeError("Campus development merge interface validation failed")


def main(workers, preflight_pid=None):
    (ROOT/"results").mkdir(exist_ok=True)
    with FileLock(str(ROOT/"results/pipeline.lock"), timeout=0):
        save(ROOT/"results/pipeline_pid.json", {"pid": os.getpid(), "started": now()})
        # Keep this experiment process awake without changing persistent power settings.
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        stage = "validation"
        try:
            seal()
            subprocess.run([sys.executable, "-X", "utf8", "embedding.py"], cwd=ROOT, check=True)
            subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, check=True)
            if preflight_pid:
                await_preflight(preflight_pid)
            for stage, args in [("A", ["study.py", "run", "--workers", str(workers)]), ("B", ["replay.py", "--workers", str(workers)])]:
                run_stage(stage, args)
                subprocess.run([sys.executable, "-X", "utf8", "analyze.py"], cwd=ROOT, check=True)
                progress(stage=stage + " finished")
                sync(f"chore(results): sync completed study {stage}")
            progress(state="complete", stage="A+B")
            sync("docs(results): finalize order-effect research report")
        except InterruptedError as exc:
            progress(state="stopped", stage=stage, error=str(exc))
        except Exception as exc:
            progress(state="needs_attention", stage=stage, error=str(exc))
            try:
                sync("chore(results): record experiment interruption")
            except Exception:
                pass
            raise
        finally:
            if os.name == "nt":
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-pid", type=int)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.seal:
        print(json.dumps(seal()))
    elif args.status:
        print((ROOT/"reports/progress.json").read_text(encoding="utf-8") if (ROOT/"reports/progress.json").exists() else "Not started")
    else:
        main(args.workers, args.preflight_pid)
