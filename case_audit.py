"""Deterministic descriptive case selection; no extra model calls or causal claims."""
from collections import defaultdict
import itertools
import json

import numpy as np

from benchmark import ROOT
from experiment import save


def audit():
    manifest = json.loads((ROOT/"manifest.json").read_text(encoding="utf-8"))
    cases = []
    left = ROOT/"results/A/reasoning_memory_ordinal_r0"
    right = ROOT/"results/A/reasoning_memory_shuffle1_r0"
    if (left/"summary.json").exists() and (right/"summary.json").exists():
        for position, task_id in enumerate(manifest["splits"]["evaluation"]):
            a = json.loads((left/f"evaluation_{position:03d}.json").read_text(encoding="utf-8"))
            b = json.loads((right/f"evaluation_{position:03d}.json").read_text(encoding="utf-8"))
            if a["success"] == b["success"]:
                continue
            item = {"task_id": task_id, "question": next(m["content"] for m in a["messages"] if m["role"] == "user")}
            for label, folder, record in [("ordinal", left, a), ("shuffle1", right, b)]:
                context = json.loads((folder/f"context_evaluation_{position:03d}.json").read_text(encoding="utf-8"))
                item[label] = {"success": record["success"], "error_type": record["evaluation"].get("error_type"), "retrieved": context["retrieved"], "memory": context["text"], "tool_errors": [m["content"][:500] for m in record["messages"] if m["role"] == "tool" and m["content"].startswith(("Error", "Invalid"))][:5]}
            cases.append(item)
            if len(cases) == 3:
                break
    rows = []
    source_sets = defaultdict(list)
    for path in sorted((ROOT/"results/B").glob("merge_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        changes, invalid = [], 0
        for step in sorted(path.parent.glob("merge_[0-9]*.json")):
            record = json.loads(step.read_text(encoding="utf-8"))
            before = {s for r in record["before"] for s in r["sources"]}
            after = {s for r in record["after"] for s in r["sources"]}
            invalid += not record["generation"].get("update_valid", True)
            changes.append({"task_id": record["task_id"], "lost_source_ids": sorted(before-after), "new_source_ids": sorted(after-before), "rule_count": len(record["after"])})
        final = json.loads((path.parent/"final_memory.json").read_text(encoding="utf-8"))["memory"]
        coverage = {s for r in final for s in r["sources"]}
        source_sets[summary["order"]].append(coverage)
        rows.append({"order": summary["order"], "rep": summary["rep"], "invalid_updates": invalid, "final_source_coverage": len(coverage), "steps": changes})
    similarities = []
    ordered_sets = [(order, values) for order, groups in source_sets.items() for values in groups]
    for (oa, a), (ob, b) in itertools.combinations(ordered_sets, 2):
        similarities.append({"same_order": oa == ob, "jaccard": len(a&b)/len(a|b) if a|b else 1.})
    canonical_hashes = set()
    for path in (ROOT/"results/B").glob("canonical_all_*/summary.json"):
        canonical_hashes.add(json.loads(path.read_text(encoding="utf-8"))["contexts_hash"])
    if len(canonical_hashes) > 1:
        raise RuntimeError("Canonical reference contexts differ between repeats")
    result = {"case_selection": "First three task-ID-ordered disagreements between ordinal r0 and shuffle1 r0, descriptive only", "cases": cases, "memory_evolution": rows, "canonical_contexts_identical": len(canonical_hashes) <= 1, "source_jaccard_descriptive": {label: float(np.mean([s["jaccard"] for s in similarities if s["same_order"] == match])) for label, match in [("within_order", True), ("between_orders", False)] if any(s["same_order"] == match for s in similarities)}, "limitation": "A source-ID disappearing is not proof that its semantic content was forgotten; pairs share memories and are not independent samples."}
    save(ROOT/"reports/case_audit.json", result)
    text = ["# 记忆与失败案例审计", "", "案例按预设规则选取：ordinal r0 与 shuffle1 r0 在固定测试任务上结果不同的前三个任务 ID。单次差异不能证明因果；这里用于定位可检查的记忆和工具行为。", ""]
    for case in cases:
        text.extend(["## " + case["task_id"], "", case["question"], ""])
        for label in ("ordinal", "shuffle1"):
            value = case[label]
            text.extend([f"**{label}**：成功={value['success']}；判分错误类型={value['error_type']}；检索来源={', '.join(value['retrieved'])}。", "", "```text", value["memory"], "```", ""])
    if not cases:
        text.append("当前没有满足该固定选择规则的已完成案例；不会另外挑选更显眼的结果。")
    if rows:
        text.extend(["", "## 融合过程", "", f"已审计 {len(rows)} 个独立融合过程。规则来源的保留/消失和格式失败见 case_audit.json。来源标签消失不等于对应知识一定被遗忘；这只是一项可复查的过程指标。", "", f"全部保留参考条件的检索上下文是否一致：{len(canonical_hashes) <= 1}。"])
    (ROOT/"reports/CASES.md").write_text("\n".join(text)+"\n", encoding="utf-8")


if __name__ == "__main__":
    audit()
