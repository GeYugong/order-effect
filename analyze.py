"""Run-level statistics; nested replay evaluations are not independent memories."""
import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy import stats

from benchmark import ROOT
from experiment import save


def contrast(groups, weights):
    if any(len(groups[k]) < 2 for k in weights):
        return {"available": False, "reason": "At least two complete runs per group are required"}
    estimate = sum(w * np.mean(groups[k]) for k, w in weights.items())
    pieces = [(w*w * np.var(groups[k], ddof=1) / len(groups[k]), len(groups[k])) for k, w in weights.items()]
    variance = sum(v for v, _ in pieces)
    if variance == 0:
        return {"available": False, "estimate": float(estimate), "reason": "Zero observed run variance; no calibrated t interval"}
    df = variance**2 / sum(v*v/(n-1) for v, n in pieces)
    se = np.sqrt(variance)
    margin = stats.t.ppf(.975, df) * se
    return {"available": True, "estimate": float(estimate), "ci95": [float(estimate-margin), float(estimate+margin)], "p_two_sided": float(2*stats.t.sf(abs(estimate/se), df)), "df": float(df)}


def holm(items):
    valid = sorted([(key, value["p_two_sided"]) for key, value in items.items() if value.get("available")], key=lambda x: x[1])
    previous = 0
    for rank, (key, p) in enumerate(valid):
        previous = min(1., max(previous, (len(valid)-rank)*p))
        items[key]["p_holm"] = previous


def summarize(study):
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "results" / study).glob("*/summary.json")]
    groups = defaultdict(list)
    for row in summaries:
        groups[(row["method"], row["order"])].append(row["evaluation_success"])
    rows = []
    for (method, order), values in sorted(groups.items()):
        mean, sd = float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        margin = float(stats.t.ppf(.975, len(values)-1)*sd/np.sqrt(len(values))) if len(values) > 1 else None
        rows.append({"method": method, "order": order, "n": len(values), "mean": mean, "sd": sd, "ci95": [mean-margin, mean+margin] if margin is not None else None, "runs": values})
    tests = {}
    if study != "B":
        weights = {("reasoning_memory", "ordinal"): -1, ("reasoning_memory", "shuffle1"): .5, ("reasoning_memory", "shuffle2"): .5, ("no_memory", "ordinal"): 1, ("no_memory", "shuffle1"): -.5, ("no_memory", "shuffle2"): -.5}
        primary = contrast(groups, weights)
        methods = ["no_memory", "reasoning_memory"]
    else:
        primary = contrast(groups, {("merge", "ordinal"): -1, ("merge", "shuffle1"): .5, ("merge", "shuffle2"): .5})
        methods = ["merge"]
    for method in methods:
        for order in ("reverse", "shuffle1", "shuffle2"):
            tests[f"{method}:{order}-ordinal"] = contrast(groups, {(method, order): 1, (method, "ordinal"): -1})
    holm(tests)
    failed = [str(p.relative_to(ROOT)) for p in (ROOT / "results" / study).glob("*/failure.json") if not (p.parent / "summary.json").exists()]
    usage = {k: sum(s["usage"].get(k, 0) for s in summaries) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    result = {"study": study, "complete_runs": len(summaries), "failed_runs": failed, "groups": rows, "primary": primary, "exploratory_tests": tests, "usage_complete_runs": usage, "transport_errors": sum(len(s.get("transport_events", [])) for s in summaries)}
    if study == "B":
        within = []
        for summary in summaries:
            per_eval = defaultdict(list)
            for r in summary["rows"]:
                per_eval[r["eval_rep"]].append(r["success"])
            within.append({"method": summary["method"], "order": summary["order"], "writer_rep": summary["rep"], "evaluation_scores": [float(np.mean(v)) for _, v in sorted(per_eval.items())], "memory_hash": summary["memory_hash"], "contexts_hash": summary["contexts_hash"]})
        result["nested_evaluations"] = within
    save(ROOT / "reports" / f"statistics_{study}.json", result)
    with (ROOT / "reports" / f"runs_{study}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["method", "order", "rep", "model", "evolution_success", "evaluation_success", "total_tokens"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({**{k: summary.get(k, "") for k in fields if k != "total_tokens"}, "total_tokens": summary["usage"]["total_tokens"]})
    if rows:
        plot(study, rows)
    return result


def plot(study, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, row in enumerate(rows):
        values = np.array(row["runs"])*100
        color = "#007d79" if row["method"] in ("reasoning_memory", "merge") else "#6c7483"
        ax.scatter(np.full(len(values), i) + np.linspace(-.12, .12, len(values)), values, color=color, alpha=.7)
        yerr = (row["ci95"][1] - row["mean"])*100 if row["ci95"] else 0
        ax.errorbar(i, row["mean"]*100, yerr=yerr, fmt="D", color=color, capsize=4)
    ax.set_xticks(range(len(rows)), [r["method"] + "\n" + r["order"] for r in rows], rotation=25, ha="right")
    ax.set_ylabel("Frozen-memory evaluation success (%)")
    ax.set_title(f"Study {study}: complete-run means and 95% t intervals")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / f"order_effect_{study}.png", dpi=180)
    fig.savefig(ROOT / "reports" / f"order_effect_{study}.pdf")
    plt.close(fig)


def report(results):
    text = ["# 顺序效应实验报告", "", "本报告基于实际完成的运行。主实验使用 MiniMax-M2.7-bf16；数据是 BFCL 的自定义跨任务划分，非原榜单评测。", ""]
    for result in results:
        study = result["study"]
        text.extend([f"## 实验 {study}", "", f"完整运行：{result['complete_runs']}；未完成运行：{len(result['failed_runs'])}；已完成运行返回的总词元：{result['usage_complete_runs']['total_tokens']:,}；服务重试事件：{result['transport_errors']}。", "", "| 方法 | 顺序 | 独立运行数 | 测试成功率均值 | 运行间标准差 |", "|---|---|---:|---:|---:|"])
        for row in result["groups"]:
            text.append(f"| {row['method']} | {row['order']} | {row['n']} | {row['mean']:.1%} | {row['sd']:.1%} |")
        text.append("")
        primary = result["primary"]
        if primary.get("available"):
            lo, hi = primary["ci95"]
            text.append(f"预先指定的主要差值：{primary['estimate']*100:+.2f} 个百分点，95% 区间 [{lo*100:+.2f}, {hi*100:+.2f}]，双侧 p={primary['p_two_sided']:.4f}。")
            text.append("区间覆盖零，当前样本不能可靠区分该差值与运行波动。" if lo <= 0 <= hi else "该区间未覆盖零，在当前模型、数据划分及统计假设下支持该差异；仍需结合效应大小和外部验证。")
        else:
            text.append("主要统计暂不可用：" + primary.get("reason", "未知"))
        text.extend(["", f"![运行分布](order_effect_{study}.png)", ""])
    text.extend(["## 解释边界", "", "A 的主要差值是记忆与无记忆条件下 shuffle−ordinal 差值的差，不是单独的 shuffle 成功率。B 的单位是独立构建的记忆；同一记忆的三次评估先取平均。所有结论条件于固定任务集，不能把几十道测试题算成几十次独立进化重复。", "", "适配方法是 ReasoningBank 风格与自定义顺序融合器，不是 ReasoningBank/ACE 原论文复现。原始顺序不等于难度课程。公开数据可能已被模型训练使用；网关模型别名和部分服务端随机性无法完全锁定。服务失败未返回的词元、失败开发运行及固定经验池抽取费用不包含在上述已完成运行合计中。", "", "详细协议见 PROTOCOL.md；每组的原始运行值、探索性检验及 Holm 校正在 statistics_*.json；逐运行表见 runs_*.csv。"])
    (ROOT / "reports/REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    results = [summarize(study) for study in ("A", "B", "C") if (ROOT / "results" / study).exists()]
    report(results)
    print(json.dumps([{k: r[k] for k in ("study", "complete_runs", "primary")} for r in results]))
