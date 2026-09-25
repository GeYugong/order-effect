"""Read-only B audit: task contributions, skipped updates, and traceable examples."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from experiment import save

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports/mechanism_v1'


def main():
    scores = defaultdict(lambda: defaultdict(list))
    skips, memories = [], {}
    for folder in sorted((ROOT/'results/B').glob('merge_shuffle*')):
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        for row in summary['rows']:
            scores[row['task_id']][summary['order']].append(row['success'])
        memories[folder.name] = json.loads((folder/'final_memory.json').read_text(encoding='utf-8'))['memory']
        for path in sorted(folder.glob('merge_[0-9]*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            if not row['generation'].get('update_valid', True):
                skips.append({'run': folder.name, 'order': summary['order'], 'position': int(path.stem.split('_')[1]), 'task_id': row['task_id'], 'error': row['generation']['error'], 'path': str(path.relative_to(ROOT)).replace('\\', '/')})
    rows = []
    for task_id, values in scores.items():
        s1, s2 = sum(values['shuffle1']), sum(values['shuffle2'])
        rows.append({'task_id': task_id, 'shuffle1_success': s1, 'shuffle2_success': s2, 'n_each': len(values['shuffle1']), 'extra_successes': s1-s2, 'contribution_pp': (s1-s2)/300*100})
    rows.sort(key=lambda r: (-r['extra_successes'], int(r['task_id'].split('_')[-1])))
    selected = rows[:3]
    pool = json.loads((ROOT/'results/B/frozen_pool.json').read_text(encoding='utf-8'))
    cases = []
    for row in selected:
        case = {'task_id': row['task_id']}
        for order, desired in [('shuffle1', True), ('shuffle2', False)]:
            candidates = []
            for folder in sorted((ROOT/'results/B').glob(f'merge_{order}_r*')):
                for path in sorted(folder.glob('eval[0-9]_[0-9]*.json')):
                    record = json.loads(path.read_text(encoding='utf-8'))
                    if record['task_id'] == row['task_id'] and record['success'] == desired:
                        candidates.append((path, record))
            path, record = candidates[0]
            rules = memories[path.parent.name]
            source_ids = sorted({s for rule in rules for s in rule['sources']})
            case[order] = {'trace_path': str(path.relative_to(ROOT)).replace('\\', '/'), 'success': record['success'], 'evaluation': record['evaluation'], 'decoded': record['decoded'], 'user_turns': [m['content'] for m in record['messages'] if m['role']=='user'], 'final_answers': [m.get('content') for m in record['messages'] if m['role']=='assistant' and not m.get('tool_calls')], 'rules': rules, 'source_experiences': {s: pool['experiences'][s] for s in source_ids}}
        cases.append(case)
    out = {'selection': 'Post-hoc top three positive task contributions, tie-break numeric task ID; first lexicographic successful shuffle1 and failed shuffle2 trace. Selected examples are not causal evidence.', 'task_rows': rows, 'skip_counts': dict(Counter(r['order'] for r in skips)), 'skips': skips, 'cases': cases, 'top3_fraction_of_net_gap': sum(r['extra_successes'] for r in selected)/sum(r['extra_successes'] for r in rows)}
    save(OUT/'offline_audit.json', out)
    lines = ['# 旧 B 实验离线机制诊断', '', '以下均为事后描述，不把同一 playbook 的三次测试当作独立记忆。未调用模型。', '', '| 测试任务 | 打乱1成功/15 | 打乱2成功/15 | 对总体差值贡献（百分点） |', '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['task_id']} | {r['shuffle1_success']} | {r['shuffle2_success']} | {r['contribution_pp']:+.2f} |")
    lines += ['', f"前三道题贡献净差值的 {out['top3_fraction_of_net_gap']:.1%}。两组融合跳过次数：{out['skip_counts']}。总跳过次数相近，不能用次数直接解释胜负；经验身份、出现位置与融合后的语义仍可能重要。", '', '## 三个可追踪案例', '', '完整用户话语、工具调用、判分差异、最终规则和对应原经验保存在 offline_audit.json；以下解释不向后续修复模型泄露测试答案。']
    explanations = {
        'multi_turn_base_130': '失败例在第一轮查询价格后要求再次确认，将用户已指定价格的下单动作拖到下一轮。官方逐轮状态检查因此失败。最终规则含广泛的价格、余额和先决条件检查；它可能增加额外确认，但尚不能从单条轨迹确定是哪条规则导致。',
        'multi_turn_base_135': '失败例完成加入自选股，却未调用 get_watchlist 来满足同一轮显示列表的要求。评分为 execution_response_mismatch；这是漏掉检索动作，不是股票代码或下单价格错误。泛化的“总结新状态”规则是否替代了真实查询尚待检验。',
        'multi_turn_base_192': '失败例已有访问令牌、预算和付款卡，仍要求姓名/生日/护照后才肯订票。最终 playbook 的第9条明确要求订票前一律验证旅客并在信息缺失时追问，来源为150/177/179号经验。该规则与拒绝继续订票的理由直接一致，适合做单条删除消融。',
    }
    for c in cases:
        lines += ['', f"### {c['task_id']}", '', explanations[c['task_id']], '', f"成功轨迹：`{c['shuffle1']['trace_path']}`；失败轨迹：`{c['shuffle2']['trace_path']}`。"]
    lines += ['', '结论：优先检验格式拒绝处理是否改变顺序差异，同时对旅行案例做一条规则的受控删除。两者分别检验过程策略和具体规则，不能合并为一个因果结论。']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'OFFLINE_AUDIT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'selected_tasks': [r['task_id'] for r in selected], 'top3_fraction': out['top3_fraction_of_net_gap'], 'skips': out['skip_counts']}))


if __name__ == '__main__':
    main()
