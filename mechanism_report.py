"""Summarize follow-up at the independent writer or fixed-context execution unit."""
from collections import defaultdict
import json
from pathlib import Path

from scipy.stats import binomtest, fisher_exact
from analyze import contrast, holm
from experiment import save
from mechanism import DATA, OUT, config, read


def risk_difference(a,b):
    """Conservative >=95% CI from two exact 97.5% binomial intervals (union bound)."""
    ca=binomtest(sum(a),len(a)).proportion_ci(confidence_level=.975,method='exact')
    cb=binomtest(sum(b),len(b)).proportion_ci(confidence_level=.975,method='exact')
    return [float(ca.low-cb.high),float(ca.high-cb.low)]


def main():
    x=config()
    factorial=[read(p) for p in sorted((DATA/'factorial').glob('*/summary.json'))]
    ablation=[read(p) for p in sorted((DATA/'ablation').glob('*/summary.json'))]
    groups=defaultdict(list)
    for r in factorial:groups[(r['policy'],r['order'])].append(r['score'])
    primary=contrast(groups,{('repair_length_once','shuffle1'):1,('repair_length_once','shuffle2'):-1,('skip_baseline','shuffle1'):-1,('skip_baseline','shuffle2'):1}) if len(factorial)==12 else {'available':False,'reason':'Wait for all 12 writer runs; no interim testing'}
    ag=defaultdict(list)
    for r in ablation:ag[r['condition']].append(r['score'])
    tests={}
    if len(ablation)==30:
        treatment=ag['remove_rule']
        for control in ['original','remove_control']:
            other=ag[control]
            p=fisher_exact([[sum(treatment),len(treatment)-sum(treatment)],[sum(other),len(other)-sum(other)]],alternative='two-sided').pvalue
            tests[control]={'available':True,'estimate':sum(treatment)/len(treatment)-sum(other)/len(other),'ci95_conservative':risk_difference(treatment,other),'p_two_sided':float(p)}
        holm(tests)
    summary={'status':'complete' if len(factorial)==12 and len(ablation)==30 else 'incomplete','factorial_runs':len(factorial),'ablation_runs':len(ablation),'factorial_primary':primary,'ablation_fisher_holm':tests,'factorial_rows':factorial,'ablation_rows':ablation,'limitations':'Post-hoc selected orders/task. n=3 writers/cell; fixed-context ablation has n=10 independent executions/condition, not 10 memories or tasks. Repair may change semantics and costs; deletion control approximates length only.'}
    save(OUT/'statistics.json',summary)
    lines=['# 机制实验结果','',f"状态：{summary['status']}；融合运行 {len(factorial)}/12；规则消融执行 {len(ablation)}/30。",'', '## 一次长度修复 × 两种顺序','', '| 策略 | 顺序 | 独立 writer 数 | 成功率 | 候选更新不合规 | 修复通过 | 最终跳过 |','|---|---|---:|---:|---:|---:|---:|']
    for (policy,order),values in sorted(groups.items()):
        selected=[r for r in factorial if r['policy']==policy and r['order']==order]
        lines.append(f"| {policy} | {order} | {len(values)} | {sum(values)/len(values):.1%} | {sum(r['proposal_invalid'] for r in selected)} | {sum(r['repair_accepted'] for r in selected)} | {sum(r['updates_skipped'] for r in selected)} |")
    if primary.get('available'):
        lo,hi=primary['ci95'];lines+=['',f"修复前后顺序差值的变化：{primary['estimate']*100:+.2f} 个百分点，95% CI [{lo*100:+.2f}, {hi*100:+.2f}]，p={primary['p_two_sided']:.4f}。"]
        lines+=['','区间跨零，未可靠区分交互差值与 writer/执行波动。' if lo<=0<=hi else '区间未跨零，当前固定设置下观察到策略与顺序的交互信号；小样本筛查仍需独立确认。']
    else:lines+=['',primary.get('reason','统计未就绪')]
    lines+=['','## 单条规则删除消融','','同一旧 playbook、同一旅行任务。original 为原记忆；remove_rule 删除79词的旅客验证规则；remove_control 删除77词的文件系统规则。每条件新执行10次。','','| 条件 | 成功/完成 | 订票请求当轮调用工具/完成 |','|---|---:|---:|']
    for c,values in sorted(ag.items()):
        selected=[r for r in ablation if r['condition']==c]
        lines.append(f"| {c} | {sum(values)}/{len(values)} | {sum(r['booking_on_requested_turn'] for r in selected)}/{len(values)} |")
    for c,r in tests.items():
        lo,hi=r['ci95_conservative']
        lines+=['',f"删除目标规则 − {c}：{r['estimate']*100:+.1f} 个百分点，保守95% CI [{lo*100:+.1f},{hi*100:+.1f}]，Fisher 双侧 p={r['p_two_sided']:.4f}，两项 Holm 校正 p={r['p_holm']:.4f}。"]
    lines+=['','差值区间由两组各97.5%的 Clopper–Pearson 区间相减构成，通过联合界给出至少95%的覆盖；通常较宽，不作跨比较的同时区间。检验与区间不保证互为反演。','','## 推断边界','','此轮是在旧数据上选出顺序和案例后固定的新采样。不能推广到所有顺序、所有任务或新的经验池。长度修复是额外的模型处理，可能改变规则语义；与原策略的差值并非纯格式效应。删除无关规则控制近似长度差异，但仍改变上下文位置和内容。目标规则还包括机场、价格与发票要求，整条删除不能单独归因到旅客验证一句。成功率以官方程序评分为准，案例解释仅适用于模拟环境。没有改变或覆盖 A/B 原始报告。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    brief=['# 给老师的一页进展汇报','', '研究问题：工具调用 Agent 的经验积累与融合是否存在超出生成波动的顺序效应。','','**已完成第一轮：**校园 DeepSeek-V4-Flash-0731-W8A8；BFCL 自定义划分；A 40 次完整运行、B 25 次完整运行，共3500次任务执行。使用固定测试集和独立运行，未训练模型权重。','','**主要结果：**A 的顺序净差值为 −8.5 个百分点，95% CI [−21.5,+4.5]，暂未可靠确认。B 的平均打乱−原顺序约为0，但打乱1与打乱2相差8个百分点；事后六组比较 Holm p=0.042，属于待确认线索。','','**离线诊断：**三个任务贡献净差值的62.5%；两种顺序的融合跳过次数分别14和15，不能用跳过次数解释优势。具体失败包括额外确认导致操作推迟、漏查自选股、将旅客验证过度泛化导致未订票。旧B有59/600次融合更新被拒绝，其中56次因超长。','','**新增机制实验：**固定两种排列，比较原跳过策略与一次长度修复（2×2×3个新writer、240次任务）；另对旅行任务比较原规则、删除目标规则、删除近似等长无关规则（共30次新执行）。不向修复模型提供测试问题或评分。', '',f"**当前状态：**{summary['status']}，融合完成{len(factorial)}/12，消融完成{len(ablation)}/30。"]
    if summary['status']=='complete':
        if primary.get('available'):brief+=['',f"**新结果：**策略×顺序交互差值 {primary['estimate']*100:+.2f} 个百分点，p={primary['p_two_sided']:.4f}。规则消融："+'；'.join(f"{c} {sum(v)}/{len(v)}" for c,v in sorted(ag.items()))+'。细节及不确定性见 RESULTS.md。']
    brief+=['','**建议讨论：**优先研究经验约束被过度泛化及更新接纳策略；当前结果是小样本机制筛查，尚不能声称普遍的顺序效应或语义遗忘机制。']
    (OUT/'TEACHER_BRIEF.md').write_text('\n'.join(brief)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    x=main();print(json.dumps({'status':x['status'],'factorial_runs':x['factorial_runs'],'ablation_runs':x['ablation_runs']}))
