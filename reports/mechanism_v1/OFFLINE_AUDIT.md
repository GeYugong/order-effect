# 旧 B 实验离线机制诊断

以下均为事后描述，不把同一 playbook 的三次测试当作独立记忆。未调用模型。

| 测试任务 | 打乱1成功/15 | 打乱2成功/15 | 对总体差值贡献（百分点） |
|---|---:|---:|---:|
| multi_turn_base_130 | 12 | 5 | +2.33 |
| multi_turn_base_135 | 14 | 10 | +1.33 |
| multi_turn_base_192 | 15 | 11 | +1.33 |
| multi_turn_base_11 | 13 | 10 | +1.00 |
| multi_turn_base_38 | 14 | 11 | +1.00 |
| multi_turn_base_81 | 15 | 13 | +0.67 |
| multi_turn_base_55 | 13 | 12 | +0.33 |
| multi_turn_base_79 | 11 | 10 | +0.33 |
| multi_turn_base_158 | 15 | 14 | +0.33 |
| multi_turn_base_169 | 15 | 14 | +0.33 |
| multi_turn_base_1 | 15 | 15 | +0.00 |
| multi_turn_base_14 | 15 | 15 | +0.00 |
| multi_turn_base_83 | 14 | 14 | +0.00 |
| multi_turn_base_101 | 15 | 15 | +0.00 |
| multi_turn_base_123 | 15 | 15 | +0.00 |
| multi_turn_base_142 | 15 | 15 | +0.00 |
| multi_turn_base_173 | 0 | 0 | +0.00 |
| multi_turn_base_191 | 15 | 15 | +0.00 |
| multi_turn_base_24 | 10 | 11 | -0.33 |
| multi_turn_base_76 | 6 | 8 | -0.67 |

前三道题贡献净差值的 62.5%。两组融合跳过次数：{'shuffle1': 14, 'shuffle2': 15}。总跳过次数相近，不能用次数直接解释胜负；经验身份、出现位置与融合后的语义仍可能重要。

## 三个可追踪案例

完整用户话语、工具调用、判分差异、最终规则和对应原经验保存在 offline_audit.json；以下解释不向后续修复模型泄露测试答案。

### multi_turn_base_130

失败例在第一轮查询价格后要求再次确认，将用户已指定价格的下单动作拖到下一轮。官方逐轮状态检查因此失败。最终规则含广泛的价格、余额和先决条件检查；它可能增加额外确认，但尚不能从单条轨迹确定是哪条规则导致。

成功轨迹：`results/B/merge_shuffle1_r0/eval0_012.json`；失败轨迹：`results/B/merge_shuffle2_r0/eval1_012.json`。

### multi_turn_base_135

失败例完成加入自选股，却未调用 get_watchlist 来满足同一轮显示列表的要求。评分为 execution_response_mismatch；这是漏掉检索动作，不是股票代码或下单价格错误。泛化的“总结新状态”规则是否替代了真实查询尚待检验。

成功轨迹：`results/B/merge_shuffle1_r0/eval0_013.json`；失败轨迹：`results/B/merge_shuffle2_r0/eval1_013.json`。

### multi_turn_base_192

失败例已有访问令牌、预算和付款卡，仍要求姓名/生日/护照后才肯订票。最终 playbook 的第9条明确要求订票前一律验证旅客并在信息缺失时追问，来源为150/177/179号经验。该规则与拒绝继续订票的理由直接一致，适合做单条删除消融。

成功轨迹：`results/B/merge_shuffle1_r0/eval0_019.json`；失败轨迹：`results/B/merge_shuffle2_r2/eval0_019.json`。

结论：优先检验格式拒绝处理是否改变顺序差异，同时对旅行案例做一条规则的受控删除。两者分别检验过程策略和具体规则，不能合并为一个因果结论。
