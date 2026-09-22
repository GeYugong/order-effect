# 暂停时的描述性快照

用户要求暂停后生成；不修改原实验方案，不进行中期显著性检验。

状态：已暂停。完成 445 次任务执行；完整运行 0/40；实验 B 尚未开始。

| 方法 | 阶段 | 成功/已完成 | 描述性成功率 |
|---|---|---:|---:|
| no_memory | evaluation | 44/50 | 88.0% |
| no_memory | evolution | 187/240 | 77.9% |
| reasoning_memory | evolution | 108/155 | 69.7% |

注意：有记忆与无记忆组的已完成任务集合、顺序、重复次数并不相同；未完成任务存在耗时选择偏差。这些比例不能用于证明记忆有效或无效，也不能证明顺序效应。测试阶段只有无记忆组的部分结果，且同一任务有重复执行，不能当作独立样本做推断。

## 失败类型

- multi_turn:instance_state_mismatch: 91
- multi_turn:empty_turn_model_response: 7
- multi_turn:execution_response_mismatch: 8

触及调用/输出上限的记录：{"output_token_limit": 1, "step_limit": 2}

失败来自官方模拟器状态与调用路径检查；其中部分状态不匹配涉及严格文本内容匹配，不能把所有失败都解释为完全无法完成用户需求。未修改官方评分规则。

断点与原始轨迹保留在本地 results/；STOP 标记保留，需用户要求继续后再恢复。
