# 工具调用 Agent 的经验顺序效应实验

目标：区分任务顺序、生成随机性和经验融合顺序对 Agent 表现的影响。主模型为校内网关的 `MiniMax-M2.7-bf16`，交叉模型为 `Qwen3.5-122B-A10B`。不进行权重训练。

本项目是 BFCL 多轮工具环境上的迁移研究，不是 Fragility / ReasoningBank 原论文分数复现。实际完成情况见 `reports/`，实验设计见 [PROTOCOL.md](PROTOCOL.md)。

## 环境与启动

Python 3.11。密钥仅从本机 `DLUT_API_KEY` 环境变量读取；Windows 可从当前用户的环境变量读取。不要把密钥写入 Git。

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe prepare.py
.venv/Scripts/python.exe benchmark.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe study.py prepare
.venv/Scripts/python.exe study.py smoke
.venv/Scripts/python.exe study.py run --workers 8
```

`prepare.py` 固定上游提交并验证 SHA-256；不会安装完整 BFCL 大模型依赖。`benchmark.py` 运行全部 200 个标准答案的自检。`manifest.json` 固定数据划分与排列。

每个完整运行独立初始化记忆；每个任务独立初始化模拟环境。测试阶段冻结记忆。模型调用不传未经验证的随机种子，也不把重复编号称为可复现生成种子。临时服务错误最多重试 3 次同一请求，全部记录，不计成任务失败；超限的未完成运行单独报告，已完成运行不会覆盖。

## 文件

- `benchmark.py`：官方工具模拟器和判分器的安全执行适配。
- `experiment.py`：Agent、经验抽取、向量检索、融合。
- `study.py`：数据划分与任务顺序实验。
- `gateway.py`：校内网关适配与接口检查。
- `source.lock.json`：上游代码和数据的固定版本与校验和。
- `results/`：本地详细轨迹，不直接纳入 Git。
- `reports/`：同步的汇总数据、统计、图表和研究报告。

第三方数据与源码位于忽略的 `vendor/`；其 Apache-2.0 许可随下载保留。项目不读取或上传微信聊天、账号密钥、用户私人文件。
