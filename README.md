# 工具调用 Agent 的经验顺序效应实验

目标：区分任务顺序、生成随机性和经验融合顺序对 Agent 表现的影响。按用户选择，任务执行和记忆生成统一用 `DeepSeek-V4-Flash-0731-W8A8`（校园网关）；检索使用本地 BGE encoder。不进行权重训练。

本项目是 BFCL 多轮工具环境上的迁移研究，不是 Fragility / ReasoningBank 原论文分数复现。实际完成情况见 `reports/`，实验设计见 [PROTOCOL.md](PROTOCOL.md)。

## 环境与启动

Python 3.11。仅调用校园端点 http://aigw.dlut.edu.cn/v1，密钥从本机 `DLUT_API_KEY` 环境变量读取；Windows 可读取当前用户环境变量。代码禁止调用官方付费 DeepSeek API，无自动回退。不要把密钥写入 Git。

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe prepare.py
.venv/Scripts/python.exe embedding.py
.venv/Scripts/python.exe benchmark.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe study.py prepare
.venv/Scripts/python.exe study.py smoke
.venv/Scripts/python.exe pipeline.py --seal
.venv/Scripts/python.exe pipeline.py --workers 4
```

`prepare.py` 固定上游提交并验证 SHA-256；不会安装完整 BFCL 大模型依赖。`benchmark.py` 运行全部 200 个标准答案的自检。`manifest.json` 固定数据划分与排列。

每个完整运行独立初始化记忆；每个任务独立初始化模拟环境。测试阶段冻结记忆。模型调用不传未经验证的随机种子，也不把重复编号称为可复现生成种子。临时服务错误最多重试 3 次同一请求，全部记录，不计成任务失败。流水线按检查点恢复，已完成步骤不重新采样。完成 A、B 后自动统计并推送汇总到当前私有仓库。

查看进度：`python pipeline.py --status` 或 `reports/STATUS.md`。停止：在 `results/` 新建 `STOP` 文件；恢复前移走该标记，再启动同一命令。运行时保持电脑联网，进程会临时防止系统自动睡眠，不修改持久电源设置。异常见 `results/stage_A.log`、`results/stage_B.log` 与 `reports/STATUS.md`。

## 文件

- `benchmark.py`：官方工具模拟器和判分器的安全执行适配。
- `experiment.py`：Agent、经验抽取、向量检索、融合。
- `study.py`：数据划分与任务顺序实验。
- `gateway.py`：校园接口、付费路由阻断与调用审计。
- `embedding.py`：本地、文件哈希固定的向量检索。
- `replay.py`：固定经验的顺序融合实验。
- `pipeline.py`：后台执行、断点恢复、状态和私有仓库同步。
- `source.lock.json`：上游代码和数据的固定版本与校验和。
- `results/`：本地详细轨迹，不直接纳入 Git。
- `reports/`：同步的汇总数据、统计、图表和研究报告。

第三方数据与源码位于忽略的 `vendor/`；其 Apache-2.0 许可随下载保留。项目不读取或上传微信聊天、账号密钥、用户私人文件。

## 校园版从头重跑

按用户要求，旧官方 API 数据整体归档到 `results/archive/official_flash_20260922/`，旧报告和协议在 `reports/archive/official_flash_20260922/`。新批次不读取旧任务、记忆和经验池。请求全局限速 24 次/分钟，超时 600 秒，使用 32768 输出上限及请求 high 思考强度。服务是否完整执行 high 参数无法独立验证；实际返回的 reasoning、模型别名和 fingerprint 均记录。

本次启动由后台 pipeline 等待正在执行的四题开发验证，随后验证记忆融合接口，再自动进入 A、B。开发验证不以任务成功率作为模型选择门槛；若无法生成任何有效记忆或融合接口失败，流水线停止并记录 needs_attention。状态页面的 development validation 表示正式任务尚未开始。
