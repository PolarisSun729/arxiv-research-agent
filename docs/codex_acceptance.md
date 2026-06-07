# Codex 修改验收标准

Codex 每次修改代码、测试、脚本或文档后，必须把统一质量门禁作为默认验收口径。除非用户明确要求只做只读分析，否则最终回复需要说明本次运行了哪些检查、哪些通过、哪些失败，以及失败是否与本次修改相关。

## 默认验收命令

仓库根目录下的默认验收命令是：

```bash
python scripts/check_quality.py
```

CI 等价主命令是：

```bash
python scripts/check_quality.py ci
```

这两个入口都只运行离线安全检查：basic doctor、后端静态检查、后端自动化测试、后端启动烟测、前端测试和前端构建。它们不要求真实 API key，不要求 Milvus 启动，不调用真实 LLM、Embedding、rerank、arXiv 或真实 PDF 下载解析。

## Codex 最终回复必须包含

每次完成修改后，Codex 最终回复至少说明：

1. 运行了哪些检查命令；
2. 每个命令的通过或失败结果；
3. 如果失败，失败阶段和关键错误摘要；
4. 判断失败是否与本次修改相关；
5. 如果无法运行，说明无法运行的具体原因，例如依赖缺失、环境缺失、命令不可用或用户明确要求不运行。

如果只运行了分阶段命令，例如 `python scripts/check_quality.py static` 或 `python -m pytest ...`，需要明确说明这不是完整默认门禁，以及为什么本次只运行了子集。

## 失败处理口径

- 如果失败与本次修改相关，优先修复后重新运行对应阶段。
- 如果失败来自当前环境缺依赖，例如 basic doctor 报缺少 Python 包，需要如实说明缺失项，并避免把环境失败描述成业务回归。
- 如果失败来自已知外部服务不可用，应确认该检查是否属于 full doctor 或 manual integration；默认门禁不应依赖真实外部服务。
- 如果用户要求继续推进但完整门禁无法运行，至少运行能覆盖本次改动风险的最小子集，并记录未覆盖风险。

## 检查分层

- 默认一键测试：`python scripts/check_quality.py`，本地提交前验收。
- CI 主命令：`python scripts/check_quality.py ci`，GitHub Actions 和干净环境验收。
- basic doctor：`python scripts/doctor.py basic`，只检查本地环境和本地依赖。
- full doctor：`python scripts/doctor.py full`，显式检查 Milvus、arXiv API/OAI 等真实连接。
- paid full doctor：`python scripts/doctor.py full --check-paid`，显式允许最小真实模型调用。
- manual integration：真实下载、真实索引构建、真实问答、真实推荐和真实数据库/向量库验证，不进入默认 CI。
