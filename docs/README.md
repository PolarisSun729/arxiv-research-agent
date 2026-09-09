# 维护者文档

本目录只描述**当前已经运行的实现**，面向需要修改、排障或扩展系统的维护者。

它不承担快速启动、完整配置清单或对外 API 参考的职责；这些内容在形成可复现的发布流程后另行维护。阅读实现时，代码始终高于本文档：如果二者不一致，应先确认代码行为，再在同一变更中修正文档。

## 阅读入口

| 关注问题 | 权威文档 | 首要代码入口 |
| --- | --- | --- |
| 应用如何分层、请求如何进入各能力 | [系统架构](architecture/system-overview.md) | [`backend/main.py`](../backend/main.py)、[`backend/dependencies.py`](../backend/dependencies.py) |
| Agent 如何规划、执行、确认和恢复 | [Agent Runtime](capabilities/agent-runtime.md) | [`backend/agents/arxiv_search_agent/`](../backend/agents/arxiv_search_agent/) |
| 单篇论文 QA、索引任务、会话与诊断如何协作 | [论文 QA](capabilities/paper-qa.md) | [`backend/services/paper_qa/`](../backend/services/paper_qa/) |
| 生成质量、正确拒答、修复增益和成本如何评分 | [生成效果评测](capabilities/evaluation.md) | [`backend/services/evaluation/`](../backend/services/evaluation/) |
| 检索、融合、rerank、上下文预算和索引资产归谁负责 | [检索与索引](capabilities/retrieval-and-indexing.md) | [`backend/services/retrieval/`](../backend/services/retrieval/) |
| 画像、行为信号和推荐排序如何保持语义一致 | [研究画像与推荐](capabilities/research-profile-and-recommendations.md) | [`backend/services/memory/`](../backend/services/memory/)、[`backend/services/recommendation/`](../backend/services/recommendation/) |
| 修改后应如何验证，以及何时必须同步文档 | [维护与质量](operations/maintenance-quality.md) | [`scripts/check_quality.py`](../scripts/check_quality.py) |
| 自动化测试的离线边界、分层命令和覆盖率如何维护 | [测试指南](operations/testing.md) | [`backend/tests/`](../backend/tests/) |

## 文档边界

- 每个业务能力只有一份说明，不按 Router 逐接口复制调用链。
- 文档记录稳定的职责、状态真源、失败边界和代码锚点，不复制容易变动的局部实现细节。
- 调试路由、trace 和内部状态可以在架构文档中说明边界，但不能被写成稳定对外接口。
- 历史改造计划、已完成的迁移基准和逐文件阅读笔记不放在仓库文档中；需要追溯时使用 Git 历史。

## 同步规则

以下变更必须在同一个提交中更新对应文档：

| 代码变更 | 必须更新 |
| --- | --- |
| Router 注册、全局错误出口、依赖装配或持久化容器 | [系统架构](architecture/system-overview.md) |
| Agent 图节点、计划/工具契约、确认或 checkpoint 语义 | [Agent Runtime](capabilities/agent-runtime.md) |
| QA 索引阶段、异步任务状态、问答证据或修复信号 | [论文 QA](capabilities/paper-qa.md) |
| 评测记录、指标定义、golden 格式、调用统计或基线比较 | [生成效果评测](capabilities/evaluation.md) |
| 召回路由、融合、rerank 降级、预算或索引版本语义 | [检索与索引](capabilities/retrieval-and-indexing.md) |
| 画像层次、行为信号、候选召回、排序或多样性策略 | [研究画像与推荐](capabilities/research-profile-and-recommendations.md) |
| 校验脚本、CI 编排或验收口径 | [维护与质量](operations/maintenance-quality.md) |
| 测试分层、离线约束、运行命令或覆盖率口径 | [测试指南](operations/testing.md) |

提交前运行 `python scripts/check_docs.py`。该校验会验证本目录及维护入口文档中的相对 Markdown 链接和代码路径，不访问网络；统一质量门禁也会执行它。
