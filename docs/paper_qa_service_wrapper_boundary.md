# Paper QA 服务边界和防回归规则

本文档记录清理旧代理函数后的 Paper QA 模块边界。
目标是让 `PaperQAService` 保持论文 QA 高层编排职责，避免重新变成底层组件能力的透传集合。

## 组件职责

| 组件 | 职责 |
| --- | --- |
| `PaperQAService` | 编排论文 QA 主流程：索引状态、上下文构建、检索、答案生成、证据校验、会话持久化 |
| `PaperQASessionService` | 会话解析、用户 ID 兜底、短期记忆合并、用户上下文、问答 turn 落库 |
| `ContextPackBuilder` | 将检索结果打包为 prompt 文本、多模态输入、source payload、上下文预算 debug |
| `AnswerGenerator` | 调用生成服务并整理答案、引用、claims、generation debug |
| `EvidenceVerifier` | 校验答案与 sources 的证据一致性，并应用最终答案 guardrail |
| `PaperQAIndexBuilder` | 构建和清理单篇论文 QA 索引 artifact |
| `routers.qa_utils` | trace 文件名规整、最新 trace 定位、QA 索引诊断等 router 工具能力 |

## PaperQAService 保留入口

以下方法仍被 router、工具入口或真实业务流程直接调用，属于当前对外契约：

- `get_qa_status()`
- `build_qa_index()`
- `delete_qa_index()`
- `build_qa_context()`
- `build_source_payload()`
- `persist_completed_turn()`
- `answer_question()`

这些入口不是底层工具函数集合；它们保留的原因是当前 router 主流程或同步工具入口仍直接依赖它们。

## 已迁出职责

以下能力不应再从 `PaperQAService` 暴露：

- 记忆开关读取：由 `PaperQASessionService.memory_flag()` 承担。
- 上下文文本、多模态输入和 source payload 构建：由 `ContextPackBuilder` 承担。
- 会话文本截断：由 `PaperQASessionService.truncate_text()` 承担。
- trace slug 与最新 trace 文件定位：由 `routers.qa_utils` 承担。

## 测试组织规则

- 组件规则应放在 `backend/tests/unit/services/paper_qa/`，直接导入对应组件。
- `PaperQAService` 集成测试只覆盖 QA 主流程编排、副作用和输出结构。
- router API 测试只覆盖 HTTP 契约、错误转换、SSE 事件和依赖注入边界。
- 不要为了断言某个底层 helper 行为而新增 `PaperQAService` 私有或公开透传方法。

## Code Review 防回归规则

评审 Paper QA 相关改动时，需要检查：

1. 新增方法是否属于 `PaperQAService` 的主流程编排职责。
2. 如果只是调用 `SessionService`、`ContextPackBuilder`、`AnswerGenerator` 或 `qa_utils` 后原样返回，应改为直接测试或调用真实组件。
3. 新测试是否依赖 `PaperQAService` 的私有兼容 wrapper；如果依赖，应迁移到组件单测或主流程集成测试。
4. 流式 QA 仍直接使用的 `build_qa_context()`、`build_source_payload()`、`persist_completed_turn()` 在迁移 router 前不能删除。
5. 修改 LLM 调用、规则兜底、参数校验、状态流转和 debug/trace 记录时，需要同步更新注释与测试断言。

## 推荐验证命令

Paper QA 边界相关改动至少运行：

```bash
pytest backend/tests/api/test_qa_router.py backend/tests/integration/test_paper_qa_service.py backend/tests/unit/services/paper_qa
pytest backend/tests/smoke/test_backend_startup.py
```
