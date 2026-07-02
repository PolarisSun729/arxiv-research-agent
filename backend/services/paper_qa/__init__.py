"""论文问答服务模块集合。

模块职责边界如下：
- `PaperQAService` 负责编排论文 QA 主流程，包括索引状态、检索上下文构建、答案生成和 turn 持久化。
- `PaperQASessionService` 负责会话解析、用户上下文、短期记忆合并和问答 turn 落库。
- `ContextPackBuilder` 负责把检索结果打包为 prompt 文本、多模态输入、source payload 和上下文预算 debug。
- `AnswerGenerator` 负责调用生成服务并规范最终答案生成结果。

不要为了测试方便在 `PaperQAService` 上新增底层组件的透传 wrapper；组件能力应由对应组件的单测或集成链路直接覆盖。
"""
