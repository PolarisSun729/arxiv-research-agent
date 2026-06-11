# Agent Runtime 正式能力与收敛分层说明

本文补充说明当前 Agent Runtime 的正式主路径、fallback 分层、compat 边界，以及澄清/推荐/LLM Planner 的真实默认行为，避免把历史模板路径误读成当前主能力。

## 1. 当前正式主路径

默认执行链路保持为：

1. `intent / goal` 构建
2. `planner`
3. `executor`
4. `observer`
5. `replan`
6. `final answer`

其中 planner 负责产出可执行计划，executor 负责按 schema 执行工具，observer 负责把工具结果转成结构化观察信号，replan 负责根据失败类别、恢复策略和安全边界决定是否修补计划或终止。

## 2. Planner 分层与默认模式

当前默认配置以 `AGENT_PLANNER_MODE=llm_preferred` 为主，含义是：

- 优先尝试 LLM Planner 生成结构化计划
- 对 LLM 输出做工具存在性、参数 schema、step 依赖、危险动作、最大步数等校验
- 校验失败时回退到规则型 planner
- 规则型 planner 也无法稳定生成时，最后才进入 `legacy_template_fallback_planner`

当前支持的 planner 运行模式：

- `rule_only`：只用规则 planner，适合稳定演示和保守发布
- `llm_preferred`：优先 LLM，失败后自动 fallback 到规则 planner
- `llm_only_strict`：只测试 LLM planner，校验失败直接报错
- `demo_rule`：显式关闭 LLM 规划，走稳定规则路径

这里的关键约束是：legacy template fallback 不再作为主要扩展对象，只承担最小安全回复职责；新增搜索、QA、推荐、偏好等业务能力应进入 LLM draft 或规则型 tool-aware planner。

## 3. Structured fallback 约定

系统已经把多处自由文本 fallback reason 收敛为统一的 `fallback_record`，主要字段包括：

- `code`：归一化后的 fallback 类型
- `stage`：发生层级，例如 `planner`、`recommendation`、`clarification`、`executor`
- `category`：更稳定的分类标签
- `message`：面向调试/trace 的说明
- `raw_reason`：历史兼容原因字符串
- `source`：触发 fallback 的模块来源
- `resolution`：系统采用的处置方式
- `detail`：附加诊断信息

推荐在日志、trace、前端 debug 面板里优先读取 `fallback_record`，不要再依赖自由拼接的 reason 字符串判断主路径。
planner 层常见 code 包括 `llm_planner_validation_failed`、`tool_aware_planner_failed`、`tool_aware_planner_disabled`、`missing_required_context`、`unsupported_request`。

## 4. 澄清模块的正式能力

澄清分析已经从固定模板升级为结构化缺失信息诊断。当前会结合用户输入和上下文判断：

- 是否缺搜索主题
- 是否缺目标论文
- 是否缺用户问题
- 是否缺推荐条件
- 是否缺偏好操作对象
- 是否缺 human confirmation 的 decision
- 是否缺 `user_id` 或上下文身份信息
- 请求是否不支持
- 请求是否过宽，需要先收窄范围

澄清结果会输出结构化字段，包括：

- 是否需要澄清
- 缺失字段列表
- 缺失原因
- 建议追问方式
- 当前可推断出的意图
- 置信度
- 是否允许系统用默认值继续执行

只有在无法安全推断时才进入澄清节点；能安全默认的场景允许继续执行，并在 trace 中记录默认值来源。

## 5. 推荐链路的正式能力

推荐链路已调整为“Agent 组织上下文，RecommendationService 负责实际计算”的分层。当前推荐会综合使用：

- 长期兴趣画像
- 当前会话意图
- 显式正向偏好
- 显式负向偏好
- 最近交互行为
- 临时主题约束
- 冷启动默认召回依据

推荐结果除了论文本身，还应包含可解释字段，例如：

- `match_reason`
- `matched_profile_terms`
- `matched_query_terms`
- `negative_filter_reason`
- `score_components`
- `preference_sources`

这部分上下文不再只进入 trace，而是实际影响候选过滤、排序、rerank 和解释生成。

## 6. Compat / Legacy 边界

旧前端或旧测试仍可能依赖历史字段，例如 `pending_action`。这类兼容逻辑已经收敛到 `compat/legacy` 层：

- 兼容层只做镜像与桥接
- 正式状态流以结构化 `pending_confirmation` / `resume` 为准
- 新功能不要继续向 legacy 入口扩展

### 6.1 论文目标解析边界

论文总结、论文问答、收藏、不喜欢、标记、下载、详情查看等任何需要定位论文目标的动作，必须统一走：

1. `paper_reference_resolver._resolve_paper_reference()` 只提取引用线索，例如 `arxiv_id`、`ordinal`、`context_paper`、`last_item`、`bare_number`、`unknown`。
2. `paper_target_resolver.resolve_paper_target()` 结合会话/页面上下文生成候选论文，输出 `resolved`、`need_confirmation` 或 `need_clarification`。
3. `PlanExecutor` 在 `need_confirmation` 时生成 `paper_target_confirmation`，通过 LangGraph checkpoint + `resume.edited_arguments` 接收 `pending_action_id` 和 `confirmed_paper_id`。
4. 下游业务工具只能消费 resolver 的明确 `paper_id/arxiv_id`，不能重新解析“第二篇”“这篇”或回退到 `selected_paper` / 第一篇结果。

`pending_action` 仍保留为前端展示镜像，不是恢复执行现场的权威来源。新增论文相关动作时，应接入 `resolve_paper` 或 `resolve_preference_target`，并按 `side_effect_level` 触发确认策略；不要在业务 adapter、旧 node 或前端上下文里新增隐式 paper 绑定。

这样可以保证审查时能明确区分“主路径状态流”和“兼容输出镜像”。

## 7. 调试能力与正式 API

调试能力应与正式业务 API 分开理解：

- 正式主链路关注搜索、论文 QA、推荐、偏好更新、human confirmation
- debug / trace 能力用于观察 planner 路径、工具调用、fallback 与恢复过程
- 判断系统默认行为时，应以 planner/executor/replan 的正式返回和 trace 结构为准，而不是以调试接口是否存在为准

## 8. 排查建议

当一轮 Agent 行为和预期不一致时，建议按下面顺序检查：

1. 看 `planner_summary.final_path` 和 `planner_summary.planner_runtime_mode`
2. 看是否出现 `fallback_record`
3. 看 `selected_plan_source` 是 LLM、规则 planner 还是 template fallback
4. 看 observer 给出的失败类别和恢复候选
5. 看 replan 是否因为安全边界、步数上限或缺少上下文而终止

如果 trace 中出现 `planner_summary.final_path=legacy_template_fallback_planner` 或 `template_fallback_used=true`，说明这轮已经不在正式主路径里，而是在用最后兜底方案保系统可用；具体触发原因看 `fallback_record.code/raw_reason`。
