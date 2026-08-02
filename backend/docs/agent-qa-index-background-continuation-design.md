# Agent QA 索引后台 Job 与 Continuation 恢复设计

## 1. 文档状态

- 状态：设计已确认，尚未实施
- 确认日期：2026-07-13
- 适用范围：Agent 论文问答缺少索引后的确认、后台建索引、真实进度展示与原问题恢复
- 不适用范围：完整用户认证系统、页面关闭后主动启动无人值守 Agent 回答、旧版未完成 continuation 迁移

本文记录本轮设计访谈已经确认的业务边界和实施方案。后续实现如果需要改变这里的核心状态语义，应先更新本文，再修改代码和测试。

## 2. 当前问题与历史事实

旧版曾实现 Agent QA 索引异步构建恢复，但后续 runtime clean-break 删除了 QA 专用 continuation 旁路，改成统一的 AgentInteraction、ApprovalGrant 和 SideEffectInvocation。

当前链路存在以下问题：

1. 用户批准 parse_and_index_paper 后，PlanExecutor 会同步调用 build_paper_qa_index。
2. 同步调用期间 Agent SSE 长时间占用，页面刷新无法恢复。
3. 前端当前只能从 tool_call 推断进度；拿不到真实 job 时使用 12% 之类的占位值。
4. IndexJobManager 和 paper_index_jobs 仍然存在，并且 builder 已经产生真实阶段与百分比。
5. 当前 IndexJobManager 依赖进程内 daemon 线程；后端重启后任务记录还在，但实际 worker 已消失。
6. 现有 agent_sessions 只保存轻量会话摘要，无法在恢复 SSE 断开后取回完整最终响应。

因此，本次目标不是回滚 clean-break，而是在新交互与授权模型上重新建立通用后台执行能力。

## 3. 已确认的核心决策

1. 一次用户批准同时覆盖启动后台索引 job，以及 job 成功后继续原问题。
2. job 成功后只恢复原 LangGraph checkpoint，不把 original_question 当作新消息重放。
3. AgentInteraction 只表达等待用户输入；批准后由 AgentWorkContinuation 接管。
4. paper_index_jobs 是真实 job 阶段、百分比与错误的唯一真源。
5. 提交或挂接后台 job 就是本次授权对应的副作用；grant 必须在此之前消费。
6. job 完成后只完成原 invocation 并投影 step 输出，不再次调用 parse_and_index_paper。
7. 后台执行方式由 ToolContract 声明；PlanExecutor 不按 QA 工具名特判。
8. 一个物理索引 job 可以服务多个彼此隔离的 continuation。
9. 只对基础设施中断进行有限自动恢复；明确业务失败后的重试需要新授权。
10. 进度只展示后端真实阶段里程碑；无数据时显示不定进度，不编造百分比。
11. 后台阶段使用 continuation 轮询；job 完成后再建立新的恢复 SSE。
12. 已经启动的恢复由 AgentResumeRun 执行到底，SSE 只负责传输。
13. 用户取消只终止自己的 continuation，不取消共享物理 job。
14. 批准时必须再次检查索引，避免并发 job 已完成后仍制造空副作用。
15. 页面只恢复当前会话的活动任务；新对话不加载旧会话 continuation。
16. job success 必须同时满足 builder 成功和 active 索引后置条件。
17. 使用短事务、持久中间状态和幂等 reconciliation，不使用长事务包住外部调用。
18. job 运行期间 continuation 不过期；成功后默认提供 7 天恢复窗口。
19. 当前 user_id 只是 demo namespace；本次只做严格业务归属校验并预留认证接口。
20. 新状态机 clean break，不迁移旧版未完成 continuation。
21. 增加 append-only 生命周期事件记录，但事件不能成为恢复真源。
22. Agent 建索引只允许后台模式，故障时禁止静默回退同步执行。

## 4. 目标执行链

    check_paper_index
        |
        | index missing
        v
    patch plan with parse_and_index_paper
        |
        v
    side_effect_approval interaction
        |
        | approve
        v
    background preflight
        |
        +-- already_satisfied --> project successful step output
        |                         |
        |                         v
        |                     observe_step
        |
        +-- active_job -------> consume grant
        |                       create invocation
        |                       attach continuation
        |
        +-- missing ----------> consume grant
                                create invocation
                                submit durable job
                                create continuation
                                        |
                                        v
                              waiting_background_job
                                        |
                         job stage/progress persisted
                                        |
                       validated success / explicit failure
                              |                    |
                              v                    v
                       ready_to_resume           failed
                              |
                        selected by user
                              |
                              v
                        AgentResumeRun
                              |
                       exact checkpoint resume
                              |
                      project IndexBuildOutput
                              |
                              v
                         observe_step
                              |
                              v
                    answer original paper question

## 5. 状态真源与职责

| 组件 | 负责的真相 | 不负责的内容 |
| --- | --- | --- |
| AgentInteraction | 尚未完成的用户选择或副作用批准 | job 进度、后台等待、恢复结果 |
| ApprovalGrant | 用户批准的 plan、step、tool 和最终参数指纹 | job 当前状态 |
| SideEffectInvocation | 一次已授权副作用调用的 prepared、invoking 与终态事实 | 前端任务展示 |
| paper_index_jobs | 物理建索引任务、lease、attempt、阶段、百分比和结果 | 用户问题与 checkpoint |
| AgentWorkContinuation | 物理 job 与特定 Agent checkpoint 的恢复绑定 | 物理 job 的阶段真值 |
| AgentResumeRun | 一次 continuation 恢复执行及其完整最终响应 | 原始批准 |
| Agent runtime checkpoint | 业务执行位置和当前等待类型 | job 阶段百分比 |
| LangGraph checkpoint | 图内部可恢复执行现场 | 用户归属与业务授权 |
| 生命周期事件表 | 排障和审计时间线 | 任何业务状态判断 |

## 6. ToolContract 扩展

建议增加内部执行策略模型：

    ExecutionPolicy
      mode: inline | background_job
      handler: optional string

parse_and_index_paper 声明：

    mode: background_job
    handler: paper_qa_index

BackgroundJobHandler 第一版只需要三个能力：

1. preflight：返回 already_satisfied、active_job、missing 或 failed。
2. submit_or_attach：幂等提交或挂接物理 job。
3. project_result：把验证后的 job result 转成工具标准输出。

执行器只判断 execution policy，不判断 tool_name。paper_qa_index handler 可以依赖 IndexJobManager 和 PaperQAService，但通用协调器不能依赖 QA 业务实现。

## 7. 业务状态模型

### 7.1 Continuation 状态

    submitting
      -> waiting_job
      -> ready_to_resume
      -> resuming
      -> resumed

可选终态：

    failed
    cancelled
    expired
    indeterminate

状态说明：

- submitting：授权事务已提交，物理 job 尚未成功挂接。
- waiting_job：已绑定 job，等待物理构建终态。
- ready_to_resume：job 和索引后置条件均已验证通过。
- resuming：已经唯一绑定 AgentResumeRun，禁止再次执行 Command(resume)。
- resumed：恢复运行完成，完整结果已经持久化。
- failed：job 或恢复前置条件明确失败。
- cancelled：用户不再希望继续自己的原问题。
- expired：超过恢复窗口。
- indeterminate：无法证明恢复是否完成，禁止自动重放。

### 7.2 Runtime checkpoint 状态

新增：

    waiting_background_job

该状态只保存 continuation_id 或等价引用，不复制 job progress。普通 waiting_interaction TTL 不得作用于 waiting_background_job。

### 7.3 Resume run 状态

    pending
      -> running
      -> completed

异常终态：

    failed
    indeterminate

同一 continuation 只能创建一个有效 resume_run_id。重复恢复请求只能返回已有 run 的状态或结果。

## 8. 持久化模型

### 8.1 paper_index_jobs 扩展

建议增加：

- recipe_version
- attempt_count
- max_attempts
- worker_id
- lease_acquired_at
- lease_expires_at
- last_heartbeat_at
- result_json
- failure_code
- stage_message
- completed_at

幂等身份建议由以下字段生成：

    arxiv_id + loading_method + recipe_version

同一篇论文同一构建配方只允许一个 active job。不同用户或 session 不进入幂等键。

### 8.2 paper_index_job_attempts

每次 worker 领取或重新领取生成独立 attempt 记录：

- attempt_id
- job_id
- attempt_no
- worker_id
- status
- started_at
- heartbeat_at
- finished_at
- failed_stage
- error_code
- error_message

基础设施恢复可以增加 attempt_no，但仍属于同一个逻辑 job。明确业务失败后，用户重试创建新的授权、invocation 和逻辑 job。

### 8.3 agent_work_continuations

建议字段：

- continuation_id，主键
- user_id
- session_id
- thread_id
- runtime_checkpoint_id
- interaction_id
- grant_id
- invocation_id
- plan_id
- step_id
- tool_name
- arguments_fingerprint
- handler_name
- job_id，可重复索引
- job_idempotency_key
- status
- display_summary_json
- validated_result_json
- resume_run_id
- error_code
- error_message
- created_at
- updated_at
- ready_at
- expires_at
- terminal_at

job_id 不能唯一，因为一个物理 job 可以关联多个 continuation。

display_summary_json 只能保存安全展示字段，例如论文标题和问题短摘要。它不能参与恢复判断。

### 8.4 agent_resume_runs

建议字段：

- resume_run_id，主键
- continuation_id，唯一有效绑定
- user_id
- session_id
- thread_id
- status
- started_at
- finished_at
- final_response_json
- error_code
- error_message
- result_retrieved_at

完整 ArxivSearchResponse 必须在发送 final_response SSE 之前持久化。

### 8.5 agent_work_events

建议记录：

- event_id
- event_type
- occurred_at
- continuation_id
- job_id
- attempt_no
- invocation_id
- resume_run_id
- run_id
- stage
- progress
- error_code
- safe_metadata_json

事件只用于审计，不参与任何恢复状态判断。heartbeat 不逐条写事件，只有 lease 变化和阶段变化写入。

## 9. 短事务与崩溃恢复

### 9.1 批准事务

在一个短事务中：

1. 校验 runtime checkpoint、interaction_id、用户与 session。
2. 校验 plan、step、tool 和参数指纹。
3. 执行 background handler preflight。
4. already_satisfied 时无副作用地解析 interaction。
5. 需要 job 时创建并消费 grant。
6. 创建 prepared invocation。
7. 创建 submitting continuation。
8. 清除 AgentInteraction。
9. runtime checkpoint 转为 waiting_background_job。
10. 写入对应生命周期事件。

事务提交后才能提交或挂接外部 job。

### 9.2 Job 提交与修复

提交使用稳定 idempotency key。以下崩溃窗口均可修复：

- 事务提交后、job 提交前：扫描 submitting continuation 并重新提交。
- job 已创建、continuation 尚未写入 job_id：按 idempotency key 查询并挂接。
- worker 领取后进程崩溃：lease 过期后创建新 attempt 并重新领取。
- job 已成功、continuation 尚未 ready：轮询或 reconciler 根据已验证结果推进。

禁止在 SQLite 写事务内启动线程或执行外部调用。

### 9.3 恢复事务

在一个短事务中：

1. 校验 continuation 所有权、状态和 TTL。
2. 再次确认 job success 和 validated_result_json。
3. CAS 将 ready_to_resume 改为 resuming。
4. 创建唯一 AgentResumeRun。
5. 写入 resume_run_started 事件。

事务提交后执行 Command(resume)。重复请求只能命中已有 resume_run_id。

## 10. 持久 Job Worker

当前 daemon 线程应升级为数据库 lease 模型：

1. submit 只负责创建或复用 job。
2. worker 原子领取 pending 或 retrying job。
3. worker 定期续租，不只依赖阶段回调刷新心跳。
4. 后端启动和定时 reconciliation 都可以回收过期 lease。
5. 多 worker 通过 worker_id 和条件更新避免重复领取。
6. 基础设施中断只在 max_attempts 内自动恢复。
7. 明确业务失败直接进入 failed。
8. 新 attempt 从整个 builder 开头执行，不承诺内部阶段续跑。
9. 版本化索引和 active build 保护必须继续有效，半成品进入 cleanup_pending 或 orphaned。

## 11. Job 成功判据

不能只相信 paper_index_jobs.status。

成功必须同时满足：

1. builder 正常返回结果。
2. 返回值包含 build_id、index_version、collection_name 和 chunk_count。
3. PaperQAService.get_qa_status 返回 has_index 为 true。
4. status 为 indexed。
5. active collection 非空。
6. active build/version 与本次构建结果一致。
7. chunk_count 大于零。

验证通过后，才允许在完成事务中：

- job 转为 success；
- invocation 转为 succeeded；
- 保存标准化 IndexBuildOutput；
- 所有关联 continuation 转为 ready_to_resume。

验证失败使用 completion_validation_failed，禁止恢复 Agent。

## 12. 真实进度契约

百分比表示已进入的阶段里程碑，不表示剩余耗时预测。

| 阶段代码 | 百分比 | 用户展示 |
| --- | ---: | --- |
| validate_loading_method | 5 | 校验文档加载方式 |
| create_build_version | 8 | 创建索引构建版本 |
| mark_index_processing | 10 | 初始化索引构建状态 |
| load_paper_metadata | 15 | 读取论文元数据 |
| download_pdf | 25 | 下载论文 PDF |
| load_pdf_document | 35 | 解析 PDF 文档 |
| chunk_document | 45 | 切分文档内容 |
| structure_table_chunks | 50 | 结构化表格片段 |
| save_chunk_file | 55 | 保存文档切片 |
| save_sparse_index | 62 | 保存稀疏关键词索引 |
| compress_chunks_for_rerank | 65 | 压缩重排文本 |
| build_retrieval_indexes | 72 | 构建检索索引 |
| save_retrieval_index | 76 | 保存检索索引产物 |
| create_retrieval_embeddings | 80 | 创建检索 embedding |
| save_embeddings | 88 | 保存 embedding |
| index_embeddings_to_vector_store | 95 | 写入向量数据库 |
| validate_new_collection | 98 | 校验新向量集合 |
| validate_sparse_index_artifact | 99 | 校验稀疏索引产物 |
| activate_index | 100 | 激活新索引版本 |

展示规则：

- 前端只读取 continuation 响应中组合的真实 job snapshot。
- 无进度时显示不定进度，不显示数字。
- 不在阶段之间编造连续百分比。
- 失败时保留最后真实百分比。
- progress 为 100 但 status 仍为 running 时，显示正在激活。
- 只有 status 为 success 时显示构建完成。
- 新 attempt 显示尝试次数并重新从该 attempt 的实际进度开始。

## 13. API 设计

### 13.1 保留入口

    POST /agent/chat/stream

普通消息和用户批准仍从统一 Agent 入口进入。批准 background_job 工具时，该请求只完成授权、提交或挂接 job，并返回后台任务安全视图，不同步构建索引。

### 13.2 Continuation 查询

    GET /agent/work-continuations/active
    GET /agent/work-continuations/{continuation_id}

服务端根据 request actor 限制用户范围，并组合当前 paper_index_job snapshot。

安全响应只包含：

- continuation_id
- session_id
- status
- safe display summary
- job_id
- job status
- current_stage
- stage label
- progress
- attempt_no
- error code/message
- can_cancel
- can_resume
- ready_at
- expires_at
- resume_run_id

不得返回 runtime checkpoint ID、grant、参数指纹、完整工具参数或内部 resume payload。

### 13.3 取消

    POST /agent/work-continuations/{continuation_id}/cancel

只取消当前用户的 continuation。共享物理 job 不受影响。

### 13.4 恢复

    POST /agent/work-continuations/{continuation_id}/resume/stream

客户端只提交 opaque continuation_id。服务端生成或返回唯一 resume_run_id，校验所有内部身份后执行恢复。

### 13.5 Resume run 查询

    GET /agent/resume-runs/{resume_run_id}

返回运行状态；完成后返回持久化 final_response。重复请求不得重新执行 Command(resume)。

## 14. 前端设计

### 14.1 展示模型

批准前显示 AgentInteraction 确认卡。

批准后确认卡立即消失，改成 AgentWorkCard：

- 论文标题
- 原问题短摘要
- 当前阶段
- 真实百分比
- attempt 次数
- 后台执行说明
- 错误与重试入口
- 停止等待并不再继续回答

不能继续展示可重复点击的确认按钮。

### 14.2 轮询

- 默认每 2.5 秒轮询 continuation。
- 页面隐藏时可以降低频率，重新可见时立即刷新。
- 进入终态后停止轮询。
- 页面只有在已经拥有当前 session 时才加载该 session 的 active continuation；新对话保持空任务列表。
- failed 和 indeterminate 属于审计终态，不得由 active 接口返回，也不得提示用户继续恢复。
- 多个 continuation 共享一个 job 时，前端可以按 job_id 合并实际 job 查询，但每张卡仍保持独立状态。

### 14.3 自动恢复

- 只有一条当前会话任务时，ready 后自动恢复。
- 多条 ready 任务只恢复当前或用户选中的一条。
- 不批量启动多个 AgentResumeRun。
- 已完成 run 从持久化结果恢复气泡，不重新执行。

### 14.4 取消文案

批准后不能显示取消构建索引，应显示：

    停止等待并不再继续回答

因为索引 job 可能被其他 continuation 共享，并且副作用已经开始。

## 15. 失败、重试和取消

### 15.1 基础设施恢复

以下情况可以在同一逻辑 job 和 invocation 下有限恢复：

- worker 在领取后崩溃；
- lease 过期；
- job 记录创建后线程未启动；
- submitting continuation 尚未挂接 job；
- 完成状态已写入但 continuation 尚未推进。

### 15.2 明确业务失败

以下错误进入 job 和 invocation 明确失败：

- PDF 下载失败；
- 文档解析失败；
- chunk 或结构化失败；
- embedding 失败；
- 向量写入失败；
- active collection 校验失败；
- 索引激活失败。

用户点击重试代表新授权，并创建新 grant、invocation、job 和 continuation。旧记录保持失败终态。

### 15.3 取消

- 批准前拒绝：不产生副作用。
- waiting_job 取消：只取消 continuation。
- ready_to_resume 取消：不创建 resume run。
- resuming 后：不撤销已启动 run，结果仍持久化。
- 第一版不提供普通用户取消物理 job。

## 16. 生命周期和清理

- waiting_job 不按普通 interaction TTL 过期。
- ready_to_resume 默认保留 7 天。
- resuming 跟随 AgentResumeRun。
- resumed、failed、cancelled、expired 和 indeterminate 再保留 7 天用于排查。
- 清理 continuation 时同步终结 runtime checkpoint。
- LangGraph checkpoint 只在业务 checkpoint 进入终态并超过保留期后删除。
- 索引和成功 job 历史不因 continuation 过期而删除。
- 页面轮询不能延长 TTL。

## 17. 身份与安全边界

当前 user_id 是 demo namespace，不是真实认证。

本次实现必须：

- 引入统一 get_request_actor 依赖；
- 当前 actor 从 demo user context 取得；
- 所有 continuation 和 resume run 操作校验 user、session、thread 与 checkpoint 归属；
- continuation_id 和 resume_run_id 使用不可预测 UUID；
- 客户端不能提交 tool、step、fingerprint、grant 或内部 resume payload；
- 文档明确当前隔离不能抵御恶意伪造 user_id。

完整 JWT、密码、刷新 token 和用户系统不属于本次范围。

## 18. 迁移策略

### 18.1 保留

- 已激活 QA 索引；
- 索引版本和构建缓存；
- paper_index_jobs 历史；
- 普通 Agent 会话记忆。

### 18.2 不迁移

- 旧 agent_qa_index_continuations 活动记录；
- 旧 pending_action；
- 旧客户端 resume_payload；
- 无参数指纹或 invocation 身份的未完成 checkpoint。

旧表不再被运行代码读取，也不在启动时自动删除。启动日志只统计并提示 legacy continuation unsupported。

## 19. 日志与事件

必须新增结构化事件：

- background_work_prepared
- background_job_submitted
- background_job_attached
- background_job_leased
- background_job_reclaimed
- background_job_stage_changed
- background_job_succeeded
- background_job_failed
- continuation_ready
- continuation_claimed
- continuation_cancelled
- continuation_expired
- resume_run_started
- resume_run_completed
- resume_run_failed
- resume_run_indeterminate
- resume_result_retrieved

INFO 日志记录 ID、状态、阶段、百分比和错误码；完整参数、问题正文、prompt 和文档内容不得进入 INFO。

## 20. 实施切片

### 切片 1：持久 Job 基础

- 扩展 paper_index_jobs schema。
- 新增 attempt 与 lease。
- 把 daemon submit 改成数据库领取模型。
- 增加启动与定时 reconciliation。
- 保存 builder result 并执行完成后置条件校验。
- 保持论文详情页原 API 可用。

验收：后端重启后，未完成 job 能被重新领取；同一 job 不会被两个 worker 同时执行。

### 切片 2：通用后台执行契约

- 增加 ExecutionPolicy。
- 增加 BackgroundJobHandler 协议和 registry。
- 注册 paper_qa_index handler。
- PlanExecutor 只按 policy 分发。
- 删除 Agent 运行路径对同步 build_paper_qa_index 的依赖。

验收：PlanExecutor 中没有 parse_and_index_paper 后台特判。

### 切片 3：Continuation 与事务

- 新增 AgentWorkContinuation、store 和事件表。
- 扩展 runtime checkpoint 状态。
- 完成批准短事务和 submitting reconciliation。
- 支持一个 job 关联多个 continuation。
- 增加批准时 preflight。

验收：批准后立即返回后台任务；页面刷新后仍能读取同一真实进度。

### 切片 4：恢复执行

- 新增 AgentResumeRun。
- 新增 continuation 恢复服务和专用 API。
- 原子 claim 后只执行一次 Command(resume)。
- 把 validated IndexBuildOutput 投影到原 step。
- 从 observe_step 继续原计划。
- 持久化完整 final_response。

验收：断网后可以取回同一个 run 的结果，重复请求不会再次恢复图。

### 切片 5：前端任务卡

- 增加 API 与类型。
- 增加后台任务列表和 AgentWorkCard。
- 接入 2.5 秒轮询。
- 删除 12% 假进度。
- 支持刷新恢复、多任务选择、取消、失败和新授权重试。

验收：所有数字来自真实 job；无数据时只显示不定进度。

### 切片 6：清理与质量门

- 删除 Agent 同步索引执行入口的有效引用。
- 更新注释、文档和架构说明。
- 补齐结构化日志与 trace。
- 执行后端单元、集成、API 测试和前端 build/test。
- 检查旧表没有被运行代码读取。

## 21. 测试矩阵

至少覆盖：

1. 已有索引时不出现确认。
2. 展示确认后其他 job 完成，批准时无副作用继续。
3. 新 job 正常提交、真实阶段推进并恢复原问题。
4. 挂接已有 active job。
5. 一个 job 唤醒多个不同用户或 session 的 continuation。
6. 取消一个 continuation 不影响共享 job 和其他 continuation。
7. 重复批准只产生一个 invocation。
8. 重复 continuation resume 只产生一个 AgentResumeRun。
9. job 进程重启后 lease 回收。
10. 两个 worker 并发领取时只有一个成功。
11. builder 成功但 active 索引后置条件失败。
12. job 明确失败后不自动重试。
13. 基础设施中断在 max_attempts 内恢复。
14. 恢复 SSE 断开后读取同一个持久结果。
15. checkpoint 缺失时明确失败，不重放原问题。
16. ready continuation 超过 7 天后过期。
17. 多条 ready 任务不批量启动 Agent。
18. user/session 不匹配时拒绝查询、取消和恢复。
19. 前端无 job progress 时不显示假百分比。
20. Agent 后台能力故障时不回退同步建索引。

## 22. Definition of Done

- Agent 批准建索引后，原请求不再同步阻塞到构建结束。
- 真实 job 阶段和百分比可在刷新后恢复。
- 所有恢复只使用精确 checkpoint。
- AgentInteraction 不承担后台 job 状态。
- 同一物理 job 可以安全关联多个 continuation。
- grant、invocation、job、continuation 和 resume run 状态可完整审计。
- 后端重启不会让活动 job 永久假运行。
- job success 经过 active 索引后置条件验证。
- continuation resume 至多执行一次，断线后结果可取回。
- 前端不存在 12% 等假进度。
- Agent 不存在同步索引 fallback。
- 旧 continuation 表和旧 resume payload 不参与新状态判断。
- 新增状态流转、异常处理、规则兜底和 trace 代码包含必要且可维护的中文注释。

## 23. 面试官问题与标准回答

### 面试官问题

为什么不能直接把旧 agent_qa_index_continuations 代码恢复回来？

### 标准回答

旧实现依赖 pending_action 和客户端 resume_payload，无法满足新架构的参数指纹、一次性 grant 和 checkpoint 归属约束。应该恢复业务能力，而不是恢复旧状态真源。

### 面试官问题

为什么一次批准可以同时覆盖建索引和建完后继续回答？

### 标准回答

用户批准的是一个边界明确的组合动作：为同一论文、同一构建参数创建索引，并在成功后继续同一个已中断问题。授权严格绑定 interaction、plan、step、参数指纹和 job，不能扩张到其他调用。

### 面试官问题

为什么 original_question 不能作为恢复真源？

### 标准回答

重新提交问题会重新执行意图识别、目标解析和规划，可能选择不同论文、重复确认或重复副作用。精确 checkpoint 才能保存原计划、绑定参数和前序工具输出。

### 面试官问题

为什么批准后不能继续使用 AgentInteraction 展示进度？

### 标准回答

AgentInteraction 只表示系统仍在等待用户输入。批准后系统已经开始工作，继续保留 interaction 会把用户等待态和后台执行态混为一谈，并导致重复确认。

### 面试官问题

为什么 grant 必须在 job 提交前消费？

### 标准回答

提交或挂接 job 就是授权对应的副作用。如果 job 已经执行后才消费授权，系统会出现未经一次性授权记录就产生外部副作用的事实漏洞。

### 面试官问题

为什么 job 成功后不再次调用 parse_and_index_paper？

### 标准回答

原 invocation 已经代表了建索引副作用，再调用一次工具会产生第二次执行语义。正确做法是验证 job 结果，把标准化输出投影到原 step，然后继续 observe_step。

### 面试官问题

为什么后台策略应由 ToolContract 声明？

### 标准回答

ToolContract 是 planner、executor 和 observer 的统一事实来源。由契约声明执行模式可以保持执行器通用，避免再次按 QA 工具名堆积业务分支。

### 面试官问题

为什么一个 job 可以关联多个 continuation？

### 标准回答

索引是论文级共享资产，重复构建浪费成本并增加激活竞态。物理工作可以共享，但每个用户的 checkpoint、invocation 和恢复结果必须独立。

### 面试官问题

为什么业务失败后的重试需要新授权？

### 标准回答

明确失败代表原 invocation 已经终结。再次构建是一项新的外部副作用，不能无限复用一次性 grant。只有结果未知的基础设施中断可以按专用 reconciliation policy 恢复。

### 面试官问题

真实进度为什么仍然可能长时间停在某个百分比？

### 标准回答

这些百分比是阶段里程碑，不是时间估算。例如下载或 embedding 可能耗时很长。真实的含义是数字来自实际 job 阶段，而不是前端为了流畅而插值。

### 面试官问题

为什么选择轮询而不是 job SSE？

### 标准回答

job 状态已经持久化，刷新恢复天然依赖查询。阶段数量很少，2 至 3 秒轮询足够；job SSE 还需要跨 worker 广播，并不能替代持久恢复。

### 面试官问题

为什么 AgentResumeRun 要与 SSE 解耦？

### 标准回答

恢复现场只能消费一次。如果 SSE 断开后重新调用 Command(resume)，可能重复执行。持久 run 让执行至多一次，并允许客户端查询同一个结果。

### 面试官问题

为什么用户取消不能取消共享 job？

### 标准回答

批准后副作用已经开始，而且同一个 job 可能服务其他用户。用户只能撤销自己继续回答的意愿，不能撤销已经发生或被共享的物理工作。

### 面试官问题

为什么批准时还要再次检查索引？

### 标准回答

展示确认卡和点击批准之间存在竞态，其他 job 可能已经完成。批准时 preflight 可以避免创建空 grant、假 invocation 和无意义 continuation。

### 面试官问题

为什么不自动恢复所有 ready continuation？

### 标准回答

一个用户可能有多个 session 或标签页。批量恢复会同时启动多条 LLM 调用并让答案缺少明确归属，因此只恢复当前或用户选择的任务。

### 面试官问题

为什么不能只相信 job 表中的 success？

### 标准回答

job 行更新和索引激活之间仍可能发生不一致。只有 active collection、build、version 和 chunk 等后置条件都成立，才证明 QA 能力真正可用。

### 面试官问题

为什么不用一个事务包住全部流程？

### 标准回答

Docling、Milvus、embedding、LangGraph 和网络传输都不能被 SQLite 事务原子覆盖。长事务只会造成锁竞争，正确方法是短事务、幂等键和持久中间状态。

### 面试官问题

为什么 ready continuation 需要独立 TTL？

### 标准回答

普通确认 TTL 太短，可能早于 job 完成；完全不清理又会永久保存图现场。job 运行期间不超时，成功后提供固定恢复窗口，能够同时保证体验和存储边界。

### 面试官问题

当前 user_id 校验为什么不等于认证？

### 标准回答

user_id 由前端 localStorage 选择，恶意调用者可以伪造。它只能防止正常业务串线，不能证明调用者身份；真实安全需要后端认证与 token。

### 面试官问题

为什么不迁移旧未完成 continuation？

### 标准回答

旧记录缺少参数指纹、grant、invocation 和新 checkpoint schema，无法证明是否安全恢复。保留已建索引比猜测旧执行现场更可靠。

### 面试官问题

为什么事件表不能成为恢复真源？

### 标准回答

事件用于解释历史顺序，可能重复、延迟或缺失。恢复必须读取当前状态表和原子状态版本，否则 debug 数据会反向污染业务判断。

### 面试官问题

为什么 Agent 不能在后台能力失败时同步 fallback？

### 标准回答

同步 fallback 会让同一个 ToolContract 在不同故障条件下具有两套执行语义，并重新引入阻塞、假进度和不可恢复问题。后台提交失败应显式失败并安全重试。
