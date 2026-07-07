# Agent 对话技术流程梳理

## 1. 一句话总览

这个项目里的 Agent 对话不是让 LLM 随机选择工具，而是一个 **FastAPI 入口 + LangGraph 显式状态机 + 结构化计划执行器 + 工具注册表 + checkpoint/resume** 组成的可观察、可中断、可恢复的执行系统。

用户的一句话会先被解析成 intent，再构造成 Goal，然后生成 ExecutablePlan。执行时每一步都会经过输入绑定、工具调用、结果观察、失败恢复，最后汇总成前端响应。如果中途遇到需要用户确认的动作，比如构建论文 QA 索引，会通过 LangGraph interrupt 暂停，并通过 checkpoint 保存现场，等用户确认后再 resume 回原执行点继续跑。

## 2. 总体流程图

```text
POST /agent/chat 或 POST /agent/chat/stream
  -> routers/agent_router.py
  -> agents/arxiv_search_agent/service.py
     1. 规范化请求
     2. 补全 session_id
     3. 加载用户记忆和 agent session memory
     4. 构造 AgentState
     5. 构造带 checkpointer 的 LangGraph
     6. 普通请求 graph.invoke(initial_state)
        resume 请求 graph.invoke(Command(resume=...))
  -> agents/arxiv_search_agent/graph.py
     parse_search_request
       -> build_goal
       -> build_plan
       -> select_next_step
       -> execute_step
       -> observe_step
       -> replan / select_next_step / finalize / error_finalize
  -> ArxivSearchResponse 或 SSE events
```

## 3. API 入口层

Agent 对话主要有两个接口：

```text
POST /agent/chat
POST /agent/chat/stream
```

`/agent/chat` 是同步接口，等整轮 Agent 执行完成后一次性返回结果。

`/agent/chat/stream` 是流式接口，使用 SSE 返回过程事件，比如：

```text
run_start
step_start
tool_call_start
tool_call_end
step_end
final_response
stream_end
exception
```

请求模型是 `ArxivSearchRequest`，核心字段包括：

```text
user_id
session_id
message
context
resume
```

含义：

- `user_id`：用于用户记忆、画像、个性化推荐和日志追踪。
- `session_id`：用于 Agent 会话，同时和 LangGraph 的 `thread_id` 对齐，支持中断恢复。
- `message`：用户本轮自然语言输入。
- `context`：前端或上游传入的上下文，比如当前选中的论文、上一轮 papers、用户画像、pending_action 等。
- `resume`：用户确认后的恢复请求，比如批准构建索引、确认目标论文。

## 4. service.py 的职责

`service.py` 是 Agent 的运行入口层。它不负责具体业务推理，而是把外部 HTTP 请求接到 LangGraph 工作流上。

同步接口的大致流程：

```text
run_arxiv_search_agent(request)
  -> _coerce_request
  -> 生成 run_id
  -> StorageContainer()
  -> 构造 MemoryService
  -> _load_agent_request_context
  -> _ensure_session_id
  -> _build_langgraph_config(session_id)
  -> _build_runtime_checkpoint_manager
  -> _build_agent_graph
  -> 如果是普通请求：构造 initial AgentState，graph.invoke(initial_state)
  -> 如果是 resume 请求：校验 checkpoint，graph.invoke(Command(resume=...))
  -> 持久化 runtime checkpoint
  -> 保存 agent session memory
  -> _state_to_response
```

逐步解释：

### 4.1 `run_arxiv_search_agent(request)`

这是同步 Agent 对话的总入口。它接收一个 `ArxivSearchRequest`，最后返回一个 `ArxivSearchResponse`。

可以把它理解成一轮 Agent 对话的“总调度函数”。它本身不负责搜索论文、不负责回答论文问题、不负责推荐算法，而是负责把请求接到正确的运行环境里，然后把 LangGraph 跑出来的最终状态转成对外响应。

它负责的事情主要是：

```text
准备运行上下文
构造 AgentState
启动 LangGraph
处理普通请求和 resume 请求
持久化执行现场
把内部状态转换成接口响应
```

### 4.2 `_coerce_request`

这一步是把外部传进来的请求统一转成标准的 `ArxivSearchRequest`。

为什么需要它：

```text
上层可能传 Pydantic 对象
测试里可能传 dict
字段里可能有多余空格
resume 字段可能是 dict 或对象
```

所以 `_coerce_request` 会做一层归一化，确保后续代码不用到处判断输入类型。

它的输出是一个干净的请求对象：

```text
normalized_request.user_id
normalized_request.session_id
normalized_request.message
normalized_request.context
normalized_request.resume
```

面试讲法：

> 入口第一步会把请求归一化成标准 schema，避免后续图节点和 service 逻辑同时兼容 dict、对象、空字符串等多种输入形态。

### 4.3 生成 `run_id`

`run_id` 是本轮 Agent 执行的日志和 trace 关联 ID。

它和 `session_id` 不一样：

```text
session_id：表示一段会话，可以跨多轮对话复用
run_id：表示本次请求的一次执行，只对应当前这一轮
```

比如同一个 `session_id` 下，用户连续问三次，会有同一个会话 ID，但每次请求都有不同的 `run_id`。

`run_id` 会进入：

```text
RequestTrace
日志 info_event
AgentState.context
AgentState.debug
下游工具链路
```

它的作用是排查问题时可以把一轮请求从入口、图节点、工具调用、RAG 服务、最终响应串起来。

### 4.4 `StorageContainer()`

`StorageContainer` 是 SQLite 存储层的统一容器。

它会把项目里需要的各种 store 聚合起来，比如：

```text
agent session store
agent runtime checkpoint store
LangGraph checkpoint store
paper catalog store
paper QA index store
research profile store
paper chat session store
preference / profile event store
```

`service.py` 不直接 new 一堆 store，而是先创建 `StorageContainer`，再从里面取需要的 store 组装服务。

面试讲法：

> StorageContainer 是运行入口侧的存储依赖聚合器，Agent 不直接操作数据库连接，而是通过 store 和 service 完成会话、checkpoint、画像和论文数据的读写。

### 4.5 构造 `MemoryService`

`MemoryService` 用来加载和保存 Agent 相关记忆。

在 Agent 对话入口，它主要负责两类记忆：

```text
用户长期记忆：
用户画像、研究兴趣、偏好、兴趣向量等

Agent 会话记忆：
上一次搜索结果、选中的论文、pending_action、paper_qa_result 等
```

为什么 Agent 对话入口要加载记忆：

```text
用户说“总结这篇论文”时，需要知道“这篇”是哪篇
用户说“推荐类似的”时，需要知道上一轮 papers 或用户画像
用户确认索引构建后，需要恢复之前的 pending 状态
个性化搜索和推荐需要用户研究兴趣
```

所以这里构造 `MemoryService`，后面 `_load_agent_request_context` 会用它读取上下文。

### 4.6 `_load_agent_request_context`

这是入口里非常关键的一步：把前端传来的 context 和后端保存的 memory 合并。

它内部大致做两件事。

第一，注入用户长期记忆：

```text
memory_service.build_user_memory_summary(user_id)
```

拿到：

```text
user_memory_summary
research_profile
preference memory 状态
interest vector 状态
```

第二，加载 Agent 会话记忆：

```text
memory_service.load_agent_memory(user_id, session_id, frontend_context=...)
```

这一步会把后端保存的上下文和前端传来的上下文合并，得到：

```text
request_context
agent_memory_payload
resolved_session_id
user_memory_debug
```

这几个返回值分别表示：

```text
request_context：
本轮真正给 Agent 使用的上下文，后面会进入 AgentState.context

agent_memory_payload：
后端会话记忆原始载荷，里面可能有 pending_action、paper_qa_result 等

resolved_session_id：
最终确定的 session_id，可能来自请求，也可能来自后端会话记忆

user_memory_debug：
记忆加载和合并的调试信息
```

一个典型例子：

```text
前端 context 里传了 selected_paper
后端 memory 里也存了上一轮 last_papers
用户长期画像里有 research_profile
```

合并后，AgentState.context 里可能同时有：

```text
selected_paper
last_papers
research_profile
user_memory_summary
run_id
```

面试讲法：

> 这一步解决的是 Agent 的上下文连续性问题。自然语言里经常有“这篇论文”“类似的”“继续刚才那个”这种引用，如果只看当前 message，Agent 无法判断目标。入口会把前端上下文、后端会话记忆和用户长期画像合并成 request_context，供后续 intent 识别和 planner 使用。

补充说明：这里的三类上下文职责不同，不能混在一起讲。

```text
本轮 message：
决定用户当前明确表达的意图和约束。

前端 context / Agent 会话记忆：
解决“这篇论文”“第一篇”“继续刚才那个”“类似的”这类当前会话引用。

用户长期画像：
只作为个性化软信号，影响默认偏好、推荐排序、搜索结果 rerank、回答风格等。
```

也就是说，并不是每一轮都必须依赖长期画像。比如用户明确说：

```text
总结 arXiv:2401.12345 这篇论文的方法
```

这种请求本轮 message 已经包含目标和任务，长期画像不是必要条件。系统可以直接识别为 `paper_qa`，解析目标论文并进入 QA 流程。

但长期画像在下面这些场景有价值：

```text
用户说“推荐几篇 RAG 相关论文”
  -> 当前 message 只说明大方向，长期画像可以帮助判断用户更关注 Agent、检索优化、医学 NLP 还是工程系统。

用户说“帮我找几篇类似的”
  -> 会话记忆解决“类似哪篇”，长期画像帮助排序哪些更符合用户长期兴趣。

用户说“总结这篇论文”
  -> 会话上下文解决“这篇是哪篇”，长期画像可以影响回答风格，比如更偏实现细节、实验对比或简洁总结。

用户搜索结果很多
  -> 当前 query 决定召回范围，长期画像只做轻量 rerank 和理由解释。
```

正确优先级应该是：

```text
1. 本轮用户明确指令优先级最高
2. 当前会话上下文用于解析引用
3. 长期画像只作为个性化软约束
4. 事实性回答仍必须来自工具结果或 RAG 证据
```

面试时可以明确说：

> 长期画像不是理解每一句话的必要条件，也不能覆盖用户当前明确表达的意图。它主要用于个性化场景，比如推荐、搜索结果重排、默认回答风格和偏好解释。真正解决“这篇论文”“刚才那个”这类引用的是当前会话 context，而不是长期画像。

### 4.7 `_ensure_session_id`

这一步保证一定有可用的 `session_id`。

逻辑很简单：

```text
如果请求或 memory 里已有 session_id，就复用
如果没有，就生成一个 uuid
```

为什么必须有 `session_id`：

```text
Agent session memory 要按 session_id 保存
LangGraph checkpoint 要按 thread_id 保存
interrupt/resume 要回到同一个 thread_id
前端也要用 session_id 关联一段对话
```

项目里有一个关键设计：

```text
LangGraph thread_id = session_id
```

所以 `session_id` 不只是前端会话 ID，也是图执行恢复 ID。

### 4.8 `_build_langgraph_config(session_id)`

LangGraph 调用时需要传 config。

这里构造的核心是：

```text
{"configurable": {"thread_id": session_id}}
```

这个 `thread_id` 用来告诉 LangGraph：

```text
这次执行属于哪个线程
checkpoint 应该写到哪里
resume 时应该从哪个线程恢复
```

面试讲法：

> LangGraph 的 checkpoint 是按 thread_id 维度保存的。项目里把业务 session_id 和 thread_id 对齐，这样用户确认后只要带着同一个 session_id，就能恢复原来的执行现场。

### 4.9 `_build_runtime_checkpoint_manager`

这里构造的是业务层 runtime checkpoint 管理器。

项目里有两层 checkpoint：

```text
LangGraph checkpoint：
保存图运行时的底层执行现场，用于 Command(resume)

Agent runtime checkpoint：
保存业务可理解的执行现场，比如 user_id、session_id、thread_id、pending_confirmation、runtime_state、状态等
```

为什么需要业务 checkpoint：

```text
只靠 LangGraph checkpoint 不方便做业务校验
前端 resume 时需要校验 pending_action_id / step_id / tool_name 是否匹配
需要防止旧确认、重复点击、跨用户恢复、跨 session 恢复
需要记录 running / waiting_confirmation / completed / failed / cancelled
```

所以 resume 前会先校验业务 checkpoint，再校验 LangGraph checkpoint。

面试讲法：

> LangGraph checkpoint 负责恢复图本身，业务 runtime checkpoint 负责确认这个恢复请求在业务上是否合法。两层都通过后才允许 resume。

### 4.10 `_build_agent_graph`

这一步构造真正的 LangGraph。

它会做：

```text
build_agent_checkpointer(langgraph_checkpoint_store)
build_arxiv_search_graph(
  generation_service=...,
  checkpointer=...,
  runtime_checkpoint_store=...
)
```

生成的图包含：

```text
parse_search_request
build_goal
build_plan
select_next_step
execute_step
observe_step
route_after_observation
replan
finalize
error_finalize
```

这里会把两个关键依赖注入进去：

```text
generation_service：
用于 intent 识别、规划、诊断等 LLM 能力

runtime_checkpoint_store：
用于执行过程中保存业务 runtime checkpoint
```

### 4.11 普通请求：构造 `initial AgentState`

如果不是 resume 请求，说明这是用户发起的新一轮正常对话。

这时会调用 `_build_initial_agent_state`，把请求和上下文组装成图的初始状态。

初始状态里包括：

```text
user_id
session_id
message
context
pending_action
paper_qa_result
debug
```

其中：

```text
context = request_context + run_id
debug = user_memory_debug + run_id
pending_action / paper_qa_result 来自后端 agent memory
```

为什么要把 `pending_action` 和 `paper_qa_result` 带进去：

```text
兼容前端展示
让 planner 能知道上一轮是否有等待确认或已有 QA 结果
但真实恢复不能只依赖 pending_action
```

然后执行：

```text
graph.invoke(initial_state.model_dump(), config=graph_config)
```

也就是从 `START -> parse_search_request` 开始跑完整张图。

### 4.12 resume 请求：校验 checkpoint 后 `Command(resume=...)`

如果请求里有 `resume`，说明用户不是发起普通对话，而是在恢复一个之前被 interrupt 暂停的执行。

比如之前执行到：

```text
parse_and_index_paper 需要用户确认
```

图已经暂停并返回 pending_action。

用户点击确认后，请求会带：

```text
resume.decision
resume.pending_action_id
resume.step_id
resume.tool_name
resume.edited_arguments
```

后端会先构造轻量 resume_payload：

```text
_build_resume_payload(normalized_request.resume)
```

然后调用：

```text
_ensure_resume_checkpoint(...)
```

这一步会校验：

```text
业务 runtime checkpoint 是否存在
user_id / session_id / thread_id 是否匹配
pending_confirmation 是否仍然有效
pending_action_id 或 step_id/tool_name 是否匹配
LangGraph checkpoint 是否存在
```

校验通过后才执行：

```text
graph.invoke(Command(resume=resume_payload), config=graph_config)
```

注意：resume 请求不会重新构造一轮新的 initial AgentState。它是从 LangGraph checkpoint 保存的中断现场继续执行。

面试讲法：

> resume 不是把用户确认当成一条新消息重新规划，而是用 `Command(resume=...)` 回到上一次 interrupt 的图执行现场。这样可以保证确认的是当时那个 step，而不是重新解析自然语言后猜测用户想确认什么。

这一块要特别补充：项目后来不是只靠 LangGraph 内存现场，而是靠 **SQLite 数据库里的两类 checkpoint** 才稳定下来。

```text
LangGraph checkpoint：
保存 LangGraph 原始执行现场，让 Command(resume=...) 能回到 interrupt 位置。

Agent runtime checkpoint：
保存业务可理解的恢复状态，比如 user_id、session_id、thread_id、runtime_state、pending_confirmation、status、expires_at。
```

为什么一开始容易出 bug：

```text
1. 只靠内存 checkpoint，服务重启、热更新、多 worker 或进程切换后，resume 找不到原执行现场。

2. 只靠前端 pending_action，页面刷新、旧缓存或重复点击时，后端无法可靠判断这个确认是否还有效。

3. LangGraph resume 有时会重放 interrupt 前的旧 waiting 快照，如果直接把旧状态写回，就会把已经 approve 的确认重新变成 waiting_confirmation。

4. 用户快速连续点击确认时，两个请求可能同时看到同一张确认卡片，导致同一个副作用工具被重复执行。
```

所以现在的实现里，数据库承担了恢复真源的职责。

### 4.12.1 LangGraph 原始 checkpoint 存数据库

`runtime_checkpoint.py` 里实现了 `SqliteAgentCheckpointer`。

它把 LangGraph 原始 checkpoint 存到：

```text
langgraph_checkpoints
langgraph_checkpoint_writes
```

核心字段包括：

```text
thread_id
checkpoint_ns
checkpoint_id
parent_checkpoint_id
checkpoint_json
metadata_json
pending_writes
```

这里的 checkpoint 是 LangGraph 自己恢复执行需要的底层快照。项目里会把它 pickle 后 base64 包进 JSON，再落到 SQLite。

它只负责一件事：

```text
让 graph.get_state / graph.invoke(Command(resume=...)) 能找到原图现场。
```

它不负责判断：

```text
这个用户能不能恢复
这个确认有没有过期
这个 pending_action_id 是否匹配
这个确认是否已经被消费
```

这些属于业务语义，放在 Agent runtime checkpoint 里。

### 4.12.2 Agent runtime checkpoint 存业务恢复真源

业务恢复状态存到：

```text
agent_runtime_checkpoints
```

这张表记录：

```text
checkpoint_id
user_id
session_id
thread_id
runtime_state_json
graph_state_json
pending_confirmation_json
current_node
next_route
status
error_summary
expires_at
created_at
updated_at
```

它的核心作用是：

```text
1. 判断当前 session/thread 是否真的有一个 waiting_confirmation
2. 校验 resume 请求是不是匹配当前 pending_confirmation
3. 原子消费 pending_confirmation，防止重复点击
4. 记录 completed / failed / cancelled / expired 等终态
5. 清理旧 pending，避免旧确认被继续恢复
```

这张表里真正重要的是：

```text
pending_confirmation_json
status
runtime_state_json
```

`pending_confirmation_json` 是后端恢复判断的真源，不是前端 `pending_action`。

### 4.12.3 resume 前先校验业务 checkpoint

`_ensure_resume_checkpoint(...)` 里先调用：

```text
checkpoint_manager.validate_resume(...)
```

它会查 `agent_runtime_checkpoints`，确认：

```text
checkpoint 存在
status = waiting_confirmation
pending_confirmation_json 不为空
step_id 匹配
tool_name 匹配
pending_action_id 匹配
```

如果不满足，就返回明确的恢复失败，而不是继续调用 `Command(resume=...)`。

这一步解决的是：

```text
旧按钮不能恢复新任务
跨 session 不能恢复
已经完成/取消/失败的现场不能恢复
pending_action_id 不一致不能恢复
```

### 4.12.4 再校验 LangGraph checkpoint 是否存在

业务 checkpoint 通过后，还会调用：

```text
graph.get_state(config={"configurable": {"thread_id": session_id}})
```

确认 LangGraph 自己也有可恢复现场。

原因是：

```text
业务 checkpoint 说明“业务上允许恢复”
LangGraph checkpoint 说明“技术上真的能回到 interrupt 位置”
```

两者必须都存在。

如果业务 checkpoint 有，但 LangGraph checkpoint 丢了，系统也不能硬 resume，否则就会变成“伪恢复”：看起来用户点了确认，但图其实已经没有原执行栈。

### 4.12.5 原子消费 pending_confirmation，防重复点击

两层 checkpoint 都校验通过后，真正 resume 前会调用：

```text
checkpoint_manager.consume_pending_confirmation(...)
```

底层会执行类似这样的数据库条件更新：

```text
UPDATE agent_runtime_checkpoints
SET status = 'running',
    pending_confirmation_json = '',
    runtime_state_json = cleaned_runtime_state,
    expires_at = NULL
WHERE user_id = ?
  AND session_id = ?
  AND thread_id = ?
  AND status = 'waiting_confirmation'
  AND pending_confirmation_json <> ''
```

这一步是关键 bug 修复点。

它的语义是：

```text
只有第一个确认请求能把 waiting_confirmation 抢占成 running。
后续重复点击进来时，WHERE 条件已经不满足，rowcount = 0，就会被拒绝恢复。
```

同时它会清理 `runtime_state_json` 里的 pending_confirmation，并把 approve 的 step 记入：

```text
approved_step_ids
```

这样后面即使 LangGraph 流式恢复过程中短暂吐出旧 waiting 快照，数据库也能识别这是“已消费确认的旧快照回放”，不会把状态重新写回 waiting。

这里的“旧 waiting 快照回放”是之前容易出 bug 的核心。

一次确认暂停时，系统会先把业务状态写成：

```text
status = waiting_confirmation
pending_confirmation = 当前确认请求
step_status[当前 step] = waiting_confirmation
```

这个状态会同时存在两个地方：

```text
LangGraph checkpoint：
保存 interrupt 当时的图执行现场。

agent_runtime_checkpoints：
保存业务上可恢复的 pending_confirmation。
```

用户点击确认后，后端会先抢占业务 checkpoint：

```text
waiting_confirmation -> running
pending_confirmation -> 清空
approved_step_ids -> 追加被批准的 step
```

但 LangGraph 的原始 checkpoint 仍然是 interrupt 那一刻的快照。恢复时，LangGraph 会从保存的现场继续执行，并且在某些流式更新或节点重入过程中，可能先暴露出 interrupt 前保存的旧状态。这个旧状态里仍然带着：

```text
pending_confirmation
turn_status = waiting_confirmation
step_status[step_id] = waiting_confirmation
```

如果持久化逻辑不加判断，看到这份旧状态后又执行一次 `persist_state`，就会把数据库里已经消费后的状态覆盖回：

```text
status = waiting_confirmation
pending_confirmation = 旧确认请求
```

最终表现就是：

```text
用户明明已经 approve
前端又看到同一个确认卡片
再次 resume 又回到 waiting_confirmation
甚至可能重复触发建索引这类副作用步骤
```

所以项目里做了两层保护：

```text
1. consume_pending_confirmation 先用数据库条件更新原子抢占确认。
2. 后续 persist_state 如果发现 incoming waiting 快照对应的 step 已经在 approved_step_ids 里，就认为它是旧 checkpoint 回放，不允许它覆盖 running 状态。
```

一句话理解：

> LangGraph checkpoint 保存的是“中断时刻的图现场”，而数据库保存的是“业务上确认是否已经被消费”。resume 过程中旧图现场可能短暂重现，所以必须让数据库业务 checkpoint 拥有最终裁决权。

### 4.12.6 为什么不能只用 LangGraph checkpoint

理论上，LangGraph checkpoint 可以让图从 interrupt 的位置继续执行，但它解决的是 **技术恢复**，不是 **业务确认消费**。

可以这样区分：

```text
LangGraph checkpoint：
我是图的执行现场。能告诉你从哪个节点、哪个中断点恢复。

Agent runtime checkpoint：
我是确认任务的业务锁。能告诉你这个确认是否属于当前用户/会话，是否过期，是否已经被消费，是否匹配当前 pending_action_id。
```

如果只依赖 LangGraph checkpoint，会有几个问题。

第一，防不了重复点击。

```text
用户连续点击两次确认
两个请求都拿同一个 thread_id 去 Command(resume=...)
LangGraph 能恢复执行，但它不天然保证这个业务确认只能被消费一次
```

SQLite 业务 checkpoint 里通过条件更新抢占：

```text
WHERE status = 'waiting_confirmation'
  AND pending_confirmation_json <> ''
```

只有第一个请求能把状态切到 `running` 并清空 `pending_confirmation`，后续请求会因为条件不满足而被拒绝。

第二，防不了旧确认误恢复。

```text
前端页面刷新后可能还保留旧 pending_action
用户可能点到旧确认按钮
或者新一轮任务已经产生了新的 pending_confirmation
```

业务 checkpoint 会校验：

```text
user_id
session_id
thread_id
pending_action_id
step_id
tool_name
status
```

不匹配就不能 resume。

第三，防不了业务过期和终态。

LangGraph 原始 checkpoint 可能还在，但业务上这次确认可能已经：

```text
completed
cancelled
failed
expired
```

这些生命周期状态不应该由前端 pending_action 或 LangGraph 内部快照决定，而应该由业务 checkpoint 统一维护。

第四，LangGraph checkpoint 不适合作为业务状态源。

它更像图引擎内部快照，格式依赖 LangGraph，用来恢复执行可以，但不适合承担：

```text
前端轮询
业务审计
过期清理
并发抢占
确认任务归属校验
防重复副作用执行
```

因此现在的边界是：

```text
LangGraph checkpoint：
负责“技术上能不能回到 interrupt 位置”。

Agent runtime checkpoint：
负责“业务上这次确认能不能被消费，以及只能被消费一次”。
```

面试讲法：

> 只用 LangGraph resume 可以完成“恢复执行”，但不能可靠完成“业务确认只被合法消费一次”。我们额外用 SQLite 存业务 runtime checkpoint，是为了把技术恢复和业务恢复解耦：LangGraph 负责回到中断点，SQLite 负责确认归属、状态、过期、并发抢占和防重复执行。尤其是建索引这种有副作用的动作，必须有数据库级别的消费锁。

### 4.12.7 为什么不能只信 `pending_action`

`pending_action` 是给前端展示的镜像，可能存在于：

```text
agent session memory
前端页面状态
debug 快照
旧响应缓存
```

它的问题是：

```text
可能过期
可能刷新后残留
可能和真实 runtime 不一致
可能被旧请求重复提交
```

所以现在的规则是：

```text
pending_action 只负责展示
pending_confirmation 才负责恢复
agent_runtime_checkpoints 才负责校验和消费
```

面试时可以这样讲：

> 这块我们后来踩过坑。最早如果只依赖 LangGraph 的内存 checkpoint 或前端 pending_action，刷新、服务重启、重复点击确认时都容易出问题。后面加了 SQLite 持久化：一张表存 LangGraph 原始 checkpoint，保证技术上能回到 interrupt；另一张 `agent_runtime_checkpoints` 存业务恢复真源，记录 pending_confirmation、status、runtime_state 和过期时间。resume 前先校验业务 checkpoint，再校验 LangGraph checkpoint，然后通过数据库条件更新原子消费 pending_confirmation，防止同一个确认被重复执行。这样前端展示状态和后端真实恢复状态就解耦了。

### 4.12.8 建索引长任务的 continuation

论文 QA 索引构建还有一个特殊点：它可能不是一个很快完成的同步工具调用，而是长任务。

所以路由里还有：

```text
POST /agent/qa-index-continuations
GET  /agent/qa-index-continuations/active
POST /agent/qa-index-continuations/{job_id}/status
```

这套 continuation 解决的是：

```text
用户确认构建索引后，索引 job 在后台跑；
前端可能刷新或轮询；
索引完成后，还要知道原来要恢复哪次 Agent 问答。
```

它会保存：

```text
job_id
user_id
session_id
arxiv_id
pending_action_id
step_id
tool_name
original_question
resume_payload
pending_action
job_snapshot
status
```

所以完整链路可以理解成两种恢复：

```text
短暂停顿恢复：
interrupt -> pending_confirmation -> resume -> Command(resume)

长任务恢复：
确认建索引 -> 创建 index job + continuation -> job 完成 -> 用 continuation 里的 resume_payload 继续原问题
```

面试时如果讲建索引 bug，要补这句：

> 对于论文索引这种长任务，除了 runtime checkpoint，还额外保存了 QA index continuation。它把 index job 和原 Agent pending_action、session_id、arxiv_id、resume_payload 绑定起来，避免前端刷新或后台任务完成后丢失“索引完成后应该继续回答哪个问题”的上下文。

### 4.13 持久化 runtime checkpoint

图执行结束后，会调用：

```text
_persist_runtime_checkpoint_after_turn(...)
```

它会根据最终状态更新业务 checkpoint。

主要分支：

```text
如果还有 pending_confirmation：
  保存 waiting_confirmation 状态，等待用户下一次 resume

如果执行完成：
  标记 completed

如果用户拒绝确认：
  标记 cancelled

如果执行失败：
  标记 failed，并清理 pending_confirmation
```

这一步的重点是防止旧确认被重复消费。

比如用户已经 approve 了一次，系统会消费并清空 pending_confirmation。后续重复点击同一按钮时，业务 checkpoint 校验会失败，不会重复执行副作用工具。

### 4.14 保存 agent session memory

图执行完后，还会保存 Agent 会话记忆：

```text
_persist_agent_session_memory(final_state, memory_service=memory_service)
```

它会把本轮最终状态写回后端 memory，例如：

```text
本轮 papers
选中的论文
paper_qa_result
pending_action 展示镜像
偏好动作结果
必要的上下文摘要
```

为什么要保存：

```text
下一轮用户可能说“总结第一篇”
下一轮用户可能说“推荐类似的”
前端刷新后仍要看到 pending_action
Agent 需要跨轮延续上下文
```

### 4.15 `_state_to_response`

最后一步是把内部 `AgentState` 转成接口响应 `ArxivSearchResponse`。

因为 `AgentState` 很大，包含：

```text
执行计划
runtime
debug
trace
checkpoint 相关字段
工具输出
中间 observation
```

这些不能全部原样返回给前端，所以 `_state_to_response` 会投影出前端需要的字段：

```text
status
intent
answer
papers
session_id
steps
warnings
next_actions
paper_qa_result
pending_action
preference_action_result
debug
errors
```

面试讲法：

> `AgentState` 是内部运行态，`ArxivSearchResponse` 是对外契约。最后要做一次投影，既保证前端拿到稳定字段，也避免把完整 runtime/checkpoint/工具原始输出直接暴露出去。

### 4.16 这一层的整体面试说法

可以这样讲：

> `run_arxiv_search_agent` 是 Agent 同步对话的总入口。它先把请求规范化，然后创建本轮 run_id 和存储容器，构造 MemoryService 去加载用户长期记忆和 Agent 会话记忆。接着它会确定 session_id，并把 session_id 作为 LangGraph 的 thread_id 构造 graph config。之后它创建业务 runtime checkpoint manager 和带 LangGraph checkpointer 的 Agent 图。
>
> 如果这是普通请求，就用请求、上下文和记忆构造 initial AgentState，从图的 START 节点开始执行。如果这是用户确认后的 resume 请求，就不会重新创建业务状态，而是先校验业务 checkpoint 和 LangGraph checkpoint，然后用 `Command(resume=...)` 回到上次 interrupt 的执行现场继续跑。
>
> 图执行结束后，service 层会根据最终状态更新 runtime checkpoint：如果还在等待确认就保留 waiting 状态，如果完成就标记 completed，如果失败或取消就清理 pending。然后把本轮 Agent 结果保存到 session memory，最后把内部 AgentState 投影成稳定的 ArxivSearchResponse 返回给前端。

这里有两个关键点。

第一，`session_id` 和 LangGraph 的 `thread_id` 对齐：

```text
thread_id = session_id
```

这样同一个会话里的中断、确认、恢复都能回到原来的图执行现场，而不是重新跑一遍 Agent。

第二，`service.py` 会加载两类上下文：

```text
用户长期记忆 / 画像
Agent 会话短期记忆
```

这些上下文会进入 `AgentState.context`，供 intent 识别、规划器、推荐、个性化排序等模块使用。

## 5. AgentState 是图里的共享状态

`AgentState` 可以理解为 LangGraph 里流转的统一状态对象。每个节点读取它的一部分字段，处理后返回新的 `AgentState`。

它里面的字段可以分成三类。

### 5.1 请求上下文

```text
user_id
session_id
message
context
```

这些字段表示用户是谁、在哪个会话、说了什么、前端传了什么上下文。

### 5.2 执行真源

```text
goal
execution_plan
plan_runtime
runtime_state
```

其中最重要的是 `plan_runtime`。

`plan_runtime` 保存当前 Agent 执行现场，包括：

```text
当前 step
每个 step 的状态
工具输出 outputs
最近一次 observation
pending_confirmation
final_answer
错误原因
重规划计数
retry 计数
```

`runtime_state` 是 `plan_runtime` 的可序列化 checkpoint 投影，用于跨请求恢复。

面试时要强调：

> 真正驱动执行的是 `plan_runtime`，不是 `pending_action`、`answer` 这些前端展示字段。

### 5.3 出站展示字段

```text
answer
papers
paper_qa_result
pending_action
preference_action_result
warnings
steps
errors
debug
```

这些字段主要用于最终响应和前端展示。

尤其是 `pending_action`，它只是 `ConfirmationRequest` 的前端展示镜像，不是恢复执行的真源。真实的可恢复状态在：

```text
plan_runtime.pending_confirmation
runtime_state.pending_confirmation
LangGraph checkpoint
AgentRuntimeCheckpointStore
```

## 6. LangGraph 主图结构

主图定义在 `agents/arxiv_search_agent/graph.py`。

核心节点：

```text
parse_search_request
build_goal
build_plan
select_next_step
execute_step
observe_step
route_after_observation
replan
finalize
error_finalize
```

图的主链路：

```text
START
  -> parse_search_request
  -> build_goal
  -> build_plan
  -> select_next_step
  -> execute_step
  -> observe_step
  -> route_after_observation
  -> replan / select_next_step / finalize / error_finalize
  -> END
```

这个图的设计重点是：把以前可能藏在一个大函数里的 planner、executor、observer、replanner 拆成显式节点。这样有几个好处：

- 每一步都能被日志和 SSE 事件观察到。
- 失败发生在哪个阶段更清楚。
- 需要用户确认时可以在图中 interrupt。
- resume 时可以回到原来的图线程继续执行。
- 重规划不再藏在工具调用内部，而是作为独立节点出现。

## 7. parse_search_request：意图识别

`parse_search_request` 是图里的第一个业务节点，职责不是执行搜索或问答，而是把用户自然语言请求收敛成后续节点能消费的结构化状态。

真实代码入口在：

```text
agents/arxiv_search_agent/node/parse_node.py
```

主流程可以按四步讲：

```text
parse_search_request(state)
  -> _prepare_parse_search_request_input
  -> _run_parse_search_request_detectors
  -> _decide_parse_search_request_intent
  -> _finalize_parse_search_request_decision
  -> _write_parse_search_request_state
```

### 7.1 输入预处理：只拿 parse 需要的最小上下文

`_prepare_parse_search_request_input` 会先把外部传入的 state 统一转成 `AgentState`，然后规范化：

```text
message
research_profile
current_state
```

这里不是把整个记忆都交给 LLM，而是只从 `state.context` 里取 `research_profile` 作为可选画像信号。也就是说，parse 阶段的主输入仍然是本轮 `message`，画像只用于帮助理解模糊研究偏好。

面试讲法：

> parse 节点第一步会把输入 state 规整成最小解析上下文，避免后续规则和 LLM 同时处理 dict、Pydantic 对象、空白字符串等多种输入形态。

### 7.2 LLM intent 识别：输出必须是 JSON

`_parse_llm_intent` 会检查 `generation_service` 是否存在，并且是否有 `complete_with_qwen` 方法。

如果没有模型服务，直接返回：

```text
ok = False
reason = "llm service is unavailable"
```

如果模型可用，会用带画像的 prompt 调用：

```text
complete_with_qwen(..., task_type="intent_recognition")
```

然后用 `_extract_json_block` 抽取 JSON，再 `json.loads`。如果模型输出不是合法 JSON dict，也会被判为不可用。

支持的 intent 包括：

```text
arxiv_search        搜索论文
paper_summary       总结论文
paper_detail        查看论文细节
paper_qa            对论文问答
recommendation      推荐论文
preference_action   喜欢/不喜欢/取消偏好
unclear             信息不足，需要澄清
unsupported         不支持
```

LLM 结果会经过 `_normalize_llm_intent_payload` 归一化：

```text
intent 不在白名单 -> unsupported
confidence 转 float，失败则 None
只有 arxiv_search 才解析 search_spec
missing_info / warnings / next_actions 做列表清洗
```

关键点是：非搜索 intent 不会在 parse 阶段生成 `search_spec`。比如 `paper_summary`、`paper_qa` 只保留 intent 和解释信息，真正解析论文目标放到后续 `resolve_paper`。

### 7.3 规则检测：始终执行，但主要是兜底和 debug 基线

`_run_parse_search_request_detectors` 里有一个容易误解的点：它不是“LLM 和规则投票”，而是无论 LLM 是否成功，都会执行规则检测：

```text
llm_result = LLM 解析结果，可能为空
rule_result = _build_rule_decision(message)
```

规则侧 `_build_rule_decision` 的分支是：

```text
1. saved-paper container 请求
   -> unsupported

2. 非搜索类硬规则命中
   -> paper_summary / paper_detail / paper_qa / recommendation / preference_action

3. 尝试从规则抽取 search_spec
   -> 抽不出：
      如果像搜索请求 -> unclear
      否则 -> unsupported

4. 抽出 search_spec 后做 rule enrichment
   -> enrichment 失败 -> unclear
   -> 成功 -> arxiv_search
```

其中非搜索类规则识别顺序是：

```text
paper_summary
paper_detail
paper_qa
recommendation
preference_action
```

这个顺序有意义。比如“总结这篇论文”不能被误判成普通搜索；“我不喜欢这篇”也不能被误判成 QA。

### 7.4 最终决策：当前策略是 LLM 主判，规则兜底

`_decide_parse_search_request_intent` 里的注释写得很明确：当前策略收紧为“LLM 主判，规则只在 LLM 不可用时兜底”。

真实分支是：

```text
1. 如果是 saved-paper container 请求
   -> 直接 unsupported

2. 如果 llm_result is None
   -> 使用 rule_result fallback

3. 如果 LLM 判定 arxiv_search
   -> 必须有合法 search_spec
   -> 没有 search_spec 则降级 unclear
   -> 有 search_spec 则做 cleaned_topic 后处理
   -> 再做 rule enrichment
   -> enrichment 失败也降级 unclear

4. 如果 LLM 判定其他 intent
   -> 直接使用 LLM intent
   -> 补齐 plan / next_actions / warnings
```

这说明规则不是没用。它的作用是：

```text
LLM 不可用时兜底
记录 rule_intent 供 debug 对比
给搜索 spec 做本地清洗和 enrichment
对 saved-paper 这类明确不支持请求做强规则拦截
```

面试可以这样讲：

> parse 阶段不是让 LLM 一句话决定所有事情。系统会先尝试 LLM 识别 intent，但规则层始终运行，作为 fallback、debug baseline 和搜索参数清洗层。对于搜索请求，LLM 不能只说“这是搜索”，还必须产出可校验的 `search_spec`，否则会被降级为 `unclear`，防止后续工具拿不到参数。

### 7.5 最终降级保护：搜索 intent 不能没有 search_spec

`_finalize_parse_search_request_decision` 会再做最后一道保护：

```text
如果 intent == arxiv_search 但 search_spec is None
  -> intent 降级为 unclear
```

原因很简单：搜索链路后面一定要构造 arXiv 查询参数。如果 parse 只输出“我要搜索”，但没有 query、title_query、category、sort 等结构化信息，后续 `build_arxiv_search_spec` 就没有可靠输入。

### 7.6 写回 AgentState

`_write_parse_search_request_state` 会把最终结果写回：

```text
state.intent
state.intent_source
state.fallback_reason
state.llm_confidence
state.search_spec
state.plan
state.warnings
state.next_actions
state.debug
```

同时它会清空上一轮搜索残留：

```text
search_retry_count = 0
fallback_specs = []
tool_name = None
tool_args = {}
tool_result = None
tool_calls = []
papers = []
answer = None
errors = []
preference_action_result = None
```

这个清理很重要。因为 parse 是新一轮图执行的入口，不能让上一轮 `papers`、`tool_result`、`errors` 污染本轮规划。

### 7.7 面试版总结

可以这样回答：

> `parse_search_request` 的职责是把自然语言请求变成结构化 intent 和必要参数。它会同时运行 LLM 解析和规则解析，但当前主策略是 LLM 可用时以 LLM 为主，规则作为兜底、debug 基线和搜索参数清洗层。对于 `arxiv_search`，系统要求必须生成合法 `search_spec`，否则会降级成 `unclear`；对于 `paper_summary`、`paper_detail`、`paper_qa` 这类请求，parse 阶段只识别阅读类 intent，不直接解析论文内容，后面会统一进入 paper QA 目标，由 planner 安排 `resolve_paper` 和 PaperQA 工具链。

## 8. build_goal：把 intent 变成 Goal

`build_goal` 对应图里的 `build_goal_node`，核心调用是 `GoalBuilder.from_state`。

它不是简单把 `state.intent` 复制到 `state.goal`，而是把 parse 阶段得到的“这句话属于什么意图”整理成后续 planner 能稳定消费的“本轮业务目标”。

这一节要抓住一句话：

> parse 负责识别“用户这句话像什么请求”，build_goal 负责定义“系统这一轮到底要完成什么目标”。

真实代码入口在：

```text
agents/arxiv_search_agent/graph.py
  -> build_goal_node

agents/arxiv_search_agent/planner.py
  -> GoalBuilder.from_state
```

整体流程是：

```text
build_goal_node(state)
  -> _coerce_state(state)
  -> deep copy 当前 AgentState
  -> GoalBuilder.from_state(next_state)
  -> next_state.goal = goal
  -> debug["goal"] = goal.model_dump()
  -> build_research_task_profile(goal, state, generation_service)
  -> next_state.research_task_profile = profile
  -> debug["research_task_profile"] = research_task_profile_debug(profile)
  -> 追加 AgentStep(step="build_goal")
  -> 返回 next_state
```

所以第 8 节实际有两层：

```text
第一层：GoalBuilder.from_state
  把 intent/context/search_spec 变成 Goal

第二层：build_goal_node
  把 Goal 写回 AgentState，并补一个 research_task_profile 语义层
```

这一步仍然不生成工具步骤。工具步骤要到下一节 `build_plan` 才生成。

Goal 的核心字段是：

```text
goal_id
goal_type
user_request
intent
constraints
success_criteria
context_refs
risk_level
task_scope
```

### 8.1 build_goal 的输入是什么

进入 `build_goal` 时，`parse_search_request` 已经写好了这些字段：

```text
state.intent
state.intent_source
state.llm_confidence
state.message
state.search_spec
state.context
state.pending_action
state.paper_qa_result
state.debug
```

也就是说，build_goal 不再重新识别 intent。它默认 parse 阶段已经给出本轮分类，然后做“目标建模”。

比如用户说：

```text
帮我总结一下这篇论文
```

parse 之后可能是：

```text
state.intent = paper_summary
state.message = 帮我总结一下这篇论文
state.context.selected_paper = {...}
```

build_goal 看到的就是这个状态。它不会问“这是不是总结请求”，而是把这个 intent 转成后续执行目标。

### 8.2 先归一化 intent

`GoalBuilder.from_state` 第一步是：

```text
intent = _normalize_intent(state.intent)
```

如果 intent 不在支持集合里，会统一降级成：

```text
unsupported
```

这是一个保护分支。因为历史调用、测试、前端传参或者 LLM 异常都有可能产生未知 intent，build_goal 不能让未知字符串继续进入 planner，否则后面工具选择会不可控。

支持的目标类型集合包括：

```text
arxiv_search
paper_detail
paper_summary
paper_qa
recommendation
preference_action
unclear
unsupported
```

面试讲法：

> build_goal 不做新的意图识别，而是消费 parse 结果。第一步会把 intent 归一化到系统支持的目标集合里，未知 intent 直接降级为 unsupported，避免后续 planner 因为未知目标暴露错误工具。

### 8.3 读取 context 和 message

接着它会拿：

```text
context = _get_context_mapping(state)
message = _normalize_text(state.message) or ""
```

这里的 `context` 是前面 service 合并过的上下文，可能包含：

```text
selected_paper
last_papers
user_memory_summary
research_profile
pending_action
paper_qa_result
前端传入的其他字段
```

GoalBuilder 只会从里面提取目标建模需要的 hint，不会在这里真正解析论文、检索证据或读数据库。

比如 `_get_selected_paper_hint(context)` 只取：

```text
selected_paper.title
selected_paper.arxiv_id
selected_paper.paper_id
selected_paper_title
selected_paper_id
```

它只是为了在 Goal.constraints 里记录：

```text
target_paper=<hint>
```

最终论文到底是哪篇，仍然交给后续 `resolve_paper` 工具。

### 8.4 计算 risk_level

然后 build_goal 根据 intent 设置风险等级：

```text
preference_action
  -> high

paper_summary / paper_detail / paper_qa / recommendation
  -> medium

其他
  -> low
```

这不是安全校验的最终门禁，只是给 planner/debug 一个目标级风险提示。

为什么偏好动作是 high？

因为 `preference_action` 最终可能写入用户长期偏好，属于持久化写操作。它不能像普通搜索一样直接执行，必须先解析目标、校验写入结果，并受工具 contract 的副作用策略约束。

为什么论文 QA 和推荐是 medium？

因为它们通常不会直接写数据库，但会调用检索、RAG、推荐等复杂业务服务，结果质量需要 evidence / validation / observer 来兜底。

为什么普通搜索是 low？

因为搜索主要是外部查询和展示，不做长期写入。真正的外部调用风险还会在工具 contract 里通过 `side_effect_level` 管。

### 8.5 计算 constraints

`constraints` 是 Goal 里最容易误解的字段，因为它看起来像参数，但在这个项目里它主要是“目标约束摘要”，不是主执行链路里的工具入参。

先把结论说死：

```text
真正用于执行搜索的参数：
  state.search_spec
  normalize_request 的输出
  build_arxiv_search_spec 的输出

Goal.constraints：
  给 Goal / planner / debug 看的目标约束摘要
  正常情况下不应该直接绑定给 build_arxiv_search_spec 当 normalized_request
```

你可以把它理解成：

```text
state.search_spec 是机器可执行参数
goal.constraints 是人和 planner 可读的目标摘要
```

代码里对应的是：

```python
constraints = _build_search_constraints(state) if intent == "arxiv_search" else []
if intent in {"paper_summary", "paper_detail", "paper_qa"}:
    constraints = _dedupe_strings([
        f"target_paper={_get_selected_paper_hint(context)}",
        "use existing QA index when available",
    ])
if intent == "unclear":
    constraints = ["do not execute business tools before clarification"]
if intent == "unsupported":
    constraints = ["do not invoke unsupported business tools"]
```

也就是说，它不是一路累加，而是按 intent 分支直接决定这一轮 Goal 的约束摘要。

#### 8.5.1 搜索场景：constraints 从 search_spec 摘要出来

如果是 `arxiv_search`，parse 阶段前面已经生成了：

```text
state.search_spec
```

比如用户说：

```text
帮我找最近 30 天 RAG query rewrite 相关论文，最多 5 篇
```

parse 后可能得到：

```text
state.intent = arxiv_search
state.search_spec = {
  query: "RAG query rewrite",
  submitted_days_ago: 30,
  max_results: 5,
  sort_by: "submittedDate",
  sort_order: "descending"
}
```

`GoalBuilder.from_state` 里调用：

```text
_build_search_constraints(state)
```

只是把这个结构化对象压成字符串摘要：


```text
constraints = [
  "query=RAG query rewrite",
  "submitted_days_ago=30",
  "max_results=5",
  "sort_by=submittedDate:descending"
]
```

所以搜索场景里，真实链路是：

```text
parse_search_request
  -> 生成 state.search_spec

build_goal
  -> 从 state.search_spec 摘要出 goal.constraints

build_plan
  -> 生成 normalize_request -> build_arxiv_search_spec -> search_arxiv

execute_step
  -> 工具输入优先来自 state.search_spec 或上一步 output
```

关键点：

```text
goal.constraints 不是 search_arxiv 的直接入参
goal.constraints 只是把 search_spec 摘要到 Goal 里，方便解释“这个目标带了哪些搜索限制”
```

为什么还要放一份摘要？

因为 Goal 是后续 planner/debug 能看到的目标对象。你看一眼 Goal 就能知道：

```text
这轮搜索主题是什么
有没有时间限制
最多要几篇
排序方式是什么
```

但真正执行时，仍然应该走结构化 `search_spec`。

#### 8.5.2 论文阅读场景：constraints 只记录目标 hint 和索引偏好

如果 intent 是：

```text
paper_summary / paper_detail / paper_qa
```

constraints 会变成：

```text
target_paper=<selected_paper hint>
use existing QA index when available
```

比如用户说：

```text
总结这篇论文
```

context 里有：

```text
selected_paper = {
  title: "Retrieval-Augmented Generation for ...",
  arxiv_id: "2401.12345"
}
```

Goal 里可能得到：

```text
goal.intent = paper_summary
goal.goal_type = paper_qa
goal.constraints = [
  "target_paper=Retrieval-Augmented Generation for ...",
  "use existing QA index when available"
]
```

这里的 `target_paper=...` 只是 hint。它的意思是：

```text
我这轮目标大概率围绕这个 selected_paper
```

但它不是最终 `paper_ref`。

真正权威的目标解析仍然在后面：

```text
resolve_paper
  -> check_paper_index
  -> answer_paper_question
```

为什么不能直接把 `target_paper` 当最终论文？

因为用户可能说：

```text
总结第一篇
继续刚才那个
总结这篇
```

这些都需要结合：

```text
selected_paper
last_papers
message 里的 arXiv ID
历史 paper_ref
```

做统一解析。这个职责属于 `resolve_paper`，不是 Goal。

`use existing QA index when available` 也不是跳过检查。它只是告诉 planner：

```text
如果已有 QA index，优先复用
如果没有，仍然要 check_paper_index 后由 observer/replanner 触发建索引确认
```

所以真实执行不会变成：

```text
看到 use existing QA index -> 直接 answer
```

而是仍然：

```text
resolve_paper
  -> check_paper_index
  -> 如果 ready，再 answer
  -> 如果 missing/stale/failed，再进入确认/建索引/replan
```

#### 8.5.3 unclear / unsupported：constraints 是安全边界提示

如果是 `unclear`：

```text
do not execute business tools before clarification
```

如果是 `unsupported`：

```text
do not invoke unsupported business tools
```

这两条是安全边界：信息不清楚或不支持时，不允许 planner 暴露搜索、QA、推荐、写入等业务工具。

比如用户只说：

```text
帮我找几篇
```

parse 可能得到：

```text
intent = unclear
```

build_goal 生成：

```text
goal.goal_type = unclear
goal.constraints = [
  "do not execute business tools before clarification"
]
```

后面 ToolCandidateSelector 看到 `goal_type=unclear`，只会开放澄清和 fallback 工具，不会开放 `search_arxiv`、`answer_paper_question`、`update_preference_store` 这些业务工具。

所以这个 constraints 不是 executor 逐字执行的命令，而是 Goal 层的安全语义；真正限制工具暴露的是后面 `goal_type=unclear/unsupported` 对 candidate tools 的筛选。

#### 8.5.4 为什么代码里又有 constraints 解析兜底

你可能会看到 `tool_adapters/search.py` 里有这段逻辑：

```text
把 LLM planner 偶尔传来的 goal.constraints 列表恢复成搜索字段
```

这容易让人误会：

```text
是不是 constraints 本来就是工具参数？
```

答案是：不是。

这是历史兼容 / 容错逻辑。因为 LLM planner 偶尔会把：

```text
goal.constraints
```

错误绑定到：

```text
build_arxiv_search_spec.normalized_request
```

所以 search adapter 做了一个兜底：

```text
如果你真的传了 ["query=...", "max_results=..."]
我尽量把它恢复成 dict，避免执行期直接炸
```

但主流程其实在 `tool_aware_planner.py` 里明确禁止这种绑定：

```text
arxiv_search build_arxiv_search_spec.normalized_request
must come from normalize_request step_output
cannot bind goal.constraints
```

这说明设计上的正确链路是：

```text
state.search_spec
  -> normalize_request
  -> build_arxiv_search_spec
  -> search_arxiv
```

而不是：

```text
goal.constraints
  -> build_arxiv_search_spec
```

#### 8.5.5 面试讲法

可以这样回答：

> `Goal.constraints` 不是工具执行参数，而是 Goal 层的约束摘要。搜索场景下，它是从 parse 阶段已经生成的 `state.search_spec` 摘要出来的，比如 query、时间范围、max_results、排序方式；真正执行搜索时仍然走 `state.search_spec -> normalize_request -> build_arxiv_search_spec -> search_arxiv`。论文 QA 场景下，constraints 只记录目标论文 hint 和“已有 QA index 优先复用”的偏好，不能替代 `resolve_paper` 和 `check_paper_index`。unclear/unsupported 场景下，constraints 是安全语义提示，真正限制工具暴露的是后续 planner 按 `goal_type` 筛 candidate tools。项目里 search adapter 对 constraints 有解析兜底，那是为了兼容 LLM planner 错绑字段，不是主设计路径。

### 8.6 计算 success_criteria

`success_criteria` 是“这轮任务怎样才算完成”的标准。它给 planner 和 debug 一个解释框架。

`arxiv_search` 的成功标准是：

```text
Extract a usable search target from the request.
Return relevant arXiv papers or explain why no suitable results were found.
Apply personalization when profile or memory context is available.
Generate a concise response with next actions when helpful.
```

`paper_summary / paper_detail / paper_qa` 的成功标准是：

```text
Resolve the requested paper.
Retrieve relevant paper evidence.
Produce an answer grounded in retrieved evidence.
```

`recommendation` 的成功标准是：

```text
Read user profile and candidate paper context.
Generate relevant recommendations.
Explain the recommendation rationale.
```

`preference_action` 的成功标准是：

```text
Resolve the preference target.
Persist the preference update.
Return a clear user-facing outcome.
```

`unclear` 和 `unsupported` 则分别要求澄清或安全 fallback。

这块面试不要讲成“代码会逐条检查这些英文句子”。它更像目标层 metadata：

```text
planner 可以把它放进 prompt / debug
trace 可以解释为什么生成这些 step
后续排查时可以看到目标完成标准
```

举例：

如果 goal 是 `paper_qa`，成功标准里明确有：

```text
Resolve the requested paper.
Retrieve relevant paper evidence.
Produce an answer grounded in retrieved evidence.
```

所以后面 planner 生成：

```text
resolve_paper
check_paper_index
answer_paper_question
assess_paper_qa_quality
```

就是合理的。因为这些 step 对应了“解析论文、检索证据、基于证据回答、质量检查”。

### 8.7 计算 goal_type：把用户表达收敛成执行目标

这是 build_goal 最核心的分支：

```text
goal_type = "paper_qa" if intent in {"paper_summary", "paper_detail", "paper_qa"} else intent
```

为什么要这么做？

因为 `intent` 更贴近用户表达，`goal_type` 更贴近系统执行能力。

比如：

```text
用户说：总结这篇论文
intent = paper_summary
goal_type = paper_qa

用户说：这篇论文用了什么数据集？
intent = paper_qa
goal_type = paper_qa

用户说：展示一下这篇论文细节
intent = paper_detail
goal_type = paper_qa
```

它们的执行目标都是：

```text
围绕单篇论文解析目标、检查索引、检索证据、生成回答
```

但是原始 intent 不能丢，因为后面 `RuleBasedToolAwarePlanBuilder._build_paper_qa` 会用：

```text
_paper_qa_draft_answer_shape(goal.intent or state.intent)
```

决定具体 answer step：

```text
paper_summary
  -> step_id = summarize_paper
  -> action_type = summarize
  -> qa_mode = summary

paper_detail
  -> step_id = inspect_paper_detail
  -> action_type = inspect_detail
  -> qa_mode = detail

paper_qa
  -> step_id = answer_paper_question
  -> action_type = answer
  -> qa_mode = qa
```

这就是为什么 Goal 同时保留：

```text
goal_type
intent
```

面试讲法：

> `goal_type` 是系统执行目标，`intent` 是用户表达类型。summary/detail/qa 都映射到 `paper_qa` goal，因为底层工具链一致；但原始 intent 会保留，后面决定 answer 工具用 summary、detail 还是 qa 模式。

### 8.8 生成 Goal 对象

最后 `GoalBuilder.from_state` 返回：

```text
Goal(
  goal_id=f"{goal_type}:{timestamp}",
  goal_type=goal_type,
  user_request=message,
  intent=intent,
  constraints=deduped_constraints,
  success_criteria=deduped_success_criteria,
  context_refs=_build_context_refs(state),
  risk_level=risk_level,
  user_goal=message,
  task_scope=goal_type,
)
```

每个字段可以这样理解：

```text
goal_id
  本轮目标 ID，包含 goal_type 和时间戳，用于 trace/debug。

goal_type
  后续 planner 的主分流依据，比如 arxiv_search / paper_qa / recommendation。

user_request
  原始用户请求，用于 prompt、debug、fallback。

intent
  parse 阶段识别出的原始意图，用于保留用户表达差异。

constraints
  目标约束摘要，不是最终工具参数。

success_criteria
  这轮任务怎样算完成的语义标准。

context_refs
  本轮目标可能依赖的上下文引用。

risk_level
  目标级风险提示。

user_goal
  兼容旧响应读取，内容等于 message。

task_scope
  兼容字段，当前等于 goal_type。
```

### 8.9 context_refs：记录本轮目标依赖了哪些上下文

`GoalBuilder` 会调用 `_build_context_refs` 收集上下文引用，比如：

```text
selected_paper
user_memory_summary
pending_action
paper_qa_result
context 里的其他 key
```

具体逻辑是：

```text
如果 context 里能提取 selected_paper hint
  -> refs 加 selected_paper

如果 context 里有 user_memory_summary 或 memory_summary
  -> refs 加 user_memory_summary

如果 state.pending_action 是 Mapping
  -> refs 加 pending_action

如果 state.paper_qa_result 是 Mapping
  -> refs 加 paper_qa_result

最后把 context 里的所有 key 也加进去
```

它的作用不是让 Goal 自己解析上下文，而是把“本轮目标依赖过哪些上下文材料”记录下来。

比如用户说：

```text
总结这篇论文
```

如果 context 里有：

```text
selected_paper
last_papers
research_profile
```

那么 Goal 可能记录：

```text
context_refs = [
  "selected_paper",
  "last_papers",
  "research_profile"
]
```

这对后面有两个价值：

```text
planner debug 能解释为什么它认为有目标论文
排查问题时能看到这轮到底消费了哪些上下文字段
```

### 8.10 build_goal_node 还会构建 ResearchTaskProfile

图节点 `build_goal_node` 在写入 Goal 后，还会调用：

```text
build_research_task_profile(goal=goal, state=next_state, generation_service=generation_service)
```

这是 Goal 和 Plan 之间的“科研任务语义层”。

它会进一步判断这轮任务更像什么科研任务：

```text
direction_exploration          方向探索
multi_paper_comparison         多论文比较
single_paper_deep_read         单篇论文深读
reading_planning               阅读路线规划
research_gap_analysis          研究空白分析
personalized_recommendation    个性化推荐
```

它的分类路线是三段式：

```text
1. 规则高置信判断
2. 规则不够时调用 LLM 语义分类
3. 本地仲裁层检查是否真的可执行
```

比如：

```text
goal_type = paper_qa
  -> 通常会得到 single_paper_deep_read

goal_type = recommendation
  -> 通常会得到 personalized_recommendation

arxiv_search + 用户说“比较这些论文”
  -> 可能得到 multi_paper_comparison

arxiv_search + 用户说“有哪些研究空白”
  -> 可能得到 research_gap_analysis
```

这个 Profile 会包含：

```text
research_task_type
task_object
constraints
intermediate_artifacts
evidence_requirements
execution_readiness
needs_clarification
classification_trace
```

它失败不会阻断主流程。代码里明确把它包在 try/except 里：

```text
语义层失败 -> debug["research_task_profile_error"] = ...
继续用原 Goal 进入 build_plan
```

面试讲法：

> Goal 是业务目标层，ResearchTaskProfile 是科研任务语义层。它不替代 Goal，也不直接执行工具，而是补充“这轮研究任务需要什么中间产物和证据”。它失败时不会影响主链路，最多影响 planner 的增强信息。

### 8.11 build_goal 输出后谁会用

`build_goal_node` 输出的 state 会进入：

```text
build_plan_node
```

下一步代码会做：

```text
goal = next_state.goal or GoalBuilder.from_state(next_state)
goal, plan, planning_debug = build_executable_plan_for_goal(goal, next_state, ...)
runtime = build_plan_runtime(next_state, goal=goal, plan=plan, ...)
```

也就是说 Goal 后面至少被三处消费：

```text
1. build_executable_plan_for_goal
   用 goal_type 决定 planner 分支和候选工具范围。

2. build_planner_context
   把 goal、intent、research_task_profile、context_refs 等整理进 PlannerContext。

3. build_plan_runtime
   把 goal 放入 PlanRuntime，后续 execute/observe/replan/finalize 都能看到同一个目标。
```

举个完整例子。

用户说：

```text
帮我总结一下这篇论文的核心贡献
```

parse 后：

```text
state.intent = paper_summary
state.message = 帮我总结一下这篇论文的核心贡献
state.context.selected_paper = {"title": "...", "arxiv_id": "..."}
```

build_goal 后：

```text
goal.intent = paper_summary
goal.goal_type = paper_qa
goal.risk_level = medium
goal.constraints = [
  "target_paper=<title or arxiv_id>",
  "use existing QA index when available"
]
goal.success_criteria = [
  "Resolve the requested paper.",
  "Retrieve relevant paper evidence.",
  "Produce an answer grounded in retrieved evidence."
]
goal.context_refs = [
  "selected_paper",
  ...
]
```

research_task_profile 可能是：

```text
research_task_type = single_paper_deep_read
task_object.object_type = paper
evidence_requirements = metadata / abstract / method_chunk / experiment_chunk / table_evidence
```

build_plan 再根据这些信息生成：

```text
resolve_paper
  -> check_paper_index
  -> summarize_paper
  -> assess_paper_qa_quality
```

所以这条链路是：

```text
自然语言 message
  -> parse 得到 intent=paper_summary
  -> build_goal 得到 goal_type=paper_qa
  -> research_task_profile 得到 single_paper_deep_read
  -> build_plan 得到具体工具步骤
```

### 8.12 面试版总结

可以这样回答：

> `build_goal` 是 parse 和 planner 之间的目标建模层。parse 只告诉系统“用户这句话是什么 intent”，而 build_goal 会把 intent、message、search_spec 和上下文整理成稳定的 `Goal`。它会做几件事：先把 intent 归一化，未知 intent 降级 unsupported；再根据 intent 设置风险等级；然后把搜索条件、目标论文 hint、澄清/unsupported 安全边界整理成 constraints；接着写入 success_criteria 和 context_refs；最后把用户表达型 intent 收敛成执行型 goal_type，比如 `paper_summary`、`paper_detail`、`paper_qa` 都统一成 `paper_qa` goal，但保留原始 intent 供后面决定 summary/detail/qa 模式。

> 在图节点里，Goal 写回 state 后还会尝试构建 `ResearchTaskProfile`，进一步判断是单篇深读、多论文比较、方向探索、研究空白分析还是个性化推荐。这个 profile 是增强语义层，失败不会阻断主流程。下一步 `build_plan` 会消费 Goal 和 Profile，才真正生成 `ExecutablePlan` 和工具步骤。

## 9. build_plan：生成 ExecutablePlan

`build_plan` 对应 `build_executable_plan_for_goal`。它的职责是把 `Goal` 转成可执行的 `ExecutablePlan`，但仍然不执行工具。

可以把它理解成：

```text
Goal: 用户要达成什么
PlanDraft: planner 产出的不可信草稿
ExecutablePlan: 经过候选工具、schema、依赖、风险和业务语义校验后的执行计划
```

### 9.0 先看总流程：build_plan 到底按什么顺序跑

图节点里真正调用的是：

```text
build_plan_node(state)
  -> current_state = _coerce_state(state)
  -> next_state = current_state.model_copy(deep=True)
  -> goal = next_state.goal or GoalBuilder.from_state(next_state)
  -> build_executable_plan_for_goal(goal, next_state, tool_registry)
  -> build_plan_runtime(next_state, goal, plan, turn_status="success")
  -> next_state.execution_plan = plan
  -> next_state.plan_runtime = runtime
  -> next_state.runtime_state = _runtime_state_from_runtime(...)
  -> debug["planner"] = planning_debug
  -> 追加 AgentStep(step="build_plan")
```

所以 `build_plan_node` 本身做三件事：

```text
1. 拿到上一节 build_goal 生成的 Goal
2. 调 build_executable_plan_for_goal 生成并校验 ExecutablePlan
3. 初始化 PlanRuntime，给后面的 select/execute/observe 使用
```

核心复杂度都在：

```text
build_executable_plan_for_goal(...)
```

它的真实执行顺序可以先背成这条链：

```text
1. 读取 planner 配置，归一化 runtime_flags
2. 如果主 planner 被关闭 -> 直接 legacy fallback
3. 构造 PlannerContext
4. 用 ToolCandidateSelector 筛候选工具
5. 如果候选工具为空 -> fallback
6. 如果开启 LLM draft -> 先尝试 LLM 生成 PlanDraft
7. LLM 草稿通过 converter/validator -> 直接返回 ExecutablePlan
8. LLM 失败但允许 fallback -> 继续走 profile-aware / rule-based
9. 尝试 ProfileAwareResearchTaskPlanBuilder
10. profile-aware 草稿通过校验 -> 返回 ExecutablePlan
11. profile-aware 不产出或失败 -> 走 RuleBasedToolAwarePlanBuilder
12. 规则草稿通过 converter/validator -> 返回 ExecutablePlan
13. 规则 planner 也失败 -> legacy template 或 unsupported fallback
14. 最后统一写 planner_summary / execution_plan debug
```

用一张简化路线图表示：

```text
Goal + AgentState
  -> runtime_flags
  -> PlannerContext
  -> candidate_tools
  -> [LLM PlanDraft 可选]
      -> PlanDraftConverter
      -> PlanValidator
      -> ExecutablePlan
  -> [Profile-aware PlanDraft 可选]
      -> PlanDraftConverter
      -> PlanValidator
      -> ExecutablePlan
  -> [Rule-based PlanDraft]
      -> PlanDraftConverter
      -> PlanValidator
      -> ExecutablePlan
  -> [Legacy fallback]
      -> generate_fallback_response plan
```

这里有一个非常重要的面试点：

```text
LLM / Profile-aware / Rule-based 只是 PlanDraft 的不同来源
真正可执行的计划必须统一经过 PlanDraftConverter + PlanValidator
```

也就是说：

```text
LLM 不是执行者
LLM 不是最终裁判
LLM 只是可能的草稿生成器
```

后面所有工具白名单、输入绑定、依赖拓扑、确认策略、业务顺序，都是本地代码兜底。

### 9.1 运行模式：先读配置决定 planner 路径

`_normalized_planner_runtime_flags` 会先读运行配置，支持四种模式：

```text
rule_only
llm_preferred
llm_only_strict
demo_rule
```

真实含义是：

```text
rule_only
  -> 开启规则 planner
  -> 关闭 LLM draft
  -> 允许 template fallback

llm_preferred
  -> 优先尝试 LLM draft
  -> 失败后可回退规则 planner
  -> 仍可回退 template fallback

llm_only_strict
  -> 尝试 LLM draft
  -> 不允许规则 fallback
  -> 不允许 template fallback
  -> LLM 失败直接抛错

demo_rule
  -> 规则 planner
  -> 关闭 LLM draft
  -> 允许 template fallback
```

这四个模式本质上不是四套 planner，而是控制下面几个开关：

```text
rule_planner_enabled
llm_draft_enabled
rule_fallback_enabled
template_fallback_enabled
strict_llm_failure
```

可以按下面这个判断树理解：

```text
planner_mode = rule_only
  -> llm_draft_enabled = False
  -> 直接跳过 LLM
  -> 后面走 profile-aware / rule-based

planner_mode = llm_preferred
  -> llm_draft_enabled = True
  -> 先尝试 LLMPlanDraftGenerator
  -> LLM 成功就返回
  -> LLM 失败就 fallback 到 profile-aware / rule-based

planner_mode = llm_only_strict
  -> llm_draft_enabled = True
  -> strict_llm_failure = True
  -> LLM 失败直接抛 PlanDraftPlanningError
  -> 不走规则 planner，不走 template fallback

planner_mode = demo_rule
  -> llm_draft_enabled = False
  -> 用规则 planner 展示稳定链路
  -> 失败时允许 template fallback
```

项目默认模式一般可以理解成：

```text
llm_preferred
```

也就是：

```text
能用 LLM 就先让 LLM 出草稿
但 LLM 草稿不可信，必须校验
LLM 不可用或输出不合法，就回到本地规则 planner
```

这一步会形成 `planner_summary`，记录：

```text
requested_path
selected_path
final_path
llm_draft_attempted
llm_draft_valid
rule_fallback_used
template_fallback_used
fallback_reason
```

面试讲法：

> build_plan 不是固定一条链路，它先根据运行配置决定主 planner 是 LLM draft 还是 rule-based。无论走哪条路径，最终都要产出同一种 `ExecutablePlan`，并把路径选择、fallback 原因、候选工具和校验状态写进 debug。

如果面试官追问“那你这个到底是不是基于 LLM”，推荐这样答：

> 它是 LLM-preferred，但不是 LLM-controlled。默认会优先尝试 LLM 生成 PlanDraft；但 LLM 只能在候选工具范围内输出 JSON 草稿，不能直接执行工具。草稿必须通过本地 converter、validator 和 ToolContract 校验。LLM 失败时会回退到规则 planner，所以整体是“LLM 参与规划，本地规则兜底和裁判”的混合式 planner。

### 9.2 planner_context：把状态整理成规划上下文

进入真实规划前，会先构造：

```text
planner_context = build_planner_context(goal, state, tool_registry)
```

它会把后续规划需要的信息整理出来，比如：

```text
raw_user_request
selected_paper
last_papers
paper_qa_result
pending_action
user_memory_summary
research_profile
intermediate_results
reusable_outputs
available_tools
high_risk_tools
```

如果 planner_context 构造失败，代码不会让整个 Agent 崩掉，而是回到 legacy template fallback：

```text
missing_required_context: planner_context_build_failed
```

这个设计说明 planner_context 是增强输入，不是不可降级的基础设施。

### 9.3 ToolCandidateSelector：先收窄工具，再让 planner 规划

工具不是全量暴露给 planner 的。`ToolCandidateSelector` 会按 `goal_type` 和上下文筛候选工具。

基础白名单包括：

```text
arxiv_search:
  normalize_request
  build_arxiv_search_spec
  search_arxiv
  validate_arxiv_results
  rewrite_arxiv_query
  personalize_paper_results
  synthesize_arxiv_response

paper_qa:
  resolve_paper
  check_paper_index
  answer_paper_question
  assess_paper_qa_quality
  request_confirmation
  parse_and_index_paper

recommendation:
  load_user_profile
  load_candidate_papers
  generate_recommendations
  validate_recommendations
  explain_recommendations

preference_action:
  resolve_preference_target
  update_preference_store
  verify_preference_update
  synthesize_preference_response
  analyze_ambiguity
  generate_clarification
  generate_fallback_response
```

然后它会继续看上下文：

```text
paper_qa 没有目标论文
  -> 只开放 resolve_paper / analyze_ambiguity / generate_clarification
  -> 防止 answer_paper_question 在缺 paper_ref 时被规划

paper_qa 的 QA index 状态是 missing/stale/failed
  -> 加入 request_confirmation / parse_and_index_paper 候选

recommendation 没有 candidate papers
  -> 不收窄工具，只记录 candidate_papers_absent_but_loader_can_degrade

preference_action 没有目标论文
  -> 只开放 resolve_preference_target / analyze_ambiguity / generate_clarification
  -> 排除 update_preference_store，防止不明确目标下写入长期偏好

unclear / unsupported
  -> 禁止暴露 search / retrieve / index / recommendation / memory_write 等业务工具
```

这就是“LLM 不乱调工具”的第一道边界：planner 只能看到候选工具，而不是看到整个后端函数集合。

### 9.4 LLMPlanDraftGenerator：LLM 只能产出 JSON 草稿

如果配置开启 LLM planner，会调用 `LLMPlanDraftGenerator.generate`。

这里非常重要：LLM 不直接生成 `ExecutablePlan`，也不能触发 executor。它只能输出 JSON-only 的 `PlanDraft`。

prompt 里会显式告诉模型：

```text
只能选择 candidate_tools 中的工具
禁止创造新工具
每个 step 必须有 why_this_step、depends_on、input_bindings、expected_output
persistent_write 目标不明确时必须澄清
paper_qa 必须先 resolve_paper，再 check_paper_index，再 answer_paper_question
preference_action 写入必须先 resolve，再 update，再 verify，再 answer
unsupported 只能选择 fallback 工具
arxiv_search 的 build_arxiv_search_spec.normalized_request 必须来自 normalize_request 的 step_output
```

LLM 输出后还会经过多层校验：

```text
必须是 JSON object
必须能通过 PlanDraft.model_validate
step 不能为空
step 数不能超过 max_steps
step_id 不能重复
tool 必须在 candidate_tools 中
tool 必须已注册
必须有 why_this_step
output_key 不能重复
depends_on 不能引用后面的 step
必填 input 必须绑定
search_spec 缺失时不能要求 search_spec 输入
persistent_write 必须目标明确且 requires_confirmation
必须有最终 answer/fallback step
paper_qa / preference_action 必须保留关键前置步骤
```

如果失败：

```text
llm_plan_valid = False
fallback_reason = llm_planner_validation_failed: ...
```

然后根据模式：

```text
strict_llm_failure=True
  -> 直接抛错

允许 rule fallback
  -> 回到 profile-aware / rule-based planner

不允许 rule fallback
  -> 进入 template 或 unsupported fallback
```

### 9.5 Profile-aware planner：增强层，失败不能阻断主链路

LLM draft 后面还有一个增强分支：

```text
ProfileAwareResearchTaskPlanBuilder
```

它可以基于用户画像、研究 profile、artifact refinement 做更细的研究任务规划。它如果产出 `profile_draft`，同样要经过：

```text
PlanDraftConverter
PlanValidator
```

如果它失败，代码会把错误写进：

```text
profile_aware_planning
```

然后回到普通规则 planner。也就是说，profile-aware 是增强层，不是主链路的单点故障。

### 9.6 RuleBasedToolAwarePlanBuilder：稳定的规则计划

如果没有开启 LLM，或者 LLM 失败后允许 fallback，就会进入规则型 planner。

规则 planner 会按 `goal_type` 选择 builder：

```text
arxiv_search        -> _build_arxiv_search
paper_qa            -> _build_paper_qa
recommendation      -> _build_recommendation
preference_action   -> _build_preference_action
unclear             -> _build_unclear
unsupported         -> _build_unsupported
```

#### arxiv_search 计划

主链路是：

```text
normalize_request
  -> build_arxiv_search_spec
  -> search_arxiv
  -> validate_arxiv_results
  -> synthesize_arxiv_response
```

如果存在用户记忆或 research_profile，并且候选工具里有 `personalize_paper_results`，会插入：

```text
search_arxiv
  -> personalize_paper_results
  -> synthesize_arxiv_response
```

如果没有画像，个性化步骤会被 skip，原因会写入 debug：

```text
candidate tool missing or no user memory/profile context
```

这里和前面你问的“长期画像有啥用”正好对应：它不是搜索必需条件，只是有画像时才启用个性化 rerank。

#### paper_qa 计划

先判断有没有目标论文：

```text
selected_paper
last_papers / candidate_papers
message 里有 arXiv ID / URL
```

如果没有目标，直接转澄清链路：

```text
analyze_ambiguity
  -> generate_clarification
```

如果有目标，生成：

```text
resolve_paper
  -> check_paper_index
  -> summarize_paper / inspect_paper_detail / answer_paper_question
  -> assess_paper_qa_quality
```

具体 answer step 由原始 intent 决定：

```text
paper_summary:
  step_id = summarize_paper
  qa_mode = summary

paper_detail:
  step_id = inspect_paper_detail
  qa_mode = detail

paper_qa:
  step_id = answer_paper_question
  qa_mode = qa
```

注意：`parse_and_index_paper` 不默认进入主计划。代码里会明确 skip：

```text
高成本索引工具不默认进入主链路，缺索引时由确认机制重规划加入。
```

也就是说，初始 paper QA 计划只负责检查索引。索引缺失后，由 `observe_step` / `replan` 判断需要用户确认，再追加 `request_confirmation` 和 `parse_and_index_paper`。

#### recommendation 计划

推荐链路是：

```text
load_user_profile
  -> load_candidate_papers   可选
  -> generate_recommendations
  -> validate_recommendations
  -> explain_recommendations
```

如果没有 `load_candidate_papers` 工具，规则 planner 不会失败，而是记录 skip，让推荐工具基于画像和消息降级。

#### preference_action 计划

偏好写入链路是高风险链路。它先判断目标是否明确：

```text
selected_paper
candidate_papers
message 里有 arXiv ID / URL
```

如果目标不明确，不生成 `update_preference_store`，而是进入澄清链路。

如果目标明确，链路是：

```text
resolve_preference_target
  -> update_preference_store
  -> verify_preference_update
  -> synthesize_preference_response
```

并且会检查：

```text
update_preference_store.side_effect_level == persistent_write
```

如果这个工具 contract 没标成持久化写，反而会 fallback，因为写入工具的风险边界必须明确。

#### unclear / unsupported 计划

`unclear` 只允许：

```text
analyze_ambiguity
  -> generate_clarification
```

`unsupported` 只允许：

```text
generate_fallback_response
```

这保证了信息不清或不支持时不会误调用业务工具。

### 9.7 PlanDraftConverter：把不可信草稿转成可执行计划

无论 PlanDraft 来自 LLM、profile-aware，还是规则 planner，都要通过：

```text
PlanDraftConverter.convert
PlanValidator().validate
```

`PlanDraftConverter` 会检查：

```text
selected_tools 必须在 allowed_tool_names 中
tool_name 必须是注册工具
step_id 不能为空且不能重复
output_key 不能重复
depends_on 必须引用已存在 step
依赖图不能有环
input_bindings 不能为空
必填 input 必须绑定
risky tool 必须有 risk strategy
只有 ToolContract.requires_confirmation 或 persistent_write 才进入人工确认门
```

最后还有业务顺序校验：

```text
arxiv_search:
  normalize_request
  -> build_arxiv_search_spec
  -> search_arxiv
  -> validate_arxiv_results
  -> synthesize_arxiv_response

paper_qa:
  如果有 answer_paper_question
  必须有 resolve_paper / check_paper_index / assess_paper_qa_quality

preference_action:
  如果有 update_preference_store
  必须有 resolve_preference_target / verify_preference_update / synthesize_preference_response
```

还有一个细节：普通 `external_call` 不一定需要人工确认。确认门由受控 `ToolContract` 决定：

```text
tool.requires_confirmation
或 side_effect_level == persistent_write
```

这样可以防止 LLM 把普通 arXiv 搜索错标成需要确认，导致搜索链路被卡住。

### 9.8 legacy template fallback：最后安全兜底

如果主 planner 关闭、planner_context 构造失败、候选工具为空、LLM/规则 planner 都失败，会进入：

```text
_fallback_to_template_or_unsupported
```

legacy fallback 只生成一个安全回复步骤：

```text
generate_fallback_response
```

它不承载搜索、QA、推荐等业务能力。代码里还写了维护边界：

```text
do_not_extend_for_new_business_capabilities
```

也就是说，新业务能力不能继续往 legacy 模板里塞，要加到 LLM draft 或 RuleBasedToolAwarePlanBuilder。

### 9.9 面试版总结

可以这样回答：

> `build_plan` 会把结构化 `Goal` 转成 `ExecutablePlan`。它不是直接让 LLM 调工具，而是先根据运行模式决定使用 LLM draft 还是规则 planner，然后通过 `ToolCandidateSelector` 按 goal 和上下文收窄候选工具。LLM 即使参与，也只能输出 JSON 形式的 `PlanDraft`，不能直接执行工具。所有草稿都会经过 `PlanDraftConverter` 和 `PlanValidator`，校验工具白名单、输入绑定、依赖拓扑、风险策略、人工确认策略和业务顺序。规则 planner 则给出稳定链路，比如搜索是 normalize、build spec、search、validate、synthesize；论文 QA 是 resolve、check index、answer、assess quality。缺索引这种高成本动作不会默认执行，而是在 observer/replanner 阶段触发确认后再加入。最后如果主 planner 全部失败，才用 legacy fallback 生成安全回复。

## 10. 工具注册表 Tool Registry

核心工具包括以下几类。

### 10.1 搜索链路

```text
normalize_request
build_arxiv_search_spec
search_arxiv
validate_arxiv_results
rewrite_arxiv_query
personalize_paper_results
synthesize_arxiv_response
```

### 10.2 论文 QA 链路

```text
resolve_paper
check_paper_index
request_confirmation
parse_and_index_paper
answer_paper_question
assess_paper_qa_quality
```

### 10.3 推荐链路

```text
load_user_profile
load_candidate_papers
generate_recommendations
validate_recommendations
explain_recommendations
```

### 10.4 偏好写入链路

```text
resolve_preference_target
update_preference_store
verify_preference_update
synthesize_preference_response
```

### 10.5 澄清和兜底链路

```text
analyze_ambiguity
generate_clarification
generate_fallback_response
```

工具 contract 的价值是：

- 限制 planner 可见工具范围。
- 明确工具输入输出 schema。
- 标记工具副作用等级。
- 指定哪些工具需要用户确认。
- 给 observer/replanner 提供失败恢复依据。

## 11. select_next_step：选择下一步

`select_next_step` 只负责选步骤，不执行工具。

选择逻辑大致是：

```text
1. 遍历 plan.steps
2. 找到 status = pending 的 step
3. 检查 depends_on 是否都完成
4. 检查 condition 是否满足
5. 检查 preconditions 是否满足
6. 写入 runtime.current_step_id
```

如果找不到可执行步骤，说明计划执行完了，进入 `finalize`。

如果出现 runtime error，则进入 `error_finalize`。

## 12. execute_step：执行工具

`execute_step` 只负责执行当前 step 的工具调用，不做结果质量判断。

流程：

```text
1. 根据 input_bindings 解析工具输入
2. 如果缺少必需输入，标记 failed
3. 如果是有副作用且已执行过的 step，尝试复用已有输出，避免重复调用
4. 如果工具需要确认，创建 ConfirmationRequest 并 interrupt
5. 否则调用 ToolAdapter
6. ToolAdapter 调用后端工具或服务
7. 对返回结果做标准化 project_tool_result
8. 写入 runtime.last_step_output
```

输入绑定的来源可以是：

```text
state
context
goal
search_spec
step_output
literal
```

所以后续 step 不需要自己猜输入，而是通过 plan 里的绑定关系明确拿上游输出。

面试讲法：

> execute 阶段只回答“这个工具是否被正确调用、输出是什么”。它不判断答案是否可信、不判断检索质量是否足够，这些交给 observer。

## 13. observe_step：观察结果质量

`observe_step` 使用 `Observer` 判断工具结果质量。

这里有一个重要思想：

```text
工具执行成功 != 业务结果可用
```

例子：

- `search_arxiv` 调用成功，但返回空列表，这属于 `empty_result`。
- `answer_paper_question` 有答案，但没有 sources，这属于低质量回答。
- `resolve_paper` 找到多个候选，这不是失败，而是需要用户确认目标论文。
- `check_paper_index` 发现索引缺失，需要触发索引构建确认。

Observer 输出 `ObservationResult`，字段包括：

```text
status
reason
confidence
details
failure_category
severity
recoverable
retryable
requires_user_input
suggested_recovery_types
suggested_action
```

常见 observation status：

```text
success
partial_success
empty_result
low_confidence
need_confirmation
need_clarification
invalid_output
tool_error
```

根据 observation，图会路由到：

```text
success / partial_success -> select_next_step
low_confidence / empty_result -> replan
need_confirmation -> finalize waiting_confirmation
need_clarification -> finalize
tool_error / invalid_output -> replan 或 error_finalize
```

## 14. replan：失败恢复和重规划

当 observer 判断结果不可用或质量不足时，进入 `replan`。

`Replanner` 不是简单重试，它会做几件事：

```text
1. FailureClassifier：把失败归类
2. LLMRecoveryDiagnoser：可选地诊断失败原因
3. RecoveryChooser：选择恢复动作
4. RecoverySafetyGuard：检查恢复动作是否安全
5. PlanPatcher：修改当前计划
```

常见恢复动作：

```text
retry_step
patch_plan
request_confirmation
ask_clarification
fallback_answer
```

例子 1：arXiv 搜索为空。

```text
search_arxiv -> empty_result
  -> replan
  -> 插入 rewrite_arxiv_query
  -> 重新 search_arxiv
```

例子 2：Paper QA 索引不存在。

```text
check_paper_index -> paper_index_missing
  -> replan
  -> request_confirmation
  -> parse_and_index_paper
  -> answer_paper_question
```

例子 3：论文目标不明确。

```text
resolve_paper -> need_confirmation 或 need_clarification
  -> 让用户确认候选论文
  或生成澄清问题
```

Replanner 还有次数限制，避免无限循环：

```text
同一失败原因重规划次数有限
同一步骤重规划次数有限
整轮计划重规划次数有限
```

超过限制后会 fallback，而不是一直重试。

## 15. finalize：汇总最终响应

`finalize` 调用执行器的 `finalize_runtime`，把 runtime 收束成 `AgentTurnResult`。

最终会投影到 `AgentState` 的出站字段：

```text
answer
papers
paper_qa_result
pending_action
preference_action_result
warnings
steps
errors
debug
```

然后 `service.py` 会：

```text
1. 持久化 runtime checkpoint 状态
2. 保存 agent session memory
3. 把 AgentState 转成 ArxivSearchResponse
4. 同步接口直接返回
5. 流式接口发送 final_response 和 stream_end
```

## 16. 用户确认和 resume 机制

这是 Agent 对话里最重要的工程亮点之一。

某些工具不能直接执行，比如：

```text
parse_and_index_paper
```

因为它会触发外部索引构建，属于有副作用的动作。工具 contract 里会标记：

```text
requires_confirmation = True
side_effect_level = external_call
```

当执行到这种 step 时，执行器不会立刻调用工具，而是：

```text
1. 构造 ConfirmationRequest
2. 写入 runtime.pending_confirmation
3. 调用 LangGraph interrupt
4. 返回 pending_action 给前端展示
5. 持久化 LangGraph checkpoint 和业务 runtime checkpoint
```

前端展示确认卡片。用户点击同意后，前端发送 resume 请求：

```text
resume.decision = approve
resume.pending_action_id = ...
或 resume.step_id + resume.tool_name = ...
```

后端收到 resume 后：

```text
1. 用 session_id 找到同一个 thread_id
2. 校验业务 runtime checkpoint 是否存在
3. 校验 LangGraph checkpoint 是否存在
4. 校验 pending_confirmation 是否匹配
5. 消费 pending_confirmation，防止重复点击导致重复执行
6. graph.invoke(Command(resume=resume_payload))
```

恢复后执行器会：

```text
1. 清空 runtime.pending_confirmation
2. 清空 runtime_state.pending_confirmation
3. 清空 plan_runtime.pending_confirmation
4. 把当前 step 放回 pending 或标记 success
5. 继续执行真正的工具
```

面试重点：

> `pending_action` 只是前端展示镜像，不能作为恢复真源。真正的恢复依赖 LangGraph checkpoint、AgentRuntimeCheckpointStore 和 runtime.pending_confirmation。

## 17. 流式接口如何展示过程

`/agent/chat/stream` 会把 LangGraph 的 update stream 转成 SSE。

典型事件顺序：

```text
run_start
step_start(parse_search_request)
step_end(parse_search_request)
step_start(build_goal)
step_end(build_goal)
step_start(build_plan)
step_end(build_plan)
step_start(select_next_step)
step_end(select_next_step)
step_start(execute_step)
tool_call_start
tool_call_end
step_end(execute_step)
step_start(observe_step)
step_end(observe_step)
...
final_response
stream_end
```

如果中间发生确认：

```text
__interrupt__
  -> 转成 waiting_confirmation 状态
  -> 返回 pending_action
  -> stream_end
```

用户确认后再次请求，带上 `resume`，后端继续从原图线程恢复。

## 18. 具体例子：总结一篇论文

用户问：

```text
帮我总结一下这篇论文的核心贡献
```

可能的执行链路：

```text
1. parse_search_request
   intent = paper_summary

2. build_goal
   goal_type = paper_qa
   success_criteria = resolve paper + retrieve evidence + grounded answer

3. build_plan
   生成计划：
   resolve_paper
     -> check_paper_index
     -> answer_paper_question
     -> assess_paper_qa_quality

4. select_next_step
   选择 resolve_paper

5. execute_step
   调用 ResolvePaperAdapter，解析当前用户指的是哪篇论文

6. observe_step
   如果目标唯一，继续
   如果多个候选，进入确认
   如果没有目标，生成澄清

7. select_next_step
   选择 check_paper_index

8. execute_step
   检查这篇论文有没有 QA 索引

9. observe_step
   如果索引存在，继续 answer_paper_question
   如果索引缺失，replan 插入确认建索引流程

10. 如果需要建索引
    request_confirmation / parse_and_index_paper
    用户确认后 resume

11. answer_paper_question
    调用 PaperQAService
    PaperQAService 内部做 RAG 检索、生成、证据校验

12. assess_paper_qa_quality
    判断答案是否有 sources、证据是否足够、是否需要修复

13. finalize
    汇总 answer、sources、paper_qa_result、debug、steps
```

## 19. 面试版回答

如果面试官问：“你这个 Agent 对话流程是怎么实现的？”

可以这样回答：

> 我的 Agent 对话不是简单调用一次大模型，而是用 LangGraph 做了一个显式的状态机。前端请求进入 FastAPI 的 `/agent/chat` 或 `/agent/chat/stream` 后，service 层会规范化请求、补全 session_id、加载用户记忆和会话上下文，然后构造 `AgentState`。这个 `session_id` 同时作为 LangGraph 的 `thread_id`，用于 checkpoint 和 resume。
>
> 图里面的主链路是 `parse_search_request -> build_goal -> build_plan -> select_next_step -> execute_step -> observe_step -> replan/finalize`。首先 parse 节点用规则和 LLM 识别用户意图，比如搜索论文、论文问答、推荐、偏好更新；然后 build_goal 把 intent 转成结构化 Goal；build_plan 根据 Goal 和工具注册表生成 ExecutablePlan。
>
> 执行阶段不是一次性跑完，而是每次选择一个可执行 step。`execute_step` 只负责输入绑定、参数校验和工具调用；`observe_step` 再判断工具结果质量，比如搜索是否为空、论文目标是否明确、Paper QA 是否有证据来源。如果结果低质量，就进入 replan，由 Replanner 根据失败类型选择 retry、patch plan、请求确认、澄清或 fallback。
>
> 对于有副作用或需要用户确认的工具，比如构建论文索引，工具 contract 会标记 `requires_confirmation`。执行器会创建 `ConfirmationRequest`，通过 LangGraph interrupt 暂停，并把 pending confirmation 写入 checkpoint。用户确认后，前端带 `resume` 请求回来，后端校验业务 checkpoint 和 LangGraph checkpoint，然后用 `Command(resume=...)` 回到原执行现场继续执行。
>
> 所以这个 Agent 的重点是可控和可恢复：LLM 主要参与 intent、规划和部分诊断，但实际执行受工具 contract、Pydantic schema、状态机路由、observer 质量判断和 checkpoint 机制约束。

## 20. 面试官可能追问的问题

### 20.1 为什么用 LangGraph，不直接写一个函数链？

推荐回答：

> 因为这个 Agent 不是固定单链路，它有搜索、论文 QA、推荐、偏好写入、确认、重规划、失败恢复等多种分支。普通函数链很容易把执行、观察、重试和确认揉在一起，后续很难调试和恢复。LangGraph 可以把每个阶段拆成显式节点，用条件边表达路由，并且天然支持 checkpoint 和 interrupt/resume。这样前端流式展示、失败定位和用户确认恢复都会更稳定。

### 20.2 Agent 和 RAG 的边界是什么？

推荐回答：

> Agent 是上层编排器，负责理解用户目标、选择工具、执行计划、观察结果、重规划和处理确认。RAG 是其中一个能力模块，负责围绕单篇论文做检索、生成和证据校验。比如 `answer_paper_question` 这个 Agent 工具会调用 PaperQAService，PaperQAService 内部才是 RAG 检索生成链路。

### 20.3 怎么保证 Agent 不乱调工具？

推荐回答：

> 工具不是直接暴露裸函数，而是统一注册成 ToolContract。每个工具声明 capability_tags、输入输出 schema、副作用等级、是否需要确认、失败模式和恢复策略。planner 只能看到当前 goal 允许的候选工具，执行器还会做 Pydantic 入参校验、依赖检查、前置条件检查和确认门，所以不是让 LLM 自由调用任意函数。

### 20.4 工具执行成功但结果不好怎么办？

推荐回答：

> 项目里专门拆了 Observer。execute 阶段只负责工具调用，observe 阶段判断业务质量。比如搜索返回空、论文 QA 没有 sources、索引缺失、目标论文不明确，这些都不一定是工具异常，但业务上不能直接 finalize。Observer 会输出结构化 ObservationResult，然后进入 replan、确认、澄清或 fallback。

### 20.5 checkpoint 和 resume 解决什么问题？

推荐回答：

> 主要解决中断后恢复和副作用防重复执行。比如用户问一篇还没建索引的论文，Agent 需要先请求用户确认是否构建索引。此时图通过 interrupt 暂停，并保存 LangGraph checkpoint 和业务 runtime checkpoint。用户确认后，用同一个 session_id/thread_id 和 resume payload 回到原执行现场。执行器会消费 pending_confirmation，并清理 runtime 中的确认状态，防止重复点击造成重复构建索引。

### 20.6 pending_action 和 pending_confirmation 有什么区别？

推荐回答：

> `pending_confirmation` 是执行真源，存在 runtime 和 checkpoint 里，用于真正恢复执行。`pending_action` 是给旧前端和展示层看的镜像，不能反向驱动业务恢复。这样可以避免前端旧缓存或展示字段污染真实执行状态。

### 20.7 为什么要把 execute 和 observe 分开？

推荐回答：

> 因为工具调用成功不代表业务结果可用。execute 只处理工具调用、输入校验、输出标准化；observe 负责判断结果质量和恢复语义。拆开后，流式事件可以清楚展示工具何时开始、何时结束、结果为什么被认为低质量，以及后续为什么进入 replan。

## 21. 复习口诀

```text
入口：FastAPI 接请求
状态：AgentState 贯穿全图
解析：parse 识别 intent
目标：build_goal 生成 Goal
规划：build_plan 生成 ExecutablePlan
选步：select_next_step 判断依赖和条件
执行：execute_step 只调工具
观察：observe_step 判断质量
恢复：replan retry/patch/confirm/clarify/fallback
确认：interrupt + checkpoint + resume
收口：finalize 投影成 response
```
