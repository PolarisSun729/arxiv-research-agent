# 后端存储架构重构实施计划

## 背景

当前 `services/storage/database_service.py` 同时承担 SQLite 连接、建表迁移、JSON 编解码、论文目录、用户偏好、研究画像、Paper QA 索引、聊天记录、Agent 会话、Agent runtime checkpoint 和 LangGraph checkpoint 等职责。虽然部分代码已经拆到 `services/storage/database/*.py` mixin 中，但外部调用方仍然依赖一个万能 `DatabaseService`，新增能力也会继续往同一个大接口上堆方法。

本轮重构的目标不是继续拆小文件，而是替换存储层架构：删除 `DatabaseService` 上帝接口和 mixin 架构，改为按业务能力暴露清晰的 SQLite store。

## 硬性设计决策

1. 删除 `DatabaseService` 作为正式存储入口。
   - 不保留 `services/storage/database_service.py`。
   - 不新增临时或长期的 `DatabaseService` 转发兼容层。

2. 删除 `services/storage/database/` mixin 包。
   - mixin 只是在一个大类上分摊实现，不能解决接口过大的根因。
   - 迁移完成后，生产代码和测试代码都不应再依赖 mixin 结构。

3. 按业务能力拆 store，不按数据库表拆。
   - store interface 表达业务动作和状态流转，不暴露表级 CRUD 给上层拼事务。
   - 一个 store 可以维护多张表，只要这些表共同承担同一个业务不变量。

4. 本轮不改数据库表结构。
   - 保留现有表名、字段名、索引和数据语义。
   - 建表、补列、legacy 数据迁移只移动位置，不主动重命名或重塑 schema。

5. 不提前抽 repository/protocol 抽象。
   - 当前只有 SQLite 一个真实实现，先避免接口文件和实现文件空转。
   - 未来出现 Postgres 或内存实现时，再基于真实第二适配器抽象接口。

6. schema 初始化集中到 `StorageSchemaMigrator`。
   - 业务 store 不负责偷偷建表或补列。
   - 应用启动或测试初始化时显式执行 schema 初始化。

7. 使用短连接，不维护全局长连接。
   - `SqliteConnectionProvider` 每次操作提供独立连接。
   - 需要原子性的业务动作由 store 在同一连接内自行管理事务。

8. `StorageContainer` 只做组合根。
   - `dependencies.py` 可以通过 container 统一组装 connection provider、schema migrator 和各 store。
   - 业务服务构造函数只接收自己需要的具体 store，不接收整个 container，避免新的万能对象复活。

9. 允许破坏后端内部兼容，不破坏外部行为。
   - 可以修改 `dependencies.py`、router、service、agent、测试桩的内部依赖方式。
   - HTTP API、前端可见字段、Paper QA 索引语义、Agent 确认恢复语义保持不变。

10. 旧入口引用为 0 是硬性验收标准。
    - 生产代码不再 import `DatabaseService`。
    - 测试代码不再围绕 `FakeDatabaseService` 这种万能假对象设计。
    - `rg "DatabaseService|services.storage.database_service|services.storage.database"` 必须只剩删除记录或文档说明，不得出现在运行路径中。

## 目标目录结构

```text
services/storage/
  sqlite/
    __init__.py
    connection.py
    container.py
    schema.py
    serialization.py
    stores/
      __init__.py
      paper_catalog.py
      user_preferences.py
      research_profiles.py
      paper_profile_evidence.py
      paper_qa_index.py
      paper_chat.py
      paper_notes.py
      agent_sessions.py
      agent_runtime_checkpoints.py
      langgraph_checkpoints.py
```

目录名显式使用 `sqlite`，因为这些 store 是 SQLite 实现，不是假装通用的存储抽象。

## Store 边界

### `PaperCatalogStore`

职责：
- 管理 `arxiv_papers`。
- 提供论文新增、查询、分类检索、统计、embedding 元数据更新和删除。

典型业务方法：
- `add_paper(paper)`
- `get_paper(arxiv_id)`
- `search_by_category(category)`
- `list_all()`
- `count_total()`
- `count_today_new()`
- `update_embedding(arxiv_id, embedding_id, embedding_model)`
- `delete_paper(arxiv_id)`

### `UserPreferenceStore`

职责：
- 管理用户对论文的显式偏好和行为事件。
- 维护 liked/disliked 与 `user_paper_actions` 的一致语义。

典型业务方法：
- `add_liked_paper(user_id, arxiv_id)`
- `remove_liked_paper(user_id, arxiv_id)`
- `add_disliked_paper(user_id, arxiv_id)`
- `remove_disliked_paper(user_id, arxiv_id)`
- `record_paper_action(user_id, arxiv_id, action_type, metadata)`
- `remove_paper_action(user_id, arxiv_id, action_type)`
- `get_preferences(user_id)`
- `get_action_state(user_id, arxiv_id)`

### `ResearchProfileStore`

职责：
- 管理用户手动画像、生成画像、有效画像和画像快照。
- 负责画像字段标准化、manual/generated/effective 合并、legacy profile 读取和迁移后的兼容读取语义。

典型业务方法：
- `get_effective_profile(user_id)`
- `get_manual_profile(user_id)`
- `get_generated_profile(user_id)`
- `get_profile_layers(user_id)`
- `upsert_manual_profile(user_id, profile)`
- `patch_manual_profile(user_id, patch)`
- `save_generated_snapshot(user_id, profile, evidence_summary, quality_report, build_config)`
- `list_snapshots(user_id, limit)`
- `activate_snapshot(user_id, snapshot_id)`

### `ProfileEventStore`

职责：
- 管理画像事件流、脏事件计数、事件消费和偏好事件去重。
- 这是画像构建的输入事件流，不应混入画像投影读写。

典型业务方法：
- `record_event(...)`
- `record_preference_event(...)`
- `deactivate_events_for_paper(...)`
- `list_events(user_id, include_consumed, limit)`
- `mark_events_consumed(user_id, job_id, event_ids)`
- `count_dirty_events(user_id)`
- `get_latest_signal_timestamp(user_id)`

### `PaperProfileEvidenceStore`

职责：
- 管理论文画像证据抽取结果。
- 保持 extractor/normalizer version 下的幂等 upsert。

典型业务方法：
- `upsert_evidence(arxiv_id, evidence)`
- `get_evidence(arxiv_id, extractor_version, normalizer_version)`

### `PaperQAIndexStore`

职责：
- 管理 `paper_qa_index`、`paper_qa_index_versions` 和 `paper_index_jobs`。
- 维护索引构建版本、active 指针、旧版本清理状态、构建任务幂等和心跳。

典型业务方法：
- `get_index(arxiv_id)`
- `create_build(arxiv_id, loading_method)`
- `get_build(build_id)`
- `get_active_build(arxiv_id)`
- `get_latest_build(arxiv_id, statuses=None)`
- `list_builds(arxiv_id, statuses=None, limit=20)`
- `update_build(build_id, **fields)`
- `activate_build(build_id)`
- `mark_build_deleted(build_id)`
- `create_job(arxiv_id, loading_method)`
- `acquire_job(job_id, stale_after_seconds)`
- `mark_stale_jobs(...)`
- `update_job(job_id, **fields)`
- `get_job(job_id)`
- `list_jobs(arxiv_id=None, limit=20)`

关键不变量：
- `activate_build()` 必须在事务内检查 build 状态和 sparse artifact 完整性。
- active 切换时，旧 active 只能标记为 `cleanup_pending`，不能在同一事务里删除产物。
- 构建任务幂等键和心跳语义必须保持现有行为。

### `PaperChatStore`

职责：
- 管理论文问答会话、消息、摘要字段和 turn 持久化。
- 维护消息追加后会话统计的同步更新。

典型业务方法：
- `create_session(user_id, arxiv_id, title)`
- `get_session(session_id, user_id)`
- `list_sessions(user_id, arxiv_id=None, limit=20)`
- `update_session(session_id, user_id, **fields)`
- `append_message(...)`
- `list_messages(session_id, user_id)`
- `get_message(message_id, user_id)`
- `clear_session(session_id, user_id)`
- `delete_session(session_id, user_id)`
- `append_qa_turn(...)`

### `PaperNoteStore`

职责：
- 管理 `paper_notes`。
- 在用户明确允许时写入画像事件流，避免普通私有笔记污染长期画像证据。

典型业务方法：
- `create_note(...)`
- `get_note(note_id, user_id)`
- `list_notes(user_id, arxiv_id=None, session_id=None)`
- `list_profile_notes(user_id)`
- `update_note(...)`
- `delete_note(note_id, user_id)`

### `AgentSessionStore`

职责：
- 管理前端/Agent 会话记录和展示态字段。
- 只承载会话级状态，不承载 runtime confirmation 真源。

典型业务方法：
- `create_or_get_session(user_id, session_id=None)`
- `get_session(session_id, user_id)`
- `update_session(session_id, user_id, **fields)`
- `clear_session(session_id, user_id)`

### `AgentRuntimeCheckpointStore`

职责：
- 管理业务级 Agent runtime checkpoint。
- 维护 pending confirmation 真源、批准态防回放、原子消费、过期和清理。

典型业务方法：
- `upsert_checkpoint(...)`
- `get_checkpoint(user_id, session_id, thread_id=None)`
- `list_by_thread(session_id, thread_id=None, limit=20)`
- `get_unique_by_thread(session_id, thread_id=None)`
- `consume_pending_confirmation(...)`
- `mark_status(..., clear_pending_confirmation=False)`
- `expire_waiting(now=None)`
- `cleanup_terminal(retention_days=7)`
- `build_context_lifecycle_stats(...)`

关键不变量：
- 已消费确认不能被 LangGraph 重放的旧 waiting 快照覆盖。
- `consume_pending_confirmation()` 必须用 `status='waiting_confirmation'` 和 pending payload 作为抢占条件。
- 清除 pending confirmation 时，要同步清理 `runtime_state_json` 内嵌 pending，并保留批准 step id。

### `LangGraphCheckpointStore`

职责：
- 管理 LangGraph 原始 checkpoint blob 和 pending writes。
- 它只负责执行恢复数据持久化，不承载业务确认语义。

典型业务方法：
- `put_checkpoint(...)`
- `get_checkpoint(thread_id, checkpoint_ns='', checkpoint_id=None)`
- `list_checkpoints(thread_id, checkpoint_ns='', limit=20)`
- `put_writes(...)`
- `get_writes(thread_id, checkpoint_ns, checkpoint_id)`
- `cleanup_for_terminal_runtime(thread_ids)`

## 实施任务清单

1. 建立 SQLite 基础设施。
   - 新增 `services/storage/sqlite/connection.py`。
   - 新增 `services/storage/sqlite/serialization.py`。
   - 新增 `services/storage/sqlite/schema.py`。
   - 新增 `services/storage/sqlite/container.py`。

2. 迁移 schema 初始化。
   - 将现有 `_initialize_database()` 中的建表和索引 SQL 移到 `StorageSchemaMigrator.ensure_schema()`。
   - 将补列逻辑移到 migrator 的私有方法。
   - 将 legacy research profile 迁移逻辑移到 migrator 或 `ResearchProfileStore` 的显式迁移辅助中，避免业务读写隐式执行全库迁移。

3. 建立业务 store。
   - 按上面的 store 边界迁移现有方法。
   - store 只接收 `SqliteConnectionProvider` 和必要的同层 store 依赖。
   - 跨 store 调用必须谨慎，优先把同一事务的不变量放在同一个 store 内。

4. 改造依赖注入。
   - `dependencies.py` 新增 `get_storage_container()`。
   - 为业务层暴露具体 store getter，例如 `get_paper_qa_index_store()`、`get_research_profile_store()`。
   - 删除 `get_database_service()`。

5. 改造业务服务和 router。
   - `PaperQAService`、`PaperQAIndexBuilder`、`PaperIndexJobManager` 改为注入 `PaperQAIndexStore`、`PaperChatStore` 等具体 store。
   - `MemoryService` 改为注入画像、偏好、证据和事件相关 store。
   - recommendation 相关服务改为注入论文目录、偏好、兴趣向量等具体 store。
   - arXiv Agent runtime 改为注入 `AgentRuntimeCheckpointStore` 和 `LangGraphCheckpointStore`。
   - router 只依赖所需 store 或服务，不依赖万能存储对象。

6. 改造测试。
   - 删除 `FakeDatabaseService` 风格的万能测试桩。
   - 为不同业务能力提供更窄的 fake store 或使用临时 SQLite store。
   - 更新 `tests/helpers/sqlite.py`，让测试显式初始化 `StorageSchemaMigrator` 和目标 store。

7. 删除旧架构。
   - 删除 `services/storage/database_service.py`。
   - 删除 `services/storage/database/`。
   - 清理 `services/storage/__init__.py` 中的旧导出。

8. 验收检查。
   - 运行旧入口引用检查。
   - 运行存储相关单元测试和集成测试。
   - 运行 Agent runtime checkpoint、Paper QA index build、recommendation、memory service 的关键回归测试。

## 验收标准

1. 文件结构符合新目录设计。
2. `services/storage/database_service.py` 不存在。
3. `services/storage/database/` 不存在。
4. 生产代码中不存在 `DatabaseService`、`services.storage.database_service`、`services.storage.database` 运行引用。
5. 测试中不存在万能 `FakeDatabaseService`。
6. 现有 SQLite 表结构没有主动重命名或重塑。
7. HTTP API 响应字段和前端可见行为保持不变。
8. Paper QA active index 切换、旧版本清理、构建任务幂等语义保持不变。
9. Agent pending confirmation 的原子消费、防 replay、过期清理语义保持不变。
10. 关键测试通过。

## 建议验证命令

```powershell
rg "DatabaseService|services\.storage\.database_service|services\.storage\.database" .
pytest tests/unit/test_database_service_sqlite.py
pytest tests/unit/agents/arxiv_search_agent/test_runtime_checkpoint.py
pytest tests/integration/test_paper_qa_service.py
pytest tests/integration/test_qa_index_build_flow.py
pytest tests/integration/test_memory_service.py
pytest tests/integration/test_recommendation_flow.py
```

测试文件名会随重构调整。最终命令应以新 store 测试文件为准，但覆盖面必须保留以上业务语义。

## 非目标

1. 不重命名 SQLite 表或字段。
2. 不切换数据库引擎。
3. 不引入 repository/protocol 抽象层。
4. 不改变 HTTP API 和前端契约。
5. 不保留 `DatabaseService` 兼容适配层。
