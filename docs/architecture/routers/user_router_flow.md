# `user_router.py` 接口处理流程

本文仅分析当前仓库中的 `backend/routers/user_router.py` 及其实际调用到的用户、偏好、推荐、记忆相关实现，不覆盖其他 router。

## 1. Router 基本信息

- Router 文件路径：`backend/routers/user_router.py`
- Router prefix：`/user`
- Router tags：`["user"]`
- 主要职责：负责用户偏好读取、论文 like / dislike、通用 paper action、研究画像读写、兴趣向量生成与读取、个性化推荐生成。
- 主要依赖的 service / class / function：
  - `RecommendationService`
  - `MemoryService`
  - `DatabaseService`
  - `dependencies.get_recommendation_service()`
  - `dependencies.get_memory_service()`
  - `dependencies.get_database_service()`
- 是否访问数据库：是
- 是否写入用户偏好：是
- 是否读取用户喜欢的论文：是
- 是否计算推荐：是
- 是否调用 embedding：是
- 是否访问向量库：是
- 是否访问 `MemoryService`：是
- 是否有缓存或推荐结果表：未发现单独的推荐结果缓存表或推荐结果持久化表；发现 `user_interest_vectors` 表用于保存用户兴趣向量

### 依赖边界说明

- `user_router.py` 本身较薄，主要做参数接收、少量状态码转换、再转发到 service。
- 用户偏好与行为落库：
  - `DatabaseService` 负责 `user_liked_papers`、`user_disliked_papers`、`user_paper_actions`、`user_research_profiles`、`user_interest_vectors` 等表的读写。
- 研究画像：
  - `MemoryService` 负责研究画像 patch / upsert 语义，以及从偏好、行为、笔记等信号更新长期画像。
- 推荐计算：
  - `RecommendationService` 负责兴趣向量生成、候选召回、候选物化、个性化打分、多样性选择。
- embedding / vector store：
  - `RecommendationService` 通过 `EmbeddingService` 生成论文 embedding 或候选临时 embedding；
  - 通过 `VectorStoreService` 读取已存论文向量、插入论文 embedding、执行基于兴趣向量或兴趣簇的相似检索。
- 用户实体：
  - 未发现独立 `users` 表或“创建用户”流程。当前系统更像以 `user_id` 作为逻辑主键，直接写偏好/画像/行为表。

## 2. 接口总览表

| 方法 | 路径 | 函数名 | 主要职责 | 主要调用 | 副作用 |
|---|---|---|---|---|---|
| POST | `/user/preferences` | `upsert_user_preferences` | 读取指定用户偏好 | `DatabaseService.get_user_preferences()` | 读数据库；未实际 upsert |
| GET | `/user/preferences/{user_id}` | `get_user_preferences` | 按 path 获取用户偏好 | `DatabaseService.get_user_preferences()` | 读数据库 |
| POST | `/user/like-paper` | `like_paper` | 记录喜欢论文 | `RecommendationService.record_user_paper_preference()` | 写 liked/disliked/action/profile，必要时物化论文并写向量库 |
| POST | `/user/dislike-paper` | `dislike_paper` | 记录不喜欢论文 | `RecommendationService.record_user_paper_preference()` | 写 disliked/liked/action/profile，必要时物化论文并写向量库 |
| POST | `/user/paper-action` | `record_paper_action` | 记录通用论文行为 | `RecommendationService.record_user_paper_action()` | 写 action，必要时物化论文、更新 profile |
| DELETE | `/user/paper-action` | `remove_paper_action` | 删除一条论文行为 | `DatabaseService.remove_user_paper_action()` | 写数据库 |
| GET | `/user/paper-actions/{user_id}` | `get_user_paper_actions` | 获取用户行为明细与 action_map | `DatabaseService.get_user_paper_actions()`、`DatabaseService.get_user_paper_action_map()` | 读数据库 |
| GET | `/user/research-profile/{user_id}` | `get_user_research_profile` | 获取研究画像 | `MemoryService.load_user_profile()` | 读数据库 |
| PUT | `/user/research-profile` | `upsert_user_research_profile` | 以整体更新语义写研究画像 | `MemoryService.patch_user_profile(source="manual_upsert")` | 写研究画像 |
| PATCH | `/user/research-profile` | `patch_user_research_profile` | 以局部 patch 语义写研究画像 | `MemoryService.patch_user_profile(source="manual")` | 写研究画像 |
| DELETE | `/user/like-paper` | `remove_like` | 取消喜欢 | `DatabaseService.remove_liked_paper()` | 删除 liked 记录，同时删除 like action |
| DELETE | `/user/dislike-paper` | `remove_dislike` | 取消不喜欢 | `DatabaseService.remove_disliked_paper()` | 删除 disliked 记录，同时删除 dislike action |
| POST | `/user/generate-interest-vector` | `generate_user_interest_vector` | 生成或重建用户兴趣向量 | `RecommendationService.generate_user_interest_vector()` | 读 liked/disliked，读写向量库，调用 embedding，写 `user_interest_vectors` |
| GET | `/user/interest-vector` | `get_user_interest_vector` | 获取已保存兴趣向量 | `DatabaseService.get_user_interest_vector()` | 读数据库 |
| POST | `/user/recommend-papers` | `recommend_papers` | 生成个性化推荐 | `RecommendationService.recommend_papers()` | 读偏好/画像/兴趣向量/OAI 数据/向量库，可能补 embedding/补论文记录；未发现保存推荐结果 |

## 3. Router 总览流程图

```mermaid
flowchart TD
    U["前端 / 调用方"] --> R["[Router] user_router.py"]

    R --> P1["[Service] DatabaseService.get_user_preferences"]
    P1 --> DB1["[DB] user_liked_papers / user_disliked_papers / user_paper_actions / user_research_profiles"]

    R --> A1["[Recommendation] RecommendationService.record_user_paper_preference"]
    A1 --> MAT1["[Service] CandidateMaterializer._ensure_paper_materialized"]
    MAT1 --> DB2["[DB] arxiv_papers"]
    MAT1 --> EMB1["[Embedding] create_single_embedding"]
    MAT1 --> VS1["[VectorStore] insert_single_embedding / get_paper_embeddings_by_arxiv_ids"]
    A1 --> DB3["[DB] add_liked_paper / add_disliked_paper / record_user_paper_action"]
    A1 --> MEM1["[Memory] update_profile_from_preference"]
    MEM1 --> DB4["[DB] user_research_profiles"]

    R --> A2["[Recommendation] RecommendationService.record_user_paper_action"]
    A2 --> MAT2["[Service] _ensure_paper_materialized"]
    A2 --> DB5["[DB] record_user_paper_action / remove_user_paper_action"]
    A2 --> MEM2["[Memory] update_profile_from_paper_action"]

    R --> P2["[Service] MemoryService.load_user_profile / patch_user_profile"]
    P2 --> DB6["[DB] user_research_profiles"]

    R --> IV1["[Recommendation] generate_user_interest_vector"]
    IV1 --> DB7["[DB] get_liked_papers / get_disliked_papers / save_user_interest_vector"]
    IV1 --> VS2["[VectorStore] get_paper_embeddings_by_arxiv_ids"]
    IV1 --> EMB2["[Embedding] fallback 生成论文向量"]
    IV1 --> CLU["[Service] HDBSCAN interest clustering"]

    R --> IV2["[DB] get_user_interest_vector"]
    IV2 --> DB8["[DB] user_interest_vectors"]

    R --> REC["[Recommendation] recommend_papers"]
    REC --> DB9["[DB] get_user_preferences / get_liked_papers_with_details / get_user_research_profile / get_user_interest_vector"]
    REC --> OAI["[Service] OAI recent papers / arXiv backfill"]
    REC --> VS3["[VectorStore] search_similar_papers / get_paper_embeddings_by_arxiv_ids"]
    REC --> EMB3["[Embedding] 候选物化或候选临时向量"]
    REC --> MEM3["[Memory] 读取研究画像信号"]
    REC --> DB10["[DB] 可能补写 arxiv_papers / user_interest_vectors"]
    REC --> RT["[Response] recommendations"]

    R --> NF["[Fallback] 未发现推荐结果缓存表 / 未发现独立 users 表"]
```

## 4. 每个接口单独流程图

## 接口：POST `/user/preferences`

### 职责

虽然路由名叫 `upsert_user_preferences`，但当前实现只是按 `user_id` 读取用户偏好聚合结果。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] upsert_user_preferences"]
    B --> C["[Validate] 解析 body.user_id"]
    C --> D["[DB] DatabaseService.get_user_preferences"]
    D --> E["[DB] get_liked_papers / get_disliked_papers / get_user_paper_action_map / get_user_research_profile"]
    E --> F["[Response] status/message/preferences"]
    D -. 异常 .-> G["[Error] 500"]
    D -. 数据缺失 .-> H["[Fallback] DatabaseService 返回空偏好结构"]
```

### 关键调用链

`upsert_user_preferences() -> DatabaseService.get_user_preferences() -> get_liked_papers()/get_disliked_papers()/get_user_paper_action_map()/get_user_research_profile()`

### 输入

- Body：
  - `user_id`

### 输出

- `status`
- `message`
- `preferences`
  - `user_id`
  - `liked_papers`
  - `disliked_papers`
  - `paper_actions`
  - `research_profile`

### 副作用

- 是否创建用户：否，未发现
- 是否更新用户信息：否
- 是否写入 like / dislike / collect：否
- 是否更新用户兴趣画像：否
- 是否更新 memory：否
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：否
- 是否访问 vector store：否
- 是否写数据库：否

### 异常 / fallback

- router 捕获异常并返回 `500`
- `DatabaseService.get_user_preferences()` 内部失败时会返回空偏好结构

## 接口：GET `/user/preferences/{user_id}`

### 职责

按 path 参数读取用户偏好聚合结果。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] get_user_preferences"]
    B --> C["[Validate] 解析 path.user_id"]
    C --> D["[DB] DatabaseService.get_user_preferences"]
    D --> E["[DB] get_liked_papers / get_disliked_papers / get_user_paper_action_map / get_user_research_profile"]
    E --> F["[Response] preferences"]
    D -. 异常 .-> G["[Error] 500"]
    D -. 数据缺失 .-> H["[Fallback] 空偏好结构"]
```

### 关键调用链

`get_user_preferences() -> DatabaseService.get_user_preferences()`

### 输入

- Path：
  - `user_id`

### 输出

- `user_id`
- `liked_papers`
- `disliked_papers`
- `paper_actions`
- `research_profile`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：否
- 是否写入 like / dislike / collect：否
- 是否更新用户兴趣画像：否
- 是否更新 memory：否
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：否
- 是否访问 vector store：否
- 是否写数据库：否

### 异常 / fallback

- router 捕获异常并返回 `500`
- service 内部失败时返回空偏好结构

## 接口：POST `/user/like-paper`

### 职责

记录“喜欢论文”的显式正反馈，并同步更新行为记录与长期研究画像；必要时先把论文物化到本地论文表和向量库。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] like_paper"]
    B --> C["[Validate] 解析 arxiv_id / user_id / paper"]
    C --> D["[Recommendation] RecommendationService.record_user_paper_preference(liked=True)"]
    D --> E["[Service] _ensure_paper_materialized"]
    E --> E1["[DB] get_paper"]
    E --> E2["[VectorStore] get_paper_embeddings_by_arxiv_ids"]
    E --> E3["[Embedding] create_single_embedding"]
    E --> E4["[VectorStore] insert_single_embedding"]
    E --> E5["[DB] add_paper / update_paper_embedding"]
    D --> F["[DB] add_liked_paper"]
    F --> F1["[DB] DELETE user_disliked_papers"]
    F --> F2["[DB] INSERT user_liked_papers"]
    F --> F3["[DB] record_user_paper_action('like')"]
    F --> F4["[DB] remove_user_paper_action('dislike'/'not_interested')"]
    D --> G["[Memory] update_profile_from_preference('like')"]
    G --> H["[DB] patch_user_research_profile / upsert_user_research_profile"]
    H --> I["[Response] status/message/arxiv_id/paper"]
    C -. arxiv_id 缺失 .-> J["[Error] 400"]
    E -. 无法物化论文 .-> K["[Error] 404/500"]
    F -. 写入失败 .-> L["[Error] 500"]
    G -. 画像更新失败 .-> M["[Fallback] 记录 warning，但不阻断主流程"]
    B -. 其他异常 .-> N["[Error] 500"]
```

### 关键调用链

`like_paper() -> RecommendationService.record_user_paper_preference() -> CandidateMaterializer._ensure_paper_materialized() -> DatabaseService.add_liked_paper() -> DatabaseService.record_user_paper_action() -> MemoryService.update_profile_from_preference() -> DatabaseService.patch_user_research_profile()/upsert_user_research_profile()`

### 输入

- Body：
  - `arxiv_id`
  - `user_id`
  - `paper`

### 输出

- `status`
- `message`
- `arxiv_id`
- `paper`

### 副作用

- 是否创建用户：否，未发现独立用户创建
- 是否更新用户信息：是，更新研究画像
- 是否写入 like / dislike / collect：是，写 like，并删除冲突 dislike
- 是否更新用户兴趣画像：是
- 是否更新 memory：是，通过 `MemoryService.update_profile_from_preference()`
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：可能会调用
- 是否访问 vector store：可能会访问
- 是否写数据库：是

### 异常 / fallback

- `arxiv_id` 缺失时返回 `400`
- 论文元数据缺失且无法物化时返回 `404`
- like 写入失败时返回 `500`
- 研究画像更新失败只记录 warning，不阻断主流程

## 接口：POST `/user/dislike-paper`

### 职责

记录“不喜欢论文”的显式负反馈，并同步更新行为记录与长期研究画像；必要时先物化论文。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] dislike_paper"]
    B --> C["[Validate] 解析 arxiv_id / user_id / paper"]
    C --> D["[Recommendation] RecommendationService.record_user_paper_preference(liked=False)"]
    D --> E["[Service] _ensure_paper_materialized"]
    E --> E1["[DB] get_paper / add_paper / update_paper_embedding"]
    E --> E2["[Embedding] create_single_embedding"]
    E --> E3["[VectorStore] get_paper_embeddings_by_arxiv_ids / insert_single_embedding"]
    D --> F["[DB] add_disliked_paper"]
    F --> F1["[DB] DELETE user_liked_papers"]
    F --> F2["[DB] INSERT user_disliked_papers"]
    F --> F3["[DB] record_user_paper_action('dislike')"]
    F --> F4["[DB] remove_user_paper_action('like')"]
    D --> G["[Memory] update_profile_from_preference('dislike')"]
    G --> H["[DB] patch_user_research_profile / upsert_user_research_profile"]
    H --> I["[Response] status/message/arxiv_id/paper"]
    C -. arxiv_id 缺失 .-> J["[Error] 400"]
    E -. 无法物化论文 .-> K["[Error] 404/500"]
    F -. 写入失败 .-> L["[Error] 500"]
    G -. 画像更新失败 .-> M["[Fallback] warning，不阻断"]
    B -. 其他异常 .-> N["[Error] 500"]
```

### 关键调用链

`dislike_paper() -> RecommendationService.record_user_paper_preference() -> CandidateMaterializer._ensure_paper_materialized() -> DatabaseService.add_disliked_paper() -> DatabaseService.record_user_paper_action() -> MemoryService.update_profile_from_preference()`

### 输入

- Body：
  - `arxiv_id`
  - `user_id`
  - `paper`

### 输出

- `status`
- `message`
- `arxiv_id`
- `paper`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：是，更新研究画像
- 是否写入 like / dislike / collect：是，写 dislike，并删除冲突 like
- 是否更新用户兴趣画像：是
- 是否更新 memory：是
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：可能会调用
- 是否访问 vector store：可能会访问
- 是否写数据库：是

### 异常 / fallback

- `arxiv_id` 缺失时返回 `400`
- 无法物化论文时返回 `404`
- dislike 写入失败时返回 `500`
- 画像更新失败只 warning，不阻断主流程

## 接口：POST `/user/paper-action`

### 职责

记录通用论文行为，如 `favorite`、`read`、`later`、`archived`、`note_saved` 等，并按行为类型决定是否更新长期画像。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] record_paper_action"]
    B --> C["[Validate] 解析 payload.user_id/arxiv_id/action_type/paper/metadata"]
    C --> D["[Recommendation] RecommendationService.record_user_paper_action"]
    D --> E["[Service] _ensure_paper_materialized"]
    E --> E1["[DB] get_paper / add_paper / update_paper_embedding"]
    E --> E2["[Embedding] create_single_embedding"]
    E --> E3["[VectorStore] get_paper_embeddings_by_arxiv_ids / insert_single_embedding"]
    D --> F["[DB] record_user_paper_action"]
    D --> G["[Memory] update_profile_from_paper_action"]
    G --> H["[Fallback] 某些 action 不更新画像，仅返回现有 profile"]
    G --> I["[DB] patch_user_research_profile / upsert_user_research_profile"]
    I --> J["[Response] status/message/arxiv_id/action_type/paper/metadata"]
    C -. 参数缺失 .-> K["[Error] 422 / 400"]
    F -. 写入失败 .-> L["[Error] 500"]
    G -. 画像更新失败 .-> M["[Fallback] warning，不阻断"]
    B -. 其他异常 .-> N["[Error] 500"]
```

### 关键调用链

`record_paper_action() -> RecommendationService.record_user_paper_action() -> CandidateMaterializer._ensure_paper_materialized() -> DatabaseService.record_user_paper_action() -> MemoryService.update_profile_from_paper_action()`

### 输入

- Body：
  - `user_id`
  - `arxiv_id`
  - `action_type`
  - `paper`
  - `metadata`

### 输出

- `status`
- `message`
- `arxiv_id`
- `action_type`
- `paper`
- `metadata`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：可能，取决于 action 类型
- 是否写入 like / dislike / collect：写入通用 `action_type`；未发现专门 `collect` 表，若使用 collect 语义更接近 `favorite` / `later`
- 是否更新用户兴趣画像：可能
- 是否更新 memory：是，经过 `MemoryService.update_profile_from_paper_action()`
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：可能会调用
- 是否访问 vector store：可能会访问
- 是否写数据库：是

### 异常 / fallback

- Pydantic 缺字段会返回 `422`
- `arxiv_id` 或 `action_type` 缺失会返回 `400`
- 不支持的 action 类型会导致 service 写入失败并转为 `500`
- 某些 action 类型不会更新画像，只返回当前画像
- 画像更新失败只 warning，不阻断主流程

## 接口：DELETE `/user/paper-action`

### 职责

删除一条已记录的用户论文行为。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] remove_paper_action"]
    B --> C["[Validate] 解析 arxiv_id / action_type / user_id"]
    C --> D["[DB] remove_user_paper_action"]
    D --> E{"[Validate] 删除成功 ?"}
    E -->|是| F["[Response] success"]
    E -->|否| G["[Error] 404"]
    B -. 异常 .-> H["[Error] 500"]
    D -. 未发现专门 fallback .-> I["[Fallback] 未发现"]
```

### 关键调用链

`remove_paper_action() -> DatabaseService.remove_user_paper_action()`

### 输入

- Body：
  - `arxiv_id`
  - `action_type`
  - `user_id`

### 输出

- `status`
- `message`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：否
- 是否写入 like / dislike / collect：是，删除一条 action
- 是否更新用户兴趣画像：否
- 是否更新 memory：否
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：否
- 是否访问 vector store：否
- 是否写数据库：是

### 异常 / fallback

- 删除不到记录时返回 `404`
- 其他异常返回 `500`
- 未发现明确 fallback

## 接口：GET `/user/paper-actions/{user_id}`

### 职责

查询用户的论文行为明细，并额外返回按 `action_type` 聚合的 `action_map`。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] get_user_paper_actions"]
    B --> C["[Validate] 解析 user_id / action_type"]
    C --> D["[DB] get_user_paper_actions"]
    D --> E["[DB] get_user_paper_action_map"]
    E --> F["[Response] status/user_id/action_type/actions/action_map"]
    D -. 空结果 .-> G["[Fallback] 返回空 actions"]
    B -. 异常 .-> H["[Error] 500"]
```

### 关键调用链

`get_user_paper_actions() -> DatabaseService.get_user_paper_actions() -> DatabaseService.get_user_paper_action_map()`

### 输入

- Path：
  - `user_id`
- Query：
  - `action_type`

### 输出

- `status`
- `user_id`
- `action_type`
- `actions`
- `action_map`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：否
- 是否写入 like / dislike / collect：否
- 是否更新用户兴趣画像：否
- 是否更新 memory：否
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：否
- 是否访问 vector store：否
- 是否写数据库：否

### 异常 / fallback

- 空结果时直接返回空列表/空 map
- 其他异常返回 `500`

## 接口：GET `/user/research-profile/{user_id}`

### 职责

读取用户长期研究画像。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] get_user_research_profile"]
    B --> C["[Validate] 解析 user_id"]
    C --> D["[Memory] MemoryService.load_user_profile"]
    D --> E["[DB] get_user_research_profile"]
    E --> F["[Response] profile"]
    E -. 未命中 .-> G["[Fallback] 返回空画像结构"]
    B -. 异常 .-> H["[Error] 500"]
```

### 关键调用链

`get_user_research_profile() -> MemoryService.load_user_profile() -> DatabaseService.get_user_research_profile()`

### 输入

- Path：
  - `user_id`

### 输出

- `user_id`
- `positive_topics`
- `negative_topics`
- `recent_topics`
- `preferred_categories`
- `preferred_answer_style`
- `common_question_types`
- `representative_papers`
- `created_at`
- `updated_at`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：否
- 是否写入 like / dislike / collect：否
- 是否更新用户兴趣画像：否
- 是否更新 memory：否
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：否
- 是否访问 vector store：否
- 是否写数据库：否

### 异常 / fallback

- 画像不存在时返回空画像结构
- 其他异常返回 `500`

## 接口：PUT `/user/research-profile`

### 职责

以“整体更新 / 补全”的语义写入研究画像，未传字段不会被写成 `null`，而是由 `manual_upsert` 语义接管。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] upsert_user_research_profile"]
    B --> C["[Validate] 解析 payload"]
    C --> D["[Validate] model_dump(exclude user_id / none)"]
    D --> E["[Memory] MemoryService.patch_user_profile(source='manual_upsert')"]
    E --> F["[DB] upsert_user_research_profile"]
    F --> G["[Response] status/profile"]
    B -. 参数不合法 .-> H["[Error] 422"]
    E -. 异常 .-> I["[Error] 500"]
    E -. 空 patch .-> J["[Fallback] 返回当前画像"]
```

### 关键调用链

`upsert_user_research_profile() -> MemoryService.patch_user_profile(source='manual_upsert') -> DatabaseService.upsert_user_research_profile()`

### 输入

- Body：
  - `user_id`
  - `positive_topics`
  - `negative_topics`
  - `recent_topics`
  - `preferred_categories`
  - `preferred_answer_style`
  - `common_question_types`
  - `representative_papers`

### 输出

- `status`
- `profile`

### 副作用

- 是否创建用户：否，未发现独立 users 表
- 是否更新用户信息：是
- 是否写入 like / dislike / collect：否
- 是否更新用户兴趣画像：是，更新研究画像表
- 是否更新 memory：是，通过 `MemoryService`
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：否
- 是否访问 vector store：否
- 是否写数据库：是

### 异常 / fallback

- 参数不合法会返回 `422`
- 空 patch 时会回退返回当前画像
- 其他异常返回 `500`

## 接口：PATCH `/user/research-profile`

### 职责

以局部 patch 的语义更新研究画像。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] patch_user_research_profile"]
    B --> C["[Validate] 解析 payload"]
    C --> D["[Validate] model_dump(exclude user_id / none)"]
    D --> E["[Memory] MemoryService.patch_user_profile(source='manual')"]
    E --> F["[DB] patch_user_research_profile -> upsert_user_research_profile"]
    F --> G["[Response] status/profile"]
    B -. 参数不合法 .-> H["[Error] 422"]
    E -. 异常 .-> I["[Error] 500"]
    E -. 空 patch .-> J["[Fallback] 返回当前画像"]
```

### 关键调用链

`patch_user_research_profile() -> MemoryService.patch_user_profile(source='manual') -> DatabaseService.patch_user_research_profile() -> DatabaseService.upsert_user_research_profile()`

### 输入

- Body：
  - `user_id`
  - `positive_topics`
  - `negative_topics`
  - `recent_topics`
  - `preferred_categories`
  - `preferred_answer_style`
  - `common_question_types`
  - `representative_papers`

### 输出

- `status`
- `profile`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：是
- 是否写入 like / dislike / collect：否
- 是否更新用户兴趣画像：是
- 是否更新 memory：是
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：否
- 是否访问 vector store：否
- 是否写数据库：是

### 异常 / fallback

- 参数不合法返回 `422`
- 空 patch 时回退到当前画像
- 其他异常返回 `500`

## 接口：DELETE `/user/like-paper`

### 职责

撤销用户对某篇论文的 like 记录，同时删除对应的 `like` action。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] remove_like"]
    B --> C["[Validate] 解析 arxiv_id / user_id"]
    C --> D["[DB] remove_liked_paper"]
    D --> E["[DB] DELETE user_liked_papers"]
    D --> F["[DB] remove_user_paper_action('like')"]
    F --> G{"[Validate] 删除成功 ?"}
    G -->|是| H["[Response] success"]
    G -->|否| I["[Error] 500"]
    B -. 异常 .-> J["[Error] 500"]
    D -. 未发现专门 fallback .-> K["[Fallback] 未发现"]
```

### 关键调用链

`remove_like() -> DatabaseService.remove_liked_paper() -> DatabaseService.remove_user_paper_action()`

### 输入

- Body：
  - `arxiv_id`
  - `user_id`

### 输出

- `status`
- `message`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：否
- 是否写入 like / dislike / collect：是，删除 like
- 是否更新用户兴趣画像：否，未发现自动重建画像
- 是否更新 memory：否
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：否
- 是否访问 vector store：否
- 是否写数据库：是

### 异常 / fallback

- 删除失败时直接返回 `500`
- 未发现专门区分“不存在记录”和“数据库异常”的异常处理

## 接口：DELETE `/user/dislike-paper`

### 职责

撤销用户对某篇论文的 dislike 记录，同时删除对应的 `dislike` action。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] remove_dislike"]
    B --> C["[Validate] 解析 arxiv_id / user_id"]
    C --> D["[DB] remove_disliked_paper"]
    D --> E["[DB] DELETE user_disliked_papers"]
    D --> F["[DB] remove_user_paper_action('dislike')"]
    F --> G{"[Validate] 删除成功 ?"}
    G -->|是| H["[Response] success"]
    G -->|否| I["[Error] 500"]
    B -. 异常 .-> J["[Error] 500"]
    D -. 未发现专门 fallback .-> K["[Fallback] 未发现"]
```

### 关键调用链

`remove_dislike() -> DatabaseService.remove_disliked_paper() -> DatabaseService.remove_user_paper_action()`

### 输入

- Body：
  - `arxiv_id`
  - `user_id`

### 输出

- `status`
- `message`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：否
- 是否写入 like / dislike / collect：是，删除 dislike
- 是否更新用户兴趣画像：否，未发现自动重建画像
- 是否更新 memory：否
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：否
- 是否访问 vector store：否
- 是否写数据库：是

### 异常 / fallback

- 删除失败时返回 `500`
- 未发现明确 fallback

## 接口：POST `/user/generate-interest-vector`

### 职责

根据用户 liked / disliked 历史生成兴趣向量，并在 liked 数量足够时尝试进行兴趣簇聚类；结果保存到 `user_interest_vectors`。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] generate_user_interest_vector"]
    B --> C["[Validate] 解析 user_id"]
    C --> D["[Recommendation] RecommendationService.generate_user_interest_vector"]
    D --> E["[DB] get_liked_papers / get_disliked_papers"]
    E --> F{"[Validate] liked_papers 是否为空 ?"}
    F -->|是| G["[Error] 400 No liked papers found for user"]
    F -->|否| H["[VectorStore] get_paper_embeddings_by_arxiv_ids"]
    H --> I["[Fallback] 缺失向量时 _hydrate_vectors_with_metadata"]
    I --> I1["[Service] arXiv/OAI backfill"]
    I --> I2["[Embedding] create_single_embedding"]
    I --> I3["[VectorStore] insert_single_embedding / add_paper"]
    D --> J["[Service] _mean_vector / _normalize_vector"]
    D --> K{"[Validate] liked 数量足够聚类 ?"}
    K -->|是| L["[Recommendation] HDBSCAN 聚类"]
    K -->|否| M["[Fallback] mean profile"]
    L --> N["[Fallback] 聚类失败回退 mean"]
    D --> O["[DB] save_user_interest_vector"]
    O --> P["[Response] status/message/paper_count/vector_dimension/cluster_summary"]
    O -. 保存失败 .-> Q["[Error] 500"]
    B -. 其他异常 .-> R["[Error] 500"]
```

### 关键调用链

`generate_user_interest_vector() -> RecommendationService.generate_user_interest_vector() -> DatabaseService.get_liked_papers()/get_disliked_papers() -> VectorStoreService.get_paper_embeddings_by_arxiv_ids() -> _hydrate_vectors_with_metadata() -> DatabaseService.save_user_interest_vector()`

### 输入

- Body：
  - `user_id`

### 输出

- `status`
- `message`
- `paper_count`
- `used_count`
- `milvus_used_count`
- `fallback_used_count`
- `unresolved_count`
- `liked_count`
- `disliked_count`
- `vector_dimension`
- `embedding_model`
- `cluster_count`
- `profile_mode`
- `weak_interest_pool`
- `weak_interest_pool_count`
- `cluster_summary`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：否
- 是否写入 like / dislike / collect：否
- 是否更新用户兴趣画像：是，更新 `user_interest_vectors`
- 是否更新 memory：否，未直接调用 `MemoryService`
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：是，缺失向量时会调用
- 是否访问 vector store：是
- 是否写数据库：是

### 异常 / fallback

- 没有 liked papers 时返回 `400`
- 没有可复用 liked embeddings 时返回 `500`
- 缺失向量时会尝试：
  - 从 arXiv / OAI 回填并写库
  - 再退化为基于本地论文文本重建 embedding
- 聚类失败时回退为 mean 向量
- 保存兴趣向量失败时返回 `500`

## 接口：GET `/user/interest-vector`

### 职责

读取已保存的用户兴趣向量。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] get_user_interest_vector"]
    B --> C["[Validate] 解析 query.user_id"]
    C --> D["[DB] get_user_interest_vector"]
    D --> E{"[Validate] 是否存在 ?"}
    E -->|是| F["[Response] interest_vector"]
    E -->|否| G["[Error] 404"]
    B -. 异常 .-> H["[Error] 500"]
    D -. 未发现专门 fallback .-> I["[Fallback] 未发现"]
```

### 关键调用链

`get_user_interest_vector() -> DatabaseService.get_user_interest_vector()`

### 输入

- Query：
  - `user_id`

### 输出

- `user_id`
- `vector_data`
- `paper_count`
- `embedding_model`
- `vector_dimension`
- `cluster_count`
- `profile_mode`
- `interest_clusters`
- `weak_interest_pool`
- `disliked_vector_data`
- `created_at`
- `updated_at`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：否
- 是否写入 like / dislike / collect：否
- 是否更新用户兴趣画像：否
- 是否更新 memory：否
- 是否实时计算推荐：否
- 是否读取缓存推荐：否
- 是否调用 embedding：否
- 是否访问 vector store：否
- 是否写数据库：否

### 异常 / fallback

- 兴趣向量不存在时返回 `404`
- 其他异常返回 `500`
- 未发现明确 fallback

## 接口：POST `/user/recommend-papers`

### 职责

根据用户兴趣向量、liked/disliked、研究画像、近期论文池和向量召回结果，实时生成个性化推荐结果。

### 处理流程图

```mermaid
flowchart TD
    A["前端 / 调用方"] --> B["[Router] recommend_papers"]
    B --> C["[Validate] 解析 user_id / top_n / max_age_months"]
    C --> D["[Recommendation] RecommendationService.recommend_papers"]
    D --> E["[DB] _get_or_refresh_interest_vector -> get_user_interest_vector / get_latest_user_signal_timestamp"]
    E --> F["[Fallback] 向量缺失或过期时 generate_user_interest_vector"]
    D --> G["[DB] get_user_preferences / get_liked_papers_with_details / get_user_research_profile / get_user_paper_action_map"]
    D --> H{"[Recommendation] 是否存在 interest_clusters ?"}
    H -->|是| I["[VectorStore] search_similar_papers(cluster recall)"]
    H -->|否| J["[Service] OAI recent pool recall"]
    I --> K["[Fallback] cluster recall 为空时 recent pool"]
    J --> L["[Service] _deduplicate_candidates(excluded_ids)"]
    K --> L
    L --> M{"[Validate] 候选是否为空 ?"}
    M -->|是| N["[Error] 400 No papers found for recommendation"]
    M -->|否| O["[Service] _materialize_candidate_papers_for_recommendation"]
    O --> O1["[DB] get_paper / add_paper / update_paper_embedding"]
    O --> O2["[VectorStore] get_paper_embeddings_by_arxiv_ids / insert_embeddings"]
    O --> O3["[Embedding] create_single_embedding"]
    D --> P["[VectorStore] get_paper_embeddings_by_arxiv_ids(候选向量复用)"]
    D --> Q["[Recommendation] _build_candidate_score"]
    Q --> Q1["[Embedding] 候选无现成向量时重算"]
    Q --> Q2["[Memory] _compute_profile_adjustment 使用研究画像"]
    D --> R["[Recommendation] _select_diverse_candidates"]
    R --> S["[Response] recommendations / recall_mode / research_profile / paper_actions"]
    B -. 其他异常 .-> T["[Error] 500"]
```

### 关键调用链

`recommend_papers() -> RecommendationService.recommend_papers() -> _get_or_refresh_interest_vector() -> DatabaseService.get_user_interest_vector()/get_latest_user_signal_timestamp()`

`recommend_papers() -> DatabaseService.get_user_preferences() -> get_liked_papers_with_details() -> _build_profile_signal_bundle()`

`recommend_papers() -> _fetch_cluster_recall_candidates() -> VectorStoreService.search_similar_papers()`

`recommend_papers() -> _fetch_recent_db_candidates() -> ArxivOaiDatabaseService.get_recent_papers()`

`recommend_papers() -> _materialize_candidate_papers_for_recommendation() -> DatabaseService.get_paper()/add_paper()/update_paper_embedding() -> VectorStoreService.get_paper_embeddings_by_arxiv_ids()/insert_embeddings() -> EmbeddingService.create_single_embedding()`

`recommend_papers() -> _build_candidate_score() -> _compute_profile_adjustment() -> _select_diverse_candidates()`

### 输入

- Body：
  - `user_id`
  - `top_n`
  - `max_age_months`

### 输出

- `status`
- `message`
- `total_found`
- `interest_profile_mode`
- `interest_cluster_count`
- `research_profile`
- `paper_actions`
- `recall_mode`
- `recommendations`

### 副作用

- 是否创建用户：否
- 是否更新用户信息：否
- 是否写入 like / dislike / collect：否
- 是否更新用户兴趣画像：可能，兴趣向量缺失或过期时会重建并写 `user_interest_vectors`
- 是否更新 memory：未直接调用 `MemoryService`，但会读取研究画像信号
- 是否实时计算推荐：是
- 是否读取缓存推荐：未发现
- 是否调用 embedding：是
- 是否访问 vector store：是
- 是否写数据库：可能，会补写 `arxiv_papers`、更新 `embedding_id`、重建 `user_interest_vectors`

### 异常 / fallback

- 用户兴趣向量缺失或过期时，自动重建
- cluster recall 为空时，回退到 recent paper pool
- 候选去重后为空时返回 `400`
- 候选物化时若缺少现成向量，会重新 embedding 或批量插入向量库
- 个别候选个性化打分失败时，会 warning 并退化为普通查询排序逻辑
- router 兜底返回 `500`

## 5. 推荐链路分析

1. 推荐是实时计算，还是读取预计算结果？
   - 基于真实代码判断，当前推荐是实时计算。
   - `POST /user/recommend-papers` 直接调用 `RecommendationService.recommend_papers()`，内部实时做兴趣向量读取/重建、候选召回、候选物化、打分与多样性选择。
   - 未发现单独的推荐结果缓存表或“读取已保存推荐结果”的逻辑。

2. 是否读取用户 liked papers？
   - 是。
   - 读取路径包括 `DatabaseService.get_liked_papers()`、`DatabaseService.get_liked_papers_with_details()`。

3. 是否使用用户 dislike 信息？
   - 是。
   - `generate_user_interest_vector()` 会读取 disliked papers，并生成 `disliked_vector_data`。
   - `recommend_papers()` 会把 disliked IDs 纳入 `excluded_ids`，并在打分阶段作为 `disliked_penalty` 参与惩罚。

4. 是否使用 collect 信息？
   - 未发现名为 `collect` 的独立表或独立逻辑。
   - 但发现通用 `user_paper_actions` 表，支持 `favorite`、`later`、`archived`、`read`、`note_saved` 等行为。
   - 如果前端把 collect 语义映射为 `favorite` 或 `later`，则这些行为会参与画像修正；代码里未发现 `collect` 这个 action_type。

5. 是否使用 embedding？
   - 是。
   - 用户兴趣向量生成、候选物化、候选打分都可能调用 `EmbeddingService.create_single_embedding()`。

6. 是否访问 vector store？
   - 是。
   - 主要通过 `VectorStoreService.get_paper_embeddings_by_arxiv_ids()`、`insert_single_embedding()`、`insert_embeddings()`、`search_similar_papers()`。

7. 是否使用 `MemoryService`？
   - 是，但不是在推荐主循环里做复杂会话记忆。
   - 主要用于：
     - like/dislike 后更新长期研究画像
     - paper action 后更新长期研究画像
     - research profile 读写
   - `recommend_papers()` 本身不直接调用 `MemoryService`，而是通过数据库读取 `user_research_profiles`。

8. 用户兴趣向量是如何生成的？
   - 先读取 liked / disliked 论文 ID。
   - 优先从 Milvus 读取对应论文向量。
   - 缺失时尝试：
     - 从 arXiv / OAI 回填论文并写入向量库
     - 再退化为基于本地论文文本重建 embedding
   - 对 liked 向量取均值；
   - 若存在 disliked 向量，则按 `liked_mean - negative_weight * disliked_mean` 做减权；
   - 最后做归一化，保存为主兴趣向量。
   - 当 liked 数量足够时，还会用 HDBSCAN 构建多兴趣簇，并保存 `interest_clusters` 与 `weak_interest_pool`。

9. 推荐结果是否保存到数据库？
   - 未发现。
   - 当前会保存的是用户兴趣向量、偏好、行为和研究画像，不会把最终推荐列表落库。

10. 是否存在冷启动处理？
   - 有，但比较有限。
   - 如果没有兴趣簇，会退化为 recent paper pool。
   - 如果兴趣向量缺失或过期，会自动重建。
   - 但如果用户根本没有 liked papers，`generate_user_interest_vector()` 会直接返回 `400`，因此严格意义上的“零偏好冷启动个性化推荐”未完全实现。

## 6. 设计观察

1. `user_router.py` 的职责整体比较清晰，确实聚焦在“用户偏好、画像、行为、推荐入口”这一组能力。
2. 用户管理、偏好管理、推荐逻辑在接口层是混在一起的，但这种混合更多是“入口聚合”而不是“逻辑杂糅”。真正复杂的推荐逻辑主要下沉在 `RecommendationService`。
3. Router 本身业务逻辑不算重，基本是参数解析和 service 转发。相对明显的逻辑只有：
   - `research-profile` 的 `model_dump(exclude_none=True)` 处理
   - `remove_paper_action` 的 404/500 区分
4. 推荐相关副作用不完全直观，尤其是：
   - like/dislike 不只是写偏好，还会写 `user_paper_actions`
   - like/dislike / paper-action 可能触发论文物化、embedding、向量库写入
   - 推荐主流程虽然不保存推荐结果，但可能补写论文表、向量映射和兴趣向量
5. 对后续个性化推荐系统的影响：
   - 当前链路已经具备“显式偏好 + 长期画像 + 向量召回 + 多样性重排”的雏形；
   - 但推荐结果未持久化，后续如果要做推荐解释、曝光反馈、A/B 或离线评估，可能需要额外的数据记录能力；
   - 当前没有独立 `users` 表，也没有真正的冷启动个性化兜底，这会影响后续更精细的人群建模与新用户体验。

## 7. 检查结论

- 文档文件已生成到 `docs/architecture/routers/user_router_flow.md`
- `user_router.py` 中全部 15 个接口已覆盖
- 每个接口都包含单独 Mermaid 流程图
- 使用的文件名、函数名、类名均来自真实代码
- 未发现凭空编造的推荐、memory、embedding、vector store 逻辑
- 未确认或未实现的点已明确标注为“未发现”
