# 配置说明

本文档整理当前项目中最常用、最容易踩坑的配置项，帮助你快速完成本地开发、联调和问题排查。

当前配置定义主要集中在：

- `backend/utils/config.py`

> 说明：项目以环境变量为主要配置入口。若代码中的默认值与你的环境不匹配，优先通过环境变量覆盖，而不是直接改源码。

## 1. 配置原则

建议遵循下面几条原则：

- **优先用环境变量覆盖默认值**，便于切换开发机、测试机和部署环境
- **不要把真实 API Key 提交到仓库**
- **先跑通最小链路，再补齐外部能力**，例如先确认前后端联通，再接 Milvus、Embedding、Rerank
- **按模块排查问题**，不要把所有能力同时接入后再定位错误

---

## 2. 最小开发配置

如果你的目标只是把项目先跑起来，建议先关注这几项：

| 变量名 | 用途 | 建议 |
| --- | --- | --- |
| `BACKEND_SERVICE_LOAD_MODE` | 控制后端加载模式 | 开发期可先用 `lazy` |
| `ARXIV_DATA_SOURCE` | arXiv 数据源类型 | 先保持 `local` |
| `ARXIV_PROXY_URL` | 访问 arXiv 的代理地址 | 没有代理时按你的环境修改 |
| `SQLITE_DATABASE_PATH` | 推荐数据库路径 | 保持默认或改为本地可写路径 |
| `OAI_SQLITE_DATABASE_PATH` | arXiv OAI 数据库路径 | 保持默认或改为本地可写路径 |
| `MILVUS_URI` | Milvus 服务地址 | 如果暂时不用向量库，可先不接完整链路 |

如果你的目标是体验完整问答和推荐链路，还需要额外关注：

- Embedding 相关配置
- Rerank 相关配置
- 模型 API Key
- Milvus 可用性

---

## 3. 核心配置

### 3.1 后端加载模式

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `BACKEND_SERVICE_LOAD_MODE` | `preload` | 控制后端服务是启动时预热，还是按需加载 |

可选值：

- `lazy`：按需加载，启动快，首次请求可能慢
- `preload`：启动时预热，启动慢，首次体验更稳定

建议：

- 本地开发联调：优先 `lazy`
- 演示或稳定体验：优先 `preload`

### 3.2 arXiv 数据源

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `ARXIV_DATA_SOURCE` | `local` | 指定 arXiv 查询和数据读取主要走哪种来源 |
| `ARXIV_PROXY_URL` | `http://127.0.0.1:7897` | 访问 arXiv 或相关远程资源时使用的代理 |

建议：

- 如果你本地没有代理，这一项通常需要改
- 如果访问 arXiv 超时，优先检查代理配置是否可用

---

## 4. 数据库与存储配置

### 4.1 SQLite

| 变量名 | 默认值 | 说明 |
 | --- | --- | --- |
 | `SQLITE_DATABASE_PATH` | `06-database/recommendation.db` | 推荐系统使用的 SQLite 数据库 |
 | `SQLITE_CHECK_SAME_THREAD` | `False` | SQLite 线程检查配置 |
 | `OAI_SQLITE_DATABASE_PATH` | `backend/06-database/arxiv_oai.db` 对应绝对路径 | arXiv OAI 本地数据库 |
 | `OAI_SQLITE_CHECK_SAME_THREAD` | `False` | OAI SQLite 线程检查配置 |

 建议：

 - 确保数据库目录存在且当前用户有写权限
 - 部署时尽量使用明确的绝对路径，减少相对路径带来的歧义

 ### 4.2 Milvus

 | 变量名 | 默认值 | 说明 |
 | --- | --- | --- |
 | `MILVUS_URI` | `http://localhost:19530` | Milvus 服务地址 |

 建议：

 - 如果问答、向量检索、推荐召回异常，优先检查 Milvus 是否可访问
 - 在跨机器或容器环境中，不要默认依赖 `localhost`

---

 ## 5. Embedding 配置

 当前项目中 Embedding 配置项较多，核心关注以下几项：

 | 变量名 | 默认值 | 说明 |
 | --- | --- | --- |
 | `EMBEDDING_PROVIDER` | `dashscope` | Embedding 提供方 |
 | `EMBEDDING_MODEL` | `qwen3-vl-embedding` | Embedding 模型名 |
 | `EMBEDDING_API_KEY` | 读取环境变量 | Embedding API Key |
 | `EMBEDDING_DASHSCOPE_API_KEY` | 读取环境变量 | DashScope Embedding Key |
 | `EMBEDDING_BASE_URL` | DashScope 默认地址 | Embedding 服务地址 |
 | `EMBEDDING_DIMENSION` | `2048` | 向量维度 |
 | `EMBEDDING_BATCH_SIZE` | `20` | 批量嵌入大小 |
 | `LOCAL_EMBEDDING_MODEL_PATH` | 仓库下 `00-models/...` | 本地 Embedding 模型目录 |
 | `LOCAL_EMBEDDING_MODEL_SCRIPTS_PATH` | 仓库下 `00-models/.../scripts` | 本地模型脚本目录 |
 | `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI Embedding 模型名 |
 | `OPENAI_EMBEDDING_BASE_URL` | 空 | OpenAI Embedding 服务地址 |

 建议：

 - 先明确你使用的是 **云端 Embedding** 还是 **本地 Embedding**
 - 如果使用云端服务，先确认 API Key 和 base URL
 - 如果使用本地模型，先确认模型目录实际存在
 - 向量维度要和向量库中的集合配置一致，否则可能导致写入或查询异常

---

 ## 6. Rerank 与检索相关配置

 README 中会提到 `RERANK_PROVIDER`，而完整实现细节建议结合代码进一步确认。对使用者来说，最重要的是理解：

 - Embedding 负责召回
 - Rerank 负责重排
 - 两者任何一环配置异常，都会影响最终问答质量

 另外，项目中还包含大量检索参数，例如：

 - chunk 长度与 overlap
 - 查询规划数量限制
 - 候选召回数量限制
 - rerank 候选上限
 - 最终上下文条数

 这类参数主要定义在 `backend/utils/config.py` 中的：

 - `CHUNKING_CONFIG`
 - `VECTOR_STORE_CONFIG`
 - `ENHANCED_RETRIEVAL_CONFIG`

 如果你现在的目标不是调优效果，而是先跑通功能，建议暂时不要大改这些参数。

---

 ## 7. 模型 API Key 配置

 当前项目涉及多类模型能力，常见变量包括：

 | 变量名 | 用途 |
 | --- | --- |
 | `OPENAI_API_KEY` | OpenAI 系列模型能力 |
 | `DEEPSEEK_API_KEY` | DeepSeek 系列模型能力 |
 | `QWEN_API_KEY` | Qwen 系列模型能力 |
 | `ALIYUN_API_KEY` | 阿里云 / DashScope 相关能力 |

 建议：

 - 不要把真实密钥硬编码进 README、脚本或前端代码
 - 本地开发建议通过环境变量注入
- 如果你更习惯看 Python 形式的配置样例，建议参考 `backend/utils/config.example.py`

---

 ## 8. 推荐系统相关配置

 推荐系统配置主要包括两类：

 ### 8.1 基础推荐参数

 例如：

 - `RECOMMENDATION_DEFAULT_TOP_N`
 - `RECOMMENDATION_DEFAULT_MAX_AGE_MONTHS`
 - `RECOMMENDATION_MIN_LIKED_PAPERS_FOR_CLUSTERING`
 - `RECOMMENDATION_MAX_INTEREST_CLUSTERS`

 这些参数决定：

 - 默认返回多少推荐结果
 - 推荐候选的时间范围
 - 何时开始聚类用户兴趣
 - 最多保留多少兴趣簇

 ### 8.2 聚类与打分参数

 例如：

 - `RECOMMENDATION_HDBSCAN_MIN_CLUSTER_SIZE`
 - `RECOMMENDATION_HDBSCAN_MIN_SAMPLES`
 - `RECOMMENDATION_SCORE_WEIGHT_SEMANTIC`
 - `RECOMMENDATION_SCORE_WEIGHT_CATEGORY`
 - `RECOMMENDATION_SCORE_WEIGHT_RECENCY`
 - `RECOMMENDATION_SCORE_WEIGHT_DISLIKED_PENALTY`

 这些参数更偏效果调优。建议：

 - 先保持默认值
 - 先验证推荐链路可用，再做效果调参

---

 ## 9. Memory 与会话相关配置

 项目中还包含一组与短期记忆、会话上下文相关的配置，例如：

 - `ENABLE_SHORT_TERM_MEMORY`
 - `ENABLE_PAPER_CHAT_SESSION`
 - `ENABLE_MEMORY_AWARE_RETRIEVAL`
 - `SHORT_TERM_MEMORY_MAX_TURNS`
 - `SHORT_TERM_MEMORY_MAX_CHARS`

 这些配置会影响：

 - 论文问答是否带会话上下文
 - 上下文保留多少轮
 - 检索是否参考历史对话

 如果你在调试“为什么连续追问表现异常”，这一组配置值得重点排查。

---

 ## 10. Docling 与文档处理配置

 文档解析和版面处理相关配置主要定义在 `DOCLING_CONFIG` 中，例如：

 - `DOCLING_OCR_ENABLED`
 - `DOCLING_ANNOTATED_PDF_EXPORT_ENABLED`
 - `DOCLING_IMAGES_SCALE`
 - 多个 legend / layout / tolerance 参数

 一般建议：

 - 如果你当前只关注检索链路，不必优先调整这组参数
 - 如果你在处理 PDF 解析质量、标注导出、表格说明等问题，再重点看这里

---

 ## 11. arXiv OAI 同步配置

 OAI 同步相关常用配置包括：

 | 变量名 | 默认值 | 说明 |
 | --- | --- | --- |
 | `ARXIV_OAI_ENDPOINT` | `https://oaipmh.arxiv.org/oai` | arXiv OAI-PMH 接口地址 |
 | `ARXIV_OAI_TARGET_CATEGORIES` | `cs.CL,cs.LG,cs.IR,cs.AI` | 目标分类列表 |
 | `ARXIV_OAI_EMBEDDING_BATCH_SIZE` | `20` | 同步时的嵌入批大小 |
 | `ARXIV_OAI_VECTOR_QUERY_BATCH_SIZE` | `100` | 向量查询批大小 |

 建议：

 - 如果同步速度慢，优先确认网络、代理和批大小
 - 如果本地数据量太大，先缩小分类范围验证链路

---

 ## 12. 分平台环境变量设置示例

 ### PowerShell

 ```powershell
 $env:BACKEND_SERVICE_LOAD_MODE = "lazy"
 $env:ARXIV_PROXY_URL = "http://127.0.0.1:7897"
 python main.py
 ```

 ### CMD

 ```cmd
 set BACKEND_SERVICE_LOAD_MODE=lazy
 set ARXIV_PROXY_URL=http://127.0.0.1:7897
 python main.py
 ```

 ### bash / zsh

 ```bash
 BACKEND_SERVICE_LOAD_MODE=lazy \
 ARXIV_PROXY_URL=http://127.0.0.1:7897 \
 python main.py
 ```

---

 ## 13. 常见排查顺序

 如果项目跑不通，建议按这个顺序排查：

 1. **先看前后端是否联通**
    - 后端是否监听 `8001`
    - 前端 `/api` 是否代理到正确地址

 2. **再看本地数据库路径是否正常**
    - SQLite 文件路径是否存在
    - 目录是否可写

 3. **再看外部服务是否可用**
    - Milvus 是否可连接
    - Embedding / Rerank / LLM 服务是否可调用

 4. **最后看效果参数是否合理**
    - chunk 大小
    - rerank 候选数
    - 推荐打分权重

 这样排查通常比一开始就盯着模型输出更高效。
