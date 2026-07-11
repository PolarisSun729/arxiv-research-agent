# 行为画像只消费稳定兴趣簇

用户对单篇论文的 like/dislike 是论文级事实，不应直接升级为长期方向偏好。行为画像构建必须先刷新并复用推荐层的兴趣模型，只允许稳定兴趣簇中的论文贡献长期 positive/negative topics；弱兴趣信号、离群论文和样本不足的行为只能进入近期上下文、弱证据或调试记录。

## Status

accepted

## Considered Options

- **直接从全部 liked/disliked 论文聚合长期画像**：实现简单，但会让一次误点或离群论文污染长期画像。
- **让长期画像单独实现一套聚类逻辑**：能做门控，但会和推荐层的 HDBSCAN 参数、离群规则和向量回填逻辑分叉。
- **复用推荐层兴趣模型并在画像构建前刷新**：推荐层和画像层共享稳定兴趣建模，画像层只消费稳定簇摘要。

## Consequences

- 画像构建流程需要在收集证据后、生成画像 evidence card 前刷新兴趣模型。
- 缺失 embedding 时应优先物化并回填，只有回填失败才降级为 unresolved evidence。
- stable cluster papers 才会主动触发画像 evidence card；weak/outlier papers 默认只复用已有缓存，并记录跳过原因。
- 手动画像保持最高优先级。稳定行为簇与手动画像冲突时记录 profile conflict，不能自动覆盖用户显式声明。
- snapshot、job metrics 和 quality report 必须记录 stable/weak/unresolved 的门控结果，便于解释为什么某次行为没有进入长期画像。
