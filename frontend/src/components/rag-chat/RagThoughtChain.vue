<script setup lang="ts">
import { computed } from 'vue'
import type { RetrievalDebug, RetrievalDebugChunk } from '@/api/papers'
import { formatQueryList } from '@/utils/format'
import { buildResearchRoundViews, buildResearchStepViews, hasResearchProcess } from '@/utils/researchProcess'

const props = defineProps<{
  question: string
  retrievalDebug: RetrievalDebug | null
}>()

function getChunkCount(chunks?: RetrievalDebugChunk[]) {
  return Array.isArray(chunks) ? chunks.length : 0
}

function formatRouteSummary() {
  const routes = props.retrievalDebug?.routes || {}
  const entries = Object.entries(routes)
  if (!entries.length) {
    return '无召回结果'
  }
  return entries.map(([name, chunks]) => `${name}: ${chunks.length} 条`).join(' · ')
}

function formatFusionSummary() {
  const fusion = props.retrievalDebug?.fusion || {}
  const algorithm = typeof fusion.algorithm === 'string' ? fusion.algorithm : 'pure_rrf'
  const rrfK = fusion.rrf_k ?? '-'
  return `算法 ${algorithm} · RRF k ${rrfK}`
}

const steps = computed(() => {
  const debug = props.retrievalDebug
  const finalContextCount =
    getChunkCount(debug?.stages?.final_context_top15) || getChunkCount(debug?.final_chunks)
  // 检索管线产出的阶段键是 reranked_top30；保留 top15 兼容旧快照。
  const rerankCount = getChunkCount(debug?.stages?.reranked_top30) || getChunkCount(debug?.stages?.reranked_top15)

  return [
    {
      key: 'original',
      title: '原始问题',
      content: debug?.original_query || props.question || '无'
    },
    {
      key: 'rewrite',
      title: 'Query Rewrite',
      content: debug?.query_rewrite?.selected_queries?.length
        ? formatQueryList(debug.query_rewrite.selected_queries)
        : formatQueryList(debug?.rewritten_queries)
    },
    {
      key: 'hyde',
      title: 'HyDE',
      content: debug?.hyde?.text || debug?.hyde_text || '未启用'
    },
    {
      key: 'keyword',
      title: 'Keyword Search',
      content: formatQueryList(debug?.keyword_search?.queries)
    },
    {
      key: 'routes',
      title: '多路召回',
      content: formatRouteSummary()
    },
    {
      key: 'fusion',
      title: 'RRF / Fusion',
      content: formatFusionSummary()
    },
    {
      key: 'rerank',
      title: 'Rerank',
      content: rerankCount ? `进入 rerank 的 chunk 数: ${rerankCount}` : '无 rerank 数据'
    },
    {
      key: 'final-context',
      title: 'Final Context',
      content: finalContextCount ? `最终上下文 chunk 数: ${finalContextCount}` : '无最终上下文数据'
    },
    {
      key: 'answer',
      title: 'Answer Generation',
      content: finalContextCount
        ? `基于 ${finalContextCount} 条最终上下文生成答案`
        : '基于当前检索结果生成答案'
    }
  ]
})

// 研究引擎会多轮检索、起草和校验，单条检索时间线描述不了它：改为"研究动作 + 每轮检索阶段"两层展示。
const isResearch = computed(() => hasResearchProcess(props.retrievalDebug))
const researchSteps = computed(() => buildResearchStepViews(props.retrievalDebug?.research_steps))
const researchRounds = computed(() => buildResearchRoundViews(props.retrievalDebug?.research_rounds))
</script>

<template>
  <div v-if="isResearch" class="rag-thought-chain rag-thought-chain--research">
    <div class="rag-thought-chain__question">
      <div class="rag-thought-chain__title">原始问题</div>
      <div class="rag-thought-chain__content">{{ retrievalDebug?.original_query || question || '无' }}</div>
    </div>

    <div class="rag-thought-chain__group-title">研究动作</div>
    <el-timeline v-if="researchSteps.length">
      <el-timeline-item v-for="step in researchSteps" :key="step.key" placement="top" type="primary">
        <div class="rag-thought-chain__title">{{ step.title }}</div>
        <div v-if="step.content" class="rag-thought-chain__content">{{ step.content }}</div>
      </el-timeline-item>
    </el-timeline>
    <div v-else class="rag-thought-chain__content">无研究动作记录</div>

    <div class="rag-thought-chain__group-title">每轮检索</div>
    <details
      v-for="(round, index) in researchRounds"
      :key="round.key"
      class="rag-thought-chain__round"
      :open="index === researchRounds.length - 1"
    >
      <summary class="rag-thought-chain__round-summary">
        <span class="rag-thought-chain__title">{{ round.title }}</span>
        <span class="rag-thought-chain__round-subtitle">{{ round.subtitle }}</span>
      </summary>
      <el-timeline class="rag-thought-chain__round-body">
        <el-timeline-item v-for="stage in round.stages" :key="stage.key" placement="top" type="primary">
          <div class="rag-thought-chain__title">{{ stage.title }}</div>
          <div class="rag-thought-chain__content">{{ stage.content }}</div>
        </el-timeline-item>
      </el-timeline>
    </details>
    <div v-if="!researchRounds.length" class="rag-thought-chain__content">本次研究没有执行检索</div>
  </div>

  <el-timeline v-else class="rag-thought-chain">
    <el-timeline-item v-for="step in steps" :key="step.key" placement="top" size="large" type="primary">
      <div class="rag-thought-chain__title">{{ step.title }}</div>
      <div class="rag-thought-chain__content">{{ step.content }}</div>
    </el-timeline-item>
  </el-timeline>
</template>

<style scoped>
.rag-thought-chain {
  padding: 6px 4px 0;
}

.rag-thought-chain__title {
  color: #0f172a;
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 6px;
}

.rag-thought-chain__content {
  color: #475569;
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.rag-thought-chain--research {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.rag-thought-chain__group-title {
  color: #0f172a;
  font-size: 13px;
  font-weight: 800;
  margin-top: 4px;
}

.rag-thought-chain__round {
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: linear-gradient(180deg, #f8fbff, #ffffff);
  padding: 12px 14px;
}

.rag-thought-chain__round-summary {
  cursor: pointer;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.rag-thought-chain__round-summary::-webkit-details-marker {
  display: none;
}

.rag-thought-chain__round-subtitle {
  color: #64748b;
  font-size: 12px;
}

.rag-thought-chain__round-body {
  margin-top: 12px;
}
</style>
