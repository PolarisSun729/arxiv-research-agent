<script setup lang="ts">
import { computed } from 'vue'
import type { RetrievalDebug, RetrievalDebugChunk } from '@/api/papers'

const props = defineProps<{
  question: string
  retrievalDebug: RetrievalDebug | null
}>()

function getChunkCount(chunks?: RetrievalDebugChunk[]) {
  return Array.isArray(chunks) ? chunks.length : 0
}

function formatQueryList(values?: string[]) {
  if (!values || values.length === 0) {
    return '无'
  }
  return values.join(' | ')
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
  const rerankCount = getChunkCount(debug?.stages?.reranked_top15)

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
</script>

<template>
  <el-timeline class="rag-thought-chain">
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
</style>
