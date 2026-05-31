<script setup lang="ts">
import { computed } from 'vue'
import RagCitationList from '@/components/rag-chat/RagCitationList.vue'
import RagThoughtChain from '@/components/rag-chat/RagThoughtChain.vue'
import type { RetrievalDebug, RetrievalDebugChunk } from '@/api/papers'
import type { RagChatSource } from '@/types/ragChat'

const props = defineProps<{
  question?: string
  sources: RagChatSource[]
  retrievalDebug: RetrievalDebug | null
  traceDownloading?: boolean
  showClose?: boolean
  emptyTitle?: string
  emptyDescription?: string
}>()

const emit = defineEmits<{
  (event: 'download-trace', format: 'md' | 'json'): void
  (event: 'close'): void
}>()

function formatDebugNumber(value?: number | null) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '-'
}

function formatQueryList(values?: string[]) {
  if (!values || values.length === 0) {
    return '无'
  }
  return values.join(' | ')
}

function formatRouteScores(routeScores?: Record<string, number>) {
  const entries = Object.entries(routeScores || {})
  if (!entries.length) {
    return '无'
  }
  return entries.map(([route, score]) => `${route} ${formatDebugNumber(score)}`).join(' · ')
}

function describeChunk(chunk: RetrievalDebugChunk, scoreKey: 'score' | 'route_score' = 'score') {
  const score = chunk[scoreKey]
  return [
    `chunk ${chunk.chunk_id ?? '-'}`,
    `page ${chunk.page_number || chunk.page_range || '-'}`,
    `score ${formatDebugNumber(score)}`
  ].join(' · ')
}

const hasEvidence = computed(() => Boolean(props.question || props.sources.length || props.retrievalDebug))
</script>

<template>
  <div class="rag-evidence-panel">
    <template v-if="hasEvidence">
      <div class="rag-evidence-panel__head">
        <div>
          <div class="rag-evidence-panel__title">来源与调试</div>
          <div v-if="question" class="rag-evidence-panel__question">{{ question }}</div>
        </div>
        <div class="rag-evidence-panel__head-actions">
          <el-button
            size="small"
            text
            :loading="traceDownloading"
            @click="emit('download-trace', 'md')"
          >
            下载 Trace
          </el-button>
          <el-button
            v-if="retrievalDebug?.trace_export?.json"
            size="small"
            text
            :loading="traceDownloading"
            @click="emit('download-trace', 'json')"
          >
            JSON
          </el-button>
          <el-button v-if="showClose" size="small" text @click="emit('close')">关闭</el-button>
        </div>
      </div>

      <section class="rag-evidence-panel__section">
        <div class="rag-evidence-panel__section-title">参考来源</div>
        <RagCitationList :sources="sources" />
      </section>

      <section v-if="retrievalDebug" class="rag-evidence-panel__section">
        <div class="rag-evidence-panel__section-title">RAG 执行过程</div>
        <RagThoughtChain :question="question || retrievalDebug.original_query" :retrieval-debug="retrievalDebug" />
      </section>

      <section v-if="retrievalDebug?.query_rewrite" class="rag-evidence-panel__section">
        <div class="rag-evidence-panel__section-title">Query Rewrite</div>
        <div class="rag-evidence-panel__meta-row">
          <span>模型输出</span>
          <strong>{{ formatQueryList(retrievalDebug.query_rewrite.model_queries) }}</strong>
        </div>
        <div class="rag-evidence-panel__meta-row">
          <span>最终参与检索</span>
          <strong>{{ formatQueryList(retrievalDebug.query_rewrite.selected_queries) }}</strong>
        </div>
      </section>

      <section v-if="retrievalDebug?.keyword_search" class="rag-evidence-panel__section">
        <div class="rag-evidence-panel__section-title">Keyword Search</div>
        <div class="rag-evidence-panel__meta-row">
          <span>Queries</span>
          <strong>{{ formatQueryList(retrievalDebug.keyword_search.queries) }}</strong>
        </div>
        <div v-if="retrievalDebug.keyword_search.keywords?.length" class="rag-evidence-panel__meta-row">
          <span>Keywords</span>
          <strong>{{ retrievalDebug.keyword_search.keywords.join(', ') }}</strong>
        </div>
      </section>

      <section v-if="retrievalDebug?.fusion" class="rag-evidence-panel__section">
        <div class="rag-evidence-panel__section-title">RRF / Fusion</div>
        <div class="rag-evidence-panel__meta-row">
          <span>算法</span>
          <strong>{{ retrievalDebug.fusion.algorithm || 'pure_rrf' }}</strong>
        </div>
        <div class="rag-evidence-panel__meta-row">
          <span>RRF k</span>
          <strong>{{ retrievalDebug.fusion.rrf_k ?? '-' }}</strong>
        </div>
        <div class="rag-evidence-panel__meta-row">
          <span>路由权重</span>
          <strong>{{ formatRouteScores(retrievalDebug.fusion.route_weights) }}</strong>
        </div>
      </section>

      <section v-if="retrievalDebug?.final_chunks?.length" class="rag-evidence-panel__section">
        <div class="rag-evidence-panel__section-title">Final Chunks</div>
        <div class="rag-evidence-panel__chunk-list">
          <details
            v-for="(chunk, index) in retrievalDebug.final_chunks"
            :key="`final-${index}`"
            class="rag-evidence-panel__chunk"
            open
          >
            <summary class="rag-evidence-panel__chunk-summary">{{ index + 1 }}. {{ describeChunk(chunk) }}</summary>
            <div class="rag-evidence-panel__chunk-body">
              <div v-if="chunk.matched_routes?.length" class="rag-evidence-panel__chunk-meta">
                命中路由: {{ formatQueryList(chunk.matched_routes) }}
              </div>
              <div v-if="chunk.source_queries?.length" class="rag-evidence-panel__chunk-meta">
                来源 Queries: {{ formatQueryList(chunk.source_queries) }}
              </div>
              <div class="rag-evidence-panel__chunk-text">{{ chunk.content || chunk.preview || '-' }}</div>
            </div>
          </details>
        </div>
      </section>

      <section v-if="retrievalDebug?.routes && Object.keys(retrievalDebug.routes).length" class="rag-evidence-panel__section">
        <div class="rag-evidence-panel__section-title">多路召回</div>
        <div class="rag-evidence-panel__group-list">
          <details v-for="(chunks, routeName) in retrievalDebug.routes" :key="routeName" class="rag-evidence-panel__group" open>
            <summary class="rag-evidence-panel__group-title">
              <span>{{ routeName }}</span>
              <span>{{ chunks.length }} 条</span>
            </summary>
            <div v-if="chunks.length" class="rag-evidence-panel__chunk-list">
              <details
                v-for="(chunk, index) in chunks"
                :key="`${routeName}-${index}`"
                class="rag-evidence-panel__chunk"
              >
                <summary class="rag-evidence-panel__chunk-summary">
                  {{ index + 1 }}. {{ describeChunk(chunk, 'route_score') }}
                </summary>
                <div class="rag-evidence-panel__chunk-body">
                  <div v-if="chunk.route_scores && Object.keys(chunk.route_scores).length" class="rag-evidence-panel__chunk-meta">
                    路由分数: {{ formatRouteScores(chunk.route_scores) }}
                  </div>
                  <div class="rag-evidence-panel__chunk-text">{{ chunk.content || chunk.preview || '-' }}</div>
                </div>
              </details>
            </div>
            <div v-else class="rag-evidence-panel__empty-inline">无召回结果</div>
          </details>
        </div>
      </section>

      <section v-if="retrievalDebug?.stages && Object.keys(retrievalDebug.stages).length" class="rag-evidence-panel__section">
        <div class="rag-evidence-panel__section-title">检索阶段</div>
        <div class="rag-evidence-panel__group-list">
          <details v-for="(chunks, stageName) in retrievalDebug.stages" :key="stageName" class="rag-evidence-panel__group" open>
            <summary class="rag-evidence-panel__group-title">
              <span>{{ stageName }}</span>
              <span>{{ chunks.length }} 条</span>
            </summary>
            <div v-if="chunks.length" class="rag-evidence-panel__chunk-list">
              <details
                v-for="(chunk, index) in chunks"
                :key="`${stageName}-${index}`"
                class="rag-evidence-panel__chunk"
              >
                <summary class="rag-evidence-panel__chunk-summary">
                  {{ index + 1 }}. {{ describeChunk(chunk) }}
                </summary>
                <div class="rag-evidence-panel__chunk-body">
                  <div v-if="chunk.retrieval_route" class="rag-evidence-panel__chunk-meta">
                    来源路由: {{ chunk.retrieval_route }}
                  </div>
                  <div v-if="chunk.source_queries?.length" class="rag-evidence-panel__chunk-meta">
                    来源 Queries: {{ formatQueryList(chunk.source_queries) }}
                  </div>
                  <div class="rag-evidence-panel__chunk-text">{{ chunk.content || chunk.preview || '-' }}</div>
                </div>
              </details>
            </div>
            <div v-else class="rag-evidence-panel__empty-inline">无阶段数据</div>
          </details>
        </div>
      </section>
    </template>

    <div v-else class="rag-evidence-panel__empty">
      <div class="rag-evidence-panel__empty-title">
        {{ emptyTitle || '选择一条回答查看证据' }}
      </div>
      <div class="rag-evidence-panel__empty-text">
        {{ emptyDescription || '右侧会展示 sources、final chunks、routes、stages 和完整的 RAG 执行过程。' }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.rag-evidence-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.rag-evidence-panel__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.rag-evidence-panel__title {
  font-size: 18px;
  font-weight: 800;
  color: #0f172a;
}

.rag-evidence-panel__question {
  margin-top: 6px;
  color: #475569;
  font-size: 13px;
  line-height: 1.6;
}

.rag-evidence-panel__head-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.rag-evidence-panel__section {
  padding: 16px;
  border-radius: 20px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
}

.rag-evidence-panel__section-title {
  font-size: 15px;
  font-weight: 800;
  color: #0f172a;
  margin-bottom: 12px;
}

.rag-evidence-panel__meta-row {
  display: grid;
  grid-template-columns: 110px minmax(0, 1fr);
  gap: 12px;
  margin-bottom: 10px;
  color: #475569;
  font-size: 13px;
  line-height: 1.6;
}

.rag-evidence-panel__meta-row strong {
  color: #0f172a;
  font-weight: 600;
  word-break: break-word;
}

.rag-evidence-panel__group-list,
.rag-evidence-panel__chunk-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.rag-evidence-panel__group,
.rag-evidence-panel__chunk {
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: linear-gradient(180deg, #f8fbff, #ffffff);
  padding: 12px 14px;
}

.rag-evidence-panel__group-title,
.rag-evidence-panel__chunk-summary {
  cursor: pointer;
  list-style: none;
  color: #0f172a;
  font-size: 13px;
  font-weight: 700;
}

.rag-evidence-panel__group-title::-webkit-details-marker,
.rag-evidence-panel__chunk-summary::-webkit-details-marker {
  display: none;
}

.rag-evidence-panel__group-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.rag-evidence-panel__chunk-body {
  margin-top: 12px;
}

.rag-evidence-panel__chunk-meta {
  margin-bottom: 8px;
  color: #64748b;
  font-size: 12px;
  line-height: 1.6;
}

.rag-evidence-panel__chunk-text {
  color: #334155;
  font-size: 13px;
  line-height: 1.75;
  white-space: pre-wrap;
}

.rag-evidence-panel__empty-inline {
  margin-top: 10px;
  color: #64748b;
  font-size: 13px;
}

.rag-evidence-panel__empty {
  padding: 22px 20px;
  border-radius: 20px;
  border: 1px dashed rgba(148, 163, 184, 0.36);
  background: rgba(255, 255, 255, 0.88);
}

.rag-evidence-panel__empty-title {
  color: #0f172a;
  font-size: 16px;
  font-weight: 800;
}

.rag-evidence-panel__empty-text {
  margin-top: 8px;
  color: #64748b;
  font-size: 13px;
  line-height: 1.7;
}

@media (max-width: 900px) {
  .rag-evidence-panel__meta-row {
    grid-template-columns: 1fr;
    gap: 4px;
  }
}
</style>
