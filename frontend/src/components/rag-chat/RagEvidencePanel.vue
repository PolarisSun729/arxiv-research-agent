<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import RagCitationList from '@/components/rag-chat/RagCitationList.vue'
import RagThoughtChain from '@/components/rag-chat/RagThoughtChain.vue'
import type { RetrievalDebug, RetrievalDebugChunk } from '@/api/papers'
import type { RagChatSource } from '@/types/ragChat'
import { useUserContext } from '@/composables/useUserContext'
import { formatQueryList } from '@/utils/format'
import { hasResearchProcess } from '@/utils/researchProcess'

const { isAdmin } = useUserContext()

const props = defineProps<{
  question?: string
  sources: RagChatSource[]
  citedSourceIds?: string[]
  citationWarning?: string | null
  highlightedSourceId?: string | null
  retrievalDebug: RetrievalDebug | null
  traceDownloading?: boolean
  showClose?: boolean
  emptyTitle?: string
  emptyDescription?: string
}>()

const emit = defineEmits<{
  (event: 'download-trace', format: 'md' | 'json'): void
  (event: 'close'): void
  (event: 'select-source', sourceId: string): void
}>()

function formatDebugNumber(value?: number | null) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '-'
}

function formatRouteScores(routeScores?: Record<string, number>) {
  const entries = Object.entries(routeScores || {})
  if (!entries.length) {
    return '无'
  }
  return entries.map(([route, score]) => `${route} ${formatDebugNumber(score)}`).join(' · ')
}

function formatYesNo(value?: boolean | null) {
  if (value === true) return '是'
  if (value === false) return '否'
  return '-'
}

function formatAnyList(values?: Array<string | number>) {
  if (!values || values.length === 0) return '无'
  return values.map(value => String(value)).join(' | ')
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

const hasMemoryDetails = computed(() =>
  Boolean(props.retrievalDebug?.memory_modules || props.retrievalDebug?.question_contextualization || props.retrievalDebug?.memory_context)
)
// 研究引擎的快照没有单次检索的 routes / stages 等字段，此时不显示"检索细节"分页。
const hasRetrievalDetails = computed(() => {
  const debug = props.retrievalDebug
  if (!debug) return false
  return Boolean(
    debug.query_rewrite ||
      hasMemoryDetails.value ||
      debug.keyword_search ||
      debug.fusion ||
      debug.final_chunks?.length ||
      (debug.routes && Object.keys(debug.routes).length) ||
      (debug.stages && Object.keys(debug.stages).length)
  )
})

type PanelTab = 'sources' | 'process' | 'details'
const activeTab = ref<PanelTab>('sources')
const tabs = computed(() =>
  [
    { key: 'sources' as PanelTab, label: '引用来源', count: props.sources.length as number | null, visible: true },
    {
      key: 'process' as PanelTab,
      label: hasResearchProcess(props.retrievalDebug) ? '研究过程' : '执行过程',
      count: null,
      visible: Boolean(props.retrievalDebug)
    },
    { key: 'details' as PanelTab, label: '检索细节', count: null, visible: hasRetrievalDetails.value }
  ].filter(tab => tab.visible)
)

// 切换到另一轮回答时回到来源页；从回答里点引用角标时也要切回来源页才能定位。
watch(
  () => props.question,
  () => {
    activeTab.value = 'sources'
  }
)
watch(
  () => props.highlightedSourceId,
  sourceId => {
    if (sourceId) activeTab.value = 'sources'
  }
)
watch(tabs, visibleTabs => {
  if (!visibleTabs.some(tab => tab.key === activeTab.value)) activeTab.value = 'sources'
})
</script>

<template>
  <div class="evidence">
    <template v-if="hasEvidence">
      <header class="evidence-head">
        <div class="evidence-head__row">
          <span class="evidence-head__eyebrow">来源与调试</span>
          <div class="evidence-head__actions">
            <button
              v-if="isAdmin"
              type="button"
              class="evidence-link"
              :disabled="traceDownloading"
              @click="emit('download-trace', 'md')"
            >
              下载 Trace
            </button>
            <button
              v-if="isAdmin && retrievalDebug?.trace_export?.json"
              type="button"
              class="evidence-link"
              :disabled="traceDownloading"
              @click="emit('download-trace', 'json')"
            >
              JSON
            </button>
            <button v-if="showClose" type="button" class="evidence-link" @click="emit('close')">关闭</button>
          </div>
        </div>
        <p v-if="question" class="evidence-head__question" :title="question">{{ question }}</p>
      </header>

      <div v-if="tabs.length > 1" class="evidence-bar">
        <div class="evidence-tabs" role="tablist">
          <button
            v-for="tab in tabs"
            :key="tab.key"
            type="button"
            role="tab"
            class="evidence-tab"
            :class="{ 'evidence-tab--active': activeTab === tab.key }"
            :aria-selected="activeTab === tab.key"
            @click="activeTab = tab.key"
          >
            {{ tab.label }}
            <span v-if="tab.count !== null" class="evidence-tab__count">{{ tab.count }}</span>
          </button>
        </div>
      </div>

      <div v-show="activeTab === 'sources'" class="evidence-pane" role="tabpanel">
        <div v-if="citationWarning" class="evidence-warning">{{ citationWarning }}</div>
        <RagCitationList
          :sources="sources"
          :cited-source-ids="citedSourceIds || []"
          :highlighted-source-id="highlightedSourceId"
          @select-source="emit('select-source', $event)"
        />
      </div>

      <div v-if="retrievalDebug" v-show="activeTab === 'process'" class="evidence-pane" role="tabpanel">
        <RagThoughtChain :question="question || retrievalDebug.original_query" :retrieval-debug="retrievalDebug" />
      </div>

      <div v-if="hasRetrievalDetails && retrievalDebug" v-show="activeTab === 'details'" class="evidence-pane" role="tabpanel">
        <section v-if="retrievalDebug.query_rewrite" class="evidence-block">
          <div class="evidence-block__title">Query Rewrite</div>
          <dl class="evidence-dl">
            <dt>模型输出</dt>
            <dd>{{ formatQueryList(retrievalDebug.query_rewrite.model_queries) }}</dd>
            <dt>最终参与检索</dt>
            <dd>{{ formatQueryList(retrievalDebug.query_rewrite.selected_queries) }}</dd>
          </dl>
        </section>

        <section v-if="hasMemoryDetails" class="evidence-block">
          <div class="evidence-block__title">记忆模块</div>
          <dl class="evidence-dl">
            <template v-if="retrievalDebug.contextualized_question">
              <dt>改写后问题</dt>
              <dd>{{ retrievalDebug.contextualized_question }}</dd>
            </template>
            <template v-if="retrievalDebug.question_contextualization?.is_follow_up !== undefined">
              <dt>识别为追问</dt>
              <dd>{{ formatYesNo(retrievalDebug.question_contextualization?.is_follow_up) }}</dd>
            </template>
          </dl>

          <div v-if="retrievalDebug.memory_modules?.short_term_memory" class="evidence-module">
            <div class="evidence-module__name">short_term_memory</div>
            <dl class="evidence-dl">
              <dt>enabled</dt><dd>{{ formatYesNo(retrievalDebug.memory_modules.short_term_memory.enabled) }}</dd>
              <dt>applied</dt><dd>{{ formatYesNo(retrievalDebug.memory_modules.short_term_memory.applied) }}</dd>
              <dt>turns</dt><dd>{{ retrievalDebug.memory_modules.short_term_memory.used_turn_count ?? 0 }} / {{ retrievalDebug.memory_modules.short_term_memory.provided_turn_count ?? 0 }}</dd>
              <template v-if="retrievalDebug.memory_modules.short_term_memory.reason"><dt>reason</dt><dd>{{ retrievalDebug.memory_modules.short_term_memory.reason }}</dd></template>
              <template v-if="retrievalDebug.memory_modules.short_term_memory.fallback_reason"><dt>fallback</dt><dd>{{ retrievalDebug.memory_modules.short_term_memory.fallback_reason }}</dd></template>
            </dl>
          </div>

          <div v-if="retrievalDebug.memory_modules?.session" class="evidence-module">
            <div class="evidence-module__name">session</div>
            <dl class="evidence-dl">
              <dt>enabled</dt><dd>{{ formatYesNo(retrievalDebug.memory_modules.session.enabled) }}</dd>
              <dt>applied</dt><dd>{{ formatYesNo(retrievalDebug.memory_modules.session.applied) }}</dd>
              <dt>session_id</dt><dd>{{ retrievalDebug.memory_modules.session.session_id || '-' }}</dd>
              <template v-if="retrievalDebug.memory_modules.session.reason"><dt>reason</dt><dd>{{ retrievalDebug.memory_modules.session.reason }}</dd></template>
              <template v-if="retrievalDebug.memory_modules.session.fallback_reason"><dt>fallback</dt><dd>{{ retrievalDebug.memory_modules.session.fallback_reason }}</dd></template>
            </dl>
          </div>

          <div v-if="retrievalDebug.memory_modules?.memory_retrieval || retrievalDebug.memory_context" class="evidence-module">
            <div class="evidence-module__name">memory_retrieval</div>
            <dl class="evidence-dl">
              <dt>enabled</dt><dd>{{ formatYesNo(retrievalDebug.memory_modules?.memory_retrieval?.enabled ?? retrievalDebug.memory_context?.enabled) }}</dd>
              <dt>applied</dt><dd>{{ formatYesNo(retrievalDebug.memory_modules?.memory_retrieval?.applied) }}</dd>
              <template v-if="retrievalDebug.memory_context?.referenced_turn_ids?.length"><dt>turn ids</dt><dd>{{ formatAnyList(retrievalDebug.memory_context.referenced_turn_ids) }}</dd></template>
              <template v-if="retrievalDebug.memory_context?.referenced_source_ids?.length"><dt>source ids</dt><dd>{{ formatAnyList(retrievalDebug.memory_context.referenced_source_ids) }}</dd></template>
              <template v-if="retrievalDebug.memory_context?.query_keywords?.length"><dt>keywords</dt><dd>{{ formatAnyList(retrievalDebug.memory_context.query_keywords) }}</dd></template>
              <template v-if="retrievalDebug.memory_modules?.memory_retrieval?.reason || retrievalDebug.memory_context?.reason"><dt>reason</dt><dd>{{ retrievalDebug.memory_modules?.memory_retrieval?.reason || retrievalDebug.memory_context?.reason }}</dd></template>
              <template v-if="retrievalDebug.memory_modules?.memory_retrieval?.fallback_reason || retrievalDebug.memory_context?.fallback_reason"><dt>fallback</dt><dd>{{ retrievalDebug.memory_modules?.memory_retrieval?.fallback_reason || retrievalDebug.memory_context?.fallback_reason }}</dd></template>
            </dl>
          </div>

          <div v-if="retrievalDebug.memory_modules?.user_profile" class="evidence-module">
            <div class="evidence-module__name">user_profile</div>
            <dl class="evidence-dl">
              <dt>enabled</dt><dd>{{ formatYesNo(retrievalDebug.memory_modules.user_profile.enabled) }}</dd>
              <dt>applied</dt><dd>{{ formatYesNo(retrievalDebug.memory_modules.user_profile.applied) }}</dd>
              <template v-if="retrievalDebug.memory_modules.user_profile.reason"><dt>reason</dt><dd>{{ retrievalDebug.memory_modules.user_profile.reason }}</dd></template>
              <template v-if="retrievalDebug.memory_modules.user_profile.fallback_reason"><dt>fallback</dt><dd>{{ retrievalDebug.memory_modules.user_profile.fallback_reason }}</dd></template>
            </dl>
          </div>
        </section>

        <section v-if="retrievalDebug.keyword_search" class="evidence-block">
          <div class="evidence-block__title">Keyword Search</div>
          <dl class="evidence-dl">
            <dt>Queries</dt>
            <dd>{{ formatQueryList(retrievalDebug.keyword_search.queries) }}</dd>
            <template v-if="retrievalDebug.keyword_search.keywords?.length">
              <dt>Keywords</dt>
              <dd>{{ retrievalDebug.keyword_search.keywords.join(', ') }}</dd>
            </template>
          </dl>
        </section>

        <section v-if="retrievalDebug.fusion" class="evidence-block">
          <div class="evidence-block__title">RRF / Fusion</div>
          <dl class="evidence-dl">
            <dt>算法</dt>
            <dd>{{ retrievalDebug.fusion.algorithm || 'pure_rrf' }}</dd>
            <dt>RRF k</dt>
            <dd>{{ retrievalDebug.fusion.rrf_k ?? '-' }}</dd>
            <dt>路由权重</dt>
            <dd>{{ formatRouteScores(retrievalDebug.fusion.route_weights) }}</dd>
          </dl>
        </section>

        <section v-if="retrievalDebug.final_chunks?.length" class="evidence-block">
          <div class="evidence-block__title">Final Chunks <span class="evidence-block__count">{{ retrievalDebug.final_chunks.length }}</span></div>
          <details v-for="(chunk, index) in retrievalDebug.final_chunks" :key="`final-${index}`" class="evidence-chunk">
            <summary>{{ index + 1 }}. {{ describeChunk(chunk) }}</summary>
            <div class="evidence-chunk__body">
              <div v-if="chunk.matched_routes?.length" class="evidence-chunk__meta">命中路由：{{ formatQueryList(chunk.matched_routes) }}</div>
              <div v-if="chunk.source_queries?.length" class="evidence-chunk__meta">来源 Queries：{{ formatQueryList(chunk.source_queries) }}</div>
              <div class="evidence-chunk__text">{{ chunk.content || chunk.preview || '-' }}</div>
            </div>
          </details>
        </section>

        <section v-if="retrievalDebug.routes && Object.keys(retrievalDebug.routes).length" class="evidence-block">
          <div class="evidence-block__title">多路召回</div>
          <details v-for="(chunks, routeName) in retrievalDebug.routes" :key="routeName" class="evidence-chunk">
            <summary>{{ routeName }} <span class="evidence-block__count">{{ chunks.length }}</span></summary>
            <div class="evidence-chunk__body">
              <details v-for="(chunk, index) in chunks" :key="`${routeName}-${index}`" class="evidence-chunk evidence-chunk--nested">
                <summary>{{ index + 1 }}. {{ describeChunk(chunk, 'route_score') }}</summary>
                <div class="evidence-chunk__body">
                  <div v-if="chunk.route_scores && Object.keys(chunk.route_scores).length" class="evidence-chunk__meta">路由分数：{{ formatRouteScores(chunk.route_scores) }}</div>
                  <div class="evidence-chunk__text">{{ chunk.content || chunk.preview || '-' }}</div>
                </div>
              </details>
              <div v-if="!chunks.length" class="evidence-chunk__meta">无召回结果</div>
            </div>
          </details>
        </section>

        <section v-if="retrievalDebug.stages && Object.keys(retrievalDebug.stages).length" class="evidence-block">
          <div class="evidence-block__title">检索阶段</div>
          <details v-for="(chunks, stageName) in retrievalDebug.stages" :key="stageName" class="evidence-chunk">
            <summary>{{ stageName }} <span class="evidence-block__count">{{ chunks.length }}</span></summary>
            <div class="evidence-chunk__body">
              <details v-for="(chunk, index) in chunks" :key="`${stageName}-${index}`" class="evidence-chunk evidence-chunk--nested">
                <summary>{{ index + 1 }}. {{ describeChunk(chunk) }}</summary>
                <div class="evidence-chunk__body">
                  <div v-if="chunk.retrieval_route" class="evidence-chunk__meta">来源路由：{{ chunk.retrieval_route }}</div>
                  <div v-if="chunk.source_queries?.length" class="evidence-chunk__meta">来源 Queries：{{ formatQueryList(chunk.source_queries) }}</div>
                  <div class="evidence-chunk__text">{{ chunk.content || chunk.preview || '-' }}</div>
                </div>
              </details>
              <div v-if="!chunks.length" class="evidence-chunk__meta">无阶段数据</div>
            </div>
          </details>
        </section>
      </div>
    </template>

    <div v-else class="evidence-empty">
      <div class="evidence-empty__title">
        {{ emptyTitle || '选择一条回答查看证据' }}
      </div>
      <div class="evidence-empty__text">
        {{ emptyDescription || '点击回答下方的「查看来源与调试」或正文里的引用角标，这里会展示引用原文和完整的检索 / 研究过程。' }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.evidence {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

/* 分页栏吸附在滚动容器顶部，长列表里也能随时切换。 */
.evidence-bar {
  position: sticky;
  /* --evidence-inset 是外层滚动容器的内边距，吸附栏要贴满容器顶部和两侧。 */
  top: calc(-1 * var(--evidence-inset, 20px));
  z-index: 2;
  margin: -6px calc(-1 * var(--evidence-inset, 20px)) 0;
  padding: 6px var(--evidence-inset, 20px) 10px;
  background: var(--color-surface);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--color-surface-sunken);
}

.evidence-head__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.evidence-head__eyebrow {
  color: var(--color-text);
  font-size: 15px;
  font-weight: 700;
}

.evidence-head__actions {
  display: flex;
  align-items: center;
  gap: 4px;
}

.evidence-link {
  padding: 3px 8px;
  border: 0;
  border-radius: 6px;
  background: none;
  color: var(--color-text-3);
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
}

.evidence-link:hover:not(:disabled) {
  background: var(--color-surface-sunken);
  color: var(--color-primary);
}

.evidence-link:disabled {
  cursor: wait;
  opacity: 0.6;
}

.evidence-head__question {
  display: -webkit-box;
  margin: 6px 0 0;
  overflow: hidden;
  color: var(--color-text-2);
  font-size: 13px;
  line-height: 1.6;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.evidence-tabs {
  display: flex;
  gap: 2px;
  padding: 3px;
  border-radius: 10px;
  background: var(--color-surface-sunken);
}

.evidence-tab {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  flex: 1;
  padding: 6px 8px;
  border: 0;
  border-radius: 8px;
  background: none;
  color: var(--color-text-3);
  cursor: pointer;
  font-family: inherit;
  font-size: 13px;
  font-weight: 500;
  transition: background 0.15s, color 0.15s;
}

.evidence-tab:hover {
  color: var(--color-text);
}

.evidence-tab--active {
  background: var(--color-surface);
  color: var(--color-text);
  font-weight: 600;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}

.evidence-tab__count {
  padding: 0 6px;
  border-radius: 999px;
  background: var(--color-border);
  color: var(--color-text-2);
  font-size: 11px;
  line-height: 17px;
}

.evidence-tab--active .evidence-tab__count {
  background: var(--color-primary-soft);
  color: var(--color-primary);
}

.evidence-pane {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.evidence-warning {
  padding: 9px 12px;
  border-radius: 10px;
  background: var(--color-warning-soft);
  border: 1px solid var(--color-warning-border);
  color: var(--color-warning);
  font-size: 12.5px;
  line-height: 1.6;
}

.evidence-block {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--color-surface-sunken);
}

.evidence-block:last-child {
  padding-bottom: 0;
  border-bottom: 0;
}

.evidence-block__title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text);
  font-size: 13.5px;
  font-weight: 600;
}

.evidence-block__count {
  padding: 0 7px;
  border-radius: 999px;
  background: var(--color-surface-sunken);
  color: var(--color-text-3);
  font-size: 11px;
  font-weight: 600;
  line-height: 18px;
}

.evidence-dl {
  display: grid;
  grid-template-columns: 92px minmax(0, 1fr);
  gap: 6px 12px;
  margin: 0;
}

.evidence-dl dt {
  color: var(--color-text-4);
  font-size: 12px;
  line-height: 1.65;
}

.evidence-dl dd {
  margin: 0;
  color: var(--color-text-2);
  font-size: 12.5px;
  line-height: 1.65;
  word-break: break-word;
}

.evidence-module {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--color-surface-muted);
}

.evidence-module__name {
  color: var(--color-text-2);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  font-weight: 600;
}

.evidence-chunk {
  border: 1px solid var(--color-border);
  border-radius: 10px;
  background: var(--color-surface);
}

.evidence-chunk--nested {
  border-color: var(--color-surface-sunken);
  background: var(--color-surface-muted);
}

.evidence-chunk summary {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  color: var(--color-text-2);
  cursor: pointer;
  font-size: 12.5px;
  font-weight: 500;
  list-style: none;
}

.evidence-chunk summary::-webkit-details-marker {
  display: none;
}

.evidence-chunk[open] > summary {
  color: var(--color-text);
  font-weight: 600;
}

.evidence-chunk__body {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 0 12px 10px;
}

.evidence-chunk__meta {
  color: var(--color-text-4);
  font-size: 12px;
  line-height: 1.6;
}

.evidence-chunk__text {
  color: var(--color-text-2);
  font-size: 12.5px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.evidence-empty {
  padding: 22px 20px;
  border-radius: var(--radius-lg);
  border: 1px dashed var(--color-border-strong);
  background: var(--color-surface-muted);
}

.evidence-empty__title {
  color: var(--color-text);
  font-size: 15px;
  font-weight: 700;
}

.evidence-empty__text {
  margin-top: 8px;
  color: var(--color-text-3);
  font-size: 13px;
  line-height: 1.7;
}

@media (max-width: 900px) {
  .evidence-dl {
    grid-template-columns: 1fr;
    gap: 2px;
  }
}
</style>
