<script setup lang="ts">
import { computed } from 'vue'
import PaperCard from '@/components/PaperCard.vue'
import type { AgentPaper, AgentPreferenceActionResult, ArxivSearchResponse } from '@/types/agent'
import type { Paper } from '@/types/paper'

const props = defineProps<{
  response: ArxivSearchResponse | null | undefined
}>()

const emit = defineEmits<{
  (event: 'view-detail', id: string): void
  (event: 'label', paper: Paper, label: 'liked' | 'disliked' | null): void
}>()

function normalizePaper(raw: AgentPaper): Paper {
  const arxivId = raw.arxiv_id || raw.arxivId || (raw.id ? String(raw.id).split('/').pop() : '')
  const authors = Array.isArray(raw.authors)
    ? raw.authors
    : String(raw.authors || '')
        .split(',')
        .map((author: string) => author.trim())
        .filter(Boolean)
  const categories = Array.isArray(raw.categories)
    ? raw.categories
    : String(raw.categories || '')
        .split(',')
        .map((category: string) => category.trim())
        .filter(Boolean)

  return {
    id: arxivId || raw.id || '',
    arxivId: arxivId || raw.id || '',
    title: raw.title || '',
    authors,
    summary: raw.abstract || raw.summary || '',
    publishedAt: raw.publishedAt || raw.published || raw.published_date || '',
    updatedAt: raw.updatedAt || raw.updated || '',
    categories,
    pdfUrl: raw.pdfUrl || raw.pdf_url || '',
    absUrl: raw.absUrl || raw.abs_url || raw.url || '',
    label: raw.label || null,
    query_match_score: raw.query_match_score,
    personalization_score: raw.personalization_score,
    final_score: raw.final_score,
    score_breakdown: raw.score_breakdown || null,
    matched_terms: raw.matched_terms || [],
    personalized_reason: raw.personalized_reason || null,
    match_reason: raw.match_reason || null,
    priority: raw.priority
  }
}

function formatJson(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function formatStepStatus(status: string) {
  if (status === 'running') return '进行中'
  if (status === 'success') return '成功'
  if (status === 'failed') return '失败'
  if (status === 'skipped') return '跳过'
  return status || 'unknown'
}

function getStepTagType(status: string) {
  if (status === 'running') return 'warning'
  if (status === 'success') return 'success'
  if (status === 'failed') return 'danger'
  return 'info'
}

function formatStepTitle(step: string) {
  const map: Record<string, string> = {
    intent_recognition: '意图识别',
    tool_argument_construction: '工具参数构造',
    search_tool_call: '搜索工具调用',
    search_result_check: '搜索结果检查',
    personalized_rerank: '个性化重排',
    final_answer_generation: '最终回答生成',
    agent_runtime: 'Agent 运行'
  }
  return map[step] || step || '未知步骤'
}

function formatToolCallStatus(status: string) {
  if (status === 'success') return '成功'
  if (status === 'failed') return '失败'
  return status || 'unknown'
}

const responseData = computed(() => props.response || ({} as ArxivSearchResponse))
const papers = computed(() => (responseData.value.papers || []).map(normalizePaper))
const steps = computed(() => responseData.value.steps || [])
const toolCalls = computed(() => responseData.value.tool_calls || [])
const warnings = computed(() => responseData.value.warnings || [])
const searchSpec = computed(() => responseData.value.search_spec || null)
const pendingAction = computed(() => responseData.value.pending_action || null)
const paperQaResult = computed(() => responseData.value.paper_qa_result || null)
const preferenceActionResult = computed(() => responseData.value.preference_action_result || null)
const streamingState = computed(() => responseData.value.streaming_state || null)

function handleViewDetail(id: string) {
  emit('view-detail', id)
}

function handleLabel(id: string, label: 'liked' | 'disliked' | null) {
  const paper = papers.value.find(item => item.id === id)
  if (!paper) return
  emit('label', paper, label)
}

function getPreferenceActionLabel(result: AgentPreferenceActionResult | null) {
  if (!result) return ''
  if (result.status === 'success' && result.action === 'like') return '已标记为感兴趣'
  if (result.status === 'success' && result.action === 'dislike') return '已标记为不感兴趣'
  if (result.status === 'success' && result.action === 'remove') return '已取消偏好标记'
  return result.message || ''
}
</script>

<template>
  <div class="agent-response-panel">
    <section class="agent-response-panel__hero">
      <div>
        <div class="section-label">最终回答区</div>
        <div class="agent-response-panel__answer">
          {{ responseData.answer || '暂无最终回答' }}
        </div>
      </div>

      <div class="agent-response-panel__meta">
        <div class="summary-chip">
          <span class="summary-chip__label">intent</span>
          <strong>{{ responseData.intent || '-' }}</strong>
        </div>
        <div class="summary-chip">
          <span class="summary-chip__label">steps</span>
          <strong>{{ steps.length }}</strong>
        </div>
        <div class="summary-chip">
          <span class="summary-chip__label">tool_calls</span>
          <strong>{{ toolCalls.length }}</strong>
        </div>
        <div class="summary-chip">
          <span class="summary-chip__label">warnings</span>
          <strong>{{ warnings.length }}</strong>
        </div>
        <div class="summary-chip">
          <span class="summary-chip__label">papers</span>
          <strong>{{ papers.length }}</strong>
        </div>
        <div v-if="streamingState" class="summary-chip">
          <span class="summary-chip__label">stream</span>
          <strong>{{ streamingState.event_type }} #{{ streamingState.sequence }}</strong>
        </div>
      </div>
    </section>

    <details class="agent-collapse">
      <summary class="agent-collapse__summary">
        <span>Agent Step Timeline</span>
        <span class="agent-collapse__count">{{ steps.length }}</span>
      </summary>
      <div v-if="!steps.length" class="empty-state">暂无步骤记录</div>
      <div v-else class="timeline-list">
        <article v-for="(step, index) in steps" :key="`${step.step}-${index}`" class="timeline-item">
          <div class="timeline-item__head">
            <strong>{{ formatStepTitle(step.step) }}</strong>
            <el-tag size="small" effect="plain" :type="getStepTagType(step.status)">
              {{ formatStepStatus(step.status) }}
            </el-tag>
          </div>
          <div class="timeline-item__action">{{ step.action }}</div>
          <details class="inline-details">
            <summary>查看输入 / 输出 / 错误</summary>
            <div v-if="Object.keys(step.inputs || {}).length" class="timeline-item__section">
              <div class="timeline-item__label">关键输入</div>
              <pre class="json-block">{{ formatJson(step.inputs) }}</pre>
            </div>
            <div v-if="Object.keys(step.outputs || {}).length" class="timeline-item__section">
              <div class="timeline-item__label">关键输出</div>
              <pre class="json-block">{{ formatJson(step.outputs) }}</pre>
            </div>
            <div v-if="step.error" class="timeline-item__error">
              {{ step.error }}
            </div>
          </details>
        </article>
      </div>
    </details>

    <details class="agent-collapse">
      <summary class="agent-collapse__summary">
        <span>Tool Call Card</span>
        <span class="agent-collapse__count">{{ toolCalls.length }}</span>
      </summary>
      <div v-if="!toolCalls.length" class="empty-state">暂无工具调用记录</div>
      <div v-else class="tool-call-list">
        <article v-for="(call, index) in toolCalls" :key="`${call.tool_name}-${index}`" class="tool-call-item">
          <div class="tool-call-top">
            <strong>{{ call.tool_name }}</strong>
            <el-tag
              size="small"
              :type="call.status === 'success' ? 'success' : call.status === 'running' ? 'warning' : 'danger'"
            >
              {{ call.status === 'running' ? '进行中' : formatToolCallStatus(call.status) }}
            </el-tag>
          </div>
          <div v-if="call.summary" class="tool-call-summary">{{ call.summary }}</div>
          <details class="inline-details">
            <summary>查看调用参数</summary>
            <pre class="json-block">{{ formatJson(call.arguments) }}</pre>
          </details>
          <details v-if="call.trace" class="inline-details">
            <summary>查看 trace</summary>
            <pre class="json-block">{{ formatJson(call.trace) }}</pre>
          </details>
          <details v-if="call.error" class="inline-details">
            <summary>查看错误</summary>
            <pre class="json-block error-block">{{ formatJson(call.error) }}</pre>
          </details>
        </article>
      </div>
    </details>

    <details class="agent-collapse">
      <summary class="agent-collapse__summary">
        <span>Search Spec</span>
        <span class="agent-collapse__count">{{ searchSpec ? 1 : 0 }}</span>
      </summary>
      <div v-if="!searchSpec" class="empty-state">暂无 search spec</div>
      <pre v-else class="json-block">{{ formatJson(searchSpec) }}</pre>
    </details>

    <details class="agent-collapse">
      <summary class="agent-collapse__summary">
        <span>Pending Action</span>
        <span class="agent-collapse__count">{{ pendingAction ? 1 : 0 }}</span>
      </summary>
      <div v-if="!pendingAction" class="empty-state">鏆傛棤寰呯‘璁や换鍔?</div>
      <pre v-else class="json-block">{{ formatJson(pendingAction) }}</pre>
    </details>

    <details class="agent-collapse">
      <summary class="agent-collapse__summary">
        <span>Paper QA Result</span>
        <span class="agent-collapse__count">{{ paperQaResult ? 1 : 0 }}</span>
      </summary>
      <div v-if="!paperQaResult" class="empty-state">鏆傛棤 paper qa 缁撴灉</div>
      <pre v-else class="json-block">{{ formatJson(paperQaResult) }}</pre>
    </details>

    <details v-if="preferenceActionResult" class="agent-collapse">
      <summary class="agent-collapse__summary">
        <span>Preference Action</span>
        <span class="agent-collapse__count">{{ preferenceActionResult.status }}</span>
      </summary>
      <div class="preference-result">
        <div class="preference-result__summary">{{ getPreferenceActionLabel(preferenceActionResult) }}</div>
        <pre class="json-block">{{ formatJson(preferenceActionResult) }}</pre>
      </div>
    </details>

    <details class="agent-collapse">
      <summary class="agent-collapse__summary">
        <span>Warnings</span>
        <span class="agent-collapse__count">{{ warnings.length }}</span>
      </summary>
      <div v-if="!warnings.length" class="empty-state">暂无 warning</div>
      <ul v-else class="warning-list">
        <li v-for="warning in warnings" :key="warning">
          {{ warning }}
        </li>
      </ul>
    </details>

    <section class="agent-response-panel__papers">
      <div class="section-label">Papers 结果展示</div>
      <div v-if="!papers.length" class="empty-state">暂无论文结果</div>
      <div v-else class="paper-list">
        <PaperCard
          v-for="paper in papers"
          :key="paper.id"
          :paper="paper"
          @view-detail="handleViewDetail"
          @label="handleLabel"
        />
      </div>
    </section>
  </div>
</template>

<style scoped>
.agent-response-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.agent-response-panel__hero {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
  border-radius: 20px;
  background: linear-gradient(180deg, #ffffff, #f8fbff);
  border: 1px solid rgba(148, 163, 184, 0.14);
}

.agent-response-panel__answer {
  margin-top: 8px;
  color: #1f2937;
  line-height: 1.85;
  font-size: 15px;
}

.agent-response-panel__meta {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
}

.summary-chip,
.agent-collapse,
.tool-call-item,
.timeline-item,
.agent-response-panel__papers {
  border-radius: 18px;
  background: linear-gradient(180deg, #ffffff, #f8fbff);
  border: 1px solid rgba(148, 163, 184, 0.14);
}

.summary-chip {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 72px;
  padding: 14px 16px;
}

.summary-chip__label,
.section-label,
.detail-section__label,
.timeline-item__label {
  font-size: 12px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.summary-chip strong {
  color: #0f172a;
  font-size: 15px;
}

.agent-collapse {
  padding: 0;
  overflow: hidden;
}

.agent-collapse__summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  cursor: pointer;
  list-style: none;
  font-weight: 700;
  color: #0f172a;
}

.agent-collapse__summary::-webkit-details-marker {
  display: none;
}

.agent-collapse__count {
  min-width: 28px;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.08);
  color: #334155;
  font-size: 12px;
  text-align: center;
}

.timeline-list,
.tool-call-list,
.paper-list,
.warning-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 0 16px 16px;
}

.timeline-item,
.tool-call-item {
  padding: 14px;
}

.timeline-item__head,
.tool-call-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.timeline-item__action,
.tool-call-summary {
  margin-top: 8px;
  color: #334155;
  line-height: 1.6;
}

.inline-details {
  margin-top: 10px;
}

.inline-details summary {
  cursor: pointer;
  color: #0f172a;
  font-size: 13px;
  font-weight: 600;
}

.timeline-item__section {
  margin-top: 10px;
}

.timeline-item__error {
  margin-top: 10px;
  color: #b91c1c;
  font-size: 13px;
}

.json-block {
  margin: 10px 0 0;
  padding: 12px;
  border-radius: 14px;
  background: #0f172a;
  color: #e2e8f0;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.6;
}

.error-block {
  background: #7f1d1d;
}

.warning-list {
  margin: 0;
  color: #92400e;
  padding-left: 34px;
}

.agent-response-panel__papers {
  padding: 16px;
}

.paper-list {
  padding: 0;
  margin-top: 12px;
}

.empty-state {
  padding: 0 16px 16px;
  color: #94a3b8;
  font-size: 13px;
}

@media (max-width: 1024px) {
  .agent-response-panel__meta {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .agent-response-panel__meta {
    grid-template-columns: 1fr;
  }
}
</style>
