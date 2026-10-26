<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { RetrievalDebug, RetrievalDebugChunk } from '@/api/papers'
import { formatQueryList } from '@/utils/format'
import {
  buildResearchRoundViews,
  buildResearchStepViews,
  buildResearchSummaryView,
  hasResearchProcess,
  splitStepDetail
} from '@/utils/researchProcess'

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

function formatHyde() {
  const debug = props.retrievalDebug
  const text = debug?.hyde?.text || debug?.hyde_text
  if (text) return text
  // 开启但生成失败时 text 为空，显示失败原因而不是“未启用”
  if (debug?.hyde?.enabled) return `生成失败（${debug.hyde.status || 'error'}），HyDE 路由已跳过`
  return '未启用'
}

const steps = computed(() => {
  const debug = props.retrievalDebug
  const finalContextCount =
    getChunkCount(debug?.stages?.final_context_top15) || getChunkCount(debug?.final_chunks)
  // 检索管线产出的阶段键是 reranked_top30；保留 top15 兼容旧快照。
  const rerankCount = getChunkCount(debug?.stages?.reranked_top30) || getChunkCount(debug?.stages?.reranked_top15)

  return [
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
      content: formatHyde()
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
const summary = computed(() => buildResearchSummaryView(props.retrievalDebug?.research_summary))
const researchSteps = computed(() =>
  buildResearchStepViews(props.retrievalDebug?.research_steps).map(step => ({ ...step, ...splitStepDetail(step.content) }))
)
const researchRounds = computed(() => buildResearchRoundViews(props.retrievalDebug?.research_rounds))

// 决策理由是模型的长段英文说明，默认收成三行，点开再看全文。
const expandedReasons = ref<Set<string>>(new Set())
const openRounds = ref<Set<string>>(new Set())

watch(
  () => props.retrievalDebug,
  () => {
    expandedReasons.value = new Set()
    const last = researchRounds.value[researchRounds.value.length - 1]
    openRounds.value = new Set(last ? [last.key] : [])
  },
  { immediate: true }
)

function toggled(set: Set<string>, key: string) {
  const next = new Set(set)
  if (next.has(key)) next.delete(key)
  else next.add(key)
  return next
}

function toggleReason(key: string) {
  expandedReasons.value = toggled(expandedReasons.value, key)
}

function toggleRound(key: string) {
  openRounds.value = toggled(openRounds.value, key)
}

const STEP_TONES: Record<string, string> = {
  search: 'blue',
  filter: 'slate',
  draft: 'violet',
  extract: 'slate',
  verify: 'green',
  restore: 'amber',
  fallback: 'amber',
  rejected: 'red',
  done: 'green'
}

function stepTone(kind?: string) {
  return STEP_TONES[kind || ''] || 'blue'
}

// Final Context 每行形如 "chunk 20 · text · p9 · 0.7813 · 正文预览"（分数可能缺省）：
// 开头连续的 chunk 号 / 类型 / 页码 / 分数是定位信息，其余是预览。
const CONTEXT_HEAD_PART = /^(chunk \S+|[a-z_]+|p[\d-]+|\d+(\.\d+)?)$/

// 快速模式不检索：整篇论文一次交给模型，过程只有"读取全文"和"生成"两段。
const isFast = computed(() => props.retrievalDebug?.engine === 'fast_fulltext')

function formatMs(value?: number | null) {
  if (typeof value !== 'number') return '-'
  return value >= 1000 ? `${(value / 1000).toFixed(1)} 秒` : `${Math.round(value)} 毫秒`
}

const fastSteps = computed(() => {
  const debug = props.retrievalDebug
  const fulltext = debug?.fulltext || {}
  const input = debug?.input || {}
  const usage = debug?.usage || {}
  const cachedTokens = usage.input_tokens_details?.cached_tokens
  const invalidIds = debug?.invalid_citation_ids || []
  return [
    {
      key: 'fulltext',
      title: '读取全文',
      content: `${fulltext.page_count ?? '-'} 页 · ${fulltext.char_count ?? '-'} 字符 · ${fulltext.image_count ?? 0} 张图表` +
        (typeof fulltext.parse_seconds === 'number' ? ` · 解析 ${fulltext.parse_seconds.toFixed(1)} 秒` : '')
    },
    { key: 'wait', title: '等待全文', content: formatMs(debug?.wait_fulltext_ms) },
    {
      key: 'input',
      title: '模型输入',
      content: `${input.image_count ?? 0} 张图片` +
        (typeof input.request_bytes === 'number' ? ` · 请求 ${(input.request_bytes / 1024 / 1024).toFixed(1)} MB` : '') +
        (input.images_downscaled ? ' · 图片已缩小' : '') +
        ` · 带最近 ${debug?.history_turn_count ?? 0} 轮对话`
    },
    {
      key: 'tokens',
      title: 'Token',
      content: typeof usage.input_tokens === 'number'
        ? `输入 ${usage.input_tokens}（缓存命中 ${cachedTokens ?? 0}）· 输出 ${usage.output_tokens ?? '-'}`
        : '无用量数据'
    },
    { key: 'first-token', title: '首字耗时', content: formatMs(debug?.first_token_ms) },
    { key: 'generation', title: '生成耗时', content: formatMs(debug?.generation_ms) },
    {
      key: 'citations',
      title: '引用整理',
      content: invalidIds.length ? `去掉 ${invalidIds.length} 个不存在的引用：${invalidIds.join(', ')}` : '引用均指向论文原文块'
    }
  ]
})

function parseContextLines(content: string) {
  return content
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => {
      const parts = line.split(' · ')
      let headCount = 0
      while (headCount < Math.min(4, parts.length - 1) && CONTEXT_HEAD_PART.test(parts[headCount])) headCount += 1
      return parts[0].startsWith('chunk ') && headCount >= 2
        ? { head: parts.slice(0, headCount).join(' · '), text: parts.slice(headCount).join(' · ') }
        : { head: '', text: line }
    })
}
</script>

<template>
  <div v-if="isResearch" class="chain">
    <div class="chain-question">
      <span class="chain-label">研究问题</span>
      <p>{{ retrievalDebug?.original_query || question || '无' }}</p>
    </div>

    <div v-if="summary" class="chain-summary">
      <div class="chain-summary__outcome" :class="`chain-summary__outcome--${summary.outcome}`">
        {{ summary.outcomeLabel }}
      </div>
      <div class="chain-summary__stats">
        <div v-for="stat in summary.stats" :key="stat.label" class="chain-summary__stat">
          <strong>{{ stat.value }}</strong>
          <span>{{ stat.label }}</span>
        </div>
      </div>
    </div>

    <section class="chain-section">
      <div class="chain-label">研究动作</div>
      <ol v-if="researchSteps.length" class="chain-steps">
        <li v-for="step in researchSteps" :key="step.key" class="chain-step" :class="`chain-step--${stepTone(step.kind)}`">
          <span class="chain-step__dot" aria-hidden="true" />
          <div class="chain-step__body">
            <div class="chain-step__title">{{ step.title }}</div>
            <div v-if="step.facts.length" class="chain-step__facts">
              <span v-for="fact in step.facts" :key="fact" class="chain-chip">{{ fact }}</span>
            </div>
            <button
              v-if="step.reason"
              type="button"
              class="chain-step__reason"
              :class="{ 'chain-step__reason--open': expandedReasons.has(step.key) }"
              :title="expandedReasons.has(step.key) ? '收起' : '展开全文'"
              @click="toggleReason(step.key)"
            >
              <span class="chain-step__reason-text">
                <span class="chain-step__reason-label">决策理由</span>
                {{ step.reason }}
              </span>
            </button>
          </div>
        </li>
      </ol>
      <div v-else class="chain-empty">无研究动作记录</div>
    </section>

    <section class="chain-section">
      <div class="chain-label">每轮检索</div>
      <div v-if="researchRounds.length" class="chain-rounds">
        <div
          v-for="round in researchRounds"
          :key="round.key"
          class="chain-round"
          :class="{ 'chain-round--open': openRounds.has(round.key) }"
        >
          <button
            type="button"
            class="chain-round__head"
            :aria-expanded="openRounds.has(round.key)"
            @click="toggleRound(round.key)"
          >
            <span class="chain-round__titles">
              <span class="chain-round__title">{{ round.title }}</span>
              <span class="chain-round__subtitle" :title="round.subtitle">{{ round.subtitle }}</span>
            </span>
            <span class="chain-chevron" aria-hidden="true" />
          </button>
          <dl v-if="openRounds.has(round.key)" class="chain-stages">
            <template v-for="stage in round.stages" :key="stage.key">
              <template v-if="stage.key === 'final-context' && parseContextLines(stage.content).some(line => line.head)">
                <dt class="chain-stages__wide">{{ stage.title }}</dt>
                <dd class="chain-stages__wide">
                  <ol class="chain-context">
                    <li v-for="(line, index) in parseContextLines(stage.content)" :key="index">
                      <span v-if="line.head" class="chain-context__head">{{ line.head }}</span>
                      <span class="chain-context__text">{{ line.text }}</span>
                    </li>
                  </ol>
                </dd>
              </template>
              <template v-else>
                <dt>{{ stage.title }}</dt>
                <dd v-if="stage.content.includes(' | ')">
                  <ul class="chain-queries">
                    <li v-for="query in stage.content.split(' | ')" :key="query">{{ query }}</li>
                  </ul>
                </dd>
                <dd v-else>{{ stage.content }}</dd>
              </template>
            </template>
          </dl>
        </div>
      </div>
      <div v-else class="chain-empty">本次研究没有执行检索</div>
    </section>
  </div>

  <div v-else-if="isFast" class="chain">
    <div class="chain-question">
      <span class="chain-label">原始问题</span>
      <p>{{ question || '无' }}</p>
    </div>
    <dl class="chain-stages chain-stages--flat">
      <template v-for="step in fastSteps" :key="step.key">
        <dt>{{ step.title }}</dt>
        <dd>{{ step.content }}</dd>
      </template>
    </dl>
  </div>

  <div v-else class="chain">
    <div class="chain-question">
      <span class="chain-label">原始问题</span>
      <p>{{ retrievalDebug?.original_query || question || '无' }}</p>
    </div>
    <dl class="chain-stages chain-stages--flat">
      <template v-for="step in steps" :key="step.key">
        <dt>{{ step.title }}</dt>
        <dd>{{ step.content }}</dd>
      </template>
    </dl>
  </div>
</template>

<style scoped>
.chain {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.chain-label {
  color: var(--color-text-3);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
}

.chain-question p {
  margin: 6px 0 0;
  color: var(--color-text);
  font-size: 13.5px;
  line-height: 1.65;
}

.chain-summary {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px 14px;
  border-radius: 14px;
  background: var(--color-surface-muted);
  border: 1px solid var(--color-surface-sunken);
}

.chain-summary__outcome {
  align-self: flex-start;
  padding: 2px 10px;
  border-radius: 999px;
  background: var(--color-border);
  color: var(--color-text-2);
  font-size: 12px;
  font-weight: 600;
}

.chain-summary__outcome--completed {
  background: var(--color-success-soft);
  color: #15803d;
}

.chain-summary__outcome--partial {
  background: var(--color-warning-soft);
  color: var(--color-warning);
}

.chain-summary__outcome--abstained {
  background: var(--color-danger-soft);
  color: #b91c1c;
}

.chain-summary__stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.chain-summary__stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.chain-summary__stat strong {
  color: var(--color-text);
  font-size: 16px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.chain-summary__stat span {
  color: var(--color-text-4);
  font-size: 12px;
}

.chain-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.chain-steps {
  margin: 0;
  padding: 0;
  list-style: none;
}

.chain-step {
  --tone: var(--color-primary);
  position: relative;
  display: flex;
  gap: 12px;
  padding-bottom: 14px;
}

.chain-step::before {
  content: '';
  position: absolute;
  top: 14px;
  bottom: -2px;
  left: 4px;
  width: 1px;
  background: var(--color-border);
}

.chain-step:last-child {
  padding-bottom: 0;
}

.chain-step:last-child::before {
  display: none;
}

.chain-step--slate { --tone: var(--color-text-4); }
.chain-step--violet { --tone: #8b5cf6; }
.chain-step--green { --tone: var(--color-success); }
.chain-step--amber { --tone: #f59e0b; }
.chain-step--red { --tone: var(--color-danger); }

.chain-step__dot {
  flex: none;
  width: 9px;
  height: 9px;
  margin-top: 5px;
  border-radius: 50%;
  background: var(--tone);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--tone) 18%, transparent);
}

.chain-step__body {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  flex: 1;
}

.chain-step__title {
  color: var(--color-text);
  font-size: 13.5px;
  font-weight: 600;
  line-height: 1.45;
}

.chain-step__facts {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.chain-chip {
  padding: 2px 8px;
  border-radius: 6px;
  background: var(--color-surface-sunken);
  color: var(--color-text-2);
  font-size: 12px;
  line-height: 1.6;
}

.chain-step__reason {
  padding: 8px 10px;
  border: 0;
  border-left: 2px solid var(--color-border);
  border-radius: 0 8px 8px 0;
  background: var(--color-surface-muted);
  color: var(--color-text-3);
  cursor: pointer;
  font-family: inherit;
  font-size: 12.5px;
  line-height: 1.65;
  text-align: left;
}

/* 行数截断放在内层：外层的内边距会让第四行露出半截。 */
.chain-step__reason-text {
  display: -webkit-box;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
}

.chain-step__reason:hover {
  border-left-color: var(--color-border-strong);
  color: var(--color-text-2);
}

.chain-step__reason--open .chain-step__reason-text {
  display: block;
}

.chain-step__reason-label {
  margin-right: 6px;
  color: var(--color-text-2);
  font-weight: 600;
}

.chain-rounds {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chain-round {
  border: 1px solid var(--color-border);
  border-radius: 14px;
  background: var(--color-surface);
}

.chain-round__head {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 11px 14px;
  border: 0;
  background: none;
  cursor: pointer;
  text-align: left;
  font-family: inherit;
}

.chain-round__titles {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  flex: 1;
}

.chain-round__title {
  color: var(--color-text);
  font-size: 13.5px;
  font-weight: 600;
}

.chain-round__subtitle {
  display: -webkit-box;
  overflow: hidden;
  color: var(--color-text-4);
  font-size: 12px;
  line-height: 1.55;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.chain-queries {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.chain-queries li {
  position: relative;
  padding-left: 12px;
}

.chain-queries li::before {
  content: '';
  position: absolute;
  top: 0.75em;
  left: 2px;
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--color-border-strong);
}

.chain-chevron {
  flex: none;
  width: 7px;
  height: 7px;
  margin: 0 4px;
  border-right: 1.5px solid var(--color-text-4);
  border-bottom: 1.5px solid var(--color-text-4);
  transform: rotate(-45deg);
  transition: transform 0.2s;
}

.chain-round--open .chain-chevron {
  transform: rotate(45deg);
}

.chain-stages {
  display: grid;
  grid-template-columns: 92px minmax(0, 1fr);
  gap: 8px 12px;
  margin: 0;
  padding: 12px 14px 14px;
  border-top: 1px solid var(--color-surface-sunken);
}

.chain-stages--flat {
  padding: 0;
  border-top: 0;
}

.chain-stages dt {
  color: var(--color-text-4);
  font-size: 12px;
  line-height: 1.65;
}

.chain-stages dd {
  margin: 0;
  color: var(--color-text-2);
  font-size: 12.5px;
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
}

.chain-stages__wide {
  grid-column: 1 / -1;
}

.chain-context {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 2px 0 0;
  padding: 0;
  list-style: none;
}

.chain-context li {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px 10px;
  border-radius: 8px;
  background: var(--color-surface-muted);
}

.chain-context__head {
  color: var(--color-text-3);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
}

.chain-context__text {
  overflow: hidden;
  color: var(--color-text-2);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chain-empty {
  color: var(--color-text-4);
  font-size: 13px;
}
</style>
