<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { listPaperIndexes } from '@/api/adminIndexes'
import type { PaperIndexSummary } from '@/api/adminIndexes'
import { runRetrievalDebug } from '@/api/adminRetrievalDebug'
import type { DebugChunkRow, RetrievalDebugRequest, RetrievalDebugRun } from '@/api/adminRetrievalDebug'
import { getErrorMessage } from '@/api/errors'
import RetrievalChunkDetail from '@/components/retrieval-debug/RetrievalChunkDetail.vue'
import { OUTCOME_LABELS, buildDebugExport, chunkOutcome, debugExportFilename, fusedKeepLimit, routeLabel, routeOverlap, routeStats } from '@/utils/retrievalDebug'
import type { ChunkOutcomeCode } from '@/utils/retrievalDebug'

type SwitchKey = 'enable_query_rewrite' | 'enable_hyde' | 'enable_keyword_search' | 'enable_table_structured_route' | 'enable_llm_rerank' | 'enable_context_expansion'

const SWITCHES: Array<{ key: SwitchKey; label: string }> = [
  { key: 'enable_query_rewrite', label: '查询改写' },
  { key: 'enable_hyde', label: 'HyDE' },
  { key: 'enable_keyword_search', label: '关键词' },
  { key: 'enable_table_structured_route', label: '表格结构化' },
  { key: 'enable_llm_rerank', label: 'LLM rerank' },
  { key: 'enable_context_expansion', label: '上下文扩展' }
]
const OUTCOME_FILTERS: Array<{ value: ChunkOutcomeCode | 'all'; label: string }> = [
  { value: 'all', label: '全部' },
  ...(['final', 'fusion_cut', 'rerank_dropped', 'not_selected', 'expansion'] as const).map(value => ({ value, label: OUTCOME_LABELS[value] }))
]
// 名次表里融合之后的三个阶段列；表头可点击，按该列名次升序排。
const STAGE_COLUMNS = [
  { value: 'fused', label: '融合' },
  { value: 'rerank', label: 'rerank' },
  { value: 'final', label: '输出' }
]
const ROUTE_STATUS_LABELS: Record<string, string> = { ok: '正常', disabled: '关闭', timeout: '超时', error: '失败', skipped: '跳过', inactive: '未启用' }
const QUERY_DROP_REASONS: Record<string, string> = { same_as_original: '与原问或译文相同', duplicate: '重复', not_english: '不是英文' }
// 改写路由没有查询时说明原因；error 由下方的「改写失败」单独展示。
const REWRITE_SKIP_REASONS: Record<string, string> = {
  covered_by_original: '改写结果与原问译文相同，改写路由跳过',
  invalid_output: '改写结果都不合格（如不是英文），改写路由跳过',
  disabled: '改写已关闭',
}

const route = useRoute()
const router = useRouter()

// ---- 运行参数：论文同步到地址栏，刷新后仍是同一篇；HyDE 默认关，其余按常用组合打开 ----
const form = ref<RetrievalDebugRequest & Record<SwitchKey, boolean>>({
  arxiv_id: String(route.query.paper || ''),
  question: '',
  top_k: null,
  enable_query_rewrite: true,
  enable_hyde: false,
  enable_keyword_search: true,
  enable_table_structured_route: true,
  enable_llm_rerank: true,
  enable_context_expansion: true
})

const paperOptions = ref<PaperIndexSummary[]>([])
const loadingPapers = ref(false)

async function searchPapers(query: string) {
  loadingPapers.value = true
  try {
    const result = await listPaperIndexes({ q: query.trim(), status: 'indexed', limit: 30 })
    paperOptions.value = result.items
  } catch (error) { ElMessage.error(getErrorMessage(error, '论文列表读取失败')) }
  finally { loadingPapers.value = false }
}

function paperLabel(item: PaperIndexSummary) {
  return item.title ? `${item.arxiv_id} · ${item.title}` : item.arxiv_id
}

const running = ref(false)
const run = ref<RetrievalDebugRun | null>(null)
// 这次运行实际用的请求；表单之后可能被改动，导出要对得上结果。
const runRequest = ref<RetrievalDebugRequest | null>(null)

async function submit() {
  const question = form.value.question.trim()
  if (!form.value.arxiv_id) return void ElMessage.warning('先选一篇已建索引的论文')
  if (!question) return void ElMessage.warning('输入要调试的问题')
  void router.replace({ query: { paper: form.value.arxiv_id } })
  running.value = true
  try {
    const request = { ...form.value, question }
    run.value = await runRetrievalDebug(request)
    runRequest.value = request
    selectedKey.value = ''
    outcomeFilter.value = 'all'
    routeFilter.value = ''
    uniqueOnly.value = false
    sortBy.value = 'fused'
    sortDesc.value = false
  } catch (error) { ElMessage.error(getErrorMessage(error, '检索调试运行失败')) }
  finally { running.value = false }
}

function exportRun() {
  if (!run.value || !runRequest.value) return
  const exportedAt = new Date()
  const payload = buildDebugExport(run.value, runRequest.value, exportedAt)
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
  const link = document.createElement('a')
  link.href = url
  link.download = debugExportFilename(run.value.arxiv_id, exportedAt)
  link.click()
  URL.revokeObjectURL(url)
}

// ---- 各路概览 ----
// 关闭或这次没召回任何 chunk 的路由只在概览里列出，不占名次表的列。
const activeRoutes = computed(() => (run.value?.routes || []).filter(item => item.enabled && item.status !== 'disabled' && item.candidate_count > 0).map(item => item.name))
const statsByRoute = computed(() => run.value ? Object.fromEntries(routeStats(run.value, activeRoutes.value).map(item => [item.name, item])) : {})
const overlapRoutes = computed(() => (run.value?.routes || []).filter(item => item.candidate_count > 0).map(item => item.name))
const overlap = computed(() => run.value ? routeOverlap(run.value, overlapRoutes.value) : [])
const finalCount = computed(() => run.value?.chunks.filter(row => row.final_rank !== null).length ?? 0)
const expansionCount = computed(() => run.value ? run.value.chunks.filter(row => chunkOutcome(row, run.value!).code === 'expansion').length : 0)
const keepLimit = computed(() => run.value ? fusedKeepLimit(run.value) : 0)

const droppedCandidates = computed(() => (run.value?.query.candidates || []).filter(item => !item.selected))

// ---- 名次表 ----
const outcomeFilter = ref<ChunkOutcomeCode | 'all'>('all')
const routeFilter = ref('')
const uniqueOnly = ref(false)
const sortBy = ref('fused')
const sortDesc = ref(false)
const selectedKey = ref('')

const rows = computed(() => {
  if (!run.value) return []
  const current = run.value
  const decorated = current.chunks.map(row => ({ row, outcome: chunkOutcome(row, current) }))
  const filtered = decorated.filter(({ row, outcome }) => {
    if (outcomeFilter.value !== 'all' && outcome.code !== outcomeFilter.value) return false
    if (routeFilter.value && !(routeFilter.value in row.route_ranks)) return false
    if (uniqueOnly.value && Object.keys(row.route_ranks).length !== 1) return false
    return true
  })
  const rankOf = (row: DebugChunkRow) => {
    if (sortBy.value === 'rerank') return row.rerank_rank
    if (sortBy.value === 'final') return row.final_rank
    if (sortBy.value.startsWith('route:')) return row.route_ranks[sortBy.value.slice(6)] ?? null
    return row.fused_rank
  }
  // 倒序只翻转有名次的 chunk；没进这一列的始终排在最后，否则倒序时最上面全是空格。
  const direction = sortDesc.value ? -1 : 1
  return filtered
    .map(item => ({ ...item, sortRank: rankOf(item.row) }))
    .sort((a, b) => {
      if (a.sortRank === null || b.sortRank === null) return Number(a.sortRank === null) - Number(b.sortRank === null)
      return (a.sortRank - b.sortRank) * direction
    })
})

// 点表头：换列时从升序开始，再点同一列在升序和倒序之间切换。
function toggleSort(key: string) {
  sortDesc.value = sortBy.value === key ? !sortDesc.value : false
  sortBy.value = key
}
function ariaSort(key: string) {
  if (sortBy.value !== key) return undefined
  return sortDesc.value ? 'descending' : 'ascending'
}

// 按融合名次升序时，在保留条数之后画一条截断线，线下的 chunk 不会进入 rerank。
const cutIndex = computed(() => sortBy.value === 'fused' && !sortDesc.value ? rows.value.findIndex(item => (item.row.fused_rank ?? Infinity) > keepLimit.value) : -1)

const selectedRow = computed(() => run.value?.chunks.find(row => row.key === selectedKey.value) || null)

const rankScale = computed(() => Math.max(10, run.value?.config.recall_candidate_limit ?? 30))
function rankStyle(rank: number | null | undefined) {
  if (rank === null || rank === undefined) return {}
  const strength = Math.max(0.06, 1 - (rank - 1) / rankScale.value)
  return { background: `color-mix(in srgb, var(--color-primary) ${Math.round(strength * 42)}%, transparent)` }
}

function overlapStyle(value: number | null) {
  if (value === null) return {}
  return { background: `color-mix(in srgb, var(--color-chart-2) ${Math.round(value * 60)}%, transparent)` }
}

function routeCellTitle(row: DebugChunkRow, name: string) {
  const vote = row.rrf_votes[name]
  const score = row.route_scores[name]
  const parts = [`${routeLabel(name)} 第 ${row.route_ranks[name]} 名`]
  if (score !== null && score !== undefined) parts.push(`原始分 ${score.toFixed(3)}`)
  if (vote?.vote !== null && vote?.vote !== undefined) parts.push(`票 ${vote.vote.toFixed(5)}`)
  return parts.join(' · ')
}

function percent(value: number) {
  return `${Math.round(value * 100)}%`
}

function fixed(value: number | null | undefined, digits: number) {
  return value === null || value === undefined ? '—' : value.toFixed(digits)
}

onMounted(() => { void searchPapers(form.value.arxiv_id) })
</script>

<template>
  <div class="page page--wide">
    <header class="page-header">
      <div class="page-header__text">
        <h1 class="page-title">检索调试</h1>
        <p class="page-desc">对一篇论文跑一次单轮检索，逐个 chunk 对照各路召回名次、融合投票、rerank 和检索输出，用来判断多路召回和融合哪里需要调。检索输出是交给下游的候选池，最终进 prompt 的部分还会再经预算规划器裁剪。</p>
      </div>
    </header>

    <form class="surface surface-pad run-form" @submit.prevent="submit">
      <div class="run-form__main">
        <el-select
          v-model="form.arxiv_id"
          class="run-form__paper"
          filterable
          remote
          :remote-method="searchPapers"
          :loading="loadingPapers"
          placeholder="搜索已建索引的论文（arXiv ID 或标题）"
          aria-label="论文"
        >
          <el-option v-for="item in paperOptions" :key="item.arxiv_id" :label="paperLabel(item)" :value="item.arxiv_id" />
        </el-select>
        <el-input v-model="form.question" class="run-form__question" placeholder="要调试的问题" aria-label="问题" clearable />
        <el-button type="primary" native-type="submit" :loading="running">运行</el-button>
      </div>
      <div class="run-form__switches">
        <el-checkbox v-for="item in SWITCHES" :key="item.key" v-model="form[item.key]" :label="item.label" />
        <span class="run-form__topk">
          <span class="muted">top_k</span>
          <el-input-number v-model="form.top_k" :min="1" :max="50" size="small" controls-position="right" placeholder="默认" :value-on-clear="null" />
        </span>
      </div>
    </form>

    <template v-if="run">
      <div class="summary">
        <span><strong>{{ run.arxiv_id }}</strong> {{ run.title }}</span>
        <span class="muted">耗时 {{ (run.elapsed_ms / 1000).toFixed(1) }}s</span>
        <span class="muted">共 {{ run.chunks.length }} 个 chunk · 检索输出 {{ finalCount }} 个（其中扩展补入 {{ expansionCount }}）· top_k {{ run.config.effective_top_k ?? '—' }}</span>
        <span class="muted">融合保留 {{ keepLimit }} · rrf_k {{ run.fusion.rrf_k ?? '—' }}</span>
        <span :class="['chip', run.rerank.applied ? 'chip--success' : run.rerank.enabled ? 'chip--warning' : '']">
          rerank {{ run.rerank.applied ? `已生效 · ${run.rerank.mode}` : run.rerank.enabled ? `未生效 · ${run.rerank.reason || run.rerank.mode}` : '关闭' }}
        </span>
      </div>

      <div class="panels">
        <section class="surface surface-pad">
          <h2 class="block-title">查询规划</h2>
          <dl class="kv">
            <div>
              <dt>意图</dt>
              <dd>
                {{ run.query.intent.main_intent || '—' }}
                <span v-if="run.query.intent.sub_intents?.length" class="muted">（{{ run.query.intent.sub_intents.join('、') }}）</span>
                <span v-if="run.query.intent.confidence !== null" class="muted"> · 置信 {{ fixed(run.query.intent.confidence, 2) }}</span>
              </dd>
            </div>
            <div v-if="run.query.original_translation?.query">
              <dt>原问译文</dt>
              <dd>
                {{ run.query.original_translation.query }}
                <span v-if="run.query.original_translation.status === 'error'" class="muted">· 翻译失败，沿用原问：{{ run.query.original_translation.error }}</span>
              </dd>
            </div>
            <div>
              <dt>改写问句</dt>
              <dd>
                <ol v-if="run.query.selected_queries.length" class="query-list">
                  <li v-for="query in run.query.selected_queries" :key="query">{{ query }}</li>
                </ol>
                <p v-else-if="REWRITE_SKIP_REASONS[run.query.rewrite_status]" class="muted">{{ REWRITE_SKIP_REASONS[run.query.rewrite_status] }}</p>
                <p v-if="run.query.rewrite_error" class="muted">改写失败，改写路由跳过：{{ run.query.rewrite_error }}</p>
                <details v-if="droppedCandidates.length" class="dropped">
                  <summary class="muted">未采用的候选 {{ droppedCandidates.length }} 条</summary>
                  <ul>
                    <li v-for="item in droppedCandidates" :key="`${item.source}-${item.query}`">
                      {{ item.query || '（空）' }} <span class="muted">· {{ QUERY_DROP_REASONS[item.reason] || item.reason }}</span>
                    </li>
                  </ul>
                </details>
              </dd>
            </div>
            <div v-if="run.query.keywords.length">
              <dt>关键词</dt>
              <dd class="chip-row"><span v-for="word in run.query.keywords" :key="word" class="chip">{{ word }}</span></dd>
            </div>
            <div v-if="run.query.hyde_text">
              <dt>HyDE</dt>
              <dd class="clamp-3">{{ run.query.hyde_text }}</dd>
            </div>
            <div v-if="run.query.rerank_query">
              <dt>rerank 查询</dt>
              <dd>{{ run.query.rerank_query }}</dd>
            </div>
          </dl>
        </section>

        <section class="surface surface-pad">
          <h2 class="block-title">各路概览</h2>
          <div class="table-scroll">
            <table class="grid-table">
              <thead>
                <tr>
                  <th>路由</th><th>状态</th><th>耗时</th><th>候选</th><th>权重</th>
                  <th title="只有这一路召回到的 chunk 数">独有</th>
                  <th title="经融合进入检索输出的 chunk 中，这一路召回到的个数">入选</th>
                  <th title="经融合进入检索输出的 chunk 的融合票里，这一路所占比例">票占比</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="item in run.routes" :key="item.name" :class="{ 'is-off': !activeRoutes.includes(item.name) }">
                  <td>{{ routeLabel(item.name) }}</td>
                  <td :title="item.error || item.fallback_reason">
                    <span :class="['chip', item.status === 'ok' ? 'chip--success' : ['timeout', 'error'].includes(item.status) ? 'chip--danger' : '']">
                      {{ ROUTE_STATUS_LABELS[item.status] || item.status }}
                    </span>
                  </td>
                  <td>{{ item.latency_ms ? `${Math.round(item.latency_ms)}ms` : '—' }}</td>
                  <td>{{ item.candidate_count }}</td>
                  <td>{{ fixed(item.weight, 2) }}</td>
                  <td>{{ statsByRoute[item.name]?.unique ?? '—' }}</td>
                  <td>{{ statsByRoute[item.name]?.finalHits ?? '—' }}</td>
                  <td>{{ statsByRoute[item.name] ? percent(statsByRoute[item.name].finalVoteShare) : '—' }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <template v-if="overlapRoutes.length > 1">
            <h3 class="sub-title">召回重叠（Jaccard）</h3>
            <div class="table-scroll">
              <table class="overlap">
                <thead>
                  <tr><th /><th v-for="name in overlapRoutes" :key="name">{{ routeLabel(name) }}</th></tr>
                </thead>
                <tbody>
                  <tr v-for="(name, i) in overlapRoutes" :key="name">
                    <th>{{ routeLabel(name) }}</th>
                    <td v-for="(value, j) in overlap[i]" :key="j" :style="i === j ? {} : overlapStyle(value)">
                      {{ i === j ? '—' : value === null ? '' : value.toFixed(2) }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p class="help">两路重叠高、独有又接近 0，说明后一路基本是在给已有结果重复投票，相当于变相加大了这批 chunk 的权重。</p>
          </template>
        </section>
      </div>

      <section class="surface matrix">
        <div class="matrix__toolbar">
          <h2 class="block-title">排名矩阵</h2>
          <el-radio-group v-model="outcomeFilter" size="small">
            <el-radio-button v-for="item in OUTCOME_FILTERS" :key="item.value" :value="item.value">{{ item.label }}</el-radio-button>
          </el-radio-group>
          <el-select v-model="routeFilter" size="small" clearable placeholder="命中路由" class="matrix__select" aria-label="命中路由">
            <el-option v-for="name in activeRoutes" :key="name" :label="routeLabel(name)" :value="name" />
          </el-select>
          <el-checkbox v-model="uniqueOnly" label="只看单路命中" size="small" />
          <el-select v-model="sortBy" size="small" class="matrix__select" aria-label="排序" @change="sortDesc = false">
            <el-option label="按融合名次" value="fused" />
            <el-option label="按 rerank 名次" value="rerank" />
            <el-option label="按输出名次" value="final" />
            <el-option v-for="name in activeRoutes" :key="name" :label="`按${routeLabel(name)}名次`" :value="`route:${name}`" />
          </el-select>
          <span class="muted">{{ rows.length }} 行</span>
          <el-button size="small" class="matrix__export" title="导出本次运行的完整结果（全部 chunk，不受筛选影响）" @click="exportRun">导出 JSON</el-button>
        </div>

        <div class="table-scroll matrix__scroll">
          <table class="grid-table rank-table">
            <thead>
              <tr>
                <th class="col-chunk">chunk</th>
                <th v-for="name in activeRoutes" :key="name" :aria-sort="ariaSort(`route:${name}`)">
                  <button type="button" class="sort-head" :class="{ 'is-active': sortBy === `route:${name}`, 'is-desc': sortDesc }" @click="toggleSort(`route:${name}`)">{{ routeLabel(name) }}</button>
                </th>
                <th v-for="stage in STAGE_COLUMNS" :key="stage.value" class="col-stage" :aria-sort="ariaSort(stage.value)">
                  <button type="button" class="sort-head" :class="{ 'is-active': sortBy === stage.value, 'is-desc': sortDesc }" @click="toggleSort(stage.value)">{{ stage.label }}</button>
                </th>
                <th>去向</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="(item, index) in rows" :key="item.row.key">
                <tr v-if="index === cutIndex" class="cut-line">
                  <td :colspan="activeRoutes.length + 5">融合只保留前 {{ keepLimit }} 名进入 rerank，以下被截断（只有被扩展按锚点补回的才会出现在输出里）</td>
                </tr>
                <tr :class="{ 'is-selected': item.row.key === selectedKey }" @click="selectedKey = item.row.key">
                  <td class="col-chunk">
                    <div class="chunk-cell__head">
                      <span class="mono">#{{ item.row.chunk_id }}</span>
                      <span v-if="item.row.chunk_type !== 'text'" class="chip">{{ item.row.chunk_type }}</span>
                      <span class="chunk-cell__section">{{ item.row.section }}</span>
                    </div>
                    <div class="chunk-cell__preview">{{ item.row.preview }}</div>
                  </td>
                  <td
                    v-for="name in activeRoutes"
                    :key="name"
                    class="rank-cell"
                    :style="rankStyle(item.row.route_ranks[name])"
                    :title="name in item.row.route_ranks ? routeCellTitle(item.row, name) : ''"
                  >
                    {{ item.row.route_ranks[name] ?? '' }}
                  </td>
                  <td class="rank-cell col-stage" :style="rankStyle(item.row.fused_rank)">
                    {{ item.row.fused_rank ?? '—' }}
                  </td>
                  <td class="rank-cell col-stage" :style="rankStyle(item.row.rerank_rank)">
                    {{ item.row.rerank_rank ?? '—' }}
                  </td>
                  <td class="rank-cell col-stage" :style="rankStyle(item.row.final_rank)">
                    {{ item.row.final_rank ?? '—' }}
                  </td>
                  <td><span :class="['chip', `outcome--${item.outcome.code}`]">{{ item.outcome.label }}</span></td>
                </tr>
              </template>
            </tbody>
          </table>
          <el-empty v-if="!rows.length" description="没有符合条件的 chunk" :image-size="80" />
        </div>
      </section>
    </template>

    <div v-else class="surface surface-pad">
      <el-empty :description="running ? '检索中…' : '选择论文、输入问题后运行'" :image-size="100" />
    </div>

    <el-drawer :model-value="!!selectedRow" size="560px" :title="selectedRow ? `chunk #${selectedRow.chunk_id}` : ''" @close="selectedKey = ''">
      <RetrievalChunkDetail v-if="selectedRow && run" :row="selectedRow" :run="run" />
    </el-drawer>
  </div>
</template>

<style scoped>
.run-form {
  display: grid;
  gap: 12px;
  margin-bottom: 16px;
}

.run-form__main {
  display: grid;
  grid-template-columns: minmax(220px, 360px) minmax(0, 1fr) auto;
  gap: 10px;
}

.run-form__switches {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 4px;
}

.run-form__topk {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-left: 12px;
}

.summary {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 18px;
  margin-bottom: 16px;
  font-size: 13px;
}

.panels {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr);
  gap: 16px;
  margin-bottom: 16px;
  align-items: start;
}

.block-title {
  margin: 0 0 12px;
  font-size: 15px;
  font-weight: 650;
  color: var(--color-text);
}

.sub-title {
  margin: 18px 0 8px;
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-2);
}

.help {
  margin: 8px 0 0;
  font-size: 12px;
  line-height: 1.5;
  color: var(--color-text-3);
}

.kv {
  display: grid;
  gap: 12px;
  margin: 0;
}

.kv dt {
  margin-bottom: 4px;
  font-size: 12px;
  color: var(--color-text-3);
}

.kv dd {
  margin: 0;
  font-size: 13px;
  line-height: 1.55;
  color: var(--color-text-2);
}

.query-list {
  display: grid;
  gap: 4px;
  margin: 0;
  padding-left: 18px;
}

.dropped {
  margin-top: 6px;
  font-size: 12px;
}

.dropped ul {
  margin: 4px 0 0;
  padding-left: 18px;
}

.table-scroll {
  overflow-x: auto;
}

.grid-table,
.overlap {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}

.grid-table th,
.grid-table td,
.overlap th,
.overlap td {
  padding: 6px 8px;
  border-bottom: 1px solid var(--color-border);
  text-align: center;
  white-space: nowrap;
}

.grid-table th,
.overlap th {
  font-weight: 500;
  color: var(--color-text-3);
}

.grid-table td:first-child,
.grid-table th:first-child,
.overlap th:first-child {
  text-align: left;
}

.is-off td {
  color: var(--color-text-4);
}

.overlap td {
  min-width: 64px;
}

.matrix {
  overflow: hidden;
}

.matrix__toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px 14px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--color-border);
}

.matrix__toolbar .block-title {
  margin: 0 8px 0 0;
}

.matrix__select {
  width: 168px;
}

.matrix__export {
  margin-left: auto;
}

.matrix__scroll {
  max-height: calc(100vh - 160px);
  overflow: auto;
}

.rank-table thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--color-surface);
}

.rank-table tbody tr {
  cursor: pointer;
}

.rank-table tbody tr:hover td {
  box-shadow: inset 0 0 0 9999px rgba(17, 24, 39, 0.03);
}

.rank-table tr.is-selected td {
  box-shadow: inset 0 0 0 9999px rgba(79, 70, 229, 0.08);
}

.col-chunk {
  min-width: 320px;
  max-width: 460px;
  white-space: normal !important;
}

.col-stage {
  border-left: 1px solid var(--color-border);
}

.sort-head {
  padding: 0;
  border: 0;
  background: none;
  font: inherit;
  color: inherit;
  cursor: pointer;
}

.sort-head:hover,
.sort-head.is-active {
  color: var(--color-primary);
}

/* 当前排序列：↑ 名次升序，↓ 倒序；没进这一列的 chunk 两种顺序下都排在最后。 */
.sort-head.is-active::after {
  content: ' ↑';
}

.sort-head.is-active.is-desc::after {
  content: ' ↓';
}

.rank-cell {
  min-width: 56px;
  font-weight: 600;
  color: var(--color-text);
}

.chunk-cell__head {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.chunk-cell__section {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 600;
  color: var(--color-text);
}

.chunk-cell__preview {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  margin-top: 2px;
  font-size: 12px;
  line-height: 1.5;
  color: var(--color-text-3);
}

.cut-line td {
  padding: 4px 8px;
  border-top: 2px dashed var(--color-danger);
  background: var(--color-danger-soft);
  font-size: 12px;
  text-align: left !important;
  color: var(--color-danger);
  cursor: default;
}

.outcome--final { background: var(--color-success-soft); border-color: var(--color-success-border); color: var(--color-success); }
.outcome--expansion { background: var(--color-primary-soft); border-color: var(--color-primary-border); color: var(--color-primary); }
.outcome--fusion_cut { background: var(--color-danger-soft); border-color: var(--color-danger-border); color: var(--color-danger); }
.outcome--rerank_dropped { background: var(--color-warning-soft); border-color: var(--color-warning-border); color: var(--color-warning); }

@media (max-width: 1100px) {
  .panels {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 720px) {
  .run-form__main {
    grid-template-columns: minmax(0, 1fr);
  }

  .run-form__topk {
    margin-left: 0;
  }
}
</style>
