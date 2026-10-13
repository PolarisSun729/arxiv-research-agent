<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { getErrorMessage } from '@/api/errors'
import { usePaperStore } from '@/stores/paperStore'
import SearchBar from '@/components/SearchBar.vue'
import PaperCard from '@/components/PaperCard.vue'
import ArxivQueryCapabilityBanner from '@/components/arxiv/ArxivQueryCapabilityBanner.vue'
import { selectSearchResults } from '@/utils/searchResults'
import { getArxivSearchSuggestions } from '@/api/papers'
import { useUserContext } from '@/composables/useUserContext'
import type { ArxivSearchParams, ArxivSearchSuggestion } from '@/types/paper'

const router = useRouter()
const store = usePaperStore()
const userContext = useUserContext()

const searchForm = ref({
  searchQuery: '',
  title: '',
  author: '',
  abstract: '',
  category: '',
  comment: '',
  journalRef: '',
  reportNumber: '',
  operator: '' as '' | 'AND' | 'OR',
  sortBy: 'relevance',
  sortOrder: 'descending' as const,
  maxResults: 30,
  submittedDateBefore: ''
})

const currentPage = ref(1)
const pageSize = ref(10)
const hasCapabilityError = computed(() => store.arxivSearchCapability?.status === 'error')

const HIDE_MARKED_STORAGE_KEY = 'search.hideMarked'

function loadHideMarked() {
  try {
    return localStorage.getItem(HIDE_MARKED_STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

// 默认展示全部；用户选过「只看未标记」后记住，下次进来仍然生效。
const hideMarked = ref(loadHideMarked())
// 上一次搜索时用户要的条数；后端为了过滤会多拉，展示时按它截断。
const searchedMaxResults = ref(searchForm.value.maxResults)
const markedIdSet = computed(() => new Set(store.searchMarkedIds))
const searchSelection = computed(() =>
  selectSearchResults(store.allPapers, markedIdSet.value, searchedMaxResults.value, hideMarked.value)
)
const visiblePapers = computed(() => searchSelection.value.items)
const pagedPapers = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return visiblePapers.value.slice(start, start + pageSize.value)
})

function setHideMarked(value: boolean) {
  hideMarked.value = value
  currentPage.value = 1
  try {
    localStorage.setItem(HIDE_MARKED_STORAGE_KEY, value ? '1' : '0')
  } catch {
    // 无法持久化时只影响下次进入页面的默认值。
  }
}

const hideMarkedModel = computed({
  get: () => hideMarked.value,
  set: (value: boolean) => setHideMarked(value)
})

function isMarked(paper: { id: string; arxivId?: string; label?: string | null }) {
  // 淡化看当前状态，搜索后新标记的论文也会立即变淡，只是不会从列表里消失。
  return Boolean(paper.label) || (store.paperActionMap.read || []).includes(paper.arxivId || paper.id)
}

function resolveSubmittedDaysAgo() {
  if (!searchForm.value.submittedDateBefore) return 30
  const selectedDate = new Date(searchForm.value.submittedDateBefore)
  const timeDiff = Date.now() - selectedDate.getTime()
  return Math.floor(timeDiff / (1000 * 60 * 60 * 24))
}

function buildSearchParams(): ArxivSearchParams {
  const submittedDaysAgo = resolveSubmittedDaysAgo()
  return {
    search_query: searchForm.value.searchQuery || undefined,
    title: searchForm.value.title || undefined,
    author: searchForm.value.author || undefined,
    abstract: searchForm.value.abstract || undefined,
    category: searchForm.value.category || undefined,
    comment: searchForm.value.comment || undefined,
    journal_ref: searchForm.value.journalRef || undefined,
    report_number: searchForm.value.reportNumber || undefined,
    operator: searchForm.value.operator || undefined,
    max_results: searchForm.value.maxResults,
    start: 0,
    sort_by: searchForm.value.sortBy,
    sort_order: searchForm.value.sortOrder,
    submitted_days_ago: submittedDaysAgo > 0 ? submittedDaysAgo : undefined
  }
}

// 至少搜过一次后，空结果才显示「没有找到」而不是「输入关键词开始搜索」。
const hasSearched = ref(false)
const searchedDaysAgo = ref<number | undefined>(30)
const searchedQuery = ref('')
const suggestions = ref<ArxivSearchSuggestion[]>([])
let suggestionRequestId = 0

const timeSuggestions = computed(() => suggestions.value.filter(item => item.kind === 'widen_time'))
const querySuggestions = computed(() => suggestions.value.filter(item => item.kind !== 'widen_time'))
// 普通搜索不接 LLM；想按语义找时引导到 Agent 搜索，只有能用 Agent 的角色才显示。
const canAskAgent = computed(() => userContext.canResearch.value && Boolean(searchedQuery.value))
// 后端返回的条数少于用户要的条数，说明当前条件下已经没有更多论文了（多拉不会让这个判断失真）。
const resultsExhausted = computed(() =>
  hasSearched.value && !store.loading && !hasCapabilityError.value && store.allPapers.length < searchedMaxResults.value
)
const showRelaxPanel = computed(() => resultsExhausted.value && (suggestions.value.length > 0 || canAskAgent.value))
const relaxSummary = computed(() => {
  const scope = searchedDaysAgo.value ? `近 ${searchedDaysAgo.value} 天内` : '当前条件下'
  const count = store.allPapers.length
  return count === 0 ? `${scope}没有命中论文，可以试试：` : `${scope}只命中 ${count} 篇，可以试试：`
})

async function loadSuggestions(params: ArxivSearchParams) {
  const requestId = ++suggestionRequestId
  try {
    const result = await getArxivSearchSuggestions({ ...params, max_results: undefined })
    if (requestId === suggestionRequestId) suggestions.value = result.suggestions
  } catch {
    // 建议只是辅助信息，取不到时不打扰用户。
  }
}

async function handleSearch() {
  currentPage.value = 1
  searchedMaxResults.value = searchForm.value.maxResults
  suggestions.value = []
  suggestionRequestId += 1
  const params = buildSearchParams()

  try {
    await store.fetchArxivPapers(params)
    hasSearched.value = true
    searchedDaysAgo.value = params.submitted_days_ago
    searchedQuery.value = searchForm.value.searchQuery.trim()
    if (resultsExhausted.value) void loadSuggestions(params)
  } catch (error) {
    // 本地 OAI 契约错误已在 store 内转成页面级 banner；这里只兜底未知请求失败。
    ElMessage.error(getErrorMessage(error, 'arXiv 搜索失败，请稍后重试'))
  }
}

function formatLocalDate(date: Date) {
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${date.getFullYear()}-${month}-${day}`
}

function applySuggestion(item: ArxivSearchSuggestion) {
  if (item.kind === 'widen_time' && item.submitted_days_ago) {
    // 表单按「从哪天起」记录时间窗口，这里换算回日期，重新搜索时会得到同样的天数。
    searchForm.value.submittedDateBefore = formatLocalDate(new Date(Date.now() - item.submitted_days_ago * 86400000))
  } else {
    searchForm.value.searchQuery = item.search_query || ''
  }
  void handleSearch()
}

function askAgent() {
  router.push({ path: '/agent-search', query: { q: `帮我找关于「${searchedQuery.value}」的论文` } })
}

onMounted(() => {
  // 卡片上的「已读」读取行为表，进入页面先加载一次。
  store.fetchPaperActions().catch(() => undefined)
})

function handleViewDetail(id: string) {
  router.push(`/paper/${id}`)
}

async function handleLabel(id: string, label: 'liked' | 'disliked' | null) {
  try {
    await store.updateLabel(id, label)
    if (label === null) {
      ElMessage.success('已取消标记')
    } else {
      ElMessage.success(`已${label === 'liked' ? '标记感兴趣' : '标记不感兴趣'}`)
    }
  } catch (error) {
    ElMessage.error('偏好保存失败')
  }
}

function handlePageChange(page: number) {
  currentPage.value = page
}
</script>

<template>
  <div class="page search-page">
    <header class="page-header">
      <div class="page-header__text">
        <h1 class="page-title">论文搜索</h1>
        <p class="page-desc">检索本地 arXiv 镜像库，支持字段前缀与布尔语法；不选日期时默认查近 30 天。</p>
      </div>
    </header>

    <SearchBar v-model="searchForm" @search="handleSearch" />

    <ArxivQueryCapabilityBanner
      v-if="store.arxivSearchCapability"
      :capability="store.arxivSearchCapability"
      class="capability-banner"
    />

    <section class="results">
      <div v-if="store.allPapers.length > 0 && !store.loading" class="results-header">
        <span>
          共 <strong>{{ visiblePapers.length }}</strong> 篇论文
          <template v-if="searchSelection.hiddenCount > 0">
            · 已隐藏 {{ searchSelection.hiddenCount }} 篇已标记论文
            <el-button link type="primary" size="small" @click="setHideMarked(false)">显示</el-button>
          </template>
        </span>
        <el-radio-group v-model="hideMarkedModel" size="small">
          <el-radio-button :value="false">全部</el-radio-button>
          <el-radio-button :value="true">只看未标记</el-radio-button>
        </el-radio-group>
      </div>

      <div v-if="showRelaxPanel" class="surface relax-panel">
        <div class="relax-summary">{{ relaxSummary }}</div>
        <div v-if="timeSuggestions.length" class="relax-row">
          <span class="relax-label">放宽时间</span>
          <el-button v-for="item in timeSuggestions" :key="item.label" size="small" @click="applySuggestion(item)">
            {{ item.label }}<span v-if="item.total !== null" class="relax-count">· {{ item.total }} 篇</span>
          </el-button>
        </div>
        <div v-if="querySuggestions.length" class="relax-row">
          <span class="relax-label">放宽写法</span>
          <el-button v-for="item in querySuggestions" :key="item.label" size="small" @click="applySuggestion(item)">
            {{ item.label }}<span v-if="item.total !== null" class="relax-count">· {{ item.total }} 篇</span>
          </el-button>
        </div>
        <div v-if="canAskAgent" class="relax-agent">
          想按意思而不是关键词找？
          <el-button link type="primary" @click="askAgent">用 Agent 搜索 →</el-button>
        </div>
      </div>

      <div v-if="store.loading" class="paper-list">
        <div v-for="n in 3" :key="n" class="surface skeleton-card">
          <el-skeleton :rows="3" animated />
        </div>
      </div>

      <div v-else-if="hasCapabilityError" class="notice notice--danger">
        <div>
          <div class="notice__title">本次查询未执行</div>
          <p>{{ store.arxivSearchError || '请按本地 OAI 镜像库支持的字段与布尔语法调整后重试。' }}</p>
        </div>
      </div>

      <div v-else-if="store.allPapers.length === 0" class="surface empty">
        <el-empty
          :description="hasSearched ? '没有找到匹配的论文' : '输入关键词开始搜索，或调整筛选条件'"
          :image-size="96"
        />
      </div>

      <div v-else-if="visiblePapers.length === 0" class="surface empty">
        <el-empty description="这次搜到的论文都已标记过" :image-size="96">
          <el-button @click="setHideMarked(false)">显示全部</el-button>
        </el-empty>
      </div>

      <div v-else class="paper-list">
        <PaperCard
          v-for="paper in pagedPapers"
          :key="paper.id"
          :paper="paper"
          :class="{ 'is-marked': isMarked(paper) }"
          @view-detail="handleViewDetail"
          @label="handleLabel"
        />
      </div>

      <div v-if="visiblePapers.length > pageSize" class="pagination">
        <el-pagination
          :current-page="currentPage"
          :page-size="pageSize"
          :total="visiblePapers.length"
          background
          layout="prev, pager, next, jumper"
          @current-change="handlePageChange"
        />
      </div>
    </section>
  </div>
</template>

<style scoped>
.search-page {
  max-width: 1080px;
}

.capability-banner {
  margin-top: 16px;
}

.results {
  margin-top: 20px;
}

.results-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
  font-size: 13px;
  color: var(--color-text-3);
}

.results-header strong {
  color: var(--color-text);
}

.paper-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.relax-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px 20px;
  margin-bottom: 12px;
  font-size: 13px;
}

.relax-summary {
  color: var(--color-text);
}

.relax-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.relax-row .el-button {
  margin-left: 0;
}

.relax-label {
  flex: 0 0 auto;
  min-width: 56px;
  color: var(--color-text-3);
}

.relax-count {
  margin-left: 4px;
  color: var(--color-text-3);
}

.relax-agent {
  color: var(--color-text-3);
}

/* 已标记的论文仍保留在结果里，只是淡化，方便扫过时跳过；悬停时恢复，避免影响阅读和操作。 */
.paper-list .is-marked {
  opacity: 0.6;
  transition: opacity 0.15s ease;
}

.paper-list .is-marked:hover,
.paper-list .is-marked:focus-within {
  opacity: 1;
}

.skeleton-card {
  padding: 20px;
}

.empty {
  padding: 24px;
}

.notice p {
  margin-top: 4px;
}

.pagination {
  display: flex;
  justify-content: center;
  margin-top: 24px;
}
</style>
