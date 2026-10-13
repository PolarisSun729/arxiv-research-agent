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

const router = useRouter()
const store = usePaperStore()

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

async function handleSearch() {
  currentPage.value = 1
  searchedMaxResults.value = searchForm.value.maxResults

  let submittedDaysAgo = 30
  
  if (searchForm.value.submittedDateBefore) {
    const selectedDate = new Date(searchForm.value.submittedDateBefore)
    const today = new Date()
    const timeDiff = today.getTime() - selectedDate.getTime()
    submittedDaysAgo = Math.floor(timeDiff / (1000 * 60 * 60 * 24))
  }
  
  try {
    await store.fetchArxivPapers({
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
    })
  } catch (error) {
    // 本地 OAI 契约错误已在 store 内转成页面级 banner；这里只兜底未知请求失败。
    ElMessage.error(getErrorMessage(error, 'arXiv 搜索失败，请稍后重试'))
  }
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
        <el-empty description="输入关键词开始搜索，或调整筛选条件" :image-size="96" />
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
