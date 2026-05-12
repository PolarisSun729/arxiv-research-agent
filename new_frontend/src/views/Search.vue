<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { usePaperStore } from '@/stores/paperStore'
import SearchBar from '@/components/SearchBar.vue'
import PaperCard from '@/components/PaperCard.vue'

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

async function handleSearch() {
  currentPage.value = 1
  
  let submittedDaysAgo = 30
  
  if (searchForm.value.submittedDateBefore) {
    const selectedDate = new Date(searchForm.value.submittedDateBefore)
    const today = new Date()
    const timeDiff = today.getTime() - selectedDate.getTime()
    submittedDaysAgo = Math.floor(timeDiff / (1000 * 60 * 60 * 24))
  }
  
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
  }, 1, pageSize.value)
}

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
  store.paginatePapers(page, pageSize.value)
}
</script>

<template>
  <div class="search-page">
    <SearchBar
      v-model="searchForm"
      @search="handleSearch"
    />

    <div class="search-results">
      <div class="results-header">
        <span class="results-count">
          共 {{ store.totalPapers }} 篇论文
        </span>
      </div>

      <div v-if="store.loading" class="loading">
        <div class="el-loading-spinner"></div>
        <p>加载中...</p>
      </div>

      <div v-else-if="store.papers.length === 0" class="empty-results">
        <el-empty description="未找到符合条件的论文" />
      </div>

      <div v-else class="paper-list">
        <PaperCard
          v-for="paper in store.papers"
          :key="paper.id"
          :paper="paper"
          @view-detail="handleViewDetail"
          @label="handleLabel"
        />
      </div>

      <div v-if="store.totalPapers > 0" class="pagination-wrapper">
        <el-pagination
          :current-page="currentPage"
          :page-size="pageSize"
          :total="store.totalPapers"
          @current-change="handlePageChange"
          layout="total, prev, pager, next, jumper"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.search-page {
  padding: 20px;
}

.search-results {
  margin-top: 20px;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.results-count {
  font-size: 14px;
  color: #6b7280;
}

.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px;
}

.empty-results {
  padding: 40px;
}

.paper-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 16px;
}

.pagination-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 24px;
}

@media (max-width: 900px) {
  .paper-list {
    grid-template-columns: 1fr;
  }
}
</style>
