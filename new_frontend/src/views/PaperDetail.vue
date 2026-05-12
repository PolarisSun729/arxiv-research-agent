<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { usePaperStore } from '@/stores/paperStore'
import {
  getPaperQaStatus,
  createPaperQaIndex,
  qaPaper,
  type QaStatusResult
} from '@/api/papers'

const route = useRoute()
const router = useRouter()
const store = usePaperStore()

const paperId = computed(() => route.params.id as string)
const loading = ref(true)

const qaMode = ref(false)
const qaLoading = ref(false)
const creatingIndex = ref(false)
const qaStatus = ref<QaStatusResult | null>(null)
const question = ref('')
const qaResults = ref<Array<{ question: string; answer: string; sources: Array<{ content: string; page_number: string }> }>>([])

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

function goBack() {
  router.back()
}

async function fetchQaStatus() {
  try {
    qaStatus.value = await getPaperQaStatus(paperId.value)
  } catch (error) {
    console.error('Failed to fetch QA status:', error)
    qaStatus.value = { arxiv_id: paperId.value, has_index: false, status: 'not_indexed' }
  }
}

async function handleCreateIndex() {
  if (creatingIndex.value) return

  creatingIndex.value = true
  try {
    await createPaperQaIndex(paperId.value)
    ElMessage.success('问答索引创建成功')
    await fetchQaStatus()
  } catch (error: any) {
    ElMessage.error(error?.response?.data?.detail || '创建索引失败')
  } finally {
    creatingIndex.value = false
  }
}

async function handleQaSubmit() {
  if (!question.value.trim() || qaLoading.value) return

  qaLoading.value = true
  try {
    const result = await qaPaper(paperId.value, question.value.trim())
    qaResults.value.unshift({
      question: result.question,
      answer: result.answer,
      sources: result.sources
    })
    question.value = ''
  } catch (error: any) {
    ElMessage.error(error?.response?.data?.detail || '问答失败')
  } finally {
    qaLoading.value = false
  }
}

async function handleAskPaper() {
  if (!qaStatus.value?.has_index) {
    try {
      await ElMessageBox.confirm(
        '这篇论文还没有建立问答索引，需要先下载PDF并创建索引。这个过程可能需要几分钟时间，是否继续？',
        '确认创建索引',
        {
          confirmButtonText: '继续',
          cancelButtonText: '取消',
          type: 'warning'
        }
      )
      await handleCreateIndex()
    } catch {
      return
    }
  }
  qaMode.value = true
}

function closeQaMode() {
  qaMode.value = false
}

onMounted(async () => {
  try {
    await store.fetchPaperById(paperId.value)
    await fetchQaStatus()
  } catch (error) {
    console.error('Failed to fetch paper:', error)
  } finally {
    loading.value = false
  }
})

watch(paperId, async () => {
  loading.value = true
  try {
    await store.fetchPaperById(paperId.value)
    await fetchQaStatus()
    qaResults.value = []
  } catch (error) {
    console.error('Failed to fetch paper:', error)
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div class="paper-detail">
    <div class="detail-header">
      <el-button type="default" @click="goBack" class="back-btn">
        <span class="btn-emoji">⬅️</span>
        返回
      </el-button>
    </div>

    <el-skeleton v-if="loading" :rows="8" animated />

    <template v-else-if="store.currentPaper">
      <el-card class="paper-detail-card">
        <div class="detail-title">
          <h1>{{ store.currentPaper.title }}</h1>
        </div>

        <div class="detail-authors">
          <span class="label">作者：</span>
          <span class="authors">{{ store.currentPaper.authors.join(', ') }}</span>
        </div>

        <div class="detail-meta">
          <div class="meta-row">
            <span class="label">发布时间：</span>
            <span>{{ formatDate(store.currentPaper.publishedAt) }}</span>
          </div>
          <div class="meta-row" v-if="store.currentPaper.updatedAt">
            <span class="label">更新时间：</span>
            <span>{{ formatDate(store.currentPaper.updatedAt) }}</span>
          </div>
          <div class="meta-row">
            <span class="label">arXiv ID：</span>
            <a :href="store.currentPaper.absUrl" target="_blank" class="arxiv-link">
              {{ store.currentPaper.arxivId }}
            </a>
          </div>
        </div>

        <div class="detail-categories">
          <span class="label">分类：</span>
          <el-tag
            v-for="cat in store.currentPaper.categories"
            :key="cat"
            type="info"
          >
            {{ cat }}
          </el-tag>
        </div>

        <div class="detail-summary">
          <h3 class="section-title">摘要</h3>
          <p>{{ store.currentPaper.summary }}</p>
        </div>

        <div class="detail-links">
          <a :href="store.currentPaper.absUrl" target="_blank" class="abs-link">
            <el-button size="large">
              <span class="btn-emoji">📎</span>
              查看原文
            </el-button>
          </a>
          <el-button
            size="large"
            type="primary"
            @click="handleAskPaper"
            :loading="creatingIndex"
          >
            <span class="btn-emoji">💬</span>
            {{ creatingIndex ? '创建索引中...' : '问这篇论文' }}
          </el-button>
        </div>

        <div v-if="qaStatus" class="qa-status">
          <el-tag :type="qaStatus.has_index ? 'success' : 'warning'">
            {{ qaStatus.has_index ? '已建立问答索引' : '未建立问答索引' }}
          </el-tag>
          <span v-if="qaStatus.chunk_count" class="chunk-info">
            ({{ qaStatus.chunk_count }} chunks)
          </span>
        </div>
      </el-card>

      <el-card v-if="qaMode" class="qa-card">
        <div class="qa-header">
          <h3 class="qa-title">论文问答</h3>
          <el-button type="text" @click="closeQaMode" class="close-btn">
            <span class="btn-emoji">✖️</span>
          </el-button>
        </div>

        <div class="qa-input-area">
          <el-input
            v-model="question"
            placeholder="请输入你的问题..."
            class="qa-input"
            @keyup.enter="handleQaSubmit"
          />
          <el-button
            type="primary"
            @click="handleQaSubmit"
            :loading="qaLoading"
            class="qa-submit-btn"
          >
            <span class="btn-emoji">🧠</span>
            {{ qaLoading ? '回答中...' : '提问' }}
          </el-button>
        </div>

        <div class="qa-history">
          <div
            v-for="(result, index) in qaResults"
            :key="index"
            class="qa-item"
          >
            <div class="qa-question">
              <span class="question-label">问：</span>
              <span>{{ result.question }}</span>
            </div>
            <div class="qa-answer">
              <span class="answer-label">答：</span>
              <p>{{ result.answer }}</p>
            </div>
            <div v-if="result.sources && result.sources.length" class="qa-sources">
              <span class="sources-label">参考来源：</span>
              <div class="sources-list">
                <div
                  v-for="(source, sIndex) in result.sources"
                  :key="sIndex"
                  class="source-item"
                >
                  <span class="page-label">页码 {{ source.page_number }}：</span>
                  <span class="source-content">{{ source.content }}...</span>
                </div>
              </div>
            </div>
          </div>

          <div v-if="qaResults.length === 0" class="qa-empty">
            <div class="empty-icon">💬</div>
            <p>还没有提问记录，开始问一问吧</p>
          </div>
        </div>
      </el-card>
    </template>

    <el-empty v-else description="论文不存在" />
  </div>
</template>

<style scoped>
.paper-detail {
  padding: 20px;
  max-width: 900px;
  margin: 0 auto;
}

.detail-header {
  margin-bottom: 20px;
}

.back-btn {
  display: flex;
  align-items: center;
  gap: 8px;
}

.btn-emoji {
  font-size: 16px;
}

.paper-detail-card {
  padding: 24px;
}

.detail-title h1 {
  font-size: 22px;
  font-weight: 700;
  color: #1f2937;
  line-height: 1.5;
  margin: 0 0 16px 0;
}

.detail-authors {
  margin-bottom: 16px;
  font-size: 15px;
}

.detail-authors .label {
  font-weight: 600;
  color: #374151;
}

.detail-authors .authors {
  color: #4b5563;
}

.detail-meta {
  margin-bottom: 16px;
}

.meta-row {
  font-size: 14px;
  color: #4b5563;
  margin-bottom: 8px;
}

.meta-row .label {
  font-weight: 500;
  color: #6b7280;
}

.arxiv-link {
  color: #1890ff;
  text-decoration: none;
}

.arxiv-link:hover {
  text-decoration: underline;
}

.detail-categories {
  margin-bottom: 20px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.detail-categories .label {
  font-weight: 500;
  color: #6b7280;
}

.detail-summary {
  margin-bottom: 20px;
  padding: 16px;
  background: #f9fafb;
  border-radius: 8px;
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  color: #374151;
  margin: 0 0 12px 0;
}

.detail-summary p {
  font-size: 14px;
  color: #4b5563;
  line-height: 1.7;
  margin: 0;
}

.detail-links {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.abs-link {
  text-decoration: none;
}

.qa-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding-top: 16px;
  border-top: 1px solid #e5e7eb;
}

.chunk-info {
  font-size: 14px;
  color: #6b7280;
}

.qa-card {
  margin-top: 20px;
  padding: 24px;
}

.qa-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.qa-title {
  font-size: 18px;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
}

.close-btn {
  padding: 0;
}

.qa-input-area {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
}

.qa-input {
  flex: 1;
}

.qa-submit-btn {
  white-space: nowrap;
}

.qa-history {
  max-height: 600px;
  overflow-y: auto;
}

.qa-item {
  margin-bottom: 20px;
  padding: 16px;
  background: #f9fafb;
  border-radius: 8px;
}

.qa-item:last-child {
  margin-bottom: 0;
}

.qa-question {
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 12px;
}

.question-label {
  color: #1890ff;
}

.qa-answer {
  color: #4b5563;
  line-height: 1.7;
}

.answer-label {
  color: #52c41a;
  font-weight: 600;
}

.qa-answer p {
  margin: 0;
}

.qa-sources {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #e5e7eb;
}

.sources-label {
  font-size: 14px;
  font-weight: 600;
  color: #6b7280;
}

.sources-list {
  margin-top: 8px;
}

.source-item {
  font-size: 14px;
  color: #4b5563;
  margin-bottom: 8px;
}

.source-item:last-child {
  margin-bottom: 0;
}

.page-label {
  color: #9ca3af;
}

.source-content {
  display: block;
  margin-left: 30px;
}

.qa-empty {
  text-align: center;
  padding: 40px;
  color: #9ca3af;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.qa-empty p {
  margin: 0;
}
</style>
