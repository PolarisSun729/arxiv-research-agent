<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import PaperCard from '@/components/PaperCard.vue'
import { runArxivSearchAgent } from '@/api/agent'
import type { ArxivSearchResponse, AgentPaper } from '@/types/agent'
import type { Paper } from '@/types/paper'

const router = useRouter()
const message = ref('')
const loading = ref(false)
const response = ref<ArxivSearchResponse | null>(null)

const papers = computed<Paper[]>(() => {
  const items = response.value?.papers || []
  return items.map(normalizePaper)
})

function normalizePaper(raw: AgentPaper): Paper {
  const arxivId = raw.arxiv_id || raw.arxivId || (raw.id ? String(raw.id).split('/').pop() : '')
  const authors = Array.isArray(raw.authors)
    ? raw.authors
    : String(raw.authors || '').split(',').map((author: string) => author.trim()).filter(Boolean)
  const categories = Array.isArray(raw.categories)
    ? raw.categories
    : String(raw.categories || '').split(',').map((category: string) => category.trim()).filter(Boolean)

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
    absUrl: raw.absUrl || raw.abs_url || raw.url || ''
  }
}

function formatJson(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function formatToolCallStatus(status: string) {
  if (status === 'success') return '成功'
  if (status === 'failed') return '失败'
  return status || 'unknown'
}

async function handleSubmit() {
  const trimmed = message.value.trim()
  if (!trimmed) {
    ElMessage.warning('请输入自然语言搜索内容')
    return
  }

  loading.value = true
  response.value = null
  try {
    response.value = await runArxivSearchAgent({
      message: trimmed,
      user_id: 'local_user'
    })
  } catch (error: any) {
    ElMessage.error(error?.response?.data?.detail || error?.message || 'Agent 调用失败')
  } finally {
    loading.value = false
  }
}

function handleViewDetail(id: string) {
  router.push(`/paper/${id}`)
}
</script>

<template>
  <div class="agent-search-page">
    <div class="hero-card">
      <div>
        <p class="eyebrow">Agent Search</p>
        <h1 class="title">自然语言 arXiv 搜索入口</h1>
        <p class="subtitle">直接输入你的搜索意图，查看 Agent 解析、调用工具和返回结果。</p>
      </div>
    </div>

    <el-card class="input-card" shadow="hover">
      <div class="input-header">
        <span class="section-title">搜索指令</span>
        <span class="section-hint">例如：帮我找最近 7 天关于 RAG 的 5 篇论文</span>
      </div>

      <el-input
        v-model="message"
        type="textarea"
        :rows="4"
        placeholder="输入自然语言搜索需求"
        resize="none"
      />

      <div class="actions">
        <el-button type="primary" :loading="loading" @click="handleSubmit">
          开始搜索
        </el-button>
        <el-button :disabled="loading" @click="message = ''">
          清空
        </el-button>
      </div>
    </el-card>

    <div v-if="response" class="result-grid">
      <el-card class="panel" shadow="hover">
        <template #header>
          <div class="panel-header">Agent 输出</div>
        </template>
        <div class="answer-box">
          <div class="answer-label">answer</div>
          <p>{{ response.answer }}</p>
        </div>
        <div class="answer-box">
          <div class="answer-label">next_actions</div>
          <ul class="inline-list">
            <li v-for="item in response.next_actions" :key="item">{{ item }}</li>
          </ul>
        </div>
      </el-card>

      <el-card class="panel" shadow="hover">
        <template #header>
          <div class="panel-header">解析与计划</div>
        </template>
        <div class="kv-item">
          <div class="kv-label">intent</div>
          <div class="kv-value">{{ response.intent }}</div>
        </div>
        <div class="kv-item">
          <div class="kv-label">search_spec</div>
          <pre class="json-block">{{ response.search_spec ? formatJson(response.search_spec) : 'null' }}</pre>
        </div>
        <div class="kv-item">
          <div class="kv-label">plan</div>
          <ul class="inline-list">
            <li v-for="item in response.plan" :key="item">{{ item }}</li>
          </ul>
        </div>
        <div class="kv-item">
          <div class="kv-label">warnings</div>
          <ul class="inline-list warning-list">
            <li v-for="item in response.warnings" :key="item">{{ item }}</li>
          </ul>
        </div>
      </el-card>

      <el-card class="panel" shadow="hover">
        <template #header>
          <div class="panel-header">工具调用</div>
        </template>
        <div v-if="response.tool_calls.length === 0" class="muted">暂无工具调用记录</div>
        <div v-else class="tool-call-list">
          <div v-for="(call, index) in response.tool_calls" :key="index" class="tool-call-item">
            <div class="tool-call-top">
              <strong>{{ call.tool_name }}</strong>
              <el-tag size="small" :type="call.status === 'success' ? 'success' : 'danger'">
                {{ formatToolCallStatus(call.status) }}
              </el-tag>
            </div>
            <div v-if="call.summary" class="tool-call-summary">{{ call.summary }}</div>
            <pre class="json-block">{{ formatJson(call.arguments) }}</pre>
            <pre v-if="call.error" class="json-block error-block">{{ formatJson(call.error) }}</pre>
          </div>
        </div>
      </el-card>
    </div>

    <el-card v-if="response" class="papers-panel" shadow="hover">
      <template #header>
        <div class="panel-header">Papers</div>
      </template>
      <div v-if="papers.length === 0" class="empty-state">
        <el-empty description="暂无论文结果" />
      </div>
      <div v-else class="paper-list">
        <PaperCard
          v-for="paper in papers"
          :key="paper.id"
          :paper="paper"
          @view-detail="handleViewDetail"
        />
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.agent-search-page {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.hero-card {
  padding: 22px 24px;
  border-radius: 20px;
  background:
    radial-gradient(circle at top right, rgba(56, 189, 248, 0.18), transparent 32%),
    linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.94));
  color: #fff;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
}

.eyebrow {
  margin: 0 0 8px;
  color: #7dd3fc;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.title {
  margin: 0;
  font-size: 30px;
  line-height: 1.15;
}

.subtitle {
  margin: 10px 0 0;
  max-width: 760px;
  color: rgba(226, 232, 240, 0.88);
}

.input-card,
.panel,
.papers-panel {
  border-radius: 18px;
}

.input-header,
.panel-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.section-title {
  font-weight: 700;
  color: #111827;
}

.section-hint {
  color: #6b7280;
  font-size: 13px;
}

.actions {
  margin-top: 14px;
  display: flex;
  gap: 10px;
}

.result-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.panel {
  min-height: 100%;
}

.answer-box {
  padding: 12px 0;
  border-bottom: 1px solid #eef2f7;
}

.answer-box:last-child {
  border-bottom: none;
}

.answer-label,
.kv-label {
  font-size: 12px;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}

.kv-item {
  margin-bottom: 16px;
}

.kv-value,
.muted {
  color: #1f2937;
}

.json-block {
  margin: 0;
  padding: 12px;
  border-radius: 12px;
  background: #0f172a;
  color: #dbeafe;
  overflow: auto;
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.error-block {
  background: #3f1d1d;
  color: #fee2e2;
}

.inline-list {
  margin: 0;
  padding-left: 18px;
  color: #1f2937;
}

.warning-list {
  color: #b45309;
}

.tool-call-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.tool-call-item {
  padding: 12px;
  border-radius: 14px;
  background: #f8fafc;
  border: 1px solid #e5e7eb;
}

.tool-call-top {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  margin-bottom: 8px;
}

.tool-call-summary {
  margin-bottom: 10px;
  color: #374151;
  font-size: 14px;
}

.papers-panel {
  border-radius: 18px;
}

.paper-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 16px;
}

.empty-state {
  padding: 24px 0;
}

@media (max-width: 1200px) {
  .result-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 900px) {
  .paper-list {
    grid-template-columns: 1fr;
  }
}
</style>
