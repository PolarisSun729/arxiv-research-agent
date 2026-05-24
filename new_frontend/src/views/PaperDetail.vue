<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  ArrowLeft,
  ArrowDown,
  ChatDotRound,
  Compass,
  Document,
  Link,
  Loading,
  RefreshRight,
  Search
} from '@element-plus/icons-vue'
import MarkdownIt from 'markdown-it'
import dollarmathPlugin from 'markdown-it-dollarmath'
import { renderToString } from 'katex'
import 'katex/dist/katex.min.css'
import { usePaperStore } from '@/stores/paperStore'
import {
  createPaperQaIndex,
  getPaperQaDiagnostic,
  getPaperQaStatus,
  getPaperRetrievalTraceDownloadUrl,
  qaPaperStream,
  type QaDiagnosticResult,
  type QaStatusResult,
  type RetrievalDebug
} from '@/api/papers'

type QaSource = {
  content: string
  page_number: string
  source?: string
}

type QaTurn = {
  id: string
  question: string
  answer: string
  sources: QaSource[]
  retrievalDebug?: RetrievalDebug | null
  createdAt: string
  streaming?: boolean
}

const route = useRoute()
const router = useRouter()
const store = usePaperStore()

const paperId = computed(() => route.params.id as string)
const loading = ref(true)
const qaMode = ref(false)
const qaLoading = ref(false)
const creatingIndex = ref(false)
const loadingMethod = ref<'pymupdf' | 'docling'>('docling')
const qaStatus = ref<QaStatusResult | null>(null)
const qaDiagnostic = ref<QaDiagnosticResult | null>(null)
const question = ref('')
const chatContainerRef = ref<HTMLElement | null>(null)
const qaResults = ref<QaTurn[]>([])
const evidenceDrawerOpen = ref(false)
const activeEvidenceTurn = ref<QaTurn | null>(null)
const traceDownloading = ref(false)
const retrievalOptions = reactive({
  enableQueryRewrite: true,
  enableHyde: true,
  enableKeywordSearch: true,
  enableLlmRerank: true,
  debug: true,
  topK: 15
})

const modelBadge = 'Qwen 3.6 Plus'
const debugRouteLabels: Record<string, string> = {
  vector_original: '原始向量召回',
  vector_rewrite: '重写向量召回',
  vector_hyde: 'HyDE 向量召回',
  keyword: '关键词召回'
}
const debugStageLabels: Record<string, string> = {
  raw_retrieval_top15: 'raw_retrieval_top15',
  fused_top15: 'fused_top15',
  reranked_top15: 'reranked_top15',
  final_context_top15: 'final_context_top15'
}

const quickPrompts = computed(() => [
  '请总结这篇论文的核心贡献。',
  '这篇论文的方法流程是怎样的？',
  '实验结果说明了什么，局限性有哪些？'
])

const markdownRenderer = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  typographer: true
}).use(dollarmathPlugin, {
  allow_space: true,
  allow_digits: true,
  double_inline: true,
  allow_labels: true,
  renderer(content: string, { displayMode }: { displayMode: boolean }) {
    return renderToString(content, {
      displayMode,
      throwOnError: false
    })
  }
})

const hasQaIndex = computed(() => Boolean(qaStatus.value?.has_index))
const chatTurns = computed(() => qaResults.value)
const loadingMethodLabel = computed(() => (loadingMethod.value === 'docling' ? 'Docling' : 'PyMuPDF'))
const loadingMethodHint = computed(() => (loadingMethod.value === 'docling' ? '更适合论文结构' : '保留传统解析'))

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

function getSourceLabel(source: QaSource, index: number) {
  const pageNumber = (source.page_number || '').trim()
  if (pageNumber && pageNumber.toUpperCase() !== 'N/A') {
    return `Page ${pageNumber}`
  }
  return `Source ${index + 1}`
}

function getDebugRouteLabel(routeName: string) {
  return debugRouteLabels[routeName] || routeName
}

function getDebugStageLabel(stageName: string) {
  return debugStageLabels[stageName] || stageName
}

function formatDebugQueryList(queries?: string[]) {
  if (!queries || queries.length === 0) {
    return '无'
  }
  return queries.join(' | ')
}

function formatDebugNumber(value?: number | null) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '-'
}

function formatRouteScores(routeScores?: Record<string, number>) {
  const entries = Object.entries(routeScores || {})
  if (!entries.length) {
    return '无'
  }
  return entries.map(([route, score]) => `${getDebugRouteLabel(route)} ${formatDebugNumber(score)}`).join(' · ')
}

function renderMarkdown(text: string) {
  const content = (text || '').replace(/\r\n/g, '\n').trim()
  if (!content) {
    return '<p class="md-empty">回答生成中...</p>'
  }
  return markdownRenderer.render(content)
}

function openEvidence(turn: QaTurn) {
  activeEvidenceTurn.value = turn
  evidenceDrawerOpen.value = true
}

function closeEvidence() {
  evidenceDrawerOpen.value = false
}

function getTraceFileName(pathValue?: string) {
  if (!pathValue) return ''
  const parts = pathValue.split(/[/\\]+/)
  return parts[parts.length - 1] || ''
}

function downloadRetrievalTrace(format: 'md' | 'json' = 'md') {
  if (traceDownloading.value || !paperId.value) return

  traceDownloading.value = true
  try {
    const traceName = activeEvidenceTurn.value?.retrievalDebug?.trace_export?.[format]
      ? getTraceFileName(activeEvidenceTurn.value.retrievalDebug.trace_export[format])
      : ''
    const url = getPaperRetrievalTraceDownloadUrl(paperId.value, format, traceName || undefined)
    window.open(url, '_blank', 'noopener,noreferrer')
  } finally {
    traceDownloading.value = false
  }
}

function goBack() {
  router.back()
}

function scrollToBottom() {
  nextTick(() => {
    const el = chatContainerRef.value
    if (!el) return
    el.scrollTo({
      top: el.scrollHeight,
      behavior: 'smooth'
    })
  })
}

async function fetchQaStatus() {
  try {
    const [statusResult, diagnosticResult] = await Promise.all([
      getPaperQaStatus(paperId.value),
      getPaperQaDiagnostic(paperId.value).catch(error => {
        console.error('Failed to fetch QA diagnostic:', error)
        return null
      })
    ])
    qaStatus.value = statusResult
    qaDiagnostic.value = diagnosticResult
  } catch (error) {
    console.error('Failed to fetch QA status:', error)
    qaStatus.value = { arxiv_id: paperId.value, has_index: false, status: 'not_indexed' }
    qaDiagnostic.value = null
  }
}

async function handleCreateIndex() {
  if (creatingIndex.value) return

  creatingIndex.value = true
  try {
    await createPaperQaIndex(paperId.value, loadingMethod.value)
    ElMessage.success('问答索引创建成功')
    await fetchQaStatus()
  } catch (error: any) {
    ElMessage.error(error?.response?.data?.detail || '创建索引失败')
  } finally {
    creatingIndex.value = false
  }
}

async function handleQaSubmit(customQuestion?: string) {
  const rawQuestion = (customQuestion ?? question.value).trim()
  if (!rawQuestion || qaLoading.value) return

  qaLoading.value = true

  const turn = reactive<QaTurn>({
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    question: rawQuestion,
    answer: '',
    sources: [],
    retrievalDebug: null,
    createdAt: new Date().toISOString(),
    streaming: true
  })
  qaResults.value.push(turn)
  question.value = ''
  scrollToBottom()

  try {
    const result = await qaPaperStream(paperId.value, rawQuestion, {
      onMeta: (meta) => {
        if (Array.isArray(meta.sources)) {
          turn.sources = meta.sources
        }
        if (meta.retrieval_debug) {
          turn.retrievalDebug = meta.retrieval_debug
        }
        scrollToBottom()
      },
      onDelta: (delta) => {
        turn.answer += delta
        scrollToBottom()
      },
      onDone: (payload) => {
        if (typeof payload.answer === 'string' && payload.answer) {
          turn.answer = payload.answer
        }
        if (Array.isArray(payload.sources)) {
          turn.sources = payload.sources
        }
        if (payload.retrieval_debug) {
          turn.retrievalDebug = payload.retrieval_debug
        }
        turn.streaming = false
        scrollToBottom()
      }
    }, {
      top_k: retrievalOptions.topK,
      enable_query_rewrite: retrievalOptions.enableQueryRewrite,
      enable_hyde: retrievalOptions.enableHyde,
      enable_keyword_search: retrievalOptions.enableKeywordSearch,
      enable_llm_rerank: retrievalOptions.enableLlmRerank,
      debug: retrievalOptions.debug
    })

    if (typeof result.answer === 'string' && !turn.answer) {
      turn.answer = result.answer
    }
    if (Array.isArray(result.sources) && turn.sources.length === 0) {
      turn.sources = result.sources
    }
    if (result.retrieval_debug && !turn.retrievalDebug) {
      turn.retrievalDebug = result.retrieval_debug
    }
    turn.streaming = false
  } catch (error: any) {
    ElMessage.error(error?.message || '问答失败')
    qaResults.value = qaResults.value.filter(item => item.id !== turn.id)
    question.value = rawQuestion
  } finally {
    qaLoading.value = false
  }
}

async function handleAskPaper() {
  if (!hasQaIndex.value) {
    try {
      await ElMessageBox.confirm(
        '这篇论文还没有建立问答索引，需要先下载 PDF 并创建索引。这个过程可能需要几分钟，是否继续？',
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
  scrollToBottom()
}

function closeQaMode() {
  qaMode.value = false
}

function applyPrompt(prompt: string) {
  question.value = prompt
  qaMode.value = true
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
    question.value = ''
  } catch (error) {
    console.error('Failed to fetch paper:', error)
  } finally {
    loading.value = false
  }
})

watch(
  () => qaResults.value.length,
  () => scrollToBottom()
)

watch(qaLoading, () => scrollToBottom())
</script>

<template>
  <div class="paper-detail-shell">
    <div class="background-glow background-glow-a" />
    <div class="background-glow background-glow-b" />

    <div class="paper-detail">
      <div class="topbar">
        <el-button class="ghost-back" @click="goBack">
          <el-icon><ArrowLeft /></el-icon>
          返回
        </el-button>
        <div class="topbar-right">
          <el-tag v-if="qaStatus" :type="qaStatus.has_index ? 'success' : 'warning'" effect="light">
            {{ qaStatus.has_index ? '已建立问答索引' : '未建立问答索引' }}
          </el-tag>
          <el-tag effect="plain" type="info">{{ modelBadge }}</el-tag>
        </div>
      </div>

      <el-skeleton v-if="loading" :rows="10" animated />

      <template v-else-if="store.currentPaper">
        <section class="hero-card">
          <div class="hero-copy">
            <div class="eyebrow">
              <el-icon><Document /></el-icon>
              <span>Paper Detail</span>
            </div>
            <h1>{{ store.currentPaper.title }}</h1>
            <p class="hero-summary">
              {{ store.currentPaper.summary }}
            </p>

            <div class="meta-row">
              <el-tag v-for="cat in store.currentPaper.categories" :key="cat" effect="plain" type="info">
                {{ cat }}
              </el-tag>
            </div>
          </div>

          <div class="hero-aside">
            <div class="metric">
              <span class="metric-label">作者</span>
              <span class="metric-value">{{ store.currentPaper.authors.join(' · ') }}</span>
            </div>
            <div class="metric">
              <span class="metric-label">发布时间</span>
              <span class="metric-value">{{ formatDate(store.currentPaper.publishedAt) }}</span>
            </div>
            <div class="metric">
              <span class="metric-label">arXiv ID</span>
              <span class="metric-value">{{ store.currentPaper.arxivId }}</span>
            </div>

            <div class="hero-actions">
              <a :href="store.currentPaper.absUrl" target="_blank" class="action-link">
                <el-button size="large" class="action-button">
                  <el-icon><Link /></el-icon>
                  查看原文
                </el-button>
              </a>
              <div class="parser-card">
                <div class="parser-card-head">
                  <span class="parser-label">PDF 解析</span>
                  <span class="parser-hint">{{ loadingMethodHint }}</span>
                </div>
                <el-radio-group v-model="loadingMethod" class="parser-segment" :disabled="creatingIndex" size="small">
                  <el-radio-button label="docling">Docling</el-radio-button>
                  <el-radio-button label="pymupdf">PyMuPDF</el-radio-button>
                </el-radio-group>
              </div>
              <el-button size="large" type="primary" class="action-button ask-button" @click="handleAskPaper" :loading="creatingIndex">
                <el-icon><ChatDotRound /></el-icon>
                {{ creatingIndex ? `创建 ${loadingMethodLabel} 索引中...` : `开始问答 · ${loadingMethodLabel}` }}
              </el-button>
            </div>

            <div class="status-note" v-if="qaStatus">
              <el-icon><Compass /></el-icon>
              <span>
                {{ qaStatus.has_index ? `索引已完成，${qaStatus.chunk_count || 0} 个 chunks 可供检索` : `先创建索引，默认解析方式：${loadingMethodLabel}` }}
              </span>
            </div>
          </div>
        </section>

        <section class="qa-layout">
          <div class="qa-main">
            <div class="qa-panel-header">
              <div>
                <div class="panel-title">论文问答</div>
                <div class="panel-subtitle">
                  检索论文片段后再交给 Qwen 生成答案，带引用来源
                </div>
              </div>

              <div class="panel-actions">
                <el-button text class="refresh-btn" @click="fetchQaStatus">
                  <el-icon><RefreshRight /></el-icon>
                  刷新状态
                </el-button>
                <el-button v-if="qaMode" text class="refresh-btn" @click="closeQaMode">
                  关闭问答
                </el-button>
              </div>
            </div>

            <div class="prompt-row">
              <button
                v-for="prompt in quickPrompts"
                :key="prompt"
                type="button"
                class="prompt-chip"
                @click="applyPrompt(prompt)"
              >
                <el-icon><Search /></el-icon>
                <span>{{ prompt }}</span>
              </button>
            </div>

            <div class="chat-window" ref="chatContainerRef">
              <div v-if="chatTurns.length === 0 && !qaLoading" class="empty-chat">
                <div class="empty-mark">
                  <el-icon><ChatDotRound /></el-icon>
                </div>
                <h3>开始向论文提问</h3>
                <p>输入一个问题，系统会先召回相关 chunk，再交给 Qwen 生成答案。</p>
              </div>

              <div v-for="turn in chatTurns" :key="turn.id" class="turn-stack">
                <div class="message message-user">
                  <div class="message-label">你</div>
                  <div class="message-bubble">{{ turn.question }}</div>
                </div>

                <div class="message message-assistant">
                  <div class="message-label">Qwen</div>
                  <div class="assistant-card">
                    <div class="assistant-markdown" v-html="renderMarkdown(turn.answer)" />
                    <div v-if="turn.streaming" class="streaming-indicator">
                      <el-icon class="spin"><Loading /></el-icon>
                      <span>正在流式生成中...</span>
                    </div>

                    <div v-if="turn.streaming && (turn.sources.length || turn.retrievalDebug)" class="streaming-evidence-note">
                      召回内容和调试信息将在回答完成后进入右侧抽屉
                    </div>

                    <div class="assistant-actions">
                      <div class="assistant-badges">
                        <el-tag size="small" effect="plain" type="info">
                          {{ turn.sources.length }} 条来源
                        </el-tag>
                        <el-tag v-if="turn.retrievalDebug" size="small" effect="plain" type="warning">
                          Debug 可查看
                        </el-tag>
                      </div>
                      <el-button
                        v-if="turn.sources.length || turn.retrievalDebug"
                        size="small"
                        text
                        type="primary"
                        class="evidence-button"
                        @click="openEvidence(turn)"
                      >
                        查看来源与调试
                      </el-button>
                    </div>

                    <div class="answer-meta">
                      <span>{{ turn.createdAt }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div class="composer">
              <el-input
                v-model="question"
                type="textarea"
                :autosize="{ minRows: 2, maxRows: 5 }"
                resize="none"
                placeholder="输入你的问题，按 Enter 发送，Shift+Enter 换行"
                @keydown.enter.exact.prevent="handleQaSubmit()"
              />
              <div class="composer-actions">
                <span class="composer-tip">
                  先做语义检索，再由模型整合回答，适合论文问答场景
                </span>
                <el-button
                  type="primary"
                  class="send-button"
                  :loading="qaLoading"
                  :disabled="!question.trim()"
                  @click="handleQaSubmit()"
                >
                  <el-icon><ChatDotRound /></el-icon>
                  {{ qaLoading ? '回答中...' : '提问' }}
                </el-button>
              </div>
            </div>
          </div>

          <aside class="qa-side">
            <div class="side-card">
              <div class="side-title">问答状态</div>
              <div class="side-list">
                <div class="side-row">
                  <span>索引状态</span>
                  <strong>{{ qaStatus?.status || 'unknown' }}</strong>
                </div>
                <div class="side-row">
                  <span>Chunk 数量</span>
                  <strong>{{ qaStatus?.chunk_count || 0 }}</strong>
                </div>
                <div class="side-row">
                  <span>模型</span>
                  <strong>{{ modelBadge }}</strong>
                </div>
              </div>
            </div>

            <div class="side-card">
              <div class="side-title">索引诊断</div>
              <div v-if="qaDiagnostic" class="diagnostic-block">
                <div class="diagnostic-summary">
                  <div class="side-row">
                    <span>Collection</span>
                    <strong>{{ qaDiagnostic.collection?.name || qaStatus?.collection_name || '-' }}</strong>
                  </div>
                  <div class="side-row">
                    <span>Milvus 实体数</span>
                    <strong>{{ qaDiagnostic.collection?.info?.num_entities ?? 0 }}</strong>
                  </div>
                  <div class="side-row">
                    <span>一致性</span>
                    <strong :class="qaDiagnostic.checks.entity_count_matches_metadata ? 'ok-text' : 'warn-text'">
                      {{ qaDiagnostic.checks.entity_count_matches_metadata ? '一致' : '不一致' }}
                    </strong>
                  </div>
                  <div class="side-row">
                    <span>关键词检索</span>
                    <strong :class="qaDiagnostic.checks.likely_keyword_search_will_work ? 'ok-text' : 'warn-text'">
                      {{ qaDiagnostic.checks.likely_keyword_search_will_work ? '可用' : '有风险' }}
                    </strong>
                  </div>
                </div>

                <details class="diagnostic-details">
                  <summary>查看详细诊断</summary>
                  <div class="diagnostic-json">
                    <div class="diagnostic-line">
                      <span>DB chunk_count</span>
                      <strong>{{ qaDiagnostic.checks.qa_chunk_count ?? 0 }}</strong>
                    </div>
                    <div class="diagnostic-line">
                      <span>Milvus num_entities</span>
                      <strong>{{ qaDiagnostic.checks.milvus_num_entities ?? 0 }}</strong>
                    </div>
                    <div class="diagnostic-line">
                      <span>collection_exists</span>
                      <strong>{{ qaDiagnostic.checks.collection_exists ? 'true' : 'false' }}</strong>
                    </div>
                    <div class="diagnostic-line">
                      <span>sample_chunks</span>
                      <strong>{{ qaDiagnostic.checks.sample_chunks_returned ?? 0 }}</strong>
                    </div>
                    <div v-if="qaDiagnostic.collection?.error" class="diagnostic-error">
                      {{ qaDiagnostic.collection.error }}
                    </div>
                    <div v-if="qaDiagnostic.sample_chunks.length" class="diagnostic-samples">
                      <div v-for="(chunk, idx) in qaDiagnostic.sample_chunks" :key="idx" class="diagnostic-sample">
                        <div class="diagnostic-sample-meta">
                          <span>#{{ idx + 1 }}</span>
                          <span>chunk {{ chunk.chunk_id ?? chunk.id ?? '-' }}</span>
                          <span>page {{ chunk.page_number || chunk.page_range || '-' }}</span>
                        </div>
                        <div class="diagnostic-sample-text">
                          {{ chunk.content || '-' }}
                        </div>
                      </div>
                    </div>
                  </div>
                </details>
              </div>
              <div v-else class="diagnostic-empty">
                暂无诊断数据，点击上方“刷新状态”加载。
              </div>
            </div>

            <div class="side-card">
              <div class="side-title">使用说明</div>
              <ul class="hint-list">
                <li>问题会先转成向量，在对应论文索引中检索相似片段。</li>
                <li>当前实现是单轮问答，没有历史上下文记忆。</li>
                <li>如果后端没有足够相关片段，会返回更保守的答案。</li>
              </ul>
            </div>
            <div class="side-card">
              <div class="side-title">检索策略</div>
              <div class="retrieval-controls">
                <div class="control-row">
                  <span>Query Rewrite</span>
                  <el-switch v-model="retrievalOptions.enableQueryRewrite" />
                </div>
                <div class="control-row">
                  <span>HyDE</span>
                  <el-switch v-model="retrievalOptions.enableHyde" />
                </div>
                <div class="control-row">
                  <span>Keyword Search</span>
                  <el-switch v-model="retrievalOptions.enableKeywordSearch" />
                </div>
                <div class="control-row">
                  <span>LLM Rerank</span>
                  <el-switch v-model="retrievalOptions.enableLlmRerank" />
                </div>
                <div class="control-row">
                  <span>Debug</span>
                  <el-switch v-model="retrievalOptions.debug" />
                </div>
                <div class="control-column">
                  <span>Top K</span>
                  <el-input-number v-model="retrievalOptions.topK" :min="1" :max="15" size="small" />
                </div>
              </div>
            </div>
          </aside>
        </section>

        <el-drawer
          v-model="evidenceDrawerOpen"
          :with-header="false"
          size="42%"
          class="evidence-drawer"
          @close="closeEvidence"
        >
          <div v-if="activeEvidenceTurn" class="evidence-drawer-body">
            <div class="evidence-drawer-head">
              <div>
                <div class="drawer-title">来源与调试</div>
                <div class="drawer-question">{{ activeEvidenceTurn.question }}</div>
              </div>
              <el-button text @click="closeEvidence">关闭</el-button>
            </div>

            <section class="drawer-section">
              <div class="drawer-section-title">参考来源</div>
              <div v-if="activeEvidenceTurn.sources.length" class="source-list drawer-source-list">
                <details v-for="(source, index) in activeEvidenceTurn.sources" :key="index" class="source-item" open>
                  <summary class="source-summary">
                    <div class="source-summary-main">
                      <span class="source-index">{{ index + 1 }}</span>
                      <div class="source-summary-text">
                        <div class="source-title-row">
                          <span class="source-title">{{ source.source || '论文片段' }}</span>
                          <span class="source-badge">{{ getSourceLabel(source, index) }}</span>
                        </div>
                      </div>
                    </div>
                    <el-icon class="source-chevron"><ArrowDown /></el-icon>
                  </summary>
                  <div class="source-body">
                    <p>{{ source.content }}</p>
                  </div>
                </details>
              </div>
              <div v-else class="source-empty">这次回答没有返回结构化来源。</div>
            </section>

            <section v-if="activeEvidenceTurn.retrievalDebug" class="drawer-section">
              <div class="drawer-section-head">
                <div class="drawer-section-title">检索调试</div>
                <el-button
                  size="small"
                  text
                  :loading="traceDownloading"
                  @click="downloadRetrievalTrace('md')"
                >
                  下载 trace
                </el-button>
              </div>

              <div class="debug-grid debug-grid-drawer">
                <section class="debug-card">
                  <div class="debug-card-title">原始问题</div>
                  <div class="debug-text">{{ activeEvidenceTurn.retrievalDebug.original_query }}</div>
                </section>

                <section class="debug-card">
                  <div class="debug-card-title">查询重写</div>
                  <div class="debug-subsection">
                    <div class="debug-subtitle">模型输出</div>
                    <div v-if="activeEvidenceTurn.retrievalDebug.query_rewrite?.model_queries?.length" class="debug-chip-group">
                      <span
                        v-for="(query, idx) in activeEvidenceTurn.retrievalDebug.query_rewrite.model_queries"
                        :key="`drawer-model-${idx}`"
                        class="debug-chip"
                      >
                        {{ query }}
                      </span>
                    </div>
                    <div v-else class="debug-empty-inline">无</div>
                  </div>
                  <div class="debug-subsection">
                    <div class="debug-subtitle">最终参与检索</div>
                    <div v-if="activeEvidenceTurn.retrievalDebug.query_rewrite?.selected_queries?.length" class="debug-chip-group">
                      <span
                        v-for="(query, idx) in activeEvidenceTurn.retrievalDebug.query_rewrite.selected_queries"
                        :key="`drawer-selected-${idx}`"
                        class="debug-chip debug-chip-primary"
                      >
                        {{ query }}
                      </span>
                    </div>
                    <div v-else class="debug-empty-inline">无</div>
                  </div>
                </section>

                <section class="debug-card">
                  <div class="debug-card-title">HyDE 与关键词</div>
                  <div class="debug-subsection">
                    <div class="debug-subtitle">HyDE</div>
                    <div class="debug-text">
                      {{ activeEvidenceTurn.retrievalDebug.hyde?.text || activeEvidenceTurn.retrievalDebug.hyde_text || '无' }}
                    </div>
                  </div>
                  <div class="debug-subsection">
                    <div class="debug-subtitle">关键词查询</div>
                    <div class="debug-text">{{ formatDebugQueryList(activeEvidenceTurn.retrievalDebug.keyword_search?.queries) }}</div>
                  </div>
                </section>

                <section class="debug-card debug-card-wide">
                  <div class="debug-card-title">各路召回</div>
                  <div
                    v-for="(routeChunks, routeName) in activeEvidenceTurn.retrievalDebug.routes"
                    :key="`drawer-route-${routeName}`"
                    class="debug-route-section"
                  >
                    <div class="debug-route-title">
                      {{ getDebugRouteLabel(routeName) }}
                      <span class="debug-route-count">{{ routeChunks.length }} 条</span>
                    </div>
                    <div v-if="routeChunks.length" class="debug-route-list">
                      <details
                        v-for="(chunk, idx) in routeChunks"
                        :key="`drawer-route-${routeName}-${idx}`"
                        class="debug-chunk-fold"
                        open
                      >
                        <summary class="debug-chunk-summary">
                          <div class="debug-chunk-meta">
                            <span>#{{ idx + 1 }}</span>
                            <span>chunk {{ chunk.chunk_id ?? '-' }}</span>
                            <span>page {{ chunk.page_number || chunk.page_range || '-' }}</span>
                            <span>score {{ formatDebugNumber(chunk.route_score) }}</span>
                          </div>
                          <el-icon class="debug-chunk-chevron"><ArrowDown /></el-icon>
                        </summary>
                        <div class="debug-chunk-body">
                          <div v-if="chunk.matched_routes?.length" class="debug-chunk-source">
                            命中路由: {{ formatDebugQueryList(chunk.matched_routes) }}
                          </div>
                          <div v-if="chunk.route_scores && Object.keys(chunk.route_scores).length" class="debug-chunk-source">
                            路由分数: {{ formatRouteScores(chunk.route_scores) }}
                          </div>
                          <div v-if="chunk.source_queries?.length" class="debug-chunk-source">
                            来源 Queries: {{ formatDebugQueryList(chunk.source_queries) }}
                          </div>
                          <div class="debug-chunk-text debug-chunk-text-full">
                            {{ chunk.content || chunk.preview || '-' }}
                          </div>
                        </div>
                      </details>
                    </div>
                    <div v-else class="debug-empty">无召回</div>
                  </div>
                </section>

                <section class="debug-card debug-card-wide">
                  <div class="debug-card-title">最终融合</div>
                  <div class="debug-mini-meta">
                    <span>算法</span>
                    <strong>{{ activeEvidenceTurn.retrievalDebug.fusion?.algorithm || 'pure_rrf' }}</strong>
                  </div>
                  <div class="debug-mini-meta">
                    <span>RRF k</span>
                    <strong>{{ activeEvidenceTurn.retrievalDebug.fusion?.rrf_k ?? '-' }}</strong>
                  </div>
                  <div class="debug-mini-meta">
                    <span>路由权重</span>
                    <strong>{{ formatRouteScores(activeEvidenceTurn.retrievalDebug.fusion?.route_weights) }}</strong>
                  </div>

                  <div class="debug-route-list">
                    <details
                      v-for="(chunk, idx) in activeEvidenceTurn.retrievalDebug.final_chunks"
                      :key="`drawer-final-${idx}`"
                      class="debug-chunk-fold"
                      open
                    >
                      <summary class="debug-chunk-summary">
                        <div class="debug-chunk-meta">
                          <span>#{{ idx + 1 }}</span>
                          <span>chunk {{ chunk.chunk_id ?? '-' }}</span>
                          <span>page {{ chunk.page_number || chunk.page_range || '-' }}</span>
                          <span>fused {{ formatDebugNumber(chunk.score) }}</span>
                        </div>
                        <el-icon class="debug-chunk-chevron"><ArrowDown /></el-icon>
                      </summary>
                      <div class="debug-chunk-body">
                        <div v-if="chunk.matched_routes?.length" class="debug-chunk-source">
                          命中路由: {{ formatDebugQueryList(chunk.matched_routes) }}
                        </div>
                        <div v-if="chunk.route_scores && Object.keys(chunk.route_scores).length" class="debug-chunk-source">
                          路由分数: {{ formatRouteScores(chunk.route_scores) }}
                        </div>
                        <div v-if="chunk.source_queries?.length" class="debug-chunk-source">
                          来源 Queries: {{ formatDebugQueryList(chunk.source_queries) }}
                        </div>
                        <div class="debug-chunk-text debug-chunk-text-full">
                          {{ chunk.content || chunk.preview || '-' }}
                        </div>
                      </div>
                    </details>
                  </div>
                </section>

                <section v-if="activeEvidenceTurn.retrievalDebug.stages" class="debug-card debug-card-wide">
                  <div class="debug-card-title">四个阶段</div>
                  <div
                    v-for="(stageChunks, stageName) in activeEvidenceTurn.retrievalDebug.stages"
                    :key="`drawer-stage-${stageName}`"
                    class="debug-route-section"
                  >
                    <div class="debug-route-title">
                      {{ getDebugStageLabel(stageName) }}
                      <span class="debug-route-count">{{ stageChunks.length }} 条</span>
                    </div>
                    <div v-if="stageChunks.length" class="debug-route-list">
                      <details
                        v-for="(chunk, idx) in stageChunks"
                        :key="`drawer-stage-${stageName}-${idx}`"
                        class="debug-chunk-fold"
                        open
                      >
                        <summary class="debug-chunk-summary">
                          <div class="debug-chunk-meta">
                            <span>#{{ idx + 1 }}</span>
                            <span>chunk {{ chunk.chunk_id ?? '-' }}</span>
                            <span>page {{ chunk.page_number || chunk.page_range || '-' }}</span>
                            <span>score {{ formatDebugNumber(chunk.score) }}</span>
                          </div>
                          <el-icon class="debug-chunk-chevron"><ArrowDown /></el-icon>
                        </summary>
                        <div class="debug-chunk-body">
                          <div v-if="chunk.retrieval_route" class="debug-chunk-source">
                            来源路由: {{ getDebugRouteLabel(chunk.retrieval_route) }}
                          </div>
                          <div v-if="chunk.matched_routes?.length" class="debug-chunk-source">
                            命中路由: {{ formatDebugQueryList(chunk.matched_routes) }}
                          </div>
                          <div v-if="chunk.source_queries?.length" class="debug-chunk-source">
                            来源 Queries: {{ formatDebugQueryList(chunk.source_queries) }}
                          </div>
                          <div class="debug-chunk-text debug-chunk-text-full">
                            {{ chunk.content || chunk.preview || '-' }}
                          </div>
                        </div>
                      </details>
                    </div>
                    <div v-else class="debug-empty">无数据</div>
                  </div>
                </section>
              </div>
            </section>
          </div>
        </el-drawer>
      </template>

      <el-empty v-else description="论文不存在" />
    </div>
  </div>
</template>

<style scoped>
.paper-detail-shell {
  position: relative;
  min-height: 100vh;
  overflow: hidden;
  background:
    radial-gradient(circle at top left, rgba(118, 170, 255, 0.18), transparent 32%),
    radial-gradient(circle at top right, rgba(255, 181, 128, 0.18), transparent 30%),
    linear-gradient(180deg, #f8fbff 0%, #f4f7fb 46%, #eef3f8 100%);
}

.background-glow {
  position: absolute;
  border-radius: 999px;
  filter: blur(20px);
  pointer-events: none;
}

.background-glow-a {
  top: 40px;
  left: -120px;
  width: 320px;
  height: 320px;
  background: rgba(100, 149, 237, 0.16);
}

.background-glow-b {
  right: -90px;
  top: 240px;
  width: 280px;
  height: 280px;
  background: rgba(255, 173, 96, 0.16);
}

.paper-detail {
  position: relative;
  max-width: 1240px;
  margin: 0 auto;
  padding: 24px 20px 36px;
  font-family: 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei UI', 'Microsoft YaHei', sans-serif;
  color: #1f2937;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
}

.ghost-back {
  border-radius: 999px;
  padding-inline: 18px;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
}

.topbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.hero-card,
.qa-main,
.qa-side .side-card {
  background: rgba(255, 255, 255, 0.84);
  border: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
  backdrop-filter: blur(14px);
}

.hero-card {
  display: grid;
  grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.85fr);
  gap: 24px;
  border-radius: 28px;
  padding: 30px;
}

.hero-copy h1 {
  font-size: clamp(26px, 3vw, 40px);
  line-height: 1.2;
  margin: 10px 0 14px;
  letter-spacing: -0.02em;
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(14, 165, 233, 0.12));
  color: #334155;
  font-size: 13px;
}

.hero-summary {
  font-size: 15px;
  line-height: 1.8;
  color: #475569;
  max-width: 70ch;
}

.meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
}

.hero-aside {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 6px 0 0;
}

.metric {
  padding: 14px 16px;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(248, 250, 252, 0.95), rgba(241, 245, 249, 0.95));
  border: 1px solid rgba(148, 163, 184, 0.18);
}

.metric-label {
  display: block;
  font-size: 12px;
  color: #64748b;
  margin-bottom: 6px;
}

.metric-value {
  font-size: 14px;
  color: #111827;
  line-height: 1.5;
  word-break: break-word;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 4px;
}

.parser-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 210px;
  padding: 12px 14px;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(248, 250, 252, 0.96) 0%, rgba(241, 245, 249, 0.82) 100%);
  border: 1px solid rgba(148, 163, 184, 0.24);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
}

.parser-card-head {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.parser-label {
  font-size: 12px;
  font-weight: 800;
  color: #334155;
  letter-spacing: 0.02em;
}

.parser-hint {
  font-size: 11px;
  color: #94a3b8;
}

.parser-segment {
  width: 100%;
}

.parser-segment :deep(.el-radio-button__inner) {
  padding: 8px 14px;
  border-color: rgba(148, 163, 184, 0.28);
  background: rgba(255, 255, 255, 0.92);
  color: #475569;
  font-weight: 700;
}

.parser-segment :deep(.el-radio-button:first-child .el-radio-button__inner) {
  border-radius: 12px 0 0 12px;
}

.parser-segment :deep(.el-radio-button:last-child .el-radio-button__inner) {
  border-radius: 0 12px 12px 0;
}

.parser-segment :deep(.el-radio-button__orig-radio:checked + .el-radio-button__inner) {
  background: linear-gradient(135deg, #4f8cff 0%, #6e7cff 100%);
  border-color: #4f8cff;
  box-shadow: 0 10px 18px rgba(79, 140, 255, 0.18);
}

.parser-segment :deep(.el-radio-button__inner:hover) {
  color: #1d4ed8;
}

.action-link {
  text-decoration: none;
}

.action-button {
  border-radius: 14px;
}

.ask-button {
  box-shadow: 0 14px 26px rgba(59, 130, 246, 0.22);
}

.status-note {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  border-radius: 16px;
  color: #475569;
  background: rgba(255, 255, 255, 0.9);
  border: 1px dashed rgba(148, 163, 184, 0.35);
}

.qa-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 18px;
  margin-top: 18px;
}

.qa-main {
  border-radius: 28px;
  padding: 22px;
  min-width: 0;
}

.qa-panel-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 16px;
}

.panel-title {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.panel-subtitle {
  margin-top: 6px;
  font-size: 13px;
  color: #64748b;
}

.refresh-btn {
  color: #475569;
}

.panel-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.prompt-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 16px;
}

.prompt-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: 1px solid rgba(148, 163, 184, 0.24);
  background: linear-gradient(180deg, #ffffff, #f8fbff);
  color: #334155;
  padding: 10px 14px;
  border-radius: 999px;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.prompt-chip:hover {
  transform: translateY(-1px);
  border-color: rgba(59, 130, 246, 0.36);
  box-shadow: 0 10px 20px rgba(59, 130, 246, 0.1);
}

.chat-window {
  max-height: 680px;
  overflow: auto;
  padding-right: 6px;
  scroll-behavior: smooth;
}

.empty-chat {
  display: grid;
  place-items: center;
  text-align: center;
  min-height: 360px;
  padding: 24px;
  color: #64748b;
}

.empty-mark {
  display: grid;
  place-items: center;
  width: 72px;
  height: 72px;
  border-radius: 24px;
  margin-bottom: 14px;
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.14), rgba(168, 85, 247, 0.14));
  color: #4f46e5;
  font-size: 28px;
}

.empty-chat h3 {
  font-size: 18px;
  color: #0f172a;
  margin-bottom: 6px;
}

.turn-stack {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 18px;
}

.message {
  display: grid;
  gap: 8px;
}

.message-label {
  font-size: 12px;
  color: #64748b;
  padding-left: 6px;
}

.message-user {
  justify-items: end;
}

.message-user .message-bubble {
  max-width: min(82%, 720px);
  background: linear-gradient(135deg, #2563eb, #4f7cff);
  color: white;
  border-radius: 20px 20px 6px 20px;
  padding: 14px 16px;
  box-shadow: 0 12px 22px rgba(37, 99, 235, 0.18);
  white-space: pre-wrap;
  line-height: 1.7;
}

.message-assistant {
  justify-items: start;
}

.assistant-card {
  width: 100%;
  border-radius: 22px;
  padding: 18px 18px 14px;
  background: #ffffff;
  border: 1px solid rgba(148, 163, 184, 0.2);
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
}

.assistant-markdown {
  max-width: 100%;
  color: #1e293b;
  line-height: 1.85;
  font-size: 15px;
  letter-spacing: 0.01em;
  overflow-wrap: anywhere;
}

.assistant-markdown :deep(p) {
  margin: 0 0 14px;
}

.assistant-markdown :deep(p:last-child) {
  margin-bottom: 0;
}

.assistant-markdown :deep(.md-paragraph) {
  white-space: pre-wrap;
}

.assistant-markdown :deep(.md-empty) {
  margin: 0;
  color: #64748b;
}

.assistant-markdown :deep(.katex-display) {
  margin: 0.8em 0;
  overflow-x: auto;
}

.assistant-markdown :deep(table) {
  display: block;
  width: 100%;
  margin: 14px 0;
  border-collapse: collapse;
  border-spacing: 0;
  overflow-x: auto;
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 16px;
  background: #ffffff;
}

.assistant-markdown :deep(th),
.assistant-markdown :deep(td) {
  padding: 10px 12px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.16);
  border-right: 1px solid rgba(148, 163, 184, 0.12);
  text-align: left;
  vertical-align: top;
}

.assistant-markdown :deep(th) {
  background: #f8fafc;
  color: #0f172a;
  font-weight: 700;
}

.assistant-markdown :deep(tr:last-child td) {
  border-bottom: none;
}

.assistant-markdown :deep(h1),
.assistant-markdown :deep(h2),
.assistant-markdown :deep(h3),
.assistant-markdown :deep(h4),
.assistant-markdown :deep(h5),
.assistant-markdown :deep(h6) {
  margin: 22px 0 10px;
  line-height: 1.35;
  color: #0f172a;
  font-weight: 800;
  letter-spacing: -0.02em;
}

.assistant-markdown :deep(h1) {
  font-size: 24px;
}

.assistant-markdown :deep(h2) {
  font-size: 20px;
}

.assistant-markdown :deep(h3) {
  font-size: 18px;
}

.assistant-markdown :deep(h4),
.assistant-markdown :deep(h5),
.assistant-markdown :deep(h6) {
  font-size: 16px;
}

.assistant-markdown :deep(blockquote) {
  margin: 14px 0;
  padding: 12px 14px;
  border-left: 4px solid rgba(59, 130, 246, 0.4);
  border-radius: 0 14px 14px 0;
  background: rgba(59, 130, 246, 0.06);
  color: #334155;
}

.assistant-markdown :deep(ul),
.assistant-markdown :deep(ol) {
  margin: 10px 0 14px;
  padding-left: 1.35em;
}

.assistant-markdown :deep(li) {
  margin: 6px 0;
}

.assistant-markdown :deep(code) {
  padding: 0.18em 0.48em;
  border-radius: 8px;
  background: rgba(37, 99, 235, 0.08);
  color: #0f172a;
  font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
  font-size: 0.92em;
}

.assistant-markdown :deep(pre) {
  margin: 14px 0;
  padding: 16px 18px;
  border-radius: 18px;
  background: linear-gradient(180deg, #0f172a, #111827);
  color: #e2e8f0;
  overflow-x: auto;
  box-shadow:
    inset 0 0 0 1px rgba(148, 163, 184, 0.08),
    0 12px 28px rgba(15, 23, 42, 0.12);
}

.assistant-markdown :deep(pre code) {
  display: block;
  padding: 0;
  background: transparent;
  color: inherit;
  white-space: pre;
  font-size: 13px;
  line-height: 1.7;
}

.assistant-markdown :deep(a) {
  color: #2563eb;
  text-decoration: none;
  border-bottom: 1px solid rgba(37, 99, 235, 0.22);
}

.assistant-markdown :deep(a:hover) {
  border-bottom-color: rgba(37, 99, 235, 0.65);
}

.assistant-markdown :deep(strong) {
  color: #0f172a;
  font-weight: 800;
}

.assistant-markdown :deep(em) {
  color: #0f172a;
  font-style: italic;
}

.assistant-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 14px;
  padding: 12px 14px;
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(241, 245, 249, 0.96));
  border: 1px solid rgba(148, 163, 184, 0.16);
}

.assistant-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.evidence-button {
  padding-inline: 10px;
}

.streaming-indicator {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  font-size: 13px;
  color: #64748b;
}

.streaming-evidence-note {
  margin-top: 12px;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(59, 130, 246, 0.08);
  color: #2563eb;
  font-size: 12px;
  line-height: 1.6;
}

.evidence-drawer {
  --el-drawer-padding-primary: 0;
}

.evidence-drawer :deep(.el-drawer__body) {
  padding: 0;
  overflow: hidden;
}

.evidence-drawer-body {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
  padding: 20px;
  overflow: auto;
  background:
    radial-gradient(circle at top right, rgba(96, 165, 250, 0.08), transparent 30%),
    linear-gradient(180deg, #ffffff, #f8fbff);
}

.evidence-drawer-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.drawer-title {
  font-size: 18px;
  font-weight: 800;
  color: #0f172a;
}

.drawer-question {
  margin-top: 6px;
  font-size: 13px;
  line-height: 1.6;
  color: #475569;
}

.drawer-section {
  padding: 16px;
  border-radius: 20px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
}

.drawer-section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.drawer-section-title {
  font-size: 15px;
  font-weight: 800;
  color: #0f172a;
  margin-bottom: 0;
}

.drawer-source-list {
  max-height: none;
  overflow: auto;
  padding-right: 4px;
}

.debug-grid-drawer {
  grid-template-columns: 1fr;
}

.source-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.source-item {
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: linear-gradient(180deg, #f8fbff, #ffffff);
  padding: 12px 14px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.source-item[open] {
  border-color: rgba(96, 165, 250, 0.38);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

.source-summary {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  cursor: pointer;
  list-style: none;
}

.source-summary::-webkit-details-marker {
  display: none;
}

.source-summary-main {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  min-width: 0;
}

.source-index {
  flex: 0 0 auto;
  width: 28px;
  height: 28px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(180deg, rgba(96, 165, 250, 0.18), rgba(99, 102, 241, 0.14));
  color: #2563eb;
  font-size: 12px;
  font-weight: 700;
}

.source-summary-text {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.source-title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.source-badge {
  font-size: 12px;
  font-weight: 600;
  color: #2563eb;
  background: rgba(37, 99, 235, 0.12);
  padding: 4px 8px;
  border-radius: 999px;
}

.source-title {
  font-size: 14px;
  font-weight: 600;
  color: #0f172a;
}

.source-chevron {
  flex: 0 0 auto;
  color: #94a3b8;
  margin-top: 4px;
  transition: transform 0.2s ease;
}

.source-item[open] .source-chevron {
  transform: rotate(180deg);
}

.source-body {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid rgba(148, 163, 184, 0.14);
}

.source-body p {
  margin: 0;
  color: #475569;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.source-empty {
  color: #64748b;
  font-size: 13px;
}

.debug-panel {
  margin-top: 16px;
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 20px;
  background: rgba(248, 250, 252, 0.96);
  padding: 14px;
}

.debug-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  cursor: pointer;
  font-weight: 700;
  color: #1e293b;
}

.debug-summary-badge {
  font-size: 12px;
  font-weight: 700;
  color: #2563eb;
  background: rgba(37, 99, 235, 0.1);
  padding: 4px 10px;
  border-radius: 999px;
}

.debug-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.debug-card,
.debug-card-wide {
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: #ffffff;
  padding: 12px 12px 10px;
}

.debug-card-wide {
  grid-column: 1 / -1;
}

.debug-card-title {
  font-size: 14px;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 10px;
}

.debug-subsection {
  margin-top: 10px;
}

.debug-subtitle {
  font-size: 12px;
  font-weight: 700;
  color: #475569;
  margin-bottom: 8px;
}

.debug-text {
  white-space: pre-wrap;
  line-height: 1.7;
  color: #1e293b;
  font-size: 13px;
}

.debug-chip-group {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.debug-chip {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.12);
  color: #334155;
  font-size: 12px;
  line-height: 1.4;
  word-break: break-word;
}

.debug-chip-primary {
  background: rgba(37, 99, 235, 0.12);
  color: #1d4ed8;
}

.debug-chip-soft {
  background: rgba(99, 102, 241, 0.1);
  color: #4f46e5;
}

.debug-detail-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.debug-detail-item {
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(248, 250, 252, 0.96);
  border: 1px solid rgba(203, 213, 225, 0.85);
}

.debug-detail-query {
  font-size: 13px;
  font-weight: 600;
  color: #0f172a;
  line-height: 1.6;
}

.debug-candidate-source {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-right: 8px;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.1);
  color: #1d4ed8;
  font-size: 11px;
  font-weight: 700;
}

.debug-detail-meta,
.debug-mini-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  font-size: 12px;
  color: #334155;
  margin-top: 6px;
}

.debug-empty-inline,
.debug-warning {
  font-size: 12px;
  line-height: 1.5;
  margin-top: 8px;
}

.debug-empty-inline {
  color: #64748b;
}

.debug-warning {
  color: #b45309;
  background: rgba(251, 191, 36, 0.12);
  border: 1px solid rgba(251, 191, 36, 0.24);
  border-radius: 12px;
  padding: 8px 10px;
}

.debug-route-section {
  margin-top: 12px;
}

.debug-route-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  font-weight: 700;
  color: #334155;
  margin-bottom: 8px;
}

.debug-route-count {
  font-size: 12px;
  font-weight: 700;
  color: #2563eb;
  background: rgba(37, 99, 235, 0.1);
  padding: 3px 8px;
  border-radius: 999px;
}

.debug-route-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.debug-chunk {
  padding: 0;
  border-radius: 16px;
  background: #ffffff;
  border: 1px solid rgba(203, 213, 225, 0.85);
  overflow: hidden;
}

.debug-chunk-source {
  margin-bottom: 6px;
  font-size: 12px;
  line-height: 1.5;
  color: #475569;
}

.debug-chunk-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 12px;
  color: #64748b;
}

.debug-chunk-fold {
  border-radius: 16px;
  background: #ffffff;
}

.debug-chunk-summary {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  cursor: pointer;
  list-style: none;
  background:
    linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.92));
  border-bottom: 1px solid rgba(203, 213, 225, 0.55);
}

.debug-chunk-summary::-webkit-details-marker {
  display: none;
}

.debug-chunk-fold:not([open]) .debug-chunk-summary {
  border-bottom-color: transparent;
}

.debug-chunk-fold[open] .debug-chunk-summary {
  background:
    linear-gradient(180deg, rgba(248, 250, 252, 1), rgba(255, 255, 255, 1));
}

.debug-chunk-chevron {
  flex: none;
  margin-top: 1px;
  color: #94a3b8;
  transition: transform 0.2s ease, color 0.2s ease;
}

.debug-chunk-fold[open] .debug-chunk-chevron {
  transform: rotate(180deg);
  color: #334155;
}

.debug-chunk-body {
  padding: 14px 16px 16px;
  background: #ffffff;
}

.debug-chunk-text,
.debug-empty {
  font-size: 13px;
  color: #1f2937;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.debug-chunk-text-full {
  margin-top: 6px;
  font-size: 14px;
  line-height: 1.75;
  color: #0f172a;
  white-space: pre-wrap;
  word-break: break-word;
}

.debug-chunk-fold .debug-chunk-body .debug-chunk-text-full {
  padding: 12px 14px;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px solid rgba(226, 232, 240, 0.95);
}

.debug-fusion-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 14px;
  margin-bottom: 10px;
}

.debug-fusion-full {
  grid-column: 1 / -1;
}

.answer-meta {
  display: flex;
  justify-content: flex-end;
  margin-top: 12px;
  font-size: 12px;
  color: #94a3b8;
}

.assistant-loading {
  padding: 18px 16px 20px;
}

.loading-line {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: #475569;
}

.spin {
  animation: spin 1s linear infinite;
}

.typing-dots {
  display: flex;
  gap: 6px;
  margin-top: 12px;
}

.typing-dots span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: linear-gradient(180deg, #60a5fa, #8b5cf6);
  animation: bounce 1.1s infinite ease-in-out;
}

.typing-dots span:nth-child(2) {
  animation-delay: 0.15s;
}

.typing-dots span:nth-child(3) {
  animation-delay: 0.3s;
}

.composer {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid rgba(148, 163, 184, 0.16);
}

.composer-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 12px;
}

.composer-tip {
  font-size: 12px;
  color: #64748b;
}

.send-button {
  min-width: 120px;
  border-radius: 14px;
}

.qa-side {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.retrieval-controls {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.control-row,
.control-column {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  color: #475569;
}

.control-column {
  align-items: flex-start;
  flex-direction: column;
}

.side-card {
  border-radius: 24px;
  padding: 18px;
}

.diagnostic-block {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.diagnostic-summary {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.diagnostic-details {
  border-top: 1px solid rgba(148, 163, 184, 0.16);
  padding-top: 10px;
}

.diagnostic-details summary {
  cursor: pointer;
  font-size: 13px;
  color: #2563eb;
  font-weight: 600;
}

.diagnostic-json {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 12px;
}

.diagnostic-line,
.diagnostic-sample-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
  color: #475569;
}

.diagnostic-error {
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(239, 68, 68, 0.08);
  color: #b91c1c;
  font-size: 12px;
  line-height: 1.5;
}

.diagnostic-samples {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.diagnostic-sample {
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(248, 250, 252, 0.92);
  border: 1px solid rgba(148, 163, 184, 0.12);
}

.diagnostic-sample-text {
  margin-top: 8px;
  font-size: 12px;
  line-height: 1.6;
  color: #334155;
  max-height: 120px;
  overflow: auto;
}

.diagnostic-empty {
  font-size: 13px;
  color: #64748b;
  line-height: 1.6;
}

.ok-text {
  color: #15803d;
}

.warn-text {
  color: #b45309;
}

.side-title {
  font-size: 16px;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 14px;
}

.side-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.side-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 0;
  border-bottom: 1px solid rgba(148, 163, 184, 0.12);
  color: #475569;
  font-size: 13px;
}

.side-row:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.side-row strong {
  color: #111827;
  text-align: right;
}

.hint-list {
  margin: 0;
  padding-left: 18px;
  color: #475569;
  line-height: 1.8;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

@keyframes bounce {
  0%,
  80%,
  100% {
    transform: scale(0.6);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

@media (max-width: 1024px) {
  .hero-card,
  .qa-layout {
    grid-template-columns: 1fr;
  }

  .qa-side {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 768px) {
  .paper-detail {
    padding: 16px 12px 28px;
  }

  .topbar,
  .qa-panel-header,
  .composer-actions {
    flex-direction: column;
    align-items: stretch;
  }

  .hero-card,
  .qa-main,
  .side-card {
    border-radius: 20px;
    padding: 18px;
  }

  .message-user .message-bubble {
    max-width: 100%;
  }

  .qa-side {
    grid-template-columns: 1fr;
  }
}
</style>
