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
import { usePaperStore } from '@/stores/paperStore'
import {
  createPaperQaIndex,
  getPaperQaStatus,
  qaPaperStream,
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
const qaStatus = ref<QaStatusResult | null>(null)
const question = ref('')
const chatContainerRef = ref<HTMLElement | null>(null)
const qaResults = ref<QaTurn[]>([])
const retrievalOptions = reactive({
  enableQueryRewrite: true,
  enableHyde: true,
  enableKeywordSearch: true,
  debug: false,
  topK: 5
})

const modelBadge = 'Qwen 3.6 Plus'

const quickPrompts = computed(() => [
  '请总结这篇论文的核心贡献。',
  '这篇论文的方法流程是怎样的？',
  '实验结果说明了什么，局限性有哪些？'
])

const hasQaIndex = computed(() => Boolean(qaStatus.value?.has_index))
const chatTurns = computed(() => qaResults.value)

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
  qaResults.value.unshift(turn)
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
              <el-button size="large" type="primary" class="action-button ask-button" @click="handleAskPaper" :loading="creatingIndex">
                <el-icon><ChatDotRound /></el-icon>
                {{ creatingIndex ? '创建索引中...' : '开始问答' }}
              </el-button>
            </div>

            <div class="status-note" v-if="qaStatus">
              <el-icon><Compass /></el-icon>
              <span>
                {{ qaStatus.has_index ? `索引已完成，${qaStatus.chunk_count || 0} 个 chunks 可供检索` : '先创建索引，再进入问答' }}
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
                    <div class="assistant-text">{{ turn.answer }}</div>
                    <div v-if="turn.streaming" class="streaming-indicator">
                      <el-icon class="spin"><Loading /></el-icon>
                      <span>正在流式生成中...</span>
                    </div>

                    <el-divider content-position="left">参考来源</el-divider>

                    <div v-if="turn.sources.length" class="source-list">
                      <details v-for="(source, index) in turn.sources" :key="index" class="source-item">
                        <summary class="source-summary">
                          <div class="source-summary-main">
                            <span class="source-index">{{ index + 1 }}</span>
                            <div class="source-summary-text">
                              <div class="source-title-row">
                                <span class="source-title">{{ source.source || '论文片段' }}</span>
                                <span class="source-badge">{{ getSourceLabel(source, index) }}</span>
                              </div>
                              <div class="source-snippet">
                                {{ source.content }}
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

                    <div v-else class="source-empty">
                      这次回答没有返回结构化来源。
                    </div>

                    <details v-if="turn.retrievalDebug" class="debug-panel">
                      <summary class="debug-summary">检索调试信息</summary>
                      <div class="debug-section">
                        <div class="debug-row"><strong>原始 Query:</strong> {{ turn.retrievalDebug.original_query }}</div>
                        <div class="debug-row"><strong>Rewrite:</strong> {{ turn.retrievalDebug.rewritten_queries.join(' | ') || '无' }}</div>
                        <div class="debug-row"><strong>HyDE:</strong> {{ turn.retrievalDebug.hyde_text || '无' }}</div>
                      </div>
                      <div v-for="(routeChunks, routeName) in turn.retrievalDebug.routes" :key="routeName" class="debug-section">
                        <div class="debug-route-title">{{ routeName }}</div>
                        <div v-if="routeChunks.length" class="debug-route-list">
                          <div v-for="(chunk, idx) in routeChunks" :key="`${routeName}-${idx}`" class="debug-chunk">
                            <div class="debug-chunk-meta">
                              <span>#{{ idx + 1 }}</span>
                              <span>chunk {{ chunk.chunk_id ?? '-' }}</span>
                              <span>page {{ chunk.page_number || chunk.page_range || '-' }}</span>
                              <span>score {{ typeof chunk.route_score === 'number' ? chunk.route_score.toFixed(4) : '-' }}</span>
                            </div>
                            <div class="debug-chunk-text">{{ chunk.preview }}</div>
                          </div>
                        </div>
                        <div v-else class="debug-empty">无召回</div>
                      </div>
                      <div class="debug-section">
                        <div class="debug-route-title">final_chunks</div>
                        <div class="debug-route-list">
                          <div v-for="(chunk, idx) in turn.retrievalDebug.final_chunks" :key="`final-${idx}`" class="debug-chunk">
                            <div class="debug-chunk-meta">
                              <span>#{{ idx + 1 }}</span>
                              <span>chunk {{ chunk.chunk_id ?? '-' }}</span>
                              <span>page {{ chunk.page_number || chunk.page_range || '-' }}</span>
                              <span>fused {{ typeof chunk.score === 'number' ? chunk.score.toFixed(4) : '-' }}</span>
                            </div>
                            <div class="debug-chunk-text">{{ chunk.preview }}</div>
                          </div>
                        </div>
                      </div>
                    </details>

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
                  <span>Debug</span>
                  <el-switch v-model="retrievalOptions.debug" />
                </div>
                <div class="control-column">
                  <span>Top K</span>
                  <el-input-number v-model="retrievalOptions.topK" :min="1" :max="12" size="small" />
                </div>
              </div>
            </div>
          </aside>
        </section>
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
  padding: 16px 16px 14px;
  background: #ffffff;
  border: 1px solid rgba(148, 163, 184, 0.2);
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
}

.assistant-text {
  white-space: pre-wrap;
  line-height: 1.85;
  color: #1e293b;
}

.streaming-indicator {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  font-size: 13px;
  color: #64748b;
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

.source-snippet {
  color: #475569;
  font-size: 13px;
  line-height: 1.65;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
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
}

.source-empty {
  color: #64748b;
  font-size: 13px;
}

.debug-panel {
  margin-top: 16px;
  border: 1px solid rgba(148, 163, 184, 0.32);
  border-radius: 16px;
  background: rgba(248, 250, 252, 0.9);
  padding: 12px 14px;
}

.debug-summary {
  cursor: pointer;
  font-weight: 700;
  color: #1e293b;
}

.debug-section {
  margin-top: 12px;
}

.debug-row,
.debug-route-title {
  font-size: 13px;
  color: #334155;
}

.debug-route-title {
  font-weight: 700;
  margin-bottom: 8px;
}

.debug-route-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.debug-chunk {
  padding: 10px 12px;
  border-radius: 12px;
  background: #ffffff;
  border: 1px solid rgba(203, 213, 225, 0.8);
}

.debug-chunk-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 12px;
  color: #64748b;
  margin-bottom: 6px;
}

.debug-chunk-text,
.debug-empty {
  font-size: 13px;
  color: #1f2937;
  line-height: 1.6;
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
