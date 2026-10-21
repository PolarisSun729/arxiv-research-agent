<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  ArrowLeft,
  ChatDotRound,
  Compass,
  Document,
  Link,
  RefreshRight
} from '@element-plus/icons-vue'
import RagChatPanel from '@/components/rag-chat/RagChatPanel.vue'
import RagEvidencePanel from '@/components/rag-chat/RagEvidencePanel.vue'
import RagRemovedClaims from '@/components/rag-chat/RagRemovedClaims.vue'
import { buildRemovedClaimsView } from '@/utils/researchProcess'
import { usePaperRagChat } from '@/composables/usePaperRagChat'
import { useUserContext } from '@/composables/useUserContext'
import { usePaperStore } from '@/stores/paperStore'
import { qaTurnToRagMessages } from '@/types/ragChat'
import { downloadApiFile } from '@/api/auth'
import { getErrorMessage } from '@/api/errors'
import {
  createPaperQaIndex,
  getLatestPaperQaIndexJob,
  getPaperQaIndexJob,
  getPaperQaDiagnostic,
  getPaperQaStatus,
  getPaperRetrievalTraceDownloadUrl,
  type QaDiagnosticResult,
  type QaIndexJobResult,
  type QaStatusResult
} from '@/api/papers'

const route = useRoute()
const router = useRouter()
const store = usePaperStore()
const { isAdmin, canResearch } = useUserContext()

const paperId = computed(() => route.params.id as string)
const loading = ref(true)
const abstractExpanded = ref(false)
const qaMode = ref(false)
const creatingIndex = ref(false)
const qaIndexJob = ref<QaIndexJobResult | null>(null)
const qaJobPolling = ref(false)
// 连续轮询失败次数，用于退避和提示“正在重试”。
const qaJobPollFailures = ref(0)
const qaStatus = ref<QaStatusResult | null>(null)
const qaDiagnostic = ref<QaDiagnosticResult | null>(null)
const traceDownloading = ref(false)
const isNarrowScreen = ref(false)
const highlightedEvidenceSourceId = ref<string | null>(null)
const sideEvidenceCard = ref<HTMLElement | null>(null)
const sideEvidenceFlash = ref(false)

const modelBadge = 'Qwen 3.6 Plus'

const {
  qaResults,
  qaLoading,
  sessionLoading,
  evidenceDrawerOpen,
  activeEvidenceTurn,
  currentSession,
  availableSessions,
  retrievalOptions,
  submitQuestion,
  resubmitTurn,
  cancelCurrentStream,
  applyPrompt,
  closeEvidence,
  resetChat,
  loadRecentSession,
  loadSession,
  createNewSession,
  clearCurrentSession
} = usePaperRagChat({
  paperId,
  onEnterQaMode: () => {
    qaMode.value = true
  }
})

const quickPrompts = computed(() => [
  '请总结这篇论文的核心贡献。',
  '这篇论文的方法流程是怎样的？',
  '实验结果说明了什么，局限性有哪些？'
])

const hasQaIndex = computed(() => Boolean(qaStatus.value?.has_index))
const hasActiveQaJob = computed(() => ['pending', 'running'].includes(String(qaIndexJob.value?.status || '')))
// 任务块只展示构建过程：进行中显示进度，失败显示原因。索引是否可用只看索引状态，
// 成功的任务只是历史记录（索引之后可能已被删除或替换），不再展示。
const showQaJobStatus = computed(() => hasActiveQaJob.value || qaIndexJob.value?.status === 'failed')
const chatTurns = computed(() => qaResults.value)
const ragChatMessages = computed(() => chatTurns.value.flatMap(qaTurnToRagMessages))
// Milvus 存的是 RetrievalIndex 向量（非原始 chunk），核对基准是后端按入库规则算出的预期向量数。
const vectorCountCheck = computed(() => {
  const checks = qaDiagnostic.value?.checks
  if (checks?.entity_count_matches_metadata == null) return { ok: false, text: '无法核对' }
  if (checks.entity_count_matches_metadata) return { ok: true, text: '与预期一致' }
  const diff = (checks.expected_vector_count ?? 0) - (checks.milvus_num_entities ?? 0)
  return { ok: false, text: diff > 0 ? `缺 ${diff} 条` : `多 ${-diff} 条` }
})
const qaJobStatusText = computed(() => {
  const status = String(qaIndexJob.value?.status || '').trim().toLowerCase()
  if (status === 'pending') return '等待开始'
  if (status === 'running') return '构建中'
  if (status === 'success') return '已完成'
  if (status === 'failed') return '失败'
  return status || '未知'
})
const qaStageText = computed(() => {
  const stage = String(qaIndexJob.value?.current_stage || '').trim()
  const stageMap: Record<string, string> = {
    pending: '等待开始',
    starting: '任务启动中',
    validate_loading_method: '校验解析方式',
    create_build_version: '创建构建版本',
    mark_index_processing: '标记索引处理中',
    load_paper_metadata: '加载论文元数据',
    download_pdf: '下载 PDF',
    load_pdf_document: '解析 PDF 文档',
    chunk_document: '切分文档 chunks',
    structure_table_chunks: '整理表格 chunks',
    save_chunk_file: '保存 chunk 文件',
    save_sparse_index_artifact: '保存关键词索引',
    compress_chunks_for_rerank: '生成 rerank 文本',
    build_retrieval_indexes: '生成检索索引',
    save_retrieval_index_artifact: '保存检索索引',
    create_chunk_embeddings: '生成 embeddings',
    save_embeddings: '保存 embeddings',
    index_embeddings_to_vector_store: '写入向量库',
    validate_new_collection: '校验新向量集合',
    validate_sparse_index_artifact: '校验关键词索引',
    activate_index: '启用新索引',
    mark_index_success: '索引完成'
  }
  return stageMap[stage] || stage || '-'
})
const currentPaperActions = computed(() => store.currentPaper?.paperActions || {})
const preferredAnswerStyle = computed(() => store.researchProfile?.preferred_answer_style || '')

let qaJobPollTimer: number | null = null

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

function getTraceFileName(pathValue?: string) {
  if (!pathValue) return ''
  const parts = pathValue.split(/[/\\]+/)
  return parts[parts.length - 1] || ''
}

async function downloadRetrievalTrace(format: 'md' | 'json' = 'md') {
  if (!isAdmin.value) return
  if (traceDownloading.value || !paperId.value) return

  traceDownloading.value = true
  try {
    const traceName = activeEvidenceTurn.value?.retrievalDebug?.trace_export?.[format]
      ? getTraceFileName(activeEvidenceTurn.value.retrievalDebug.trace_export[format])
      : ''
    const url = getPaperRetrievalTraceDownloadUrl(paperId.value, format, traceName || undefined)
    await downloadApiFile(url, traceName || `retrieval-trace.${format}`)
  } catch (error) {
    ElMessage.error(getErrorMessage(error, '下载失败'))
  } finally {
    traceDownloading.value = false
  }
}

function goBack() {
  router.back()
}

function syncViewport() {
  if (typeof window === 'undefined') return
  isNarrowScreen.value = window.innerWidth < 1280
  if (!isNarrowScreen.value) {
    evidenceDrawerOpen.value = false
  }
}

// 宽屏下证据面板在右栏顶部，用户在回答底部点按钮时它多半在视口外：切换后把面板滚进视野并闪一下，否则看起来"点了没反应"。
function revealSideEvidence(resetScroll: boolean) {
  if (isNarrowScreen.value) return
  sideEvidenceFlash.value = false
  nextTick(() => {
    const card = sideEvidenceCard.value
    if (!card) return
    if (resetScroll) {
      card.scrollTop = 0
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
    requestAnimationFrame(() => {
      sideEvidenceFlash.value = true
    })
  })
}

function handlePanelOpenEvidence(turnId: string) {
  const turn = chatTurns.value.find(item => item.id === turnId)
  if (!turn) return
  activeEvidenceTurn.value = turn
  highlightedEvidenceSourceId.value = null
  evidenceDrawerOpen.value = isNarrowScreen.value
  revealSideEvidence(true)
}

function handleCitationSelect(item: any, sourceId: string) {
  const turn = item?.response && typeof item.response === 'object' ? item.response : null
  if (!turn) return
  activeEvidenceTurn.value = turn
  // 先清空再设置：重复点同一个角标时高亮值不变，来源列表的 watch 不会再次滚动定位。
  highlightedEvidenceSourceId.value = null
  nextTick(() => {
    highlightedEvidenceSourceId.value = sourceId
  })
  evidenceDrawerOpen.value = isNarrowScreen.value
  // 滚动由来源列表把被点的来源居中完成，这里只负责闪一下提示。
  revealSideEvidence(false)
}

async function handleToggleRead() {
  if (!store.currentPaper) return
  try {
    const enabled = !currentPaperActions.value.read
    await store.togglePaperAction(store.currentPaper, 'read', enabled)
    ElMessage.success(enabled ? '已标记已读' : '已取消已读')
  } catch (error: any) {
    ElMessage.error(error?.response?.data?.detail || '操作失败')
  }
}

// 与列表卡片共用 like/dislike 偏好；再次点击已选中的标记即取消。
async function handleToggleLabel(label: 'liked' | 'disliked') {
  if (!store.currentPaper) return
  const next = store.currentPaper.label === label ? null : label
  try {
    await store.updateLabel(store.currentPaper.id, next)
    ElMessage.success(next === null ? '已取消标记' : `已标记为${next === 'liked' ? '感兴趣' : '不感兴趣'}`)
  } catch {
    ElMessage.error('偏好保存失败')
  }
}

async function fetchQaStatus() {
  try {
    const [statusResult, diagnosticResult] = await Promise.all([
      getPaperQaStatus(paperId.value),
      isAdmin.value ? getPaperQaDiagnostic(paperId.value).catch(error => {
        console.error('Failed to fetch QA diagnostic:', error)
        return null
      }) : Promise.resolve(null)
    ])
    qaStatus.value = statusResult
    qaDiagnostic.value = diagnosticResult
  } catch (error) {
    console.error('Failed to fetch QA status:', error)
    qaStatus.value = { arxiv_id: paperId.value, has_index: false, status: 'not_indexed' }
    qaDiagnostic.value = null
  }
}

async function handleRefreshQaState() {
  await fetchQaStatus()
  await resumeLatestQaJobPolling()
}

function clearQaJobPollTimer() {
  if (qaJobPollTimer) {
    clearTimeout(qaJobPollTimer)
    qaJobPollTimer = null
  }
}

function stopQaJobPolling() {
  qaJobPolling.value = false
  qaJobPollFailures.value = 0
  clearQaJobPollTimer()
}

function scheduleQaJobPolling(jobId: string, delayMs: number = 2500) {
  clearQaJobPollTimer()
  qaJobPolling.value = true
  qaJobPollTimer = window.setTimeout(() => {
    pollQaJobStatus(jobId).catch(error => {
      console.error('Failed to poll QA index job:', error)
    })
  }, delayMs)
}

async function pollQaJobStatus(jobId: string) {
  if (!paperId.value || !jobId) return

  let job: QaIndexJobResult
  try {
    job = await getPaperQaIndexJob(paperId.value, jobId)
  } catch (error: any) {
    const status = error?.httpStatus
    // 任务不存在或无权查看时重试也没用，直接停止；登录失效由请求层统一处理。
    if (status === 404 || status === 401 || status === 403) {
      stopQaJobPolling()
      return
    }
    // 其余多为暂时性错误（服务重启时的 502、网络抖动、超时），任务仍在后台运行，退避后继续轮询。
    qaJobPollFailures.value += 1
    console.warn('QA index job poll failed, retrying:', error)
    scheduleQaJobPolling(jobId, Math.min(2500 * 2 ** qaJobPollFailures.value, 15000))
    return
  }

  qaJobPollFailures.value = 0
  qaIndexJob.value = job

  if (job.status === 'success') {
    stopQaJobPolling()
    ElMessage.success('问答索引创建完成')
    await fetchQaStatus()
    return
  }

  if (job.status === 'failed') {
    stopQaJobPolling()
    await fetchQaStatus()
    ElMessage.error(job.error_message || `索引创建失败：${qaStageText.value}`)
    return
  }

  if (job.status === 'pending' || job.status === 'running') {
    scheduleQaJobPolling(jobId)
    return
  }

  stopQaJobPolling()
}

async function resumeLatestQaJobPolling() {
  stopQaJobPolling()
  qaIndexJob.value = null
  if (!paperId.value) return

  try {
    const job = await getLatestPaperQaIndexJob(paperId.value)
    qaIndexJob.value = job
    if (job.status === 'pending' || job.status === 'running') {
      scheduleQaJobPolling(job.job_id, 1500)
    }
  } catch (error: any) {
    if (error?.httpStatus !== 404) {
      console.error('Failed to fetch latest QA index job:', error)
    }
  }
}

async function handleCreateIndex() {
  if (creatingIndex.value || !canResearch.value) return

  creatingIndex.value = true
  try {
    const result = await createPaperQaIndex(paperId.value)
    if (result.job_id) {
      qaIndexJob.value = {
        job_id: result.job_id,
        arxiv_id: result.arxiv_id,
        status: result.job_status || 'pending',
        current_stage: result.current_stage || 'pending',
        progress: typeof result.progress === 'number' ? result.progress : 0,
        loading_method: result.loading_method || 'docling',
        error_message: null
      }
      ElMessage.success('问答索引任务已提交，正在后台构建')
      scheduleQaJobPolling(result.job_id, 1500)
      return
    }

    ElMessage.success('问答索引创建成功')
    await fetchQaStatus()
  } catch (error: any) {
    ElMessage.error(error?.response?.data?.detail || '创建索引失败')
  } finally {
    creatingIndex.value = false
  }
}

async function handleAskPaper() {
  if (!hasQaIndex.value) {
    if (!canResearch.value) {
      ElMessage.info('这篇论文还没有问答索引，请联系研究者或管理员创建。')
      return
    }
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
  if (!currentSession.value && !sessionLoading.value) {
    await loadRecentSession()
  }
}

function closeQaMode() {
  qaMode.value = false
}

async function handleCreateSession() {
  qaMode.value = true
  await createNewSession()
}

async function handleClearSession() {
  await clearCurrentSession()
}

function handleStopGeneration() {
  cancelCurrentStream('user')
}

function getTurnStatusHint(item: { response?: any }) {
  const turn = item.response
  if (!turn) return ''
  if (turn.status === 'aborted') return '已取消生成，当前内容未完整保存。'
  if (turn.status === 'partial' || turn.status === 'interrupted') return '回答中断，已保留当前部分内容。'
  if (turn.status === 'persistence_failed' || turn.persistenceStatus === 'failed') {
    return '答案已生成，但未保存到历史记录。'
  }
  if (turn.status === 'failed') return turn.error || '问答失败，请稍后重试。'
  // 研究引擎的部分回答：执行成功，但有主张没找到论文证据支持，已从正文中删去。
  if (turn.retrievalDebug?.qa_observation?.outcome === 'partial') {
    // 旧记录没有 removed_claims，不能指引用户去看不存在的列表。
    const removed = buildRemovedClaimsView(turn.retrievalDebug)
    return removed.items.length || removed.omittedCount
      ? '部分内容未找到论文证据支持，已从回答中移除，可在下方查看被移除的句子与原因。'
      : '部分内容未找到论文证据支持，已从回答中移除。'
  }
  return ''
}

function canResubmitTurn(item: { response?: any }) {
  // 面板消息同时兼容 Agent/QA 响应，只有 QA turn 的终态非 completed 时才展示重试入口。
  return Boolean(item.response && item.response.status !== 'completed')
}

async function copyAssistantAnswer(item: { content?: string }) {
  const content = String(item.content || '').trim()
  if (!content) {
    ElMessage.warning('当前回答没有可复制的内容')
    return
  }
  try {
    await navigator.clipboard.writeText(content)
    ElMessage.success('回答已复制')
  } catch {
    ElMessage.error('复制失败，请手动选择文本复制')
  }
}

async function handleResumeSession(sessionId: string) {
  qaMode.value = true
  await loadSession(sessionId)
}

onMounted(async () => {
  syncViewport()
  window.addEventListener('resize', syncViewport)
  try {
    await Promise.all([store.fetchResearchProfile(), store.fetchPaperActions()])
    await store.fetchPaperById(paperId.value)
    await fetchQaStatus()
    await resumeLatestQaJobPolling()
    await loadRecentSession()
  } catch (error) {
    console.error('Failed to fetch paper:', error)
  } finally {
    loading.value = false
  }
})

onBeforeUnmount(() => {
  cancelCurrentStream('cleanup')
  stopQaJobPolling()
  if (typeof window === 'undefined') return
  window.removeEventListener('resize', syncViewport)
})

watch(paperId, async () => {
  stopQaJobPolling()
  loading.value = true
  try {
    await Promise.all([store.fetchResearchProfile(), store.fetchPaperActions()])
    await store.fetchPaperById(paperId.value)
    await fetchQaStatus()
    await resumeLatestQaJobPolling()
    resetChat()
    await loadRecentSession()
  } catch (error) {
    console.error('Failed to fetch paper:', error)
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div class="page page--wide paper-detail">
    <div class="topbar">
      <el-button text :icon="ArrowLeft" class="back-btn" @click="goBack">返回</el-button>
      <div class="topbar-right">
        <span v-if="qaStatus" :class="['chip', qaStatus.has_index ? 'chip--success' : 'chip--warning']">
          {{ qaStatus.has_index ? '已建立问答索引' : '未建立问答索引' }}
        </span>
        <span class="chip">{{ modelBadge }}</span>
      </div>
    </div>

    <div v-if="loading" class="surface skeleton-card">
      <el-skeleton :rows="8" animated />
    </div>

    <template v-else-if="store.currentPaper">
      <section class="surface hero">
        <div class="hero-main">
          <div class="hero-meta">
            <span class="chip chip--primary mono">arXiv:{{ store.currentPaper.arxivId }}</span>
            <span v-for="cat in store.currentPaper.categories" :key="cat" class="chip mono">{{ cat }}</span>
            <span class="hero-date">{{ formatDate(store.currentPaper.publishedAt) }}</span>
          </div>
          <h1 class="hero-title">{{ store.currentPaper.title }}</h1>
          <p class="hero-authors">{{ store.currentPaper.authors.join(', ') }}</p>

          <div class="abstract">
            <div class="abstract-label">
              <el-icon><Document /></el-icon>
              摘要
            </div>
            <p :class="['abstract-text', { 'clamp-abstract': !abstractExpanded }]">{{ store.currentPaper.summary }}</p>
            <button
              v-if="(store.currentPaper.summary || '').length > 420"
              type="button"
              class="link-btn"
              @click="abstractExpanded = !abstractExpanded"
            >
              {{ abstractExpanded ? '收起' : '展开全文' }}
            </button>
          </div>
        </div>

        <aside class="hero-aside">
          <div class="aside-block">
            <el-button
              size="large"
              type="primary"
              class="ask-button"
              :icon="ChatDotRound"
              :loading="creatingIndex"
              @click="handleAskPaper"
            >
              {{ creatingIndex ? '提交索引任务中...' : (hasActiveQaJob ? `索引构建中 · ${qaJobStatusText}` : '开始问答') }}
            </el-button>
            <a :href="store.currentPaper.absUrl" target="_blank" rel="noopener" class="action-link">
              <el-button size="large" class="action-button" :icon="Link">查看 arXiv 原文</el-button>
            </a>
          </div>

          <div v-if="qaIndexJob && showQaJobStatus" class="aside-block job-status">
            <div class="job-status-head">
              <span :class="['chip', qaIndexJob.status === 'failed' ? 'chip--danger' : 'chip--warning']">
                {{ qaJobStatusText }}
              </span>
              <span class="muted">{{ qaStageText }}</span>
            </div>
            <el-progress
              :percentage="Number(qaIndexJob.progress || 0)"
              :stroke-width="6"
              :status="qaIndexJob.status === 'failed' ? 'exception' : undefined"
            />
            <div v-if="qaIndexJob.error_message" class="job-error-text">{{ qaIndexJob.error_message }}</div>
            <div v-else-if="qaJobPollFailures > 0" class="muted">暂时无法获取最新进度，正在重试…</div>
          </div>

          <div v-if="qaStatus" class="status-note">
            <el-icon><Compass /></el-icon>
            <span>
              {{ qaStatus.has_index ? `索引已完成，${qaStatus.chunk_count || 0} 个片段可供检索` : canResearch ? '先创建索引' : '请联系研究者或管理员创建问答索引' }}
            </span>
          </div>

          <div class="aside-block">
            <div class="aside-label"><span>我的标记</span></div>
            <div class="action-toggles">
              <button type="button" :class="['toggle', { 'is-on': store.currentPaper.label === 'liked', 'toggle--success': true }]" @click="handleToggleLabel('liked')">
                感兴趣
              </button>
              <button type="button" :class="['toggle', { 'is-on': currentPaperActions.read, 'toggle--primary': true }]" @click="handleToggleRead">
                {{ currentPaperActions.read ? '已读' : '标记已读' }}
              </button>
              <button type="button" :class="['toggle', { 'is-on': store.currentPaper.label === 'disliked', 'toggle--danger': true }]" @click="handleToggleLabel('disliked')">
                不感兴趣
              </button>
            </div>
          </div>

          <div v-if="preferredAnswerStyle" class="answer-style-note">
            回答风格偏好：{{ preferredAnswerStyle }}
          </div>
        </aside>
      </section>

      <section class="qa-layout">
        <div class="surface qa-main">
          <div class="qa-panel-header">
            <div>
              <h2 class="section-title">论文问答</h2>
              <p class="section-desc">先检索论文片段，再交给大模型生成带引用来源的答案</p>
            </div>

            <div class="panel-actions">
              <el-button text size="small" :icon="RefreshRight" @click="handleRefreshQaState">刷新状态</el-button>
              <el-button text size="small" @click="handleCreateSession">新建会话</el-button>
              <el-button v-if="currentSession" text size="small" @click="handleClearSession">清空会话</el-button>
              <el-button v-if="qaMode" text size="small" @click="closeQaMode">关闭问答</el-button>
            </div>
          </div>

          <div v-if="currentSession || availableSessions.length" class="session-strip">
            <span class="session-label">会话</span>
            <div class="session-list">
              <button
                v-for="session in availableSessions.slice(0, 5)"
                :key="session.session_id"
                type="button"
                :class="['session-chip', { 'is-active': session.session_id === currentSession?.session_id }]"
                @click="handleResumeSession(session.session_id)"
              >
                {{ session.title || '未命名会话' }}
              </button>
              <span v-if="currentSession && !availableSessions.some(item => item.session_id === currentSession?.session_id)" class="session-chip is-active">
                {{ currentSession.title || '未命名会话' }}
              </span>
            </div>
          </div>

          <RagChatPanel
            :messages="ragChatMessages"
            :loading="qaLoading || sessionLoading"
            :quick-prompts="quickPrompts"
            title="关于这篇论文，你想了解什么？"
            :description="`围绕《${store.currentPaper.title}》提问，系统会先召回相关片段，再生成答案。`"
            prompt-title="快捷问题"
            assistant-label="Qwen"
            sender-placeholder="输入你想问论文的问题，Enter 发送，Shift+Enter 换行"
            sender-tip="回答基于检索到的论文片段生成，点击角标可定位原文证据"
            :can-stop="qaLoading"
            @submit-question="submitQuestion"
            @select-prompt="applyPrompt"
            @stop-generation="handleStopGeneration"
            @select-source="handleCitationSelect"
          >
            <template #message-footer="{ item }">
              <div v-if="item.role === 'assistant' && getTurnStatusHint(item)" class="turn-status-hint">
                {{ getTurnStatusHint(item) }}
                <el-button
                  v-if="canResubmitTurn(item)"
                  size="small"
                  text
                  type="primary"
                  @click="resubmitTurn(item.turnId || '')"
                >
                  重新生成
                </el-button>
              </div>
              <RagRemovedClaims v-if="item.role === 'assistant'" :debug="item.retrievalDebug" />
              <div
                v-if="item.role === 'assistant' && ((item.sources?.length || 0) > 0 || item.retrievalDebug)"
                class="assistant-actions"
              >
                <div class="assistant-badges">
                  <span class="chip">{{ item.sources?.length || 0 }} 条来源</span>
                  <span v-if="item.retrievalDebug" class="chip chip--warning">含调试信息</span>
                </div>
                <div class="assistant-buttons">
                  <el-button size="small" text type="primary" @click="handlePanelOpenEvidence(item.turnId || '')">
                    查看来源与调试
                  </el-button>
                  <el-button size="small" text @click="copyAssistantAnswer(item)">复制回答</el-button>
                </div>
              </div>
            </template>
          </RagChatPanel>
        </div>

        <aside class="qa-side">
          <div
            v-if="!isNarrowScreen"
            ref="sideEvidenceCard"
            class="surface side-card qa-side-evidence"
            :class="{ 'qa-side-evidence--flash': sideEvidenceFlash }"
            @animationend="sideEvidenceFlash = false"
          >
            <RagEvidencePanel
              :question="activeEvidenceTurn?.question"
              :sources="activeEvidenceTurn?.sources || []"
              :cited-source-ids="activeEvidenceTurn?.citedSourceIds || []"
              :citation-warning="activeEvidenceTurn?.citationWarning"
              :highlighted-source-id="highlightedEvidenceSourceId"
              :retrieval-debug="activeEvidenceTurn?.retrievalDebug || null"
              :trace-downloading="traceDownloading"
              @download-trace="downloadRetrievalTrace"
              @select-source="highlightedEvidenceSourceId = $event"
            />
          </div>

          <div class="surface side-card">
            <div class="side-title">问答状态</div>
            <div class="kv-list">
              <div class="kv-row">
                <span>索引状态</span>
                <strong>{{ qaStatus?.status || 'unknown' }}</strong>
              </div>
              <div v-if="qaIndexJob" class="kv-row">
                <span>任务状态</span>
                <strong>{{ qaJobStatusText }}</strong>
              </div>
              <div v-if="qaIndexJob" class="kv-row">
                <span>当前阶段</span>
                <strong>{{ qaStageText }}</strong>
              </div>
              <div v-if="qaIndexJob" class="kv-row">
                <span>任务进度</span>
                <strong>{{ Number(qaIndexJob.progress || 0) }}%</strong>
              </div>
              <div v-if="qaIndexJob" class="kv-row">
                <span>轮询状态</span>
                <strong>{{ qaJobPolling ? '进行中' : '已停止' }}</strong>
              </div>
              <div class="kv-row">
                <span>片段数量</span>
                <strong>{{ qaStatus?.chunk_count || 0 }}</strong>
              </div>
              <div class="kv-row">
                <span>模型</span>
                <strong>{{ modelBadge }}</strong>
              </div>
            </div>
          </div>

          <div class="surface side-card">
            <div class="side-title">检索策略</div>
            <div class="retrieval-controls">
              <div class="control-row">
                <span>Query Rewrite</span>
                <el-switch v-model="retrievalOptions.enableQueryRewrite" size="small" />
              </div>
              <div class="control-row">
                <span>HyDE</span>
                <el-switch v-model="retrievalOptions.enableHyde" size="small" />
              </div>
              <div class="control-row">
                <span>Keyword Search</span>
                <el-switch v-model="retrievalOptions.enableKeywordSearch" size="small" />
              </div>
              <div class="control-row">
                <span>LLM Rerank</span>
                <el-switch v-model="retrievalOptions.enableLlmRerank" size="small" />
              </div>
              <div class="control-row">
                <span>Debug</span>
                <el-switch v-model="retrievalOptions.debug" size="small" />
              </div>
              <div class="control-row">
                <span>Top K</span>
                <el-input-number v-model="retrievalOptions.topK" :min="1" :max="15" size="small" controls-position="right" />
              </div>
            </div>
          </div>

          <div v-if="isAdmin" class="surface side-card">
            <div class="side-title">索引诊断</div>
            <div v-if="qaDiagnostic" class="diagnostic-block">
              <div class="kv-list">
                <div class="kv-row">
                  <span>Collection</span>
                  <strong class="mono">{{ qaDiagnostic.collection?.name || qaStatus?.collection_name || '-' }}</strong>
                </div>
                <div class="kv-row">
                  <span>Milvus 实体数</span>
                  <strong>{{ qaDiagnostic.collection?.info?.num_entities ?? 0 }}</strong>
                </div>
                <div class="kv-row">
                  <span>向量数核对</span>
                  <strong :class="vectorCountCheck.ok ? 'ok-text' : 'warn-text'">{{ vectorCountCheck.text }}</strong>
                </div>
                <div class="kv-row">
                  <span>关键词检索</span>
                  <strong :class="qaDiagnostic.checks.likely_keyword_search_will_work ? 'ok-text' : 'warn-text'">
                    {{ qaDiagnostic.checks.likely_keyword_search_will_work ? '可用' : '有风险' }}
                  </strong>
                </div>
              </div>

              <details class="diagnostic-details">
                <summary>查看详细诊断</summary>
                <div class="kv-list diagnostic-json">
                  <div class="kv-row">
                    <span>DB chunk_count</span>
                    <strong>{{ qaDiagnostic.checks.qa_chunk_count ?? 0 }}</strong>
                  </div>
                  <div class="kv-row">
                    <span>预期向量数</span>
                    <strong>{{ qaDiagnostic.checks.expected_vector_count ?? '-' }}</strong>
                  </div>
                  <div class="kv-row">
                    <span>Milvus num_entities</span>
                    <strong>{{ qaDiagnostic.checks.milvus_num_entities ?? 0 }}</strong>
                  </div>
                  <div class="kv-row">
                    <span>collection_exists</span>
                    <strong>{{ qaDiagnostic.checks.collection_exists ? 'true' : 'false' }}</strong>
                  </div>
                  <div class="kv-row">
                    <span>sample_chunks</span>
                    <strong>{{ qaDiagnostic.checks.sample_chunks_returned ?? 0 }}</strong>
                  </div>
                </div>
                <div v-if="qaDiagnostic.collection?.error" class="notice notice--danger diagnostic-error">
                  {{ qaDiagnostic.collection.error }}
                </div>
                <div v-if="qaDiagnostic.sample_chunks.length" class="diagnostic-samples">
                  <div v-for="(chunk, idx) in qaDiagnostic.sample_chunks" :key="idx" class="surface-muted diagnostic-sample">
                    <div class="diagnostic-sample-meta">
                      <span>#{{ idx + 1 }}</span>
                      <span>chunk {{ chunk.chunk_id ?? chunk.id ?? '-' }}</span>
                      <span>page {{ chunk.page_number || chunk.page_range || '-' }}</span>
                    </div>
                    <div class="diagnostic-sample-text">{{ chunk.content || '-' }}</div>
                  </div>
                </div>
              </details>
            </div>
            <div v-else class="muted diagnostic-empty">暂无诊断数据，点击「刷新状态」加载。</div>
          </div>

          <details class="surface side-card side-help">
            <summary class="side-title">问答是如何工作的</summary>
            <ul class="hint-list">
              <li>问题会先转成向量，在对应论文索引中检索相似片段。</li>
              <li>短期上下文会参与问题改写与记忆检索，但不会直接覆盖最终证据。</li>
              <li>如果没有足够相关的片段，会返回更保守的答案。</li>
            </ul>
          </details>
        </aside>
      </section>

      <el-drawer
        v-if="isNarrowScreen"
        v-model="evidenceDrawerOpen"
        :with-header="false"
        size="min(560px, 92vw)"
        class="evidence-drawer"
        @close="closeEvidence"
      >
        <div class="evidence-drawer-body">
          <RagEvidencePanel
            :question="activeEvidenceTurn?.question"
            :sources="activeEvidenceTurn?.sources || []"
            :cited-source-ids="activeEvidenceTurn?.citedSourceIds || []"
            :citation-warning="activeEvidenceTurn?.citationWarning"
            :highlighted-source-id="highlightedEvidenceSourceId"
            :retrieval-debug="activeEvidenceTurn?.retrievalDebug || null"
            :trace-downloading="traceDownloading"
            :show-close="true"
            @download-trace="downloadRetrievalTrace"
            @close="closeEvidence"
            @select-source="highlightedEvidenceSourceId = $event"
          />
        </div>
      </el-drawer>
    </template>

    <div v-else class="surface empty">
      <el-empty description="论文不存在或已被移除" :image-size="96">
        <el-button type="primary" plain @click="goBack">返回</el-button>
      </el-empty>
    </div>
  </div>
</template>

<style scoped>
.paper-detail {
  padding-top: 16px;
  /* clip 而不是 hidden：hidden 的容器仍可被 scrollIntoView 横向滚动，定位证据面板时整页会左移。 */
  overflow: clip;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 12px;
}

.back-btn {
  margin-left: -8px;
  color: var(--color-text-3);
}

.topbar-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

.skeleton-card,
.empty {
  padding: 24px;
}

/* ---------- 论文信息 ---------- */

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: 28px;
  padding: 24px 28px;
}

.hero-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}

.hero-date {
  margin-left: 4px;
  font-size: 12.5px;
  color: var(--color-text-3);
}

.hero-title {
  margin-top: 12px;
  font-size: 24px;
  font-weight: 650;
  line-height: 1.35;
  letter-spacing: -0.01em;
  color: var(--color-text);
}

.hero-authors {
  margin-top: 8px;
  font-size: 13.5px;
  color: var(--color-text-2);
}

.abstract {
  margin-top: 18px;
  padding-top: 16px;
  border-top: 1px solid var(--color-border);
}

.abstract-label {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
  font-size: 12.5px;
  font-weight: 600;
  color: var(--color-text-3);
}

.abstract-text {
  font-size: 14.5px;
  line-height: 1.8;
  color: var(--color-text-2);
}

.clamp-abstract {
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.link-btn {
  margin-top: 4px;
  padding: 0;
  border: none;
  background: none;
  color: var(--color-primary);
  font: inherit;
  font-size: 13px;
  cursor: pointer;
}

.link-btn:hover {
  text-decoration: underline;
}

.hero-aside {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding-left: 24px;
  border-left: 1px solid var(--color-border);
}

.aside-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.aside-label {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  font-size: 12.5px;
  font-weight: 550;
  color: var(--color-text-2);
}

.aside-label .muted {
  font-weight: 400;
  font-size: 12px;
  text-align: right;
}

.ask-button,
.action-button {
  width: 100%;
  margin: 0;
}

.action-link {
  display: block;
}

.job-status-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12.5px;
}

.job-error-text {
  font-size: 12.5px;
  color: var(--color-danger);
}

.status-note {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 12px;
  border-radius: var(--radius);
  background: var(--color-surface-muted);
  border: 1px solid var(--color-border);
  font-size: 12.5px;
  line-height: 1.6;
  color: var(--color-text-2);
}

.status-note .el-icon {
  margin-top: 3px;
  flex-shrink: 0;
  color: var(--color-text-3);
}

.action-toggles {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.toggle {
  --tone: var(--color-text-2);
  --tone-soft: var(--color-surface-sunken);
  height: 28px;
  padding: 0 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-pill);
  background: var(--color-surface);
  color: var(--color-text-2);
  font: inherit;
  font-size: 12.5px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.toggle--success {
  --tone: var(--color-success);
  --tone-soft: var(--color-success-soft);
}

.toggle--primary {
  --tone: var(--color-primary);
  --tone-soft: var(--color-primary-soft);
}

.toggle--danger {
  --tone: var(--color-danger);
  --tone-soft: var(--color-danger-soft);
}

.toggle:hover {
  color: var(--tone);
}

.toggle.is-on {
  background: var(--tone-soft);
  border-color: currentColor;
  color: var(--tone);
}

.answer-style-note {
  font-size: 12.5px;
  color: var(--color-text-3);
}

/* ---------- 问答区 ---------- */

.qa-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) clamp(340px, 26vw, 420px);
  gap: 16px;
  margin-top: 16px;
  align-items: start;
}

.qa-main {
  min-width: 0;
  padding: 18px 20px;
}

.qa-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 12px;
}

.panel-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
}

.panel-actions .el-button + .el-button {
  margin-left: 0;
}

.session-strip {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--color-border);
}

.session-label {
  flex-shrink: 0;
  font-size: 12.5px;
  color: var(--color-text-3);
}

.session-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  min-width: 0;
}

.session-chip {
  max-width: 220px;
  height: 26px;
  padding: 0 10px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-pill);
  background: var(--color-surface);
  color: var(--color-text-2);
  font: inherit;
  font-size: 12.5px;
  line-height: 24px;
  cursor: pointer;
}

.session-chip:hover {
  border-color: var(--color-primary-border);
  color: var(--color-primary);
}

.session-chip.is-active {
  border-color: var(--color-primary-border);
  background: var(--color-primary-soft);
  color: var(--color-primary);
  font-weight: 550;
}

.turn-status-hint {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: var(--radius);
  border: 1px solid var(--color-warning-border);
  background: var(--color-warning-soft);
  color: var(--color-warning);
  font-size: 12.5px;
  line-height: 1.5;
}

.assistant-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  padding-top: 10px;
  border-top: 1px dashed var(--color-border);
}

.assistant-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.assistant-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
}

.assistant-buttons .el-button + .el-button {
  margin-left: 0;
}

/* ---------- 右侧栏 ---------- */

.qa-side {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}

.side-card {
  padding: 16px 18px;
}

.side-title {
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text);
}

/* 证据面板内容可能很长：限制在视口高度内，并在卡片内部滚动，避免把下方状态卡推出视口。 */
.qa-side-evidence {
  --evidence-inset: 18px;
  position: sticky;
  top: 16px;
  z-index: 1;
  max-height: calc(100vh - 32px);
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: var(--color-border-strong) transparent;
  overscroll-behavior: contain;
  scroll-margin-top: 16px;
}

.qa-side-evidence--flash {
  animation: side-evidence-flash 1.2s ease;
}

.retrieval-controls {
  display: flex;
  flex-direction: column;
}

.control-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 34px;
  font-size: 13px;
  color: var(--color-text-2);
}

.control-row :deep(.el-input-number) {
  width: 96px;
}

.ok-text {
  color: var(--color-success) !important;
}

.warn-text {
  color: var(--color-warning) !important;
}

.diagnostic-details {
  margin-top: 8px;
}

.diagnostic-details summary {
  cursor: pointer;
  font-size: 12.5px;
  color: var(--color-primary);
}

.diagnostic-json {
  margin-top: 6px;
}

.diagnostic-error {
  margin-top: 8px;
}

.diagnostic-samples {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 10px;
}

.diagnostic-sample {
  padding: 10px 12px;
}

.diagnostic-sample-meta {
  display: flex;
  gap: 10px;
  margin-bottom: 4px;
  font-size: 11.5px;
  color: var(--color-text-3);
  font-family: var(--font-mono);
}

.diagnostic-sample-text {
  font-size: 12.5px;
  line-height: 1.6;
  color: var(--color-text-2);
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.diagnostic-empty {
  font-size: 12.5px;
}

.side-help summary {
  margin-bottom: 0;
  cursor: pointer;
  list-style: none;
}

.side-help summary::-webkit-details-marker {
  display: none;
}

.side-help summary::after {
  content: '+';
  float: right;
  color: var(--color-text-4);
  font-weight: 400;
}

.side-help[open] summary::after {
  content: '−';
}

.hint-list {
  margin-top: 10px;
  padding-left: 18px;
  display: grid;
  gap: 6px;
  font-size: 12.5px;
  line-height: 1.6;
  color: var(--color-text-2);
}

.evidence-drawer :deep(.el-drawer__body) {
  padding: 0;
  overflow: hidden;
}

.evidence-drawer-body {
  height: 100%;
  padding: 20px;
  overflow: auto;
  scrollbar-width: thin;
  scrollbar-color: var(--color-border-strong) transparent;
  background: var(--color-surface);
}

@keyframes side-evidence-flash {
  0%,
  100% {
    box-shadow: var(--shadow-xs);
  }

  25% {
    box-shadow: 0 0 0 3px var(--color-primary-border);
  }
}

@media (max-width: 1279px) {
  .qa-layout {
    grid-template-columns: 1fr;
  }

  .qa-side {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 960px) {
  .hero {
    grid-template-columns: 1fr;
    padding: 20px;
  }

  .hero-aside {
    padding-left: 0;
    padding-top: 16px;
    border-left: none;
    border-top: 1px solid var(--color-border);
  }
}

@media (max-width: 768px) {
  .hero-title {
    font-size: 20px;
  }

  .qa-panel-header {
    flex-direction: column;
  }

  .qa-side {
    grid-template-columns: 1fr;
  }
}
</style>
