import { ElMessage } from 'element-plus'
import { reactive, ref, toValue, watch, type MaybeRefOrGetter } from 'vue'
import {
  clearPaperChatSession,
  createPaperChatSession,
  getPaperChatMessages,
  getRecentPaperChatSession,
  listPaperChatSessions,
  qaPaperStream
} from '@/api/papers'
import { getErrorMessage } from '@/api/errors'
import { ApiError } from '@/api/errors'
import type {
  PaperChatMessage,
  PaperChatSession,
  QaConversationContextTurn,
  QaRequestOptions
} from '@/api/papers'
import type {
  QaPersistenceStatus,
  QaTurnForRagChat as QaTurn,
  QaTurnStatus
} from '@/types/ragChat'
import { useUserContext } from '@/composables/useUserContext'
import { extractCitedSourceIds, normalizeEvidenceSources } from '@/utils/evidence'

const MAX_CONVERSATION_TURNS = 5
const MAX_ANSWER_SUMMARY_LENGTH = 280
const MAX_SOURCE_CONTENT_LENGTH = 180
const MAX_SOURCES_PER_TURN = 3

function getMemoryConfig() {
  return {
    enableShortTermMemory: true,
    enablePaperChatSession: true,
    enableMemoryAwareRetrieval: true,
    enableUserResearchProfile: false,
    shortTermMemoryMaxTurns: MAX_CONVERSATION_TURNS,
    shortTermMemoryMaxChars: MAX_ANSWER_SUMMARY_LENGTH,
    memoryContextDebug: true
  }
}

export interface PaperRagRetrievalOptions {
  enableQueryRewrite: boolean
  enableHyde: boolean
  enableKeywordSearch: boolean
  enableLlmRerank: boolean
  debug: boolean
  topK: number
}

interface UsePaperRagChatOptions {
  paperId: MaybeRefOrGetter<string>
  onEnterQaMode?: () => void
  onScrollToBottom?: () => void
  userId?: MaybeRefOrGetter<string>
}

function truncateText(value: string | undefined, maxLength: number) {
  const normalized = String(value || '').replace(/\s+/g, ' ').trim()
  if (normalized.length <= maxLength) {
    return normalized
  }
  return `${normalized.slice(0, maxLength - 3).trimEnd()}...`
}

function buildConversationContext(qaResults: QaTurn[], pendingTurnId?: string): QaConversationContextTurn[] {
  const memoryConfig = getMemoryConfig()
  if (!memoryConfig.enableShortTermMemory) {
    return []
  }
  return qaResults
    .filter(turn => turn.id !== pendingTurnId && !turn.streaming)
    .filter(turn => !turn.status || turn.status === 'completed')
    .slice(-memoryConfig.shortTermMemoryMaxTurns)
    .map(turn => ({
      turn_id: turn.id,
      question: truncateText(turn.question, 220),
      answer_summary: truncateText(turn.answer, memoryConfig.shortTermMemoryMaxChars),
      created_at: turn.createdAt,
      sources: (turn.sources || []).slice(0, MAX_SOURCES_PER_TURN).map(source => ({
        source_id: source.source_id,
        content: truncateText(source.content || source.asset_summary || source.asset_preview_text || '', MAX_SOURCE_CONTENT_LENGTH),
        page_number: source.page_number,
        source: source.source,
        section_path: source.section_path,
        chunk_type: source.chunk_type,
        asset_summary: truncateText(source.asset_summary, MAX_SOURCE_CONTENT_LENGTH),
        asset_preview_text: truncateText(source.asset_preview_text, MAX_SOURCE_CONTENT_LENGTH)
      }))
    }))
    .filter(turn => Boolean(turn.question || turn.answer_summary || turn.sources.length))
}

function groupMessagesToTurns(messages: PaperChatMessage[]): QaTurn[] {
  const turnsById = new Map<string, QaTurn>()

  for (const message of messages) {
    const turnId = message.turn_id || message.message_id
    const existing = turnsById.get(turnId) ?? {
      id: turnId,
      question: '',
      answer: '',
      sources: [],
      retrievalDebug: null,
      createdAt: message.created_at,
      streaming: false,
      status: 'completed',
      error: null,
      partial: false,
      completedAt: message.created_at,
      interruptedReason: null,
      persistenceStatus: 'saved',
      originalQuestion: '',
      contextualizedQuestion: message.contextualized_question || '',
      usedShortTermMemory: Boolean(message.question_contextualization?.used_short_term_memory),
      questionContextualization: message.question_contextualization ?? null,
      citedSourceIds: [],
      citationWarning: null
    }

    if (message.role === 'user') {
      existing.question = message.content || existing.question
      existing.originalQuestion = message.content || existing.originalQuestion
      existing.createdAt = message.created_at || existing.createdAt
    } else {
      existing.answer = message.content || existing.answer
      existing.sources = normalizeEvidenceSources(message.sources)
      existing.citedSourceIds = extractCitedSourceIds(existing.answer, existing.sources)
      existing.retrievalDebug = message.retrieval_debug_snapshot ?? existing.retrievalDebug
      existing.contextualizedQuestion = message.contextualized_question || existing.contextualizedQuestion
      existing.questionContextualization = message.question_contextualization ?? existing.questionContextualization
      existing.usedShortTermMemory = Boolean(
        message.question_contextualization?.used_short_term_memory ?? existing.usedShortTermMemory
      )
      existing.citedSourceIds = Array.isArray(message.cited_source_ids)
        ? message.cited_source_ids.map(value => String(value))
        : existing.citedSourceIds || []
      existing.citationWarning = typeof message.citation_warning === 'string'
        ? message.citation_warning
        : existing.citationWarning || null
    }

    turnsById.set(turnId, existing)
  }

  return Array.from(turnsById.values()).sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime())
}

function getApiErrorCode(error: unknown) {
  if (error instanceof ApiError) {
    return String(error.payload.code || 'unknown_error')
  }
  if (error && typeof error === 'object' && 'payload' in error) {
    const payload = (error as { payload?: { code?: unknown } }).payload
    if (payload?.code) return String(payload.code)
  }
  return 'unknown_error'
}

function mapResultToTurnStatus(resultStatus?: string, persistenceStatus?: QaPersistenceStatus): QaTurnStatus {
  const status = String(resultStatus || '').trim().toLowerCase()
  if (persistenceStatus === 'failed' || ['persistence_failed', 'partial_success', 'database_write_failed'].includes(status)) {
    return 'persistence_failed'
  }
  if (status === 'aborted') return 'aborted'
  if (status === 'failed') return 'failed'
  if (status === 'partial' || status === 'stream_interrupted') return 'partial'
  return 'completed'
}

function hasEffectiveTurnOutput(turn: QaTurn) {
  return Boolean(turn.answer.trim() || turn.sources.length || turn.retrievalDebug)
}

function applyTurnTerminalState(
  turn: QaTurn,
  status: QaTurnStatus,
  options: {
    error?: string | null
    partial?: boolean
    interruptedReason?: string | null
    persistenceStatus?: QaPersistenceStatus
    completedAt?: string | null
  } = {}
) {
  turn.streaming = false
  turn.status = status
  turn.error = options.error ?? null
  turn.partial = options.partial ?? status !== 'completed'
  turn.interruptedReason = options.interruptedReason ?? null
  turn.persistenceStatus = options.persistenceStatus ?? (status === 'completed' ? 'saved' : 'not_saved')
  turn.completedAt = options.completedAt ?? (status === 'completed' ? new Date().toISOString() : null)
}

export function usePaperRagChat(options: UsePaperRagChatOptions) {
  const userContext = useUserContext()
  const memoryConfig = getMemoryConfig()
  const qaResults = ref<QaTurn[]>([])
  const question = ref('')
  const qaLoading = ref(false)
  const sessionLoading = ref(false)
  const evidenceDrawerOpen = ref(false)
  const activeEvidenceTurn = ref<QaTurn | null>(null)
  const currentSession = ref<PaperChatSession | null>(null)
  const availableSessions = ref<PaperChatSession[]>([])
  const activeStreamController = ref<AbortController | null>(null)
  const activeAbortIntent = ref<'user' | 'cleanup' | null>(null)
  const retrievalOptions = reactive<PaperRagRetrievalOptions>({
    enableQueryRewrite: true,
    enableHyde: false,
    enableKeywordSearch: true,
    enableLlmRerank: true,
    debug: true,
    topK: 15
  })

  function scrollToBottom() {
    options.onScrollToBottom?.()
  }

  function getUserId() {
    const resolved = options.userId ? toValue(options.userId) : userContext.userId.value
    return String(resolved || userContext.defaultUserId).trim() || userContext.defaultUserId
  }

  function openEvidence(turn: QaTurn) {
    activeEvidenceTurn.value = turn
    evidenceDrawerOpen.value = true
  }

  function closeEvidence() {
    evidenceDrawerOpen.value = false
  }

  function applyPrompt(prompt: string) {
    question.value = prompt
    options.onEnterQaMode?.()
  }

  function cancelCurrentStream(intent: 'user' | 'cleanup' = 'user') {
    const controller = activeStreamController.value
    if (!controller || controller.signal.aborted) return
    // 用户停止和页面清理都走 AbortController，但后续错误处理需要知道是否恢复输入框。
    activeAbortIntent.value = intent
    controller.abort()
  }

  function resetChat() {
    cancelCurrentStream('cleanup')
    qaResults.value = []
    question.value = ''
    currentSession.value = null
    availableSessions.value = []
    closeEvidence()
    activeEvidenceTurn.value = null
  }

  watch(() => getUserId(), () => {
    // 用户切换后必须丢弃当前 QA 现场，避免用旧用户的 session 和短期记忆继续发问。
    resetChat()
  })

  async function refreshSessions() {
    if (!memoryConfig.enablePaperChatSession) {
      availableSessions.value = []
      return
    }
    const currentPaperId = toValue(options.paperId)
    if (!currentPaperId) {
      availableSessions.value = []
      return
    }
    const response = await listPaperChatSessions(currentPaperId, {
      user_id: getUserId(),
      limit: 20
    })
    availableSessions.value = response.items || []
  }

  async function loadSession(sessionId: string) {
    if (!memoryConfig.enablePaperChatSession) return
    const currentPaperId = toValue(options.paperId)
    if (!currentPaperId || !sessionId) return

    cancelCurrentStream('cleanup')
    sessionLoading.value = true
    try {
      const response = await getPaperChatMessages(currentPaperId, sessionId, getUserId())
      currentSession.value = response.session
      qaResults.value = groupMessagesToTurns(response.items || [])
      if (qaResults.value.length > 0) {
        options.onEnterQaMode?.()
      }
      scrollToBottom()
    } finally {
      sessionLoading.value = false
    }
  }

  async function loadRecentSession() {
    if (!memoryConfig.enablePaperChatSession) {
      currentSession.value = null
      qaResults.value = []
      return
    }
    const currentPaperId = toValue(options.paperId)
    if (!currentPaperId) return

    sessionLoading.value = true
    try {
      await refreshSessions()
      const response = await getRecentPaperChatSession(currentPaperId, getUserId())
      const recentSession = response.item
      currentSession.value = recentSession
      if (recentSession?.session_id) {
        await loadSession(recentSession.session_id)
      } else {
        qaResults.value = []
      }
    } finally {
      sessionLoading.value = false
    }
  }

  async function createNewSession() {
    if (!memoryConfig.enablePaperChatSession) return null
    const currentPaperId = toValue(options.paperId)
    if (!currentPaperId) return null

    cancelCurrentStream('cleanup')
    sessionLoading.value = true
    try {
      const response = await createPaperChatSession(currentPaperId, { user_id: getUserId() })
      currentSession.value = response.item
      qaResults.value = []
      question.value = ''
      closeEvidence()
      activeEvidenceTurn.value = null
      await refreshSessions()
      options.onEnterQaMode?.()
      return response.item
    } finally {
      sessionLoading.value = false
    }
  }

  async function clearCurrentSession() {
    cancelCurrentStream('cleanup')
    if (!memoryConfig.enablePaperChatSession) {
      qaResults.value = []
      return
    }
    const currentPaperId = toValue(options.paperId)
    const sessionId = currentSession.value?.session_id
    if (!currentPaperId || !sessionId) {
      qaResults.value = []
      return
    }

    sessionLoading.value = true
    try {
      const response = await clearPaperChatSession(currentPaperId, sessionId, getUserId())
      currentSession.value = response.item
      qaResults.value = []
      question.value = ''
      closeEvidence()
      activeEvidenceTurn.value = null
      await refreshSessions()
    } finally {
      sessionLoading.value = false
    }
  }

  async function submitQuestion(customQuestion?: string) {
    const rawQuestion = (customQuestion ?? question.value).trim()
    if (!rawQuestion || qaLoading.value) return

    const currentPaperId = toValue(options.paperId)
    if (!currentPaperId) return

    qaLoading.value = true

    const turn = reactive<QaTurn>({
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      question: rawQuestion,
      answer: '',
      sources: [],
      retrievalDebug: null,
      createdAt: new Date().toISOString(),
      streaming: true,
      status: 'preparing',
      error: null,
      partial: false,
      completedAt: null,
      interruptedReason: null,
      persistenceStatus: 'unknown',
      originalQuestion: rawQuestion,
      contextualizedQuestion: rawQuestion,
      usedShortTermMemory: false,
      questionContextualization: null,
      citedSourceIds: [],
      citationWarning: null
    })
    qaResults.value.push(turn)
    question.value = ''
    scrollToBottom()
    const streamController = new AbortController()
    activeStreamController.value = streamController
    activeAbortIntent.value = null

    try {
      const conversationContext = buildConversationContext(qaResults.value, turn.id)
      const requestOptions: QaRequestOptions = {
        user_id: getUserId(),
        session_id: memoryConfig.enablePaperChatSession ? currentSession.value?.session_id : undefined,
        top_k: retrievalOptions.topK,
        enable_query_rewrite: retrievalOptions.enableQueryRewrite,
        enable_hyde: retrievalOptions.enableHyde,
        enable_keyword_search: retrievalOptions.enableKeywordSearch,
        enable_llm_rerank: retrievalOptions.enableLlmRerank,
        debug: retrievalOptions.debug,
        conversation_context: memoryConfig.enableShortTermMemory && conversationContext.length ? conversationContext : undefined
      }

      const result = await qaPaperStream(
        currentPaperId,
        rawQuestion,
        {
          signal: streamController.signal,
          onMeta: meta => {
            // meta 到达表示后端已完成检索准备，后续 turn 可以展示来源和调试信息。
            turn.status = 'streaming'
            if (Array.isArray(meta.sources)) {
              turn.sources = meta.sources
            }
            if (meta.retrieval_debug) {
              turn.retrievalDebug = meta.retrieval_debug
            }
            if (meta.chat_session) {
              currentSession.value = meta.chat_session
            }
            if (typeof meta.original_question === 'string' && meta.original_question) {
              turn.originalQuestion = meta.original_question
            }
            if (typeof meta.contextualized_question === 'string' && meta.contextualized_question) {
              turn.contextualizedQuestion = meta.contextualized_question
            }
            if (typeof meta.used_short_term_memory === 'boolean') {
              turn.usedShortTermMemory = meta.used_short_term_memory
            }
            if (meta.question_contextualization) {
              turn.questionContextualization = meta.question_contextualization
            }
            if (Array.isArray(meta.cited_source_ids)) {
              turn.citedSourceIds = meta.cited_source_ids.map(value => String(value))
            }
            if (typeof meta.citation_warning === 'string') {
              turn.citationWarning = meta.citation_warning
            }
            scrollToBottom()
          },
          onDelta: delta => {
            turn.status = 'streaming'
            turn.answer += delta
            scrollToBottom()
          },
          onDone: payload => {
            if (typeof payload.answer === 'string' && payload.answer) {
              turn.answer = payload.answer
            }
            if (Array.isArray(payload.sources)) {
              turn.sources = payload.sources
            }
            if (payload.retrieval_debug) {
              turn.retrievalDebug = payload.retrieval_debug
            }
            if (payload.chat_session) {
              currentSession.value = payload.chat_session
            }
            if (typeof payload.original_question === 'string' && payload.original_question) {
              turn.originalQuestion = payload.original_question
            }
            if (typeof payload.contextualized_question === 'string' && payload.contextualized_question) {
              turn.contextualizedQuestion = payload.contextualized_question
            }
            if (typeof payload.used_short_term_memory === 'boolean') {
              turn.usedShortTermMemory = payload.used_short_term_memory
            }
            if (payload.question_contextualization) {
              turn.questionContextualization = payload.question_contextualization
            }
            if (Array.isArray(payload.cited_source_ids)) {
              turn.citedSourceIds = payload.cited_source_ids.map(value => String(value))
            }
            if (typeof payload.citation_warning === 'string') {
              turn.citationWarning = payload.citation_warning
            }
            const persistenceStatus = (payload.persistence_status || payload.persistenceStatus || 'unknown') as QaPersistenceStatus
            const terminalStatus = mapResultToTurnStatus(payload.status, persistenceStatus)
            applyTurnTerminalState(turn, terminalStatus, {
              partial: terminalStatus !== 'completed',
              interruptedReason: payload.interrupted_reason || payload.interruptedReason || null,
              persistenceStatus,
              completedAt: payload.completed_at || payload.completedAt || null,
              error: terminalStatus === 'persistence_failed' ? '答案已生成，但未保存到历史记录。' : null
            })
            scrollToBottom()
          },
          onError: () => {
            turn.streaming = false
          }
        },
        requestOptions
      )

      if (typeof result.answer === 'string' && !turn.answer) {
        turn.answer = result.answer
      }
      if (Array.isArray(result.sources) && turn.sources.length === 0) {
        turn.sources = result.sources
      }
      if (result.retrieval_debug && !turn.retrievalDebug) {
        turn.retrievalDebug = result.retrieval_debug
      }
      if (result.chat_session) {
        currentSession.value = result.chat_session
      }
      if (typeof result.original_question === 'string' && result.original_question) {
        turn.originalQuestion = result.original_question
      }
      if (typeof result.contextualized_question === 'string' && result.contextualized_question) {
        turn.contextualizedQuestion = result.contextualized_question
      }
      if (typeof result.used_short_term_memory === 'boolean') {
        turn.usedShortTermMemory = result.used_short_term_memory
      }
      if (result.question_contextualization) {
        turn.questionContextualization = result.question_contextualization
      }
      const persistenceStatus = result.persistence_status || 'unknown'
      const terminalStatus = mapResultToTurnStatus(result.status, persistenceStatus)
      applyTurnTerminalState(turn, terminalStatus, {
        partial: Boolean(result.partial) || terminalStatus !== 'completed',
        interruptedReason: result.interrupted_reason || null,
        persistenceStatus,
        completedAt: result.completed_at || null,
        error: terminalStatus === 'persistence_failed' ? '答案已生成，但未保存到历史记录。' : null
      })
      await refreshSessions()
      if (terminalStatus === 'persistence_failed') {
        ElMessage.warning('答案已生成，但未保存到历史记录。')
      }
    } catch (error) {
      const errorCode = getApiErrorCode(error)
      const message = getErrorMessage(error, '问答失败，请稍后重试。')
      const hasOutput = hasEffectiveTurnOutput(turn)

      if (!hasOutput) {
        qaResults.value = qaResults.value.filter(item => item.id !== turn.id)
        if (errorCode !== 'aborted' || activeAbortIntent.value === 'user') {
          question.value = rawQuestion
        }
      } else if (errorCode === 'aborted') {
        applyTurnTerminalState(turn, 'aborted', {
          error: '已取消生成',
          partial: true,
          interruptedReason: activeAbortIntent.value === 'cleanup' ? 'cleanup' : 'user_abort',
          persistenceStatus: 'not_saved'
        })
      } else {
        // 已有 answer/sources 时保留临时 turn，避免把用户可复制的半截回答直接丢掉。
        applyTurnTerminalState(turn, errorCode === 'stream_incomplete' ? 'partial' : 'failed', {
          error: message,
          partial: true,
          interruptedReason: errorCode,
          persistenceStatus: errorCode === 'database_write_failed' ? 'failed' : 'not_saved'
        })
      }

      if (activeAbortIntent.value !== 'cleanup') {
        if (errorCode === 'aborted') {
          ElMessage.info(message)
        } else {
          ElMessage.error(message)
        }
      }
    } finally {
      if (activeStreamController.value === streamController) {
        activeStreamController.value = null
      }
      activeAbortIntent.value = null
      qaLoading.value = false
    }
  }

  function resubmitTurn(turnId: string) {
    const target = qaResults.value.find(turn => turn.id === turnId)
    if (!target?.question || qaLoading.value) return
    void submitQuestion(target.question)
  }

  return {
    qaResults,
    question,
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
    openEvidence,
    closeEvidence,
    resetChat,
    refreshSessions,
    loadSession,
    loadRecentSession,
    createNewSession,
    clearCurrentSession
  }
}
