import { ElMessage } from 'element-plus'
import { reactive, ref, toValue, type MaybeRefOrGetter } from 'vue'
import {
  clearPaperChatSession,
  createPaperChatSession,
  getPaperChatMessages,
  getRecentPaperChatSession,
  listPaperChatSessions,
  qaPaperStream
} from '@/api/papers'
import type {
  PaperChatMessage,
  PaperChatSession,
  QaConversationContextTurn,
  QaRequestOptions
} from '@/api/papers'
import type { QaTurnForRagChat as QaTurn } from '@/types/ragChat'

const MAX_CONVERSATION_TURNS = 5
const MAX_ANSWER_SUMMARY_LENGTH = 280
const MAX_SOURCE_CONTENT_LENGTH = 180
const MAX_SOURCES_PER_TURN = 3
const DEFAULT_USER_ID = 'local_user'

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

function getErrorMessage(error: unknown, fallback: string) {
  if (error && typeof error === 'object' && 'message' in error && typeof error.message === 'string' && error.message) {
    return error.message
  }
  return fallback
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
    .slice(-memoryConfig.shortTermMemoryMaxTurns)
    .map(turn => ({
      turn_id: turn.id,
      question: truncateText(turn.question, 220),
      answer_summary: truncateText(turn.answer, memoryConfig.shortTermMemoryMaxChars),
      created_at: turn.createdAt,
      sources: (turn.sources || []).slice(0, MAX_SOURCES_PER_TURN).map(source => ({
        source_id: source.parent_chunk_id,
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
      originalQuestion: '',
      contextualizedQuestion: message.contextualized_question || '',
      usedShortTermMemory: Boolean(message.question_contextualization?.used_short_term_memory),
      questionContextualization: message.question_contextualization ?? null
    }

    if (message.role === 'user') {
      existing.question = message.content || existing.question
      existing.originalQuestion = message.content || existing.originalQuestion
      existing.createdAt = message.created_at || existing.createdAt
    } else {
      existing.answer = message.content || existing.answer
      existing.sources = (message.sources || []).map(source => ({
        content: source.content || source.asset_summary || source.asset_preview_text || '',
        page_number: String(source.page_number || ''),
        source: source.source,
        section_path: source.section_path,
        parent_chunk_id: source.parent_chunk_id,
        chunk_type: source.chunk_type,
        asset_summary: source.asset_summary,
        asset_preview_text: source.asset_preview_text
      }))
      existing.retrievalDebug = message.retrieval_debug_snapshot ?? existing.retrievalDebug
      existing.contextualizedQuestion = message.contextualized_question || existing.contextualizedQuestion
      existing.questionContextualization = message.question_contextualization ?? existing.questionContextualization
      existing.usedShortTermMemory = Boolean(
        message.question_contextualization?.used_short_term_memory ?? existing.usedShortTermMemory
      )
    }

    turnsById.set(turnId, existing)
  }

  return Array.from(turnsById.values()).sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime())
}

export function usePaperRagChat(options: UsePaperRagChatOptions) {
  const memoryConfig = getMemoryConfig()
  const qaResults = ref<QaTurn[]>([])
  const question = ref('')
  const qaLoading = ref(false)
  const sessionLoading = ref(false)
  const evidenceDrawerOpen = ref(false)
  const activeEvidenceTurn = ref<QaTurn | null>(null)
  const currentSession = ref<PaperChatSession | null>(null)
  const availableSessions = ref<PaperChatSession[]>([])
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
    const resolved = options.userId ? toValue(options.userId) : DEFAULT_USER_ID
    return String(resolved || DEFAULT_USER_ID).trim() || DEFAULT_USER_ID
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

  function resetChat() {
    qaResults.value = []
    question.value = ''
    currentSession.value = null
    availableSessions.value = []
    closeEvidence()
    activeEvidenceTurn.value = null
  }

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
      originalQuestion: rawQuestion,
      contextualizedQuestion: rawQuestion,
      usedShortTermMemory: false,
      questionContextualization: null
    })
    qaResults.value.push(turn)
    question.value = ''
    scrollToBottom()

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
          onMeta: meta => {
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
            scrollToBottom()
          },
          onDelta: delta => {
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
            turn.streaming = false
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
      turn.streaming = false
      await refreshSessions()
    } catch (error) {
      ElMessage.error(getErrorMessage(error, '\u95ee\u7b54\u5931\u8d25'))
      qaResults.value = qaResults.value.filter(item => item.id !== turn.id)
      question.value = rawQuestion
    } finally {
      qaLoading.value = false
    }
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
