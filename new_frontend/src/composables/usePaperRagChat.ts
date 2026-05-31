import { ElMessage } from 'element-plus'
import { reactive, ref, toValue, type MaybeRefOrGetter } from 'vue'
import { qaPaperStream } from '@/api/papers'
import type { QaRequestOptions } from '@/api/papers'
import type { QaTurnForRagChat as QaTurn } from '@/types/ragChat'

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
}

function getErrorMessage(error: unknown, fallback: string) {
  if (error && typeof error === 'object' && 'message' in error && typeof error.message === 'string' && error.message) {
    return error.message
  }
  return fallback
}

export function usePaperRagChat(options: UsePaperRagChatOptions) {
  const qaResults = ref<QaTurn[]>([])
  const question = ref('')
  const qaLoading = ref(false)
  const evidenceDrawerOpen = ref(false)
  const activeEvidenceTurn = ref<QaTurn | null>(null)
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
    closeEvidence()
    activeEvidenceTurn.value = null
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
      streaming: true
    })
    qaResults.value.push(turn)
    question.value = ''
    scrollToBottom()

    try {
      const requestOptions: QaRequestOptions = {
        top_k: retrievalOptions.topK,
        enable_query_rewrite: retrievalOptions.enableQueryRewrite,
        enable_hyde: retrievalOptions.enableHyde,
        enable_keyword_search: retrievalOptions.enableKeywordSearch,
        enable_llm_rerank: retrievalOptions.enableLlmRerank,
        debug: retrievalOptions.debug
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
      turn.streaming = false
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
    evidenceDrawerOpen,
    activeEvidenceTurn,
    retrievalOptions,
    submitQuestion,
    applyPrompt,
    openEvidence,
    closeEvidence,
    resetChat
  }
}
