import { ref, watch, type Ref } from 'vue'
import { ElMessage } from 'element-plus'
import { runAgentChat, streamAgentChat } from '@/api/agent'
import { getErrorMessage } from '@/api/errors'
import type {
  AgentInteraction,
  AgentPaper,
  AgentStreamEvent,
  ArxivSearchRequest,
  ArxivSearchResponse
} from '@/types/agent'
import type { AgentChatMessage } from '@/types/agentChat'
import type { UserResearchProfile } from '@/types/paper'
import { usePaperStore } from '@/stores/paperStore'
import { useUserContext } from '@/composables/useUserContext'

type ResumeDecision = 'approve' | 'reject' | 'select' | 'cancel'

interface SubmitMessageOptions {
  resume?: NonNullable<ArxivSearchRequest['resume']>
  skipUserMessage?: boolean
}

function createMessageId(prefix: 'user' | 'assistant') {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

function createDraftResponse(): ArxivSearchResponse {
  return {
    session_id: null,
    intent: 'loading',
    answer: '',
    search_spec: null,
    interaction: null,
    paper_qa_result: null,
    preference_action_result: null,
    plan: [],
    tool_calls: [],
    papers: [],
    warnings: [],
    next_actions: [],
    steps: [],
    debug: {},
    streaming_state: null
  }
}

function applyStreamEvent(target: AgentChatMessage<ArxivSearchResponse>, event: AgentStreamEvent) {
  const response = target.response || createDraftResponse()
  const payload = event.data || {}
  if (event.event_type === 'final_response' && payload.response) {
    target.response = payload.response as ArxivSearchResponse
    target.content = target.response.answer || ''
    return
  }
  response.streaming_state = {
    run_id: String(event.run_id || ''),
    sequence: Number(event.sequence || 0),
    event_type: event.event_type,
    active_step: typeof payload.step_id === 'string' ? payload.step_id : null,
    active_tool_call: null
  }
  target.response = response
}

export function useAgentSearchChat() {
  const paperStore = usePaperStore()
  const userContext = useUserContext()
  const inputMessage = ref('')
  const loading = ref(false)
  const interactionSubmitting = ref(false)
  const messages: Ref<AgentChatMessage<ArxivSearchResponse>[]> = ref([])
  const latestResponse = ref<ArxivSearchResponse | null>(null)
  const lastSearchPapers = ref<AgentPaper[]>([])
  const currentInteraction = ref<AgentInteraction | null>(null)
  const selectedPaper = ref<AgentPaper | null>(null)
  const paperQaResult = ref<Record<string, any> | null>(null)
  const activeSessionId = ref<string | null>(null)
  const activeSessionUserId = ref<string | null>(null)

  function getUserId() {
    return userContext.getUserId()
  }

  function setInputMessage(value: string) {
    inputMessage.value = value
  }

  function clearConversation() {
    messages.value = []
    latestResponse.value = null
    lastSearchPapers.value = []
    currentInteraction.value = null
    selectedPaper.value = null
    paperQaResult.value = null
    activeSessionId.value = null
    activeSessionUserId.value = null
    interactionSubmitting.value = false
    inputMessage.value = ''
  }

  function rememberResponse(response: ArxivSearchResponse) {
    latestResponse.value = response
    currentInteraction.value = response.interaction?.status === 'pending' ? response.interaction : null
    paperQaResult.value = response.paper_qa_result ? { ...response.paper_qa_result } : null

    const sessionId = String(response.session_id || '').trim()
    if (sessionId) {
      activeSessionId.value = sessionId
      activeSessionUserId.value = getUserId()
    }
    if (response.intent === 'arxiv_search' && response.papers.length) {
      lastSearchPapers.value = response.papers.map(paper => ({ ...paper }))
      selectedPaper.value = null
    }
    const qaArxivId = response.paper_qa_result?.arxiv_id
    if (qaArxivId) {
      selectedPaper.value = lastSearchPapers.value.find(paper =>
        [paper.arxiv_id, paper.arxivId, paper.id].includes(qaArxivId)
      ) || { arxiv_id: qaArxivId, title: response.paper_qa_result?.title || '' }
    }
  }

  function buildRequestContext(): ArxivSearchRequest['context'] {
    const context: {
      selected_paper?: AgentPaper | null
      last_papers?: AgentPaper[]
      paper_qa_result?: Record<string, any> | null
      research_profile?: UserResearchProfile | null
      arxiv_id?: string | null
      source?: 'chat'
    } = {}
    if (lastSearchPapers.value.length) context.last_papers = lastSearchPapers.value.map(paper => ({ ...paper }))
    if (selectedPaper.value) context.selected_paper = { ...selectedPaper.value }
    if (paperQaResult.value) context.paper_qa_result = { ...paperQaResult.value }
    if (paperStore.researchProfile) context.research_profile = { ...paperStore.researchProfile }
    context.arxiv_id = selectedPaper.value?.arxiv_id || selectedPaper.value?.arxivId || selectedPaper.value?.id || null
    context.source = 'chat'
    return Object.keys(context).length ? context : undefined
  }

  async function submitMessage(rawMessage?: string, options?: SubmitMessageOptions) {
    const message = (rawMessage ?? inputMessage.value).trim()
    if (!message || loading.value) return

    const assistantId = createMessageId('assistant')
    const createdAt = new Date().toISOString()
    if (!options?.skipUserMessage) {
      messages.value.push({
        id: createMessageId('user'),
        role: 'user',
        content: message,
        loading: false,
        createdAt,
        response: null,
        error: null
      })
    }
    const assistantMessage: AgentChatMessage<ArxivSearchResponse> = {
      id: assistantId,
      role: 'assistant',
      content: '',
      loading: true,
      createdAt,
      response: createDraftResponse(),
      error: null
    }
    messages.value.push(assistantMessage)
    inputMessage.value = ''
    loading.value = true

    const request: ArxivSearchRequest = {
      message,
      user_id: getUserId(),
      session_id: activeSessionUserId.value === getUserId() ? activeSessionId.value : null,
      resume: options?.resume,
      context: buildRequestContext()
    }

    try {
      let response: ArxivSearchResponse
      try {
        response = await streamAgentChat(request, {
          onEvent: event => applyStreamEvent(assistantMessage, event)
        })
      } catch (streamError) {
        // resume 可能已在服务端原子消费，传输失败时禁止自动重放；普通消息才允许同步 fallback。
        if (request.resume) throw streamError
        response = await runAgentChat(request)
      }
      rememberResponse(response)
      assistantMessage.response = response
      assistantMessage.content = response.answer || ''
    } catch (error) {
      const messageText = getErrorMessage(error, 'Agent 调用失败')
      assistantMessage.error = messageText
      assistantMessage.content = messageText
      ElMessage.error(messageText)
    } finally {
      assistantMessage.loading = false
      loading.value = false
    }
  }

  async function submitResume(
    decision: ResumeDecision,
    note?: string,
    response: Record<string, any> = {}
  ) {
    const interaction = currentInteraction.value
    if (!interaction || loading.value || interactionSubmitting.value) return
    interactionSubmitting.value = true
    try {
      await submitMessage('继续处理当前交互', {
        skipUserMessage: true,
        resume: {
          interaction_id: interaction.interaction_id,
          decision,
          response,
          note: note || null
        }
      })
    } finally {
      interactionSubmitting.value = false
    }
  }

  watch(() => getUserId(), clearConversation)

  return {
    inputMessage,
    loading,
    interactionSubmitting,
    messages,
    latestResponse,
    lastSearchPapers,
    currentInteraction,
    selectedPaper,
    paperQaResult,
    activeSessionId,
    setInputMessage,
    submitMessage,
    submitResume,
    clearConversation
  }
}
