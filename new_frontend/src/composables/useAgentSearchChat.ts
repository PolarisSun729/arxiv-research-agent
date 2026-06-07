import { ref, watch, type Ref } from 'vue'
import { ElMessage } from 'element-plus'
import { runAgentChat, streamAgentChat } from '@/api/agent'
import { getErrorMessage } from '@/api/errors'
import type { AgentPaper, AgentStep, AgentStreamEvent, AgentToolCall, ArxivSearchResponse } from '@/types/agent'
import type { AgentChatMessage } from '@/types/agentChat'
import type { UserResearchProfile } from '@/types/paper'
import { usePaperStore } from '@/stores/paperStore'
import { useUserContext } from '@/composables/useUserContext'

type ResumeDecision = 'approve' | 'reject'
const RESUME_CHECKPOINT_NOT_FOUND_CODE = 'resume_checkpoint_not_found'
const RESUME_CHECKPOINT_NOT_FOUND_MESSAGE = '原执行现场已失效，请重新发起论文解析或问答请求。'

interface AgentResumePayload {
  decision: ResumeDecision
  note?: string | null
  step_id?: string | null
  interrupt_id?: string | null
  edited_arguments?: Record<string, any> | null
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
    pending_action: null,
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

function getAgentErrorCode(response: ArxivSearchResponse | null | undefined) {
  const runtimeCode = response?.debug?.runtime_error?.code
  if (typeof runtimeCode === 'string' && runtimeCode) return runtimeCode

  const paperQaCode = response?.paper_qa_result?.error_code
  if (typeof paperQaCode === 'string' && paperQaCode) return paperQaCode

  const stepError = response?.steps?.find(step => step.error)?.error
  return typeof stepError === 'string' ? stepError : null
}

function isResumeCheckpointNotFound(response: ArxivSearchResponse | null | undefined) {
  return getAgentErrorCode(response) === RESUME_CHECKPOINT_NOT_FOUND_CODE
}

function normalizeResumeCheckpointFailure(response: ArxivSearchResponse) {
  // checkpoint 已失效时，pending_action 只是旧展示镜像；必须清掉，避免用户反复点同一个 resume。
  response.pending_action = null
  response.paper_qa_result = {
    ...(response.paper_qa_result || {}),
    status: 'failed',
    error_code: RESUME_CHECKPOINT_NOT_FOUND_CODE,
    message: RESUME_CHECKPOINT_NOT_FOUND_MESSAGE
  }
  response.answer = response.answer || RESUME_CHECKPOINT_NOT_FOUND_MESSAGE
  response.next_actions = response.next_actions?.length
    ? response.next_actions
    : ['请重新发起论文解析或问答请求']
  return response
}

function ensureAssistantResponse(target: AgentChatMessage<ArxivSearchResponse>) {
  if (!target.response) {
    target.response = createDraftResponse()
  }
  return target.response
}

function upsertStep(steps: AgentStep[], nextStep: AgentStep) {
  const index = steps.findIndex(item => item.step === nextStep.step)
  if (index === -1) {
    steps.push(nextStep)
    return
  }

  steps[index] = {
    ...steps[index],
    ...nextStep
  }
}

function upsertToolCall(response: ArxivSearchResponse, toolCall: AgentToolCall) {
  const index = response.tool_calls.findIndex(
    item => item.tool_name === toolCall.tool_name && item.status === 'running'
  )
  if (index >= 0) {
    response.tool_calls[index] = {
      ...(response.tool_calls[index] || {}),
      ...toolCall
    }
    return
  }

  response.tool_calls.push(toolCall)
}

function applyStreamEvent(
  target: AgentChatMessage<ArxivSearchResponse>,
  event: AgentStreamEvent
) {
  const response = ensureAssistantResponse(target)
  response.streaming_state = {
    run_id: event.run_id,
    sequence: event.sequence,
    event_type: event.event_type,
    active_step: response.streaming_state?.active_step ?? null,
    active_tool_call: response.streaming_state?.active_tool_call ?? null
  }

  if (event.event_type === 'run_start') {
    target.error = null
    return
  }

  if (event.event_type === 'step_start') {
    const stepName = String(event.data?.step || '')
    const stepAction = String(
      event.data?.state?.next_actions?.[0] || event.data?.action || '执行步骤'
    )
    upsertStep(response.steps, {
      step: stepName,
      status: 'running',
      action: stepAction,
      inputs: event.data?.state || {},
      outputs: {},
      error: null
    })
    response.streaming_state.active_step = stepName
    return
  }

  if (event.event_type === 'step_end') {
    const stepName = String(event.data?.step || '')
    const stepDetail = event.data?.step_detail || {}
    upsertStep(response.steps, {
      step: stepName,
      status: (stepDetail.status || 'success') as AgentStep['status'],
      action: String(stepDetail.action || event.data?.action || '步骤结束'),
      inputs: stepDetail.inputs || event.data?.state || {},
      outputs: stepDetail.outputs || {},
      error: stepDetail.error || null
    })

    if (event.data?.state) {
      if (typeof event.data.state.intent === 'string') {
        response.intent = event.data.state.intent
      }
      if (event.data.state.search_spec) {
        response.search_spec = event.data.state.search_spec
      }
      if (Object.prototype.hasOwnProperty.call(event.data.state, 'pending_action')) {
        response.pending_action = event.data.state.pending_action || null
      }
      if (Object.prototype.hasOwnProperty.call(event.data.state, 'paper_qa_result')) {
        response.paper_qa_result = event.data.state.paper_qa_result || null
      }
      if (Array.isArray(event.data.state.next_actions)) {
        response.next_actions = event.data.state.next_actions
      }
    }

    if (response.streaming_state.active_step === stepName) {
      response.streaming_state.active_step = null
    }
    return
  }

  if (event.event_type === 'tool_call_start') {
    const toolCall = event.data?.tool_call || {}
    const activeToolCall = {
      tool_name: String(toolCall.tool_name || 'unknown'),
      arguments: toolCall.arguments || {},
      status: 'running',
      summary: toolCall.summary || `正在调用 ${toolCall.tool_name || '工具'}`,
      trace: toolCall.trace || null,
      error: null
    }
    upsertToolCall(response, activeToolCall)
    response.streaming_state.active_tool_call = activeToolCall
    return
  }

  if (event.event_type === 'tool_call_end') {
    const toolCall = event.data?.tool_call || {}
    const normalizedToolCall = {
      tool_name: String(toolCall.tool_name || 'unknown'),
      arguments: toolCall.arguments || {},
      status: String(toolCall.status || 'success'),
      summary: toolCall.summary || '',
      trace: toolCall.trace || null,
      error: toolCall.error || null
    }
    const index = response.tool_calls.findIndex(
      item => item.tool_name === normalizedToolCall.tool_name && item.status === 'running'
    )
    if (index >= 0) {
      response.tool_calls[index] = normalizedToolCall
    } else {
      upsertToolCall(response, normalizedToolCall)
    }
    response.streaming_state.active_tool_call = null
    return
  }

  if (event.event_type === 'final_response') {
    const finalResponse = event.data?.response
    if (finalResponse && typeof finalResponse === 'object') {
      Object.assign(response, finalResponse)
      response.streaming_state = {
        run_id: event.run_id,
        sequence: event.sequence,
        event_type: event.event_type,
        active_step: null,
        active_tool_call: null
      }
      target.content = response.answer || ''
    }
    return
  }

  if (event.event_type === 'exception') {
    const exceptionResponse = event.data?.response
    if (exceptionResponse && typeof exceptionResponse === 'object') {
      Object.assign(response, exceptionResponse)
      target.content = response.answer || String(event.data?.detail || 'Agent 流式请求失败')
    }
    target.error = String(event.data?.detail || 'Agent 流式请求失败')
    return
  }

  if (event.event_type === 'stream_end') {
    const finalResponse = event.data?.response
    if (finalResponse && typeof finalResponse === 'object') {
      Object.assign(response, finalResponse)
      target.content = response.answer || target.content
    }
    target.loading = false
    response.streaming_state = {
      run_id: event.run_id,
      sequence: event.sequence,
      event_type: event.event_type,
      active_step: null,
      active_tool_call: null
    }
  }
}

function setAssistantResponse(
  target: AgentChatMessage<ArxivSearchResponse>,
  response: ArxivSearchResponse
) {
  target.content = response.answer || ''
  target.response = response
  target.error = null
  target.loading = false
}

function buildFallbackErrorResponse(message: string, detail: string) {
  return {
    intent: 'unsupported',
    answer: message,
    search_spec: null,
    pending_action: null,
    paper_qa_result: null,
    plan: [],
    tool_calls: [],
    papers: [],
    warnings: [detail],
    next_actions: ['请修正输入后重试', '后续可以接入论文总结或 QA 功能'],
    steps: [
      {
        step: 'agent_runtime',
        status: 'failed' as const,
        action: '流式请求失败后回退到错误响应',
        inputs: { detail },
        outputs: {},
        error: detail
      }
    ],
    debug: {},
    streaming_state: null
  } as ArxivSearchResponse
}

export function useAgentSearchChat() {
  const paperStore = usePaperStore()
  const userContext = useUserContext()
  const inputMessage = ref('')
  const loading = ref(false)
  const messages: Ref<AgentChatMessage<ArxivSearchResponse>[]> = ref([])
  const latestResponse = ref<ArxivSearchResponse | null>(null)
  const lastSearchPapers = ref<AgentPaper[]>([])
  const pendingAction = ref<Record<string, any> | null>(null)
  const selectedPaper = ref<AgentPaper | null>(null)
  const paperQaResult = ref<Record<string, any> | null>(null)
  const activeSessionId = ref<string | null>(null)
  const activeSessionUserId = ref<string | null>(null)

  function getUserId() {
    return userContext.getUserId()
  }

  function getActiveSessionId(userId: string) {
    return activeSessionUserId.value === userId ? activeSessionId.value : null
  }

  function setInputMessage(value: string) {
    inputMessage.value = value
  }

  function clearConversation() {
    messages.value = []
    latestResponse.value = null
    lastSearchPapers.value = []
    pendingAction.value = null
    selectedPaper.value = null
    paperQaResult.value = null
    activeSessionId.value = null
    activeSessionUserId.value = null
    inputMessage.value = ''
  }

  function rememberSearchPapers(response: ArxivSearchResponse | null | undefined) {
    if (response?.intent === 'arxiv_search' && Array.isArray(response.papers) && response.papers.length > 0) {
      lastSearchPapers.value = response.papers.map(item => ({ ...item }))
      selectedPaper.value = response.papers[0] ? { ...response.papers[0] } : null
    }
  }

  function rememberSelectedPaper(response: ArxivSearchResponse | null | undefined) {
    const qaArxivId = response?.paper_qa_result?.arxiv_id
    const qaTitle = response?.paper_qa_result?.title

    if (qaArxivId || qaTitle) {
      const fromSearch = lastSearchPapers.value.find(item => {
        const itemId = item.arxiv_id || item.arxivId || item.id
        return qaArxivId && itemId === qaArxivId
      })
      if (fromSearch) {
        selectedPaper.value = { ...fromSearch }
        return
      }
      selectedPaper.value = {
        ...(selectedPaper.value || {}),
        ...(qaArxivId ? { arxiv_id: qaArxivId } : {}),
        ...(qaTitle ? { title: qaTitle } : {})
      }
    }
  }

  function rememberPaperQaResult(response: ArxivSearchResponse | null | undefined) {
    paperQaResult.value = response?.paper_qa_result ? { ...response.paper_qa_result } : null
  }

  function rememberPendingAction(response: ArxivSearchResponse | null | undefined) {
    if (isResumeCheckpointNotFound(response)) {
      pendingAction.value = null
      return
    }

    const resultStatus = response?.paper_qa_result?.status
    if (response?.pending_action && resultStatus === 'waiting_confirmation') {
      pendingAction.value = { ...response.pending_action }
      return
    }

    if (resultStatus === 'success' || resultStatus === 'failed') {
      pendingAction.value = null
      return
    }

    if (!response?.pending_action && resultStatus !== 'waiting_confirmation') {
      pendingAction.value = null
    }
  }

  function rememberSessionId(response: ArxivSearchResponse | null | undefined) {
    const sessionId = typeof response?.session_id === 'string' ? response.session_id.trim() : ''
    if (sessionId) {
      activeSessionId.value = sessionId
      activeSessionUserId.value = getUserId()
    }
  }

  watch(() => getUserId(), () => {
    // Agent 的 pending_action/resume 与后端 checkpoint 绑定用户，切换用户时必须清空现场。
    clearConversation()
  })

  async function submitMessage(
    rawMessage?: string,
    options?: {
      resume?: AgentResumePayload
    }
  ) {
    const message = (rawMessage ?? inputMessage.value).trim()
    if (!message || loading.value) return

    const effectiveUserId = getUserId()
    const userMessageId = createMessageId('user')
    const assistantId = createMessageId('assistant')
    const createdAt = new Date().toISOString()

    const userMessage: AgentChatMessage<ArxivSearchResponse> = {
      id: userMessageId,
      role: 'user',
      content: message,
      loading: false,
      createdAt,
      response: null,
      error: null
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

    messages.value.push(userMessage, assistantMessage)
    latestResponse.value = null
    inputMessage.value = ''
    loading.value = true

    const target = messages.value.find(item => item.id === assistantId)
    if (!target) {
      loading.value = false
      return
    }

    const requestContext: {
      selected_paper?: AgentPaper | null
      last_papers?: AgentPaper[]
      pending_action?: Record<string, any> | null
      paper_qa_result?: Record<string, any> | null
      research_profile?: UserResearchProfile | null
      arxiv_id?: string | null
      source?: 'button' | 'chat' | 'detail_page'
    } = lastSearchPapers.value.length || selectedPaper.value || paperQaResult.value
      || paperStore.researchProfile
      ? {
          ...(lastSearchPapers.value.length
            ? { last_papers: lastSearchPapers.value.map(item => ({ ...item })) }
            : {}),
          selected_paper: selectedPaper.value
            ? { ...selectedPaper.value }
            : lastSearchPapers.value[0]
              ? { ...lastSearchPapers.value[0] }
              : null,
          ...(paperQaResult.value ? { paper_qa_result: { ...paperQaResult.value } } : {}),
          ...(paperStore.researchProfile ? { research_profile: { ...paperStore.researchProfile } } : {}),
          arxiv_id:
            selectedPaper.value?.arxiv_id ||
            selectedPaper.value?.arxivId ||
            selectedPaper.value?.id ||
            paperQaResult.value?.arxiv_id ||
            null,
          source: 'chat'
        }
      : {}
    if (pendingAction.value) {
      requestContext.pending_action = { ...pendingAction.value }
    }

    try {
      const response = await streamAgentChat(
        {
          message,
          user_id: effectiveUserId,
          session_id: getActiveSessionId(effectiveUserId),
          ...(options?.resume ? { resume: options.resume } : {}),
          context: Object.keys(requestContext).length ? requestContext : undefined
        },
        {
          onEvent: event => {
            const currentTarget = messages.value.find(item => item.id === assistantId)
            if (currentTarget) {
              applyStreamEvent(currentTarget, event)
            }
          }
        }
      )

      if (isResumeCheckpointNotFound(response)) {
        normalizeResumeCheckpointFailure(response)
        ElMessage.warning(RESUME_CHECKPOINT_NOT_FOUND_MESSAGE)
      }
      latestResponse.value = response
      rememberSessionId(response)
      rememberSearchPapers(response)
      rememberSelectedPaper(response)
      rememberPaperQaResult(response)
      rememberPendingAction(response)
      if (target.response) {
        target.response = {
          ...target.response,
          ...response
        }
      } else {
        target.response = response
      }
      setAssistantResponse(target, response)
    } catch (streamError) {
      // stream 已经返回明确 resume 失效语义时，不再把同一份 resume 请求 fallback 到普通接口重复尝试。
      if (options?.resume && target.response && isResumeCheckpointNotFound(target.response)) {
        const failedResponse = normalizeResumeCheckpointFailure(target.response)
        latestResponse.value = failedResponse
        rememberPaperQaResult(failedResponse)
        rememberPendingAction(failedResponse)
        setAssistantResponse(target, failedResponse)
        ElMessage.warning(RESUME_CHECKPOINT_NOT_FOUND_MESSAGE)
        return
      }

      try {
        const fallbackResponse = await runAgentChat({
          message,
          user_id: effectiveUserId,
          session_id: getActiveSessionId(effectiveUserId),
          ...(options?.resume ? { resume: options.resume } : {}),
          context: Object.keys(requestContext).length ? requestContext : undefined
        })
        if (isResumeCheckpointNotFound(fallbackResponse)) {
          normalizeResumeCheckpointFailure(fallbackResponse)
          ElMessage.warning(RESUME_CHECKPOINT_NOT_FOUND_MESSAGE)
        }
        latestResponse.value = fallbackResponse
        rememberSessionId(fallbackResponse)
        rememberSearchPapers(fallbackResponse)
        rememberSelectedPaper(fallbackResponse)
        rememberPaperQaResult(fallbackResponse)
        rememberPendingAction(fallbackResponse)
        setAssistantResponse(target, fallbackResponse)
      } catch (fallbackError) {
        const errorMessage = getErrorMessage(
          fallbackError,
          getErrorMessage(streamError, 'Agent 调用失败')
        )
        ElMessage.error(errorMessage)
        const errorResponse = buildFallbackErrorResponse(
          'Agent 流式请求失败，已切换到错误响应',
          errorMessage
        )
        latestResponse.value = errorResponse
        target.content = errorMessage
        target.response = errorResponse
        target.error = errorMessage
        target.loading = false
        inputMessage.value = message
      }
    } finally {
      loading.value = false
      const currentTarget = messages.value.find(item => item.id === assistantId)
      if (currentTarget) {
        currentTarget.loading = false
      }
    }
  }

  async function submitResume(decision: ResumeDecision, note?: string) {
    if (!pendingAction.value || loading.value) return
    const confirmationRequest = pendingAction.value.confirmation_request || {}
    const resumePayload: AgentResumePayload = {
      decision,
      note: note || null,
      step_id: pendingAction.value.step_id || confirmationRequest.step_id || null,
      interrupt_id: pendingAction.value.interrupt_id || confirmationRequest.interrupt_id || null
    }
    if (pendingAction.value.edited_arguments) {
      resumePayload.edited_arguments = { ...pendingAction.value.edited_arguments }
    }

    await submitMessage(decision === 'approve' ? '确认执行当前工具操作' : '拒绝执行当前工具操作', {
      // 确认按钮必须走结构化 resume；message 只是满足后端请求模型的可读占位文本。
      resume: resumePayload
    })
  }

  return {
    inputMessage,
    loading,
    messages,
    latestResponse,
    lastSearchPapers,
    pendingAction,
    selectedPaper,
    paperQaResult,
    activeSessionId,
    setInputMessage,
    submitMessage,
    submitResume,
    clearConversation
  }
}
