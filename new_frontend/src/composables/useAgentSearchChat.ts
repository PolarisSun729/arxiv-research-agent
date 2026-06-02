import { ref, type Ref } from 'vue'
import { ElMessage } from 'element-plus'
import { runAgentChat, streamAgentChat } from '@/api/agent'
import type { AgentPaper, AgentStep, AgentStreamEvent, AgentToolCall, ArxivSearchResponse } from '@/types/agent'
import type { AgentChatMessage } from '@/types/agentChat'

function createMessageId(prefix: 'user' | 'assistant') {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

function getErrorMessage(error: unknown, fallback: string) {
  if (
    error &&
    typeof error === 'object' &&
    'message' in error &&
    typeof (error as { message?: unknown }).message === 'string' &&
    (error as { message?: string }).message
  ) {
    return (error as { message: string }).message
  }
  return fallback
}

function createDraftResponse(): ArxivSearchResponse {
  return {
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
    streaming_state: null
  }
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
    streaming_state: null
  } as ArxivSearchResponse
}

export function useAgentSearchChat() {
  const inputMessage = ref('')
  const loading = ref(false)
  const messages: Ref<AgentChatMessage<ArxivSearchResponse>[]> = ref([])
  const latestResponse = ref<ArxivSearchResponse | null>(null)
  const lastSearchPapers = ref<AgentPaper[]>([])
  const pendingAction = ref<Record<string, any> | null>(null)

  function setInputMessage(value: string) {
    inputMessage.value = value
  }

  function clearConversation() {
    messages.value = []
    latestResponse.value = null
    lastSearchPapers.value = []
    pendingAction.value = null
    inputMessage.value = ''
  }

  function rememberSearchPapers(response: ArxivSearchResponse | null | undefined) {
    if (response?.intent === 'arxiv_search' && Array.isArray(response.papers) && response.papers.length > 0) {
      lastSearchPapers.value = response.papers.map(item => ({ ...item }))
    }
  }

  function rememberPendingAction(response: ArxivSearchResponse | null | undefined) {
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

  async function submitMessage(rawMessage?: string) {
    const message = (rawMessage ?? inputMessage.value).trim()
    if (!message || loading.value) return

    const userId = createMessageId('user')
    const assistantId = createMessageId('assistant')
    const createdAt = new Date().toISOString()

    const userMessage: AgentChatMessage<ArxivSearchResponse> = {
      id: userId,
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
      source?: 'button' | 'chat' | 'detail_page'
    } = lastSearchPapers.value.length
      ? {
          last_papers: lastSearchPapers.value.map(item => ({ ...item })),
          selected_paper: lastSearchPapers.value[0] ? { ...lastSearchPapers.value[0] } : null,
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
          user_id: 'local_user',
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

      latestResponse.value = response
      rememberSearchPapers(response)
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
      try {
        const fallbackResponse = await runAgentChat({
          message,
          user_id: 'local_user',
          context: Object.keys(requestContext).length ? requestContext : undefined
        })
        latestResponse.value = fallbackResponse
        rememberSearchPapers(fallbackResponse)
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

  return {
    inputMessage,
    loading,
    messages,
    latestResponse,
    lastSearchPapers,
    pendingAction,
    setInputMessage,
    submitMessage,
    clearConversation
  }
}
