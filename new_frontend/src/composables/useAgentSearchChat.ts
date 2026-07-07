import { ref, watch, type Ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  createAgentQaIndexContinuation,
  listActiveAgentQaIndexContinuations,
  runAgentChat,
  streamAgentChat,
  updateAgentQaIndexContinuationStatus
} from '@/api/agent'
import { getErrorMessage } from '@/api/errors'
import { getPaperQaIndexJob } from '@/api/papers'
import type {
  AgentPaper,
  AgentPendingAction,
  AgentQaIndexContinuation,
  AgentQaIndexJob,
  AgentStep,
  AgentStreamEvent,
  AgentToolCall,
  ArxivSearchResponse
} from '@/types/agent'
import type { AgentChatMessage } from '@/types/agentChat'
import type { UserResearchProfile } from '@/types/paper'
import { usePaperStore } from '@/stores/paperStore'
import { useUserContext } from '@/composables/useUserContext'

type ResumeDecision = 'approve' | 'reject'
const RESUME_CHECKPOINT_NOT_FOUND_CODE = 'resume_checkpoint_not_found'
const RESUME_CHECKPOINT_NOT_FOUND_MESSAGE = '原执行现场已失效，请重新发起论文解析或问答请求。'
const VISIBLE_PENDING_ACTION_STATUSES = new Set([
  'waiting_confirmation',
  'confirming',
  'index_building',
  'index_failed',
  'ready_to_resume'
])

interface AgentResumePayload {
  decision: ResumeDecision
  note?: string | null
  step_id?: string | null
  interrupt_id?: string | null
  tool_name?: string | null
  pending_action_id?: string | null
  edited_arguments?: Record<string, any> | null
}

interface ActiveConfirmationSubmission {
  key: string
  decision: ResumeDecision
  step_id?: string | null
  tool_name?: string | null
  pending_action_id?: string | null
}

interface SubmitMessageOptions {
  resume?: AgentResumePayload
  optimisticToolCall?: AgentToolCall
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

function normalizePendingActionForDisplay(action: AgentPendingAction | Record<string, any> | null | undefined): AgentPendingAction | null {
  if (!action || typeof action !== 'object') return null
  // pending_action 只是后端确认请求的展示镜像；这里只额外保留前端接管的索引构建中间态。
  const status = String(action.status || '').trim()
  return VISIBLE_PENDING_ACTION_STATUSES.has(status)
    ? { ...action } as AgentPendingAction
    : null
}

function applyFinalResponse(response: ArxivSearchResponse, finalResponse: Record<string, any>) {
  const existingToolCalls = Array.isArray(response.tool_calls) ? [...response.tool_calls] : []
  Object.assign(response, finalResponse)
  if ((!Array.isArray(response.tool_calls) || response.tool_calls.length === 0) && existingToolCalls.length) {
    // 兼容旧后端或异常路径：最终响应缺少 tool_calls 时，保留流式阶段已经展示的工具进度。
    response.tool_calls = existingToolCalls
  }
  response.pending_action = normalizePendingActionForDisplay(response.pending_action)
}

function getPendingActionKey(action: AgentPendingAction | Record<string, any> | null | undefined) {
  if (!action || typeof action !== 'object') return ''
  const pendingActionId = String(action.pending_action_id || action.confirmation_request?.pending_action_id || '').trim()
  if (pendingActionId) return `pending:${pendingActionId}`
  const sessionId = String(action.session_id || action.confirmation_request?.session_id || '').trim()
  const stepId = String(action.step_id || action.confirmation_request?.step_id || '').trim()
  const toolName = String(action.tool_name || action.confirmation_request?.tool_name || '').trim()
  return [sessionId, stepId, toolName].some(Boolean) ? `step:${sessionId}:${stepId}:${toolName}` : ''
}

function isIndexBuildPendingAction(action: AgentPendingAction | null | undefined) {
  const toolName = String(action?.tool_name || action?.confirmation_request?.tool_name || '').trim()
  return toolName === 'parse_and_index_paper'
}

function resolveIndexBuildArxivId(action: AgentPendingAction | null | undefined) {
  const targetPaper = action?.target_paper || action?.confirmation_request?.target_paper || {}
  const args = action?.arguments_summary || action?.confirmation_request?.arguments_summary || {}
  const paperReference = args.paper_reference || args.paper_ref || {}
  return String(
    action?.arxiv_id ||
    targetPaper.arxiv_id ||
    targetPaper.arxivId ||
    paperReference.arxiv_id ||
    paperReference.arxivId ||
    paperReference.id ||
    ''
  ).trim()
}

function resolveIndexBuildLoadingMethod(action: AgentPendingAction | null | undefined) {
  const args = action?.arguments_summary || action?.confirmation_request?.arguments_summary || {}
  return String(args.loading_method || args.loadingMethod || 'docling').trim() || 'docling'
}

function normalizeIndexActionStatusFromJob(job: AgentQaIndexJob | null | undefined) {
  const status = String(job?.status || '').trim().toLowerCase()
  if (status === 'success') return 'ready_to_resume'
  if (['failed', 'stale', 'cancelled'].includes(status)) return 'index_failed'
  return 'index_building'
}

function withIndexBuildState(
  action: AgentPendingAction,
  job: AgentQaIndexJob,
  continuation?: AgentQaIndexContinuation | null
): AgentPendingAction {
  return {
    ...action,
    status: normalizeIndexActionStatusFromJob(job),
    index_job: { ...job },
    index_continuation: continuation
      ? {
          ...continuation,
          job: { ...job }
        }
      : action.index_continuation || null
  }
}

function isPendingActionVisible(action: AgentPendingAction | Record<string, any> | null | undefined): action is AgentPendingAction {
  return Boolean(action && typeof action === 'object' && VISIBLE_PENDING_ACTION_STATUSES.has(String(action.status || '').trim()))
}

function normalizePendingActionForBanner(action: AgentPendingAction | Record<string, any> | null | undefined): AgentPendingAction | null {
  if (!isPendingActionVisible(action)) return null
  return { ...action }
}

function buildConfirmationPlaceholder(decision: ResumeDecision, action: AgentPendingAction) {
  const toolName = String(action.tool_name || '').trim() || '当前工具'
  const stepId = String(action.step_id || '').trim()
  const pendingActionId = String(action.pending_action_id || '').trim()
  if (decision === 'approve') {
    return `已提交确认请求，等待执行 ${toolName}${stepId ? ` (${stepId})` : ''}`
  }
  return `已提交取消请求，等待后端消费 ${toolName}${pendingActionId ? ` [${pendingActionId}]` : ''}`
}

function shouldIgnoreBlockedPendingAction(
  action: AgentPendingAction | Record<string, any> | null | undefined,
  blockedPendingActionKey?: string | null
) {
  if (!blockedPendingActionKey) return false
  const incomingKey = getPendingActionKey(action)
  return Boolean(incomingKey) && incomingKey === blockedPendingActionKey
}

function applyFinalResponseWithConfirmationGuard(
  response: ArxivSearchResponse,
  finalResponse: Record<string, any>,
  blockedPendingActionKey?: string | null
) {
  if (!blockedPendingActionKey) {
    applyFinalResponse(response, finalResponse)
    return
  }
  const existingToolCalls = Array.isArray(response.tool_calls) ? [...response.tool_calls] : []
  Object.assign(response, finalResponse)
  if ((!Array.isArray(response.tool_calls) || response.tool_calls.length === 0) && existingToolCalls.length) {
    response.tool_calls = existingToolCalls
  }
  response.pending_action = shouldIgnoreBlockedPendingAction(response.pending_action, blockedPendingActionKey)
    ? null
    : normalizePendingActionForBanner(response.pending_action)
}

function applyStreamEvent(
  target: AgentChatMessage<ArxivSearchResponse>,
  event: AgentStreamEvent,
  options?: {
    blockedPendingActionKey?: string | null
  }
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
        const incomingPendingAction = normalizePendingActionForBanner(event.data.state.pending_action)
        response.pending_action = shouldIgnoreBlockedPendingAction(incomingPendingAction, options?.blockedPendingActionKey)
          ? null
          : incomingPendingAction
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
      applyFinalResponseWithConfirmationGuard(response, finalResponse, options?.blockedPendingActionKey)
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
      applyFinalResponseWithConfirmationGuard(response, exceptionResponse, options?.blockedPendingActionKey)
      target.content = response.answer || String(event.data?.detail || 'Agent 流式请求失败')
    }
    target.error = String(event.data?.detail || 'Agent 流式请求失败')
    return
  }

  if (event.event_type === 'stream_end') {
    const finalResponse = event.data?.response
    if (finalResponse && typeof finalResponse === 'object') {
      applyFinalResponseWithConfirmationGuard(response, finalResponse, options?.blockedPendingActionKey)
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
  const pendingAction = ref<AgentPendingAction | null>(null)
  const activeConfirmationSubmission = ref<ActiveConfirmationSubmission | null>(null)
  const selectedPaper = ref<AgentPaper | null>(null)
  const paperQaResult = ref<Record<string, any> | null>(null)
  const activeSessionId = ref<string | null>(null)
  const activeSessionUserId = ref<string | null>(null)
  const indexContinuationTimers = new Map<string, number>()

  function getUserId() {
    return userContext.getUserId()
  }

  function getActiveSessionId(userId: string) {
    return activeSessionUserId.value === userId ? activeSessionId.value : null
  }

  function setInputMessage(value: string) {
    inputMessage.value = value
  }

  function updateVisiblePendingAction(nextAction: AgentPendingAction | null) {
    const nextKey = getPendingActionKey(nextAction)
    pendingAction.value = nextAction ? { ...nextAction } : null
    for (const message of messages.value) {
      const response = message.response
      if (!response?.pending_action) continue
      const responseKey = getPendingActionKey(response.pending_action)
      if (!nextKey || !responseKey || responseKey !== nextKey) continue
      response.pending_action = nextAction ? { ...nextAction } : null
    }
    if (latestResponse.value?.pending_action && nextKey && getPendingActionKey(latestResponse.value.pending_action) === nextKey) {
      latestResponse.value.pending_action = nextAction ? { ...nextAction } : null
    }
  }

  function buildResumePayloadFromAction(
    action: AgentPendingAction,
    decision: ResumeDecision,
    note?: string,
    editedArguments?: Record<string, any>
  ): AgentResumePayload {
    const confirmationRequest = action.confirmation_request || {}
    const mergedEditedArguments = {
      ...(action.edited_arguments || {}),
      ...(editedArguments || {})
    }
    const resumePayload: AgentResumePayload = {
      decision,
      note: note || null,
      step_id: action.step_id || confirmationRequest.step_id || null,
      interrupt_id: action.interrupt_id || confirmationRequest.interrupt_id || null,
      tool_name: action.tool_name || confirmationRequest.tool_name || null,
      pending_action_id: action.pending_action_id || confirmationRequest.pending_action_id || null
    }
    if (Object.keys(mergedEditedArguments).length) {
      // 目标论文确认只通过 edited_arguments 传稳定 paper_id/arxiv_id；message 仍只是占位文本。
      resumePayload.edited_arguments = mergedEditedArguments
    }
    return resumePayload
  }

  function ensureContinuationMessage(action: AgentPendingAction) {
    const actionKey = getPendingActionKey(action)
    if (messages.value.some(message => message.response?.pending_action && getPendingActionKey(message.response.pending_action) === actionKey)) {
      return
    }
    const assistantId = createMessageId('assistant')
    const response = createDraftResponse()
    response.session_id = action.session_id || action.confirmation_request?.session_id || activeSessionId.value
    response.answer = action.description || '索引构建完成后会继续回答原问题。'
    response.pending_action = { ...action }
    response.paper_qa_result = {
      status: 'waiting_confirmation',
      arxiv_id: resolveIndexBuildArxivId(action),
      question: action.original_question || action.qa_question || action.confirmation_request?.original_question || ''
    }
    messages.value.push({
      id: assistantId,
      role: 'assistant',
      content: response.answer,
      loading: false,
      createdAt: new Date().toISOString(),
      response,
      error: null
    })
  }

  function clearConversation() {
    for (const timerId of indexContinuationTimers.values()) {
      window.clearTimeout(timerId)
    }
    indexContinuationTimers.clear()
    messages.value = []
    latestResponse.value = null
    lastSearchPapers.value = []
    pendingAction.value = null
    activeConfirmationSubmission.value = null
    selectedPaper.value = null
    paperQaResult.value = null
    activeSessionId.value = null
    activeSessionUserId.value = null
    inputMessage.value = ''
  }

  function rememberSearchPapers(response: ArxivSearchResponse | null | undefined) {
    if (response?.intent === 'arxiv_search' && Array.isArray(response.papers) && response.papers.length > 0) {
      lastSearchPapers.value = response.papers.map(item => ({ ...item }))
      // 搜索结果第一篇只是列表项，不代表用户当前选中；伪造 selected_paper 会让“这篇论文”误指向第一篇。
      selectedPaper.value = null
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
      activeConfirmationSubmission.value = null
      return
    }

    const resultStatus = response?.paper_qa_result?.status
    const displayPendingAction = normalizePendingActionForBanner(response?.pending_action)
    const activePendingActionKey = activeConfirmationSubmission.value?.key || null
    if (shouldIgnoreBlockedPendingAction(displayPendingAction, activePendingActionKey)) {
      return
    }
    if (displayPendingAction && resultStatus === 'waiting_confirmation') {
      pendingAction.value = { ...displayPendingAction }
      activeConfirmationSubmission.value = null
      return
    }

    const responsePendingStatus = String(response?.pending_action?.status || '').trim()
    if (
      resultStatus === 'success'
      || resultStatus === 'failed'
      || responsePendingStatus === 'approved'
      || responsePendingStatus === 'rejected'
      || response?.pending_action?.confirmation_consumed === true
    ) {
      pendingAction.value = null
      activeConfirmationSubmission.value = null
      return
    }

    if (!response?.pending_action && resultStatus !== 'waiting_confirmation') {
      pendingAction.value = null
      activeConfirmationSubmission.value = null
    }
  }

  function rememberSessionId(response: ArxivSearchResponse | null | undefined) {
    const sessionId = typeof response?.session_id === 'string' ? response.session_id.trim() : ''
    if (sessionId) {
      activeSessionId.value = sessionId
      activeSessionUserId.value = getUserId()
    }
  }

  function scheduleIndexContinuationPolling(action: AgentPendingAction, delayMs = 1500) {
    const jobId = String(action.index_job?.job_id || action.index_continuation?.job_id || '').trim()
    if (!jobId) return
    const previousTimer = indexContinuationTimers.get(jobId)
    if (previousTimer) {
      window.clearTimeout(previousTimer)
    }
    const timerId = window.setTimeout(() => {
      indexContinuationTimers.delete(jobId)
      pollIndexContinuation(action)
    }, delayMs)
    indexContinuationTimers.set(jobId, timerId)
  }

  async function pollIndexContinuation(action: AgentPendingAction) {
    const jobId = String(action.index_job?.job_id || action.index_continuation?.job_id || '').trim()
    const arxivId = String(action.index_job?.arxiv_id || action.index_continuation?.arxiv_id || resolveIndexBuildArxivId(action)).trim()
    if (!jobId || !arxivId) return

    try {
      const job = await getPaperQaIndexJob(arxivId, jobId) as AgentQaIndexJob
      const nextAction = withIndexBuildState(action, job, action.index_continuation || null)
      updateVisiblePendingAction(nextAction)
      const status = String(job.status || '').trim().toLowerCase()
      if (['pending', 'running', 'retrying'].includes(status)) {
        scheduleIndexContinuationPolling(nextAction, 1500)
        return
      }
      if (status === 'success') {
        await updateAgentQaIndexContinuationStatus(jobId, { user_id: getUserId(), status: 'ready_to_resume' })
        // 异步 job 已完成后再消费原来的结构化确认，Agent 会复查索引并继续回答原问题。
        await submitResume('approve', '索引构建完成，继续回答原问题')
        await updateAgentQaIndexContinuationStatus(jobId, { user_id: getUserId(), status: 'resumed' })
        return
      }
      await updateAgentQaIndexContinuationStatus(jobId, {
        user_id: getUserId(),
        status: 'failed',
        error_message: job.error_message || '索引构建失败，请重试。'
      })
    } catch (error) {
      const failedAction: AgentPendingAction = {
        ...action,
        status: 'index_failed',
        index_job: {
          ...(action.index_job || {}),
          job_id: jobId,
          arxiv_id: arxivId,
          status: 'failed',
          progress: action.index_job?.progress || 0,
          current_stage: action.index_job?.current_stage || 'failed',
          error_message: getErrorMessage(error, '查询索引任务状态失败')
        }
      }
      updateVisiblePendingAction(failedAction)
    }
  }

  async function submitIndexBuildContinuation() {
    const action = pendingAction.value
    if (!action || loading.value || action.status === 'confirming') return false
    if (!isIndexBuildPendingAction(action)) return false
    if (action.status === 'ready_to_resume') {
      await submitResume('approve', '索引构建完成，继续回答原问题')
      const jobId = String(action.index_job?.job_id || action.index_continuation?.job_id || '').trim()
      if (jobId) {
        await updateAgentQaIndexContinuationStatus(jobId, { user_id: getUserId(), status: 'resumed' }).catch(() => null)
      }
      return true
    }
    if (action.status === 'index_building') return true

    const arxivId = resolveIndexBuildArxivId(action)
    if (!arxivId) {
      ElMessage.error('缺少论文 arXiv ID，无法创建索引任务')
      return true
    }
    const resumePayload = buildResumePayloadFromAction(action, 'approve', '索引构建完成，继续回答原问题')
    try {
      const result = await createAgentQaIndexContinuation({
        user_id: getUserId(),
        session_id: activeSessionId.value || action.session_id || action.confirmation_request?.session_id || null,
        arxiv_id: arxivId,
        loading_method: resolveIndexBuildLoadingMethod(action),
        original_question: action.original_question || action.qa_question || action.confirmation_request?.original_question || null,
        pending_action: action,
        resume_payload: resumePayload
      })
      const nextAction = withIndexBuildState(action, result.job, result.continuation)
      updateVisiblePendingAction(nextAction)
      ElMessage.success('问答索引任务已提交，正在后台构建')
      scheduleIndexContinuationPolling(nextAction, 1200)
      return true
    } catch (error) {
      ElMessage.error(getErrorMessage(error, '提交索引构建任务失败'))
      return true
    }
  }

  async function cancelIndexBuildContinuation() {
    const action = pendingAction.value
    const jobId = String(action?.index_job?.job_id || action?.index_continuation?.job_id || '').trim()
    if (!action || !jobId) return false
    const timerId = indexContinuationTimers.get(jobId)
    if (timerId) {
      window.clearTimeout(timerId)
      indexContinuationTimers.delete(jobId)
    }
    await updateAgentQaIndexContinuationStatus(jobId, {
      user_id: getUserId(),
      status: 'cancelled',
      error_message: '用户取消本次索引构建后的自动问答'
    }).catch(() => null)
    updateVisiblePendingAction(null)
    return true
  }

  async function restoreActiveIndexContinuations() {
    try {
      const result = await listActiveAgentQaIndexContinuations({
        user_id: getUserId(),
        session_id: activeSessionId.value || undefined,
        limit: 5
      })
      const continuation = (result.continuations || [])[0]
      const pending = continuation?.pending_action as AgentPendingAction | null
      const job = continuation?.job
      if (!continuation || !pending || !job) return
      activeSessionId.value = continuation.session_id
      activeSessionUserId.value = getUserId()
      const restoredAction = withIndexBuildState(
        {
          ...pending,
          session_id: continuation.session_id,
          pending_action_id: continuation.pending_action_id || pending.pending_action_id || null,
          step_id: continuation.step_id || pending.step_id || null,
          tool_name: continuation.tool_name || pending.tool_name || 'parse_and_index_paper'
        },
        job,
        continuation
      )
      ensureContinuationMessage(restoredAction)
      updateVisiblePendingAction(restoredAction)
      if (String(job.status || '').toLowerCase() === 'success') {
        await submitResume('approve', '索引构建完成，继续回答原问题')
        await updateAgentQaIndexContinuationStatus(job.job_id, { user_id: getUserId(), status: 'resumed' })
      } else if (!['failed', 'stale', 'cancelled'].includes(String(job.status || '').toLowerCase())) {
        scheduleIndexContinuationPolling(restoredAction, 1200)
      }
    } catch {
      // 恢复失败只影响刷新后的进度卡展示，不应该阻塞用户正常发起新的 Agent 对话。
    }
  }

  watch(() => getUserId(), () => {
    // Agent 的 pending_action/resume 与后端 checkpoint 绑定用户，切换用户时必须清空现场。
    clearConversation()
  })

  async function submitMessage(
    rawMessage?: string,
    options?: SubmitMessageOptions
  ) {
    const message = (rawMessage ?? inputMessage.value).trim()
    if (!message || loading.value) return

    const effectiveUserId = getUserId()
    const userMessageId = createMessageId('user')
    const assistantId = createMessageId('assistant')
    const createdAt = new Date().toISOString()

    const userMessage: AgentChatMessage<ArxivSearchResponse> = {
      id: userMessageId,
      role: options?.skipUserMessage ? 'system' : 'user',
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
    if (options?.optimisticToolCall) {
      // resume approve 后后端会立刻进入耗时工具，先放一张 running 卡片，避免用户误以为点击没有生效。
      assistantMessage.response?.tool_calls.push(options.optimisticToolCall)
      assistantMessage.response!.streaming_state = {
        run_id: assistantId,
        sequence: 0,
        event_type: 'optimistic_tool_call',
        active_step: options.optimisticToolCall.trace?.step_id || null,
        active_tool_call: options.optimisticToolCall
      }
    }

    if (!options?.skipUserMessage) {
      messages.value.push(userMessage)
    }
    messages.value.push(assistantMessage)
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
      pending_action?: AgentPendingAction | Record<string, any> | null
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
          selected_paper: selectedPaper.value ? { ...selectedPaper.value } : null,
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
              applyStreamEvent(currentTarget, event, {
                blockedPendingActionKey: activeConfirmationSubmission.value?.key || null
              })
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
      const displayResponse: ArxivSearchResponse = {
        ...response,
        pending_action: shouldIgnoreBlockedPendingAction(response.pending_action, activeConfirmationSubmission.value?.key || null)
          ? null
          : normalizePendingActionForBanner(response.pending_action)
      }
      if (target.response) {
        target.response = {
          ...target.response,
          ...displayResponse
        }
      } else {
        target.response = displayResponse
      }
      setAssistantResponse(target, (target.response || displayResponse) as ArxivSearchResponse)
    } catch (streamError) {
      // stream 已经返回明确 resume 失效语义时，不再把同一份 resume 请求 fallback 到普通接口重复尝试。
      if (options?.resume && target.response && isResumeCheckpointNotFound(target.response)) {
        const failedResponse = normalizeResumeCheckpointFailure(target.response)
        const failedDisplayResponse: ArxivSearchResponse = {
          ...failedResponse,
          pending_action: normalizePendingActionForBanner(failedResponse.pending_action)
        }
        latestResponse.value = failedResponse
        rememberPaperQaResult(failedResponse)
        rememberPendingAction(failedResponse)
        setAssistantResponse(target, failedDisplayResponse)
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
        const fallbackDisplayResponse: ArxivSearchResponse = {
          ...fallbackResponse,
          pending_action: normalizePendingActionForBanner(fallbackResponse.pending_action)
        }
        latestResponse.value = fallbackResponse
        rememberSessionId(fallbackResponse)
        rememberSearchPapers(fallbackResponse)
        rememberSelectedPaper(fallbackResponse)
        rememberPaperQaResult(fallbackResponse)
        rememberPendingAction(fallbackResponse)
        setAssistantResponse(target, fallbackDisplayResponse)
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
        pendingAction.value = null
        activeConfirmationSubmission.value = null
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

  async function submitResume(decision: ResumeDecision, note?: string, editedArguments?: Record<string, any>) {
    if (!pendingAction.value || loading.value || pendingAction.value.status === 'confirming') return
    const currentPendingAction = { ...pendingAction.value }
    const resumePayload = buildResumePayloadFromAction(currentPendingAction, decision, note, editedArguments)
    const pendingActionKey = getPendingActionKey(currentPendingAction)
    activeConfirmationSubmission.value = pendingActionKey
      ? {
          key: pendingActionKey,
          decision,
          step_id: resumePayload.step_id,
          tool_name: resumePayload.tool_name,
          pending_action_id: resumePayload.pending_action_id
        }
      : null
    pendingAction.value = {
      ...currentPendingAction,
      status: 'confirming',
      decision,
      edited_arguments: resumePayload.edited_arguments || currentPendingAction.edited_arguments || null
    }
    const isPaperTargetConfirmation = currentPendingAction.request_type === 'paper_target_confirmation'
      || currentPendingAction.type === 'paper_target_confirmation'

    await submitMessage(buildConfirmationPlaceholder(decision, currentPendingAction), {
      // 确认按钮必须走结构化 resume；这里的 message 只保留给请求体做可读占位，不能再驱动后端判断确认对象。
      resume: resumePayload,
      skipUserMessage: true,
      optimisticToolCall: decision === 'approve'
        ? {
            tool_name: String(currentPendingAction.tool_name || (isPaperTargetConfirmation ? 'resolve_paper' : 'parse_and_index_paper')),
            arguments: isPaperTargetConfirmation
              ? { ...(currentPendingAction.arguments_summary || {}), ...(resumePayload.edited_arguments || {}) }
              : currentPendingAction.arguments_summary || {},
            status: 'running',
            summary: isPaperTargetConfirmation
              ? '已确认目标论文，正在继续执行原动作'
              : '用户已确认，正在执行索引构建工具',
            trace: {
              step_id: resumePayload.step_id,
              source: 'resume_approve'
            },
            error: null
          }
        : undefined
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
    submitIndexBuildContinuation,
    cancelIndexBuildContinuation,
    restoreActiveIndexContinuations,
    clearConversation
  }
}
