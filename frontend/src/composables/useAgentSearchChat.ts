import { reactive, ref, watch, type Ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  cancelAgentWorkContinuation,
  getAgentChatMessages,
  getAgentResumeRun,
  listAgentChatSessions,
  listAgentWorkContinuations,
  runAgentChat,
  streamAgentChat,
  streamAgentWorkContinuationResume
} from '@/api/agent'
import { getApiErrorMessage, getErrorMessage } from '@/api/errors'
import { normalizeServerTimestamp } from '@/utils/format'
import { isRecord } from '@/utils/guards'
import { applyProgressEvent, createRunProgress, finishRunProgress } from '@/utils/agentRunProgress'
import type {
  AgentChatSessionSummary,
  AgentInteraction,
  AgentPaper,
  AgentWorkContinuation,
  AgentStoredChatMessage,
  AgentStreamEvent,
  AgentToolCall,
  ArxivSearchRequest,
  ArxivSearchResponse
} from '@/types/agent'
import type { AgentChatMessage } from '@/types/agentChat'
import type { UserResearchProfile } from '@/types/paper'
import { usePaperStore } from '@/stores/paperStore'
import { useUserContext } from '@/composables/useUserContext'

type ResumeDecision = 'approve' | 'reject' | 'select' | 'cancel'

const CONTINUATION_POLL_INTERVAL_MS = 2500
const HIDDEN_CONTINUATION_POLL_INTERVAL_MS = 10000
const LAST_SESSION_STORAGE_PREFIX = 'agent_search_last_session:'
const INTERRUPTED_TURN_MESSAGE = '这一轮回答没有完成（页面刷新或连接中断），可以重新发送。'

interface SubmitMessageOptions {
  resume?: NonNullable<ArxivSearchRequest['resume']>
  skipUserMessage?: boolean
  sessionId?: string | null
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

function lastSessionStorageKey(userId: string) {
  return `${LAST_SESSION_STORAGE_PREFIX}${userId}`
}

// 只记一个指针：从菜单直接进入页面时回到上次对话；内容本身始终以后端流水为准。
function readLastSessionId(userId: string) {
  if (!userId) return null
  try {
    return typeof window === 'undefined' ? null : window.localStorage.getItem(lastSessionStorageKey(userId))
  } catch {
    return null
  }
}

function writeLastSessionId(userId: string, sessionId: string | null) {
  if (!userId || typeof window === 'undefined') return
  try {
    if (sessionId) {
      window.localStorage.setItem(lastSessionStorageKey(userId), sessionId)
    } else {
      window.localStorage.removeItem(lastSessionStorageKey(userId))
    }
  } catch {
    // 存储不可用只影响“进入页面回到上次对话”，URL 中的 session 仍可恢复。
  }
}

function buildConfirmationPlaceholder(interaction: AgentInteraction | null) {
  if (interaction?.kind === 'target_selection') return '正在提交论文选择并恢复 Agent…'
  if (interaction?.kind === 'side_effect_approval') return '正在保存授权并创建后台任务…'
  return ''
}

function getAgentRuntimeErrorMessage(response: ArxivSearchResponse | null | undefined) {
  const runtimeError = response?.debug?.runtime_error
  if (!isRecord(runtimeError)) return null
  const code = typeof runtimeError.code === 'string' && runtimeError.code ? runtimeError.code : 'agent_runtime_error'
  const fallback = typeof runtimeError.message === 'string' && runtimeError.message
    ? runtimeError.message
    : response?.answer || 'Agent 调用失败'
  return getApiErrorMessage(code, fallback)
}

function getStreamEventErrorMessage(event: AgentStreamEvent, response: ArxivSearchResponse | null | undefined) {
  const responseError = getAgentRuntimeErrorMessage(response)
  if (responseError) return responseError
  const payload = event.data || {}
  const code = typeof payload.code === 'string' && payload.code
    ? payload.code
    : typeof payload.error?.code === 'string' ? payload.error.code : 'agent_runtime_error'
  const fallback = typeof payload.message === 'string' && payload.message ? payload.message : 'Agent 调用失败'
  return getApiErrorMessage(code, fallback)
}

function toRestoredMessages(items: AgentStoredChatMessage[]): AgentChatMessage<ArxivSearchResponse>[] {
  const restored: AgentChatMessage<ArxivSearchResponse>[] = []
  let unansweredUser: AgentStoredChatMessage | null = null
  const pushInterrupted = (userItem: AgentStoredChatMessage) => {
    restored.push({
      id: `interrupted-${userItem.message_id}`,
      role: 'assistant',
      content: INTERRUPTED_TURN_MESSAGE,
      loading: false,
      createdAt: normalizeServerTimestamp(userItem.created_at),
      response: null,
      error: INTERRUPTED_TURN_MESSAGE,
      retryMessage: userItem.content
    })
  }
  for (const item of items) {
    if (item.role === 'user') {
      // 用户消息在执行前就落库；后面紧跟的不是回答，说明那一轮中途断开、没有结果。
      if (unansweredUser) pushInterrupted(unansweredUser)
      unansweredUser = item
      restored.push({
        id: item.message_id,
        role: 'user',
        content: item.content,
        loading: false,
        createdAt: normalizeServerTimestamp(item.created_at),
        response: null,
        error: null
      })
      continue
    }
    unansweredUser = null
    restored.push({
      id: item.message_id,
      role: 'assistant',
      content: item.content,
      loading: false,
      createdAt: normalizeServerTimestamp(item.created_at),
      response: item.response,
      error: getAgentRuntimeErrorMessage(item.response) || (item.status === 'error' ? item.content || 'Agent 调用失败' : null)
    })
  }
  if (unansweredUser) pushInterrupted(unansweredUser)
  return restored
}

function applyStreamEvent(target: AgentChatMessage<ArxivSearchResponse>, event: AgentStreamEvent) {
  const response = target.response || createDraftResponse()
  const payload = event.data || {}
  // 中间步骤时间线与最终响应并行维护：final_response 会替换 response，但进度要保留下来供用户回看。
  target.progress = applyProgressEvent(target.progress || createRunProgress(Date.now()), event, Date.now())
  if (event.event_type === 'final_response' && payload.response) {
    target.response = payload.response as ArxivSearchResponse
    target.content = target.response.answer || ''
    const runtimeErrorMessage = getAgentRuntimeErrorMessage(target.response)
    if (runtimeErrorMessage) {
      // 后端以 200 + final_response 回传 Agent 运行错误时，仍要把气泡标成错误态，避免看起来像普通空回复。
      target.error = runtimeErrorMessage
      target.loading = false
    }
    return
  }
  if (event.event_type === 'exception') {
    const errorResponse = payload.response ? payload.response as ArxivSearchResponse : response
    target.response = errorResponse
    target.content = errorResponse.answer || payload.message || 'Agent 调用失败'
    // exception 是流式协议里的失败终态信号之一；先落到当前气泡，避免后续网络中断时 UI 卡在 loading。
    target.error = getStreamEventErrorMessage(event, errorResponse)
    target.loading = false
    return
  }
  if (event.event_type === 'stream_end' && payload.status === 'error') {
    // 没有 final_response 的异常流也必须显式结束当前气泡，不能继续展示“正在生成”。
    target.error = target.error || getStreamEventErrorMessage(event, response)
    target.content = target.content || target.error || 'Agent 调用失败'
    target.loading = false
    return
  }
  const activeToolCall = isRecord(payload.tool_call) ? payload.tool_call as AgentToolCall : null
  // tool_call_start/tool_call_end 是前端展示耗时工具进度的唯一实时来源；其它事件保留上一条工具状态，
  // 避免 step_end 先于 tool_call_end 到达时进度条短暂消失。
  const nextActiveToolCall = activeToolCall || response.streaming_state?.active_tool_call || null
  response.streaming_state = {
    run_id: String(event.run_id || ''),
    sequence: Number(event.sequence || 0),
    event_type: event.event_type,
    active_step: typeof payload.step_id === 'string'
      ? payload.step_id
      : typeof payload.step === 'string' ? payload.step : null,
    active_tool_call: nextActiveToolCall
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
  const workContinuations = ref<AgentWorkContinuation[]>([])
  const chatSessions = ref<AgentChatSessionSummary[]>([])
  const restoringConversation = ref(false)
  const continuationRefreshing = ref(false)
  const resumingContinuationIds = reactive(new Set<string>())
  const retryingContinuationIds = reactive(new Set<string>())
  const deliveredResumeRunIds = new Set<string>()
  let continuationPollTimer: ReturnType<typeof setTimeout> | null = null
  let continuationPollingActive = false
  let visibilityListenerAttached = false
  // 每次清空或切换会话都会递增；历史加载返回时版本已变就丢弃，避免旧会话回流覆盖当前页面。
  let conversationVersion = 0

  function getUserId() {
    return userContext.getUserId()
  }

  function setInputMessage(value: string) {
    inputMessage.value = value
  }

  function clearConversation() {
    conversationVersion += 1
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
    workContinuations.value = []
    deliveredResumeRunIds.clear()
  }

  function startNewConversation() {
    // 新对话只重置本页状态：旧会话的流水和已批准的后台任务都继续保留，可从历史对话切回查看。
    clearConversation()
    writeLastSessionId(getUserId(), null)
  }

  async function refreshChatSessions() {
    try {
      chatSessions.value = await listAgentChatSessions()
    } catch (error) {
      ElMessage.error(getErrorMessage(error, '历史对话加载失败'))
    }
  }

  async function loadConversation(sessionId: string) {
    const normalizedSessionId = sessionId.trim()
    if (!normalizedSessionId || loading.value) return false
    clearConversation()
    const version = conversationVersion
    const userId = getUserId()
    restoringConversation.value = true
    try {
      const items = await getAgentChatMessages(normalizedSessionId)
      if (version !== conversationVersion) return false
      if (!items.length) {
        // 他人会话、已不存在或从未产生消息的会话都按空处理；指针一并清掉，下次进入不再尝试。
        if (readLastSessionId(userId) === normalizedSessionId) writeLastSessionId(userId, null)
        return false
      }
      messages.value = toRestoredMessages(items)
      // 按时间顺序重放每个回答，最终的论文候选、选中论文和待处理交互与实时对话时一致。
      for (const item of items) {
        if (item.role !== 'assistant' || !item.response) continue
        rememberResponse(item.response)
        if (item.resume_run_id) deliveredResumeRunIds.add(item.resume_run_id)
      }
      activeSessionId.value = normalizedSessionId
      activeSessionUserId.value = userId
      return true
    } catch (error) {
      if (version === conversationVersion) ElMessage.error(getErrorMessage(error, '对话历史加载失败'))
      return false
    } finally {
      if (version === conversationVersion) restoringConversation.value = false
    }
  }

  // 页面进入时：URL 指定的会话优先，否则回到该用户上次所在的会话；都没有就是新对话。
  async function restoreConversation(preferredSessionId?: string | null) {
    const candidate = preferredSessionId || readLastSessionId(getUserId())
    if (!candidate) return false
    return loadConversation(candidate)
  }

  async function switchConversation(sessionId: string) {
    if (loading.value || sessionId === activeSessionId.value) return
    if (!await loadConversation(sessionId)) return
    await refreshWorkContinuations()
    scheduleContinuationPoll()
  }

  function replaceWorkContinuation(updated: AgentWorkContinuation) {
    const index = workContinuations.value.findIndex(item => item.continuation_id === updated.continuation_id)
    if (index < 0) {
      workContinuations.value = [updated, ...workContinuations.value]
      return
    }
    workContinuations.value[index] = updated
  }

  // 返回 false 表示这次续跑的结果已经展示过（例如历史回放先一步取回），调用方不能再展示一次。
  function markResumeDelivered(resumeRunId?: string | null) {
    const normalizedRunId = String(resumeRunId || '').trim()
    if (!normalizedRunId) return true
    if (deliveredResumeRunIds.has(normalizedRunId)) return false
    deliveredResumeRunIds.add(normalizedRunId)
    return true
  }

  function appendResumeResponse(response: ArxivSearchResponse, resumeRunId?: string | null) {
    if (!markResumeDelivered(resumeRunId)) return

    rememberResponse(response)
    messages.value.push({
      id: createMessageId('assistant'),
      role: 'assistant',
      content: response.answer || '',
      loading: false,
      createdAt: new Date().toISOString(),
      response,
      error: getAgentRuntimeErrorMessage(response)
    })
  }

  async function recoverResumeRun(continuation: AgentWorkContinuation) {
    if (!continuation.resume_run_id || resumingContinuationIds.has(continuation.continuation_id)) return
    const run = await getAgentResumeRun(continuation.resume_run_id, continuation.session_id)
    if (run.status === 'completed' && run.final_response) {
      // completed 结果已经持久化；这里只恢复气泡，绝不能再次调用 continuation resume 接口。
      appendResumeResponse(run.final_response, run.resume_run_id)
      workContinuations.value = workContinuations.value.filter(
        item => item.continuation_id !== continuation.continuation_id
      )
      return
    }
    if (run.status === 'failed' || run.status === 'indeterminate') {
      replaceWorkContinuation({
        ...continuation,
        status: run.status,
        error_code: run.error_code,
        error_message: run.error_message,
        can_cancel: false,
        can_resume: false
      })
    }
  }

  async function resumeWorkContinuation(continuation: AgentWorkContinuation) {
    const continuationId = continuation.continuation_id
    if (!continuation.can_resume || resumingContinuationIds.has(continuationId)) return
    resumingContinuationIds.add(continuationId)
    replaceWorkContinuation({ ...continuation, status: 'resuming', can_resume: false })

    // 续跑要重新检索、调模型，可能持续一两分钟；先放一个运行中的回答气泡，后端推来的节点事件逐条落到它的执行时间线上。
    const assistantMessage = reactive<AgentChatMessage<ArxivSearchResponse>>({
      id: createMessageId('assistant'),
      role: 'assistant',
      content: '',
      loading: true,
      createdAt: new Date().toISOString(),
      response: createDraftResponse(),
      error: null,
      progress: createRunProgress(Date.now())
    })
    messages.value.push(assistantMessage)
    // 续跑期间切换会话或清空对话后，气泡已不在列表里，迟到的事件和结果都不能再回写。
    const isAssistantMessageActive = () => messages.value.includes(assistantMessage)

    let resumeRunId = continuation.resume_run_id
    try {
      const response = await streamAgentWorkContinuationResume(
        continuationId,
        continuation.session_id,
        {
          onEvent: (event, payload) => {
            if (event === 'progress') {
              if (isAssistantMessageActive()) applyStreamEvent(assistantMessage, payload as AgentStreamEvent)
              return
            }
            if (event !== 'run_start') return
            resumeRunId = String(payload.resume_run_id || '') || resumeRunId
            const current = workContinuations.value.find(item => item.continuation_id === continuationId)
            if (current) {
              replaceWorkContinuation({ ...current, status: 'resuming', resume_run_id: resumeRunId || null })
            }
          }
        }
      )
      if (isAssistantMessageActive()) {
        if (markResumeDelivered(resumeRunId)) {
          rememberResponse(response)
          assistantMessage.response = response
          assistantMessage.content = response.answer || ''
          assistantMessage.error = getAgentRuntimeErrorMessage(response)
        } else {
          // 同一结果已经展示过，去掉占位气泡，避免出现两条一样的回答。
          messages.value = messages.value.filter(item => item !== assistantMessage)
        }
      }
      workContinuations.value = workContinuations.value.filter(item => item.continuation_id !== continuationId)
    } catch (error) {
      // resume claim 可能已经消费；失败后只刷新持久状态，禁止自动再次 POST 导致语义不确定。
      const messageText = getErrorMessage(error, 'Agent 恢复失败')
      if (isAssistantMessageActive()) {
        assistantMessage.error = messageText
        assistantMessage.content = messageText
      }
      ElMessage.error(messageText)
    } finally {
      if (isAssistantMessageActive()) {
        assistantMessage.loading = false
        // 断线时收不到结束事件，这里兜底收尾，避免时间线一直显示运行中。
        if (assistantMessage.progress) {
          assistantMessage.progress = finishRunProgress(
            assistantMessage.progress,
            Date.now(),
            assistantMessage.error ? 'failed' : 'done'
          )
        }
      }
      resumingContinuationIds.delete(continuationId)
      await refreshWorkContinuations({ allowAutoResume: false })
    }
  }

  async function cancelWorkContinuation(continuation: AgentWorkContinuation) {
    if (!continuation.can_cancel) return
    try {
      const cancelled = await cancelAgentWorkContinuation(
        continuation.continuation_id,
        continuation.session_id
      )
      replaceWorkContinuation(cancelled)
    } catch (error) {
      ElMessage.error(getErrorMessage(error, '停止等待失败'))
    }
  }

  async function retryWorkContinuation(continuation: AgentWorkContinuation) {
    const arxivId = String(continuation.display_summary.arxiv_id || '').trim()
    if (!arxivId || retryingContinuationIds.has(continuation.continuation_id) || loading.value) return
    retryingContinuationIds.add(continuation.continuation_id)
    const question = String(continuation.display_summary.question_summary || '').trim()
    const retryMessage = question
      ? `请重新为 arXiv:${arxivId} 构建问答索引，索引完成后继续回答：${question}`
      : `请重新为 arXiv:${arxivId} 构建问答索引。`
    try {
      // 明确业务失败不能复用旧授权；重新发送用户请求，让后端生成新的确认、grant 和 continuation。
      await submitMessage(retryMessage, { sessionId: continuation.session_id })
    } finally {
      retryingContinuationIds.delete(continuation.continuation_id)
    }
  }

  async function refreshWorkContinuations(options: { allowAutoResume?: boolean } = {}) {
    if (continuationRefreshing.value) return
    const sessionId = activeSessionId.value
    if (!sessionId) {
      // 页面首次进入代表一段新对话，没有当前 session 时禁止跨会话恢复旧任务。
      workContinuations.value = []
      return
    }
    continuationRefreshing.value = true
    try {
      const items = await listAgentWorkContinuations(sessionId)
      if (activeSessionId.value !== sessionId) {
        // 清空对话期间旧轮询可能刚好返回；会话已切换时必须丢弃该响应，避免旧任务卡回流。
        return
      }
      workContinuations.value = items

      // resuming 或“已完成但尚未取回”的任务只查询同一个 run，页面刷新不会重放原问题。
      await Promise.all(items
        .filter(item => Boolean(item.resume_run_id) && ['resuming', 'resumed'].includes(item.status))
        .map(item => recoverResumeRun(item)))

      if (options.allowAutoResume === false) return
      const currentSessionItems = workContinuations.value.filter(item => item.session_id === sessionId)
      if (currentSessionItems.length !== 1) return
      const [onlyItem] = currentSessionItems
      if (onlyItem.status === 'ready_to_resume' && onlyItem.can_resume) {
        // 只有当前会话恰好一条任务时才自动恢复；多任务必须由用户明确选择。
        await resumeWorkContinuation(onlyItem)
      }
    } catch (error) {
      ElMessage.error(getErrorMessage(error, '后台任务刷新失败'))
    } finally {
      continuationRefreshing.value = false
    }
  }

  function continuationPollDelay() {
    if (typeof document !== 'undefined' && document.hidden) return HIDDEN_CONTINUATION_POLL_INTERVAL_MS
    return CONTINUATION_POLL_INTERVAL_MS
  }

  function hasContinuationsNeedingPolling() {
    return workContinuations.value.some(item =>
      ['submitting', 'waiting_job', 'resuming'].includes(item.status)
    )
  }

  function scheduleContinuationPoll() {
    if (!continuationPollingActive) return
    if (continuationPollTimer) clearTimeout(continuationPollTimer)
    continuationPollTimer = null
    // ready/failed 等稳定状态不再持续 2.5 秒请求；新批准或页面重新可见时会主动刷新并重启。
    if (!hasContinuationsNeedingPolling()) return
    continuationPollTimer = setTimeout(async () => {
      await refreshWorkContinuations()
      scheduleContinuationPoll()
    }, continuationPollDelay())
  }

  async function handleVisibilityChange() {
    if (!continuationPollingActive) return
    if (typeof document !== 'undefined' && !document.hidden) {
      await refreshWorkContinuations()
    }
    scheduleContinuationPoll()
  }

  async function startWorkContinuationPolling() {
    if (continuationPollingActive) return
    continuationPollingActive = true
    if (typeof document !== 'undefined' && !visibilityListenerAttached) {
      document.addEventListener('visibilitychange', handleVisibilityChange)
      visibilityListenerAttached = true
    }
    await refreshWorkContinuations()
    scheduleContinuationPoll()
  }

  function stopWorkContinuationPolling() {
    continuationPollingActive = false
    if (continuationPollTimer) clearTimeout(continuationPollTimer)
    continuationPollTimer = null
    if (typeof document !== 'undefined' && visibilityListenerAttached) {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      visibilityListenerAttached = false
    }
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
    const resolvedPaper = response.resolved_paper
    const resolvedArxivId = resolvedPaper?.arxiv_id
    if (resolvedArxivId) {
      selectedPaper.value = lastSearchPapers.value.find(paper =>
        [paper.arxiv_id, paper.arxivId, paper.id].includes(resolvedArxivId)
      ) || { arxiv_id: resolvedArxivId, title: resolvedPaper?.title || '' }
    }
  }

  function buildRequestContext(): ArxivSearchRequest['context'] {
    const context: {
      frontend_visible_paper?: AgentPaper
      last_papers?: AgentPaper[]
      paper_qa_result?: Record<string, any> | null
      research_profile?: UserResearchProfile | null
      source?: 'chat'
    } = {}
    if (lastSearchPapers.value.length) context.last_papers = lastSearchPapers.value.map(paper => ({ ...paper }))
    if (selectedPaper.value) {
      // 前端只上报当前论文候选；后端负责判断它是否是本轮用户指代的最终目标。
      const arxivId = selectedPaper.value.arxiv_id || selectedPaper.value.arxivId || selectedPaper.value.id
      if (arxivId) {
        context.frontend_visible_paper = {
          ...selectedPaper.value,
          arxiv_id: arxivId
        }
      }
    }
    if (paperQaResult.value) context.paper_qa_result = { ...paperQaResult.value }
    if (paperStore.researchProfile) context.research_profile = { ...paperStore.researchProfile }
    context.source = 'chat'
    return Object.keys(context).length ? context : undefined
  }

  async function submitMessage(rawMessage?: string, options?: SubmitMessageOptions) {
    const message = (rawMessage ?? inputMessage.value).trim()
    if (!message || loading.value) return
    // 同一次请求固定 user_id，避免用户切换恰好发生在 request/session 字段读取之间而造成归属不一致。
    const effectiveUserId = getUserId()

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
    // 流式回调会长期持有 assistantMessage 引用；这里直接创建响应式对象，避免后续事件写到非渲染依赖上。
    const assistantMessage = reactive<AgentChatMessage<ArxivSearchResponse>>({
      id: assistantId,
      role: 'assistant',
      content: options?.resume ? buildConfirmationPlaceholder(currentInteraction.value) : '',
      loading: true,
      createdAt,
      response: createDraftResponse(),
      error: null,
      progress: createRunProgress(Date.now())
    })
    messages.value.push(assistantMessage)
    // 用户切换账号或清空对话后，旧请求可能仍会返回；此时只结束全局 loading，不再回写当前会话状态。
    const isAssistantMessageActive = () => messages.value.includes(assistantMessage)
    inputMessage.value = ''
    loading.value = true

    const request: ArxivSearchRequest = {
      message,
      user_id: effectiveUserId,
      session_id: options?.sessionId ?? (
        activeSessionUserId.value === effectiveUserId ? activeSessionId.value : null
      ),
      resume: options?.resume,
      context: buildRequestContext()
    }

    try {
      let response: ArxivSearchResponse
      try {
        response = await streamAgentChat(request, {
          onEvent: event => {
            if (!isAssistantMessageActive()) return
            applyStreamEvent(assistantMessage, event)
          }
        })
      } catch (streamError) {
        // resume 可能已在服务端原子消费，传输失败时禁止自动重放；普通消息才允许同步 fallback。
        if (request.resume) throw streamError
        response = await runAgentChat(request)
      }
      if (!isAssistantMessageActive()) return
      rememberResponse(response)
      assistantMessage.response = response
      assistantMessage.content = response.answer || ''
      const runtimeErrorMessage = getAgentRuntimeErrorMessage(response)
      if (runtimeErrorMessage) {
        assistantMessage.error = runtimeErrorMessage
        assistantMessage.content = assistantMessage.content || runtimeErrorMessage
        ElMessage.error(runtimeErrorMessage)
      }
    } catch (error) {
      const messageText = getErrorMessage(error, 'Agent 调用失败')
      if (isAssistantMessageActive()) {
        assistantMessage.error = messageText
        assistantMessage.content = messageText
        ElMessage.error(messageText)
      }
    } finally {
      if (isAssistantMessageActive()) {
        assistantMessage.loading = false
        // 同步 fallback 或网络中断时收不到结束事件，这里兜底收尾，避免时间线一直显示运行中。
        if (assistantMessage.progress) {
          assistantMessage.progress = finishRunProgress(
            assistantMessage.progress,
            Date.now(),
            assistantMessage.error ? 'failed' : 'done'
          )
        }
      }
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
      await refreshWorkContinuations()
      scheduleContinuationPoll()
    } finally {
      interactionSubmitting.value = false
    }
  }

  watch(activeSessionId, sessionId => {
    const userId = getUserId()
    if (sessionId && activeSessionUserId.value === userId) writeLastSessionId(userId, sessionId)
  })

  watch(() => getUserId(), () => {
    // 切换用户时当前 API actor 已经变化，只做本地清理；上一用户的会话和指针都原样保留。
    clearConversation()
    chatSessions.value = []
    // continuation 属于用户命名空间；切换用户后必须立即清空旧卡，再从新用户范围重新加载。
    workContinuations.value = []
    deliveredResumeRunIds.clear()
    void refreshWorkContinuations({ allowAutoResume: false })
  })

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
    workContinuations,
    chatSessions,
    restoringConversation,
    continuationRefreshing,
    resumingContinuationIds,
    retryingContinuationIds,
    setInputMessage,
    submitMessage,
    submitResume,
    refreshWorkContinuations,
    startWorkContinuationPolling,
    stopWorkContinuationPolling,
    resumeWorkContinuation,
    cancelWorkContinuation,
    retryWorkContinuation,
    startNewConversation,
    refreshChatSessions,
    restoreConversation,
    switchConversation
  }
}
