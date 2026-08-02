import request from './request'
import { ApiError, normalizeApiError, parseFetchErrorResponse } from './errors'
import type {
  AgentGraphResponse,
  AgentResumeRun,
  AgentStreamEvent,
  AgentWorkContinuation,
  ArxivSearchRequest,
  ArxivSearchResponse
} from '@/types/agent'
import { getCurrentUserId } from '@/composables/useUserContext'

export interface AgentStreamHandlers {
  onEvent?: (event: AgentStreamEvent) => void
  onDone?: (response: ArxivSearchResponse) => void
  onError?: (error: Error) => void
  signal?: AbortSignal
}

export interface AgentResumeStreamHandlers {
  onEvent?: (event: string, payload: Record<string, any>) => void
  signal?: AbortSignal
}

function parseSseEvent(payload: string) {
  const lines = payload.split(/\r?\n/)
  const eventLine = lines.find(line => line.startsWith('event:'))
  const dataLines = lines.filter(line => line.startsWith('data:'))

  if (!eventLine || !dataLines.length) return null

  const event = eventLine.slice('event:'.length).trim()
  const dataText = dataLines
    .map(line => line.slice('data:'.length).trim())
    .join('\n')

  try {
    return { event, data: JSON.parse(dataText) as AgentStreamEvent }
  } catch {
    return { event, data: null as AgentStreamEvent | null, raw: dataText }
  }
}

function ensureErrorMessage(error: unknown, fallback: string) {
  if (error instanceof Error && error.message) return error
  if (error && typeof error === 'object' && 'message' in error && typeof error.message === 'string' && error.message) {
    return new Error(error.message)
  }
  return new Error(fallback)
}

function withResolvedUserId(payload: ArxivSearchRequest): ArxivSearchRequest {
  const normalizedUserId = String(payload.user_id || '').trim()
  return {
    ...payload,
    // Agent API 保留兜底入口，但默认值只来自统一用户上下文，避免 resume 与 chat 用户不一致。
    user_id: normalizedUserId || getCurrentUserId()
  }
}

export async function runAgentChat(payload: ArxivSearchRequest): Promise<ArxivSearchResponse> {
  return request.post('/agent/chat', withResolvedUserId(payload))
}

export async function fetchAgentGraph(): Promise<AgentGraphResponse> {
  return request.get('/agent/graph')
}

export async function streamAgentChat(
  payload: ArxivSearchRequest,
  handlers: AgentStreamHandlers = {}
): Promise<ArxivSearchResponse> {
  const requestPayload = withResolvedUserId(payload)
  const response = await fetch('/api/agent/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream'
    },
    body: JSON.stringify(requestPayload),
    signal: handlers.signal
  })

  if (!response.ok) {
    throw await parseFetchErrorResponse(response, 'Agent 调用失败')
  }

  if (!response.body) {
    throw new Error('Streaming response body is empty')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let finalResponse: ArxivSearchResponse | null = null

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    const parts = buffer.split(/\r?\n\r?\n/)
    buffer = parts.pop() || ''

    for (const part of parts) {
      const parsed = parseSseEvent(part)
      if (!parsed || !parsed.data) continue

      handlers.onEvent?.(parsed.data)

      if (parsed.data.event_type === 'final_response') {
        const responsePayload = parsed.data.data?.response
        if (responsePayload) {
          finalResponse = responsePayload as ArxivSearchResponse
        }
      } else if (parsed.data.event_type === 'exception') {
        const responsePayload = parsed.data.data?.response
        if (responsePayload) {
          finalResponse = responsePayload as ArxivSearchResponse
          // 后端异常分支会继续发送 final_response / stream_end；这里先保留错误响应，
          // 不能提前 throw，否则调用方会丢失可展示的 Agent 错误气泡并触发一次多余的同步重试。
          continue
        }
        if (parsed.data.data?.code || parsed.data.data?.error?.code) {
          // Agent stream 的异常事件也遵守统一错误契约，调用方可以直接读取 payload.code。
          throw new ApiError(normalizeApiError(parsed.data.data?.error || parsed.data.data, 'Agent 调用失败'))
        }
      } else if (parsed.data.event_type === 'stream_end') {
        const responsePayload = parsed.data.data?.response
        if (responsePayload) {
          finalResponse = responsePayload as ArxivSearchResponse
        }
        if (!finalResponse) {
          if (parsed.data.data?.status === 'error' || parsed.data.data?.code) {
            // stream_end 可能只携带错误状态而不带完整响应；此时仍要走统一错误契约，避免 UI 永远停在 loading。
            throw new ApiError(normalizeApiError(parsed.data.data, 'Agent 调用失败'))
          }
          throw new Error('Stream ended without a final response')
        }
        handlers.onDone?.(finalResponse)
        return finalResponse
      }
    }
  }

  if (finalResponse) {
    handlers.onDone?.(finalResponse)
    return finalResponse
  }

  throw ensureErrorMessage(new Error('Streaming response ended unexpectedly'), 'Streaming response ended unexpectedly')
}

export async function listAgentWorkContinuations(sessionId?: string | null): Promise<AgentWorkContinuation[]> {
  const response = await request.get('/agent/work-continuations/active', {
    params: {
      user_id: getCurrentUserId(),
      ...(sessionId ? { session_id: sessionId } : {})
    }
  }) as unknown as { items?: AgentWorkContinuation[] }
  return Array.isArray(response?.items) ? response.items : []
}

export async function clearAgentSession(sessionId: string): Promise<void> {
  await request.post(`/agent/sessions/${encodeURIComponent(sessionId)}/clear`, null, {
    params: { user_id: getCurrentUserId() }
  })
}

export async function getAgentWorkContinuation(
  continuationId: string,
  sessionId: string
): Promise<AgentWorkContinuation> {
  return request.get(`/agent/work-continuations/${encodeURIComponent(continuationId)}`, {
    params: { user_id: getCurrentUserId(), session_id: sessionId }
  }) as unknown as Promise<AgentWorkContinuation>
}

export async function cancelAgentWorkContinuation(
  continuationId: string,
  sessionId: string
): Promise<AgentWorkContinuation> {
  return request.post(`/agent/work-continuations/${encodeURIComponent(continuationId)}/cancel`, null, {
    params: { user_id: getCurrentUserId(), session_id: sessionId }
  }) as unknown as Promise<AgentWorkContinuation>
}

export async function getAgentResumeRun(resumeRunId: string, sessionId?: string | null): Promise<AgentResumeRun> {
  return request.get(`/agent/resume-runs/${encodeURIComponent(resumeRunId)}`, {
    params: {
      user_id: getCurrentUserId(),
      ...(sessionId ? { session_id: sessionId } : {})
    }
  }) as unknown as Promise<AgentResumeRun>
}

export async function streamAgentWorkContinuationResume(
  continuationId: string,
  sessionId: string,
  handlers: AgentResumeStreamHandlers = {}
): Promise<ArxivSearchResponse> {
  const params = new URLSearchParams({ user_id: getCurrentUserId(), session_id: sessionId })
  const response = await fetch(
    `/api/agent/work-continuations/${encodeURIComponent(continuationId)}/resume/stream?${params.toString()}`,
    { method: 'POST', headers: { Accept: 'text/event-stream' }, signal: handlers.signal }
  )
  if (!response.ok) throw await parseFetchErrorResponse(response, 'Agent 恢复失败')
  if (!response.body) throw new Error('Agent resume stream body is empty')

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let finalResponse: ArxivSearchResponse | null = null
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split(/\r?\n\r?\n/)
    buffer = parts.pop() || ''
    for (const part of parts) {
      const lines = part.split(/\r?\n/)
      const event = lines.find(line => line.startsWith('event:'))?.slice('event:'.length).trim()
      const rawData = lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n')
      if (!event || !rawData) continue
      const payload = JSON.parse(rawData) as Record<string, any>
      handlers.onEvent?.(event, payload)
      if (event === 'final_response' && payload.response) finalResponse = payload.response as ArxivSearchResponse
      if (event === 'exception') throw new Error(String(payload.message || payload.error_code || 'Agent 恢复失败'))
      if (event === 'stream_end' && finalResponse) return finalResponse
    }
  }
  if (finalResponse) return finalResponse
  throw new Error('Agent resume stream ended without a persisted final response')
}
