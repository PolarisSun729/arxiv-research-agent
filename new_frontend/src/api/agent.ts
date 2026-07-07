import request from './request'
import { ApiError, normalizeApiError, parseFetchErrorResponse } from './errors'
import type {
  AgentGraphResponse,
  AgentPendingAction,
  AgentQaIndexContinuation,
  AgentQaIndexJob,
  AgentStreamEvent,
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

export interface CreateAgentQaIndexContinuationPayload {
  user_id?: string | null
  session_id?: string | null
  arxiv_id?: string | null
  loading_method?: string
  original_question?: string | null
  pending_action?: AgentPendingAction | Record<string, any> | null
  resume_payload?: ArxivSearchRequest['resume'] | Record<string, any> | null
}

export interface AgentQaIndexContinuationResponse {
  status?: string
  job: AgentQaIndexJob
  continuation: AgentQaIndexContinuation
}

export async function createAgentQaIndexContinuation(
  payload: CreateAgentQaIndexContinuationPayload
): Promise<AgentQaIndexContinuationResponse> {
  return request.post('/agent/qa-index-continuations', {
    ...payload,
    user_id: payload.user_id || getCurrentUserId()
  })
}

export async function listActiveAgentQaIndexContinuations(params?: {
  user_id?: string | null
  session_id?: string | null
  limit?: number
}): Promise<{ continuations: AgentQaIndexContinuation[] }> {
  return request.get('/agent/qa-index-continuations/active', {
    params: {
      ...(params || {}),
      user_id: params?.user_id || getCurrentUserId()
    }
  })
}

export async function updateAgentQaIndexContinuationStatus(
  jobId: string,
  payload: { user_id?: string | null; status: string; error_message?: string | null }
): Promise<{ continuation: AgentQaIndexContinuation }> {
  return request.post(`/agent/qa-index-continuations/${jobId}/status`, {
    ...payload,
    user_id: payload.user_id || getCurrentUserId()
  })
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
