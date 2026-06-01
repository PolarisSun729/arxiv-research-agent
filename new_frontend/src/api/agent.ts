import request from './request'
import type { AgentStreamEvent, ArxivSearchRequest, ArxivSearchResponse } from '@/types/agent'

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

export async function runArxivSearchAgent(payload: ArxivSearchRequest): Promise<ArxivSearchResponse> {
  return request.post('/agent/arxiv-search', payload)
}

export async function streamArxivSearchAgent(
  payload: ArxivSearchRequest,
  handlers: AgentStreamHandlers = {}
): Promise<ArxivSearchResponse> {
  const response = await fetch('/api/agent/arxiv-search/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream'
    },
    body: JSON.stringify(payload),
    signal: handlers.signal
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed with status ${response.status}`)
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
