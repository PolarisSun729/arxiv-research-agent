import type { RetrievalDebug } from '@/api/papers'
import type { AgentChatMessage } from '@/types/agentChat'

export type RagChatRole = 'user' | 'assistant'

export interface RagChatSource {
  content: string
  page_number: string
  source?: string
}

// This mirrors the current QA turn shape so existing page state can be
// converted into future chat-component messages without changing behavior now.
export interface QaTurnForRagChat {
  id: string
  question: string
  answer: string
  sources: RagChatSource[]
  retrievalDebug?: RetrievalDebug | null
  createdAt: string
  streaming?: boolean
}

export interface RagChatMessage extends AgentChatMessage<QaTurnForRagChat> {
  turnId: string
  sources: RagChatSource[]
  retrievalDebug: RetrievalDebug | null
}

// Keep the transformation centralized so the future UI swap can reuse the same
// message mapping instead of rebuilding it inside view components.
export function qaTurnToRagMessages(turn: QaTurnForRagChat): RagChatMessage[] {
  const userMessage: RagChatMessage = {
    id: `${turn.id}-user`,
    turnId: turn.id,
    role: 'user',
    content: turn.question,
    loading: false,
    createdAt: turn.createdAt,
    response: null,
    error: null,
    sources: [],
    retrievalDebug: null
  }

  const assistantMessage: RagChatMessage = {
    id: `${turn.id}-assistant`,
    turnId: turn.id,
    role: 'assistant',
    content: turn.answer,
    loading: Boolean(turn.streaming),
    createdAt: turn.createdAt,
    response: turn,
    error: null,
    sources: turn.sources,
    retrievalDebug: turn.retrievalDebug ?? null
  }

  return [userMessage, assistantMessage]
}
