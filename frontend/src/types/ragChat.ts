import type { QaMode, RetrievalDebug } from '@/api/papers'
import type { AgentChatMessage } from '@/types/agentChat'

export type QaTurnStatus =
  | 'idle'
  | 'preparing'
  | 'streaming'
  | 'completed'
  | 'interrupted'
  | 'aborted'
  | 'partial'
  | 'failed'
  | 'persistence_failed'

export type QaPersistenceStatus = 'unknown' | 'saved' | 'failed' | 'not_saved'

export interface RagChatSource {
  source_id: string
  content: string
  page_number: string
  source?: string
  section_title?: string
  section_path?: string
  parent_chunk_id?: string | number
  chunk_type?: string
  asset_summary?: string
  asset_preview_text?: string
  asset_url?: string
  is_cited?: boolean
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
  status?: QaTurnStatus
  error?: string | null
  partial?: boolean
  completedAt?: string | null
  interruptedReason?: string | null
  persistenceStatus?: QaPersistenceStatus
  originalQuestion?: string
  contextualizedQuestion?: string
  usedShortTermMemory?: boolean
  questionContextualization?: Record<string, any> | null
  citedSourceIds?: string[]
  citationWarning?: string | null
  // 这一轮走的问答模式；历史记录按 retrieval_debug.engine 还原。
  mode?: QaMode
  // 生成前的等待说明（如快速模式正在读取全文），缺省时显示通用的“正在生成回复”。
  progressText?: string | null
}

export interface RagChatMessage extends AgentChatMessage<QaTurnForRagChat> {
  turnId: string
  sources: RagChatSource[]
  retrievalDebug: RetrievalDebug | null
  citedSourceIds: string[]
  citationWarning: string | null
  loadingText?: string
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
    retrievalDebug: null,
    citedSourceIds: [],
    citationWarning: null
  }

  const assistantMessage: RagChatMessage = {
    id: `${turn.id}-assistant`,
    turnId: turn.id,
    role: 'assistant',
    content: turn.answer,
    loading: Boolean(turn.streaming),
    createdAt: turn.createdAt,
    response: turn,
    error: turn.error ?? null,
    sources: turn.sources,
    retrievalDebug: turn.retrievalDebug ?? null,
    citedSourceIds: turn.citedSourceIds || [],
    citationWarning: turn.citationWarning || null,
    loadingText: turn.progressText || undefined
  }

  return [userMessage, assistantMessage]
}
