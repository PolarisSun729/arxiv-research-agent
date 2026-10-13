import type { AgentRunProgress } from '@/types/agent'

export type AgentChatRole = 'user' | 'assistant' | 'system'

export interface AgentChatMessage<ResponseType = unknown> {
  id: string
  role: AgentChatRole
  content: string
  loading: boolean
  createdAt: string
  response?: ResponseType | null
  error?: string | null
  // 只有 Agent 流式消息会带运行进度；论文问答等其它消息不设置。
  progress?: AgentRunProgress | null
}
