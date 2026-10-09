export type AgentChatRole = 'user' | 'assistant' | 'system'

export interface AgentChatMessage<ResponseType = unknown> {
  id: string
  role: AgentChatRole
  content: string
  loading: boolean
  createdAt: string
  response?: ResponseType | null
  error?: string | null
}
