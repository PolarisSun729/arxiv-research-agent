export interface ArxivSearchRequest {
  user_id?: string | null
  session_id?: string | null
  message: string
  context?: {
    selected_paper?: AgentPaper | null
    last_papers?: AgentPaper[]
    pending_action?: Record<string, any> | null
    paper_qa_result?: Record<string, any> | null
    arxiv_id?: string | null
    source?: 'button' | 'chat' | 'detail_page'
  } | null
}

export type AgentStepStatus = 'running' | 'success' | 'failed' | 'skipped'

export interface ArxivSearchSpec {
  intent: string
  query?: string | null
  title_query?: string | null
  abstract_query?: string | null
  categories: string[]
  submitted_days_ago?: number | null
  max_results: number
  sort_by: string
  sort_order: string
  field_operator: string
  category_operator: string
  reasoning_summary?: string | null
}

export interface AgentToolCall {
  tool_name: string
  arguments: Record<string, any>
  status: string
  summary?: string | null
  trace?: Record<string, any> | null
  error?: Record<string, any> | null
}

export interface AgentStep {
  step: string
  status: AgentStepStatus
  action: string
  inputs: Record<string, any>
  outputs: Record<string, any>
  error?: string | null
}

export interface AgentStreamingState {
  run_id: string
  sequence: number
  event_type: string
  active_step?: string | null
  active_tool_call?: AgentToolCall | null
}

export interface AgentStreamEvent {
  event_type:
    | 'run_start'
    | 'step_start'
    | 'step_end'
    | 'tool_call_start'
    | 'tool_call_end'
    | 'final_response'
    | 'exception'
    | 'stream_end'
  sequence: number
  run_id: string
  timestamp: string
  data: Record<string, any>
}

export interface AgentPaper {
  arxiv_id?: string
  arxivId?: string
  id?: string
  title?: string
  authors?: string[] | string
  abstract?: string
  summary?: string
  published?: string
  publishedAt?: string
  published_date?: string
  updated?: string
  updatedAt?: string
  categories?: string[] | string
  pdf_url?: string
  pdfUrl?: string
  abs_url?: string
  absUrl?: string
  url?: string
  query_match_score?: number
  personalization_score?: number
  final_score?: number
  score_breakdown?: Record<string, any> | null
  matched_terms?: string[]
  personalized_reason?: string | null
  match_reason?: string | null
  priority?: number
  label?: 'liked' | 'disliked' | null
}

export interface AgentPreferenceActionResult {
  status: 'success' | 'failed'
  action: 'like' | 'dislike' | 'remove'
  label: 'liked' | 'disliked' | 'none'
  arxiv_id?: string | null
  title?: string | null
  message: string
  paper?: AgentPaper | null
  error?: string | null
}

export interface ArxivSearchResponse {
  intent: string
  answer: string
  search_spec?: ArxivSearchSpec | null
  pending_action?: Record<string, any> | null
  paper_qa_result?: Record<string, any> | null
  preference_action_result?: AgentPreferenceActionResult | null
  plan: string[]
  tool_calls: AgentToolCall[]
  papers: AgentPaper[]
  warnings: string[]
  next_actions: string[]
  steps: AgentStep[]
  streaming_state?: AgentStreamingState | null
}
