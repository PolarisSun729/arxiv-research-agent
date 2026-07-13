import type { BackendArxivQueryCapability } from '@/types/arxivCapability'
import type { UserResearchProfile } from '@/types/paper'

export interface ArxivSearchRequest {
  user_id?: string | null
  session_id?: string | null
  message: string
  resume?: {
    interaction_id: string
    decision: 'approve' | 'reject' | 'select' | 'cancel'
    response?: Record<string, any>
    note?: string | null
  } | null
  context?: {
    selected_paper?: AgentPaper | null
    last_papers?: AgentPaper[]
    paper_qa_result?: Record<string, any> | null
    research_profile?: UserResearchProfile | null
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
  paper_id?: string
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

export interface PaperTargetCandidate extends AgentPaper {
  candidate_id?: string
  rank?: number
  source?: string
  source_type?: string
  source_key?: string
  source_label?: string
  list_name?: string
  authors_summary?: string
}

export interface TargetSelectionPayload {
  candidates: PaperTargetCandidate[]
  recommended_candidate_id?: string | null
  reference_hint: Record<string, any>
}

export interface SideEffectApprovalPayload {
  tool_name: string
  action_type: string
  reason: string
  arguments_summary: Record<string, any>
  arguments_fingerprint: string
}

export interface AgentInteraction {
  interaction_id: string
  kind: 'target_selection' | 'side_effect_approval'
  status: 'pending' | 'resolved' | 'cancelled' | 'expired'
  plan_id: string
  step_id: string
  payload: TargetSelectionPayload | SideEffectApprovalPayload
  created_at: string
  expires_at: string
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
  session_id?: string | null
  intent: string
  answer: string
  query_capability?: BackendArxivQueryCapability | null
  search_spec?: ArxivSearchSpec | null
  interaction?: AgentInteraction | null
  paper_qa_result?: Record<string, any> | null
  preference_action_result?: AgentPreferenceActionResult | null
  plan: string[]
  tool_calls: AgentToolCall[]
  papers: AgentPaper[]
  warnings: string[]
  next_actions: string[]
  steps: AgentStep[]
  debug?: Record<string, any>
  streaming_state?: AgentStreamingState | null
}

export interface AgentGraphResponse {
  graph_name: string
  render_source: string
  node_names: string[]
  mermaid: string
  supports_png: boolean
}

export interface AgentIndexJobSnapshot {
  status: string | null
  current_stage: string | null
  stage_label: string | null
  progress: number | null
  attempt_no: number | null
  max_attempts: number | null
  error_code: string | null
  error_message: string | null
}

export interface AgentWorkContinuation {
  continuation_id: string
  session_id: string
  status: 'submitting' | 'waiting_job' | 'ready_to_resume' | 'resuming' | 'resumed' | 'failed' | 'cancelled' | 'expired' | 'indeterminate'
  display_summary: {
    arxiv_id?: string | null
    paper_title?: string | null
    question_summary?: string | null
  }
  job_id: string | null
  job: AgentIndexJobSnapshot | null
  error_code: string | null
  error_message: string | null
  can_cancel: boolean
  can_resume: boolean
  ready_at: string | null
  expires_at: string | null
  resume_run_id: string | null
}

export interface AgentResumeRun {
  resume_run_id: string
  continuation_id: string
  user_id: string
  session_id: string
  thread_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'indeterminate'
  started_at: string | null
  finished_at: string | null
  final_response: ArxivSearchResponse | null
  error_code: string | null
  error_message: string | null
  result_retrieved_at: string | null
}
