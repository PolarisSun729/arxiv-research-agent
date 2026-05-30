export interface ArxivSearchRequest {
  user_id?: string | null
  session_id?: string | null
  message: string
}

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
}

export interface ArxivSearchResponse {
  intent: string
  answer: string
  search_spec?: ArxivSearchSpec | null
  plan: string[]
  tool_calls: AgentToolCall[]
  papers: AgentPaper[]
  warnings: string[]
  next_actions: string[]
}
