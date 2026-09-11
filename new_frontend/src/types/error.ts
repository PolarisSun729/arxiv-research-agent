export type ApiErrorCode =
  | 'request_validation_error'
  | 'paper_not_found'
  | 'qa_index_not_found'
  | 'qa_index_build_failed'
  | 'vector_store_error'
  | 'llm_generation_failed'
  | 'aborted'
  | 'stream_incomplete'
  | 'stream_interrupted'
  | 'resume_checkpoint_not_found'
  | 'agent_runtime_error'
  | 'database_write_failed'
  | 'local_arxiv_search_error'
  | 'unsupported_local_arxiv_query'
  | 'local_search_index_unavailable'
  | 'unknown_error'

export interface ApiErrorPayload {
  status: 'failed'
  code: ApiErrorCode | string
  message: string
  detail?: string | null
  details?: Record<string, any> | null
  recoverable: boolean
  retry_after?: number | null
}
