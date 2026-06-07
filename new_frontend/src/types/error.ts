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
  | 'unknown_error'

export interface ApiErrorPayload {
  status: 'failed'
  code: ApiErrorCode | string
  message: string
  detail?: string | null
  recoverable: boolean
}
