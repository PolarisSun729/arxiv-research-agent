export type PaperLabel = 'liked' | 'disliked'

export type PaperActionType =
  | 'like'
  | 'dislike'
  | 'favorite'
  | 'read'
  | 'later'
  | 'archived'
  | 'note_saved'
  | 'not_interested'

export type UserPaperActionMap = Partial<Record<PaperActionType, string[]>>

export type PaperNoteType =
  | 'summary'
  | 'method'
  | 'experiment'
  | 'result'
  | 'limitation'
  | 'idea'
  | 'todo'
  | 'custom'

export interface UserResearchProfile {
  user_id: string
  positive_topics: string[]
  negative_topics: string[]
  recent_topics: string[]
  preferred_categories: string[]
  preferred_answer_style: string
  common_question_types: string[]
  representative_papers: string[]
  created_at?: string | null
  updated_at?: string | null
}

export interface UserPaperAction {
  id?: number
  user_id: string
  arxiv_id: string
  action_type: PaperActionType
  metadata?: Record<string, any>
  created_at?: string
  updated_at?: string
}

export interface PaperNote {
  note_id: string
  user_id: string
  arxiv_id: string
  session_id?: string | null
  source_message_id?: string | null
  source_turn_id?: string | null
  title: string
  content: string
  note_type: PaperNoteType
  source_chunk_ids: string[]
  tags: string[]
  include_in_profile: boolean
  created_at?: string | null
  updated_at?: string | null
  sources?: Array<{
    content?: string
    page_number?: string
    source?: string
    section_path?: string
    parent_chunk_id?: string | number
    chunk_type?: string
    asset_summary?: string
    asset_preview_text?: string
  }>
}

export interface Paper {
  id: string
  arxivId: string
  title: string
  authors: string[]
  summary: string
  publishedAt: string
  updatedAt?: string
  categories: string[]
  pdfUrl: string
  absUrl: string
  label?: PaperLabel | null
  paperActions?: Partial<Record<PaperActionType, boolean>>
  query_match_score?: number
  personalization_score?: number
  final_score?: number
  score_breakdown?: Record<string, any> | null
  matched_terms?: string[]
  personalized_reason?: string | null
  match_reason?: string | null
  priority?: number
}

export interface RecommendationScoreBreakdown {
  semantic_score: number
  category_score: number
  recency_score: number
  diversity_score: number
  relevance_score?: number
  selection_score?: number
  semantic_diversity_score?: number | null
  cluster_diversity_score?: number | null
  category_diversity_score?: number | null
  disliked_penalty?: number
}

export interface RecommendedPaper extends Paper {
  similarityScore: number
  finalScore?: number
  reason?: string
  scoreBreakdown?: RecommendationScoreBreakdown
  recall_source?: string
  recall_cluster_id?: string | null
  recall_cluster_similarity?: number | null
  recall_cluster_rank?: number | null
  recall_cluster_hits?: Array<{
    cluster_id?: string | null
    similarity?: number
    rank?: number | null
  }>
  best_matched_cluster_id?: string | null
  best_matched_cluster_similarity?: number | null
  cluster_similarities?: Array<{
    cluster_id?: string | null
    similarity?: number
  }>
  disliked_penalty?: number
  diversityDebug?: {
    diversity_reason?: string
    diversity_penalty_source?: string | null
    diversity_penalty_value?: number | null
    semantic_diversity_score?: number | null
    cluster_diversity_score?: number | null
    category_diversity_score?: number | null
  }
}

export interface LabeledPaper extends Paper {
  label: PaperLabel
  labeledAt: string
}

export interface PaperMaterializationPayload {
  arxiv_id: string
  title: string
  authors: string[]
  abstract: string
  categories: string[]
  published_date: string
  url: string
  abs_url?: string
  pdf_url?: string
  publishedAt?: string
}

export interface PaperPreferenceRequest {
  user_id?: string
  arxiv_id: string
  paper: PaperMaterializationPayload
}

export interface SearchParams {
  keyword?: string
  category?: string
  page: number
  pageSize: number
  sortBy?: string
}

export interface ArxivSearchParams {
  search_query?: string
  id_list?: string[]
  title?: string
  author?: string
  abstract?: string
  category?: string
  comment?: string
  journal_ref?: string
  report_number?: string
  operator?: 'AND' | 'OR'
  max_results?: number
  start?: number
  sort_by?: string
  sort_order?: 'ascending' | 'descending'
  submitted_days_ago?: number
}

export interface LabelParams {
  label: PaperLabel
}

export interface PaginatedResponse<T> {
  total: number
  items: T[]
}

export interface InterestClusterSummary {
  cluster_id: string
  centroid_vector: number[]
  paper_count: number
  paper_ids: string[]
}

export interface InterestVector {
  user_id: string
  vector_data: number[]
  paper_count: number
  embedding_model: string
  vector_dimension: number
  cluster_count?: number
  profile_mode?: string
  interest_clusters?: InterestClusterSummary[]
  disliked_vector_data?: number[] | null
  created_at: string
  updated_at: string
}

export interface InterestVectorResult {
  status: string
  message: string
  paper_count: number
  used_count?: number
  milvus_used_count?: number
  fallback_used_count?: number
  unresolved_count?: number
  liked_count?: number
  disliked_count?: number
  liked_milvus_count?: number
  disliked_milvus_count?: number
  liked_fallback_count?: number
  disliked_fallback_count?: number
  liked_unresolved_count?: number
  disliked_unresolved_count?: number
  vector_dimension: number
  embedding_model: string
  cluster_count?: number
  profile_mode?: string
  cluster_summary?: Array<{
    cluster_id?: string
    paper_count?: number
    paper_ids?: string[]
  }>
}

export interface RecommendationResult {
  status: string
  message: string
  total_found: number
  interest_profile_mode?: string
  interest_cluster_count?: number
  recall_mode?: string
  research_profile?: UserResearchProfile | null
  paper_actions?: UserPaperActionMap | null
  recommendations: RecommendedPaper[]
}
