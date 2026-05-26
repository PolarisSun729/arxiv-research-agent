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
  label?: 'liked' | 'disliked' | null
}

export interface RecommendationScoreBreakdown {
  semantic_score: number
  category_score: number
  recency_score: number
  diversity_score: number
}

export interface RecommendedPaper extends Paper {
  similarityScore: number
  finalScore?: number
  reason?: string
  scoreBreakdown?: RecommendationScoreBreakdown
}

export interface LabeledPaper extends Paper {
  label: 'liked' | 'disliked'
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
  label: 'liked' | 'disliked'
}

export interface PaginatedResponse<T> {
  total: number
  items: T[]
}
