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

export interface RecommendedPaper extends Paper {
  similarityScore: number
  reason?: string
}

export interface LabeledPaper extends Paper {
  label: 'liked' | 'disliked'
  labeledAt: string
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
