import request from './request'
import type {
  Paper,
  RecommendedPaper,
  LabeledPaper,
  SearchParams,
  LabelParams,
  PaginatedResponse,
  ArxivSearchParams,
  PaperMaterializationPayload,
  PaperPreferenceRequest,
  InterestVector,
  InterestVectorResult,
  RecommendationResult
} from '@/types/paper'
import { mockPapers, mockRecommendedPapers, mockLabeledPapers, mockStats } from '@/mock/papers'

const isMockMode = false

function getPaperArxivId(paper: Pick<Paper, 'id' | 'arxivId'>) {
  return paper.arxivId || paper.id.split('/').pop() || paper.id
}

function normalizePaper(raw: any): Paper {
  const arxivId = raw.arxiv_id || raw.arxivId || (raw.id ? String(raw.id).split('/').pop() : '')
  return {
    id: arxivId || raw.id || '',
    arxivId: arxivId || raw.id || '',
    title: raw.title || '',
    authors: Array.isArray(raw.authors)
      ? raw.authors
      : String(raw.authors || '').split(',').map(author => author.trim()).filter(Boolean),
    summary: raw.summary || raw.abstract || '',
    publishedAt: raw.published || raw.publishedAt || raw.published_date || '',
    updatedAt: raw.updated || raw.updatedAt || '',
    categories: Array.isArray(raw.categories)
      ? raw.categories
      : String(raw.categories || '').split(',').map(category => category.trim()).filter(Boolean),
    pdfUrl: raw.pdf_url || raw.pdfUrl || '',
    absUrl: raw.abs_url || raw.absUrl || raw.url || ''
  }
}

export async function getUserPreferences(): Promise<{
  liked_papers: string[]
  disliked_papers: string[]
}> {
  try {
    return await request.get('/user/preferences/local_user')
  } catch (error: any) {
    if (error?.response?.status === 404) {
      return { liked_papers: [], disliked_papers: [] }
    }
    throw error
  }
}

function buildPaperMaterializationPayload(paper: Paper): PaperMaterializationPayload {
  const arxivId = getPaperArxivId(paper)
  return {
    arxiv_id: arxivId,
    title: paper.title,
    authors: paper.authors,
    abstract: paper.summary,
    categories: paper.categories,
    published_date: paper.publishedAt,
    url: paper.absUrl || paper.pdfUrl,
    abs_url: paper.absUrl,
    pdf_url: paper.pdfUrl,
    publishedAt: paper.publishedAt
  }
}

export async function likePaper(paper: Paper): Promise<void> {
  const arxivId = getPaperArxivId(paper)
  const payload: PaperPreferenceRequest = {
    arxiv_id: arxivId,
    paper: buildPaperMaterializationPayload(paper)
  }
  await request.post('/user/like-paper', payload)
}

export async function dislikePaper(paper: Paper): Promise<void> {
  const arxivId = getPaperArxivId(paper)
  const payload: PaperPreferenceRequest = {
    arxiv_id: arxivId,
    paper: buildPaperMaterializationPayload(paper)
  }
  await request.post('/user/dislike-paper', payload)
}

export async function removePaperPreference(paper: Paper, label: LabelParams['label']): Promise<void> {
  const endpoint = label === 'liked' ? '/user/like-paper' : '/user/dislike-paper'
  await request.delete(endpoint, {
    data: { arxiv_id: getPaperArxivId(paper) }
  })
}

export async function searchPapers(params: SearchParams): Promise<PaginatedResponse<Paper>> {
  if (isMockMode) {
    let filtered = [...mockPapers]
    
    if (params.keyword) {
      const keyword = params.keyword.toLowerCase()
      filtered = filtered.filter(p => 
        p.title.toLowerCase().includes(keyword) ||
        p.authors.some(a => a.toLowerCase().includes(keyword)) ||
        p.summary.toLowerCase().includes(keyword)
      )
    }
    
    if (params.category) {
      const category = params.category
      filtered = filtered.filter(p => p.categories.includes(category))
    }
    
    if (params.sortBy === 'newest') {
      filtered.sort((a, b) => new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime())
    } else if (params.sortBy === 'oldest') {
      filtered.sort((a, b) => new Date(a.publishedAt).getTime() - new Date(b.publishedAt).getTime())
    }
    
    const start = (params.page - 1) * params.pageSize
    const end = start + params.pageSize
    
    return {
      total: filtered.length,
      items: filtered.slice(start, end)
    }
  }
  
  return request.get('/papers/search', { params })
}

export async function getPaperById(id: string): Promise<Paper> {
  if (isMockMode) {
    const paper = mockPapers.find(p => p.id === id)
    if (!paper) {
      throw new Error('Paper not found')
    }
    return paper
  }
  
  return normalizePaper(await request.get(`/paper/${id}`))
}

export async function getRecommendations(params: { page: number; pageSize: number }): Promise<PaginatedResponse<RecommendedPaper>> {
  if (isMockMode) {
    const start = (params.page - 1) * params.pageSize
    const end = start + params.pageSize
    return {
      total: mockRecommendedPapers.length,
      items: mockRecommendedPapers.slice(start, end)
    }
  }
  
  return request.get('/papers/recommendations', { params })
}

export async function labelPaper(id: string, data: LabelParams): Promise<void> {
  if (isMockMode) {
    const paper = mockPapers.find(p => p.id === id)
    if (paper) {
      paper.label = data.label
    }
    return
  }
  
  const paper = mockPapers.find(p => p.id === id)
  if (!paper) {
    throw new Error('Paper not found')
  }
  if (data.label === 'liked') {
    return likePaper(paper)
  }
  return dislikePaper(paper)
}

export async function getLabeledPapers(params: { 
  label?: 'liked' | 'disliked'
  page: number
  pageSize: number
}): Promise<PaginatedResponse<LabeledPaper>> {
  if (isMockMode) {
    let filtered = [...mockLabeledPapers]
    
    if (params.label) {
      filtered = filtered.filter(p => p.label === params.label)
    }
    
    const start = (params.page - 1) * params.pageSize
    const end = start + params.pageSize
    
    return {
      total: filtered.length,
      items: filtered.slice(start, end)
    }
  }
  
  const preferences = await getUserPreferences()
  const liked = preferences.liked_papers || []
  const disliked = preferences.disliked_papers || []
  const targetIds = params.label === 'liked'
    ? liked
    : params.label === 'disliked'
      ? disliked
      : [...liked, ...disliked]

  const papers = await Promise.all(
    targetIds.map(async id => {
      const paper = normalizePaper(await request.get(`/paper/${id}`))
      return {
        ...paper,
        label: liked.includes(id) ? 'liked' as const : 'disliked' as const,
        labeledAt: new Date().toISOString()
      }
    })
  )

  const start = (params.page - 1) * params.pageSize
  const end = start + params.pageSize
  return {
    total: papers.length,
    items: papers.slice(start, end)
  }
}

export async function getStats(): Promise<typeof mockStats> {
  if (isMockMode) {
    return mockStats
  }
  
  return request.get('/stats')
}

export async function searchArxiv(params: ArxivSearchParams): Promise<PaginatedResponse<Paper>> {
  const defaultParams: ArxivSearchParams = {
    submitted_days_ago: 30,
    ...params
  }
  const response: any = await request.post('/arxiv/search', defaultParams)
  const papers = response.papers || response.items || []
  const items = papers.map(normalizePaper)
  const maxResults = defaultParams.max_results || 10
  const actualTotal = Math.min(response.total_results || items.length, maxResults)
  return {
    total: actualTotal,
    items
  }
}

export async function generateInterestVector(): Promise<InterestVectorResult> {
  return request.post('/user/generate-interest-vector')
}

export async function getInterestVector(): Promise<InterestVector> {
  return request.get('/user/interest-vector')
}

export async function recommendPapers(topN: number = 10, maxAgeMonths: number = 6): Promise<RecommendationResult> {
  return request.post('/user/recommend-papers', { top_n: topN, max_age_months: maxAgeMonths })
}

export interface QaStatusResult {
  arxiv_id: string
  has_index: boolean
  status: string
  collection_name?: string
  chunk_count?: number
  embedding_model?: string
}

export async function getPaperQaStatus(arxivId: string): Promise<QaStatusResult> {
  return request.get(`/paper/${arxivId}/qa-status`)
}

export interface QaDiagnosticCollectionInfo {
  name?: string
  exists_in_milvus?: boolean
  info?: {
    name?: string
    num_entities?: number
    schema?: Record<string, any>
  } | null
  error?: string | null
}

export interface QaDiagnosticResult {
  arxiv_id: string
  qa_index: QaStatusResult | null
  milvus: {
    provider: string
    collections: string[]
  }
  collection: QaDiagnosticCollectionInfo | null
  sample_chunks: Array<{
    id?: number
    content?: string
    page_number?: string
    page_range?: string
    chunk_id?: number
    chunk_index?: number
    subchunk_label?: string
    source?: string
  }>
  checks: {
    has_qa_index?: boolean
    indexed_status?: boolean
    collection_exists?: boolean
    qa_chunk_count?: number
    milvus_num_entities?: number
    entity_count_matches_metadata?: boolean
    milvus_has_entities?: boolean
    sample_chunks_returned?: number
    likely_keyword_search_will_work?: boolean
  }
}

export async function getPaperQaDiagnostic(arxivId: string, sampleLimit: number = 3): Promise<QaDiagnosticResult> {
  return request.get(`/paper/${arxivId}/qa-diagnose`, {
    params: { sample_limit: sampleLimit }
  })
}

export function getPaperRetrievalTraceDownloadUrl(
  arxivId: string,
  format: 'md' | 'json' = 'md',
  traceName?: string
): string {
  const params = new URLSearchParams({ format })
  if (traceName) {
    params.set('trace_name', traceName)
  }
  return `/api/paper/${arxivId}/qa-trace/latest?${params.toString()}`
}

export interface CreateQaIndexResult {
  status: string
  message: string
  arxiv_id: string
  loading_method?: string
  pdf_path: string
  document_path?: string
  filename?: string
  total_pages?: number
  document_text_length?: number
  document_markdown_length?: number
}

export async function createPaperQaIndex(arxivId: string, loadingMethod: string = 'pymupdf'): Promise<CreateQaIndexResult> {
  return request.post(`/paper/${arxivId}/create-qa-index`, null, {
    params: {
      loading_method: loadingMethod
    }
  })
}

export interface QaResult {
  status: string
  arxiv_id: string
  question: string
  answer: string
  retrieval_debug?: RetrievalDebug | null
  sources: Array<{
    content: string
    page_number: string
    source?: string
    subchunk_label?: string
  }>
}

export interface QaRequestOptions {
  top_k?: number
  enable_query_rewrite?: boolean
  enable_hyde?: boolean
  enable_keyword_search?: boolean
  enable_llm_rerank?: boolean
  debug?: boolean
}

export interface RetrievalDebugChunk {
  chunk_id?: number
  original_chunk_id?: number
  page_number?: string
  page_range?: string
  route_rank?: number
  score?: number
  route_score?: number
  retrieval_route?: string
  matched_routes?: string[]
  route_scores?: Record<string, number>
  source_query?: string
  source_queries?: string[]
  subchunk_label?: string
  content?: string
  preview?: string
}

export interface RetrievalDebugQueryDetail {
  query: string
  keywords: string[]
  keyword_count?: number
}

export interface RetrievalDebugQueryCandidate {
  query: string
  source?: 'model' | 'heuristic' | string
  source_index?: number
  normalized?: string
  selected?: boolean
  reason?: string
}

export interface RetrievalDebugQueryRewrite {
  enabled?: boolean
  original_query?: string
  query_plan?: Record<string, any>
  model_queries?: string[]
  heuristic_queries?: string[]
  selected_queries?: string[]
  selected_keywords?: string[]
  selected_query_details?: RetrievalDebugQueryDetail[]
  candidates?: RetrievalDebugQueryCandidate[]
  llm_error?: string | null
}

export interface RetrievalDebugHyde {
  enabled?: boolean
  text?: string
  source_queries?: string[]
  focus_queries?: string[]
}

export interface RetrievalDebugKeywordSearch {
  enabled?: boolean
  queries?: string[]
  selected_rewrite_queries?: string[]
  query_details?: RetrievalDebugQueryDetail[]
  keywords?: string[]
}

export interface RetrievalDebug {
  original_query: string
  rewritten_queries: string[]
  hyde_text: string
  query_plan?: Record<string, any>
  query_rewrite?: RetrievalDebugQueryRewrite
  hyde?: RetrievalDebugHyde
  keyword_search?: RetrievalDebugKeywordSearch
  routes: Record<string, RetrievalDebugChunk[]>
  stages?: Record<string, RetrievalDebugChunk[]>
  final_chunks: RetrievalDebugChunk[]
  config?: Record<string, any>
  fusion?: Record<string, any>
  trace_export?: Record<string, string>
}

export async function qaPaper(arxivId: string, question: string, options: QaRequestOptions = {}): Promise<QaResult> {
  return request.post(`/paper/${arxivId}/qa`, { question, ...options })
}

export interface QaStreamHandlers {
  onMeta?: (meta: {
    status: string
    arxiv_id: string
    question: string
      sources: Array<{
        content: string
        page_number: string
        source?: string
      }>
      retrieval_debug?: RetrievalDebug | null
  }) => void
  onDelta?: (delta: string) => void
  onDone?: (payload: {
    status: string
    answer: string
    sources: Array<{
      content: string
      page_number: string
      source?: string
    }>
    retrieval_debug?: RetrievalDebug | null
    usage?: {
      input_tokens?: number | null
      output_tokens?: number | null
      total_tokens?: number | null
    } | null
  }) => void
  onError?: (detail: string) => void
  signal?: AbortSignal
}

function parseSseEvent(rawEvent: string): { event: string; data: any } | null {
  const lines = rawEvent
    .split(/\r?\n/)
    .filter(Boolean)

  if (!lines.length) return null

  let event = 'message'
  const dataLines: string[] = []

  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).replace(/^ /, ''))
    }
  }

  const dataText = dataLines.join('\n')
  if (!dataText) return { event, data: null }

  try {
    return { event, data: JSON.parse(dataText) }
  } catch {
    return { event, data: dataText }
  }
}

export async function qaPaperStream(
  arxivId: string,
  question: string,
  handlers: QaStreamHandlers = {},
  options: QaRequestOptions = {}
): Promise<QaResult> {
  const response = await fetch(`/api/paper/${arxivId}/qa/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream'
    },
    body: JSON.stringify({ question, ...options }),
    signal: handlers.signal
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed with status ${response.status}`)
  }

  if (!response.body) {
    throw new Error('Streaming response body is empty')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let finalAnswer = ''
  let finalSources: Array<{
    content: string
    page_number: string
    source?: string
  }> = []
  let finalRetrievalDebug: RetrievalDebug | null = null

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    const parts = buffer.split(/\r?\n\r?\n/)
    buffer = parts.pop() || ''

    for (const part of parts) {
      const parsed = parseSseEvent(part)
      if (!parsed) continue

      if (parsed.event === 'meta' && parsed.data) {
        handlers.onMeta?.(parsed.data)
        if (Array.isArray(parsed.data.sources)) {
          finalSources = parsed.data.sources
        }
        if (parsed.data.retrieval_debug) {
          finalRetrievalDebug = parsed.data.retrieval_debug
        }
      } else if (parsed.event === 'delta' && parsed.data?.delta) {
        finalAnswer += parsed.data.delta
        handlers.onDelta?.(parsed.data.delta)
      } else if (parsed.event === 'done' && parsed.data) {
        if (typeof parsed.data.answer === 'string') {
          finalAnswer = parsed.data.answer
        }
        if (Array.isArray(parsed.data.sources)) {
          finalSources = parsed.data.sources
        }
        if (parsed.data.retrieval_debug) {
          finalRetrievalDebug = parsed.data.retrieval_debug
        }
        handlers.onDone?.(parsed.data)
        return {
          status: parsed.data.status || 'success',
          arxiv_id: arxivId,
          question,
          answer: finalAnswer,
          sources: finalSources,
          retrieval_debug: finalRetrievalDebug
        }
      } else if (parsed.event === 'error' && parsed.data) {
        const detail = parsed.data.detail || 'Streaming request failed'
        handlers.onError?.(detail)
        throw new Error(detail)
      }
    }
  }

  return {
    status: 'success',
    arxiv_id: arxivId,
    question,
    answer: finalAnswer,
    sources: finalSources,
    retrieval_debug: finalRetrievalDebug
  }
}
