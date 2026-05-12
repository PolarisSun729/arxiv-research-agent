import request from './request'
import type { Paper, RecommendedPaper, LabeledPaper, SearchParams, LabelParams, PaginatedResponse, ArxivSearchParams } from '@/types/paper'
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

export async function savePaperMetadata(paper: Paper): Promise<void> {
  await request.post('/paper', {
    arxiv_id: getPaperArxivId(paper),
    title: paper.title,
    authors: paper.authors.join(', '),
    abstract: paper.summary,
    categories: paper.categories.join(', '),
    published_date: paper.publishedAt,
    url: paper.absUrl || paper.pdfUrl
  })
}

export async function likePaper(paper: Paper): Promise<void> {
  const arxivId = getPaperArxivId(paper)
  await savePaperMetadata(paper)
  await request.post('/user/like-paper', { arxiv_id: arxivId })
}

export async function dislikePaper(paper: Paper): Promise<void> {
  const arxivId = getPaperArxivId(paper)
  await savePaperMetadata(paper)
  await request.post('/user/dislike-paper', { arxiv_id: arxivId })
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

export interface InterestVectorResult {
  status: string
  message: string
  paper_count: number
  vector_dimension: number
  embedding_model: string
}

export interface InterestVector {
  user_id: string
  vector_data: number[]
  paper_count: number
  embedding_model: string
  vector_dimension: number
  created_at: string
  updated_at: string
}

export async function generateInterestVector(): Promise<InterestVectorResult> {
  return request.post('/user/generate-interest-vector')
}

export async function getInterestVector(): Promise<InterestVector> {
  return request.get('/user/interest-vector')
}

export interface RecommendationResult {
  status: string
  message: string
  total_found: number
  recommendations: Paper[]
}

export async function recommendPapers(topN: number = 10): Promise<RecommendationResult> {
  return request.post('/user/recommend-papers', { top_n: topN })
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

export interface CreateQaIndexResult {
  status: string
  message: string
  arxiv_id: string
  collection_name: string
  chunk_count: number
  embedding_model: string
  pdf_path: string
}

export async function createPaperQaIndex(arxivId: string): Promise<CreateQaIndexResult> {
  return request.post(`/paper/${arxivId}/create-qa-index`)
}

export interface QaResult {
  status: string
  arxiv_id: string
  question: string
  answer: string
  sources: Array<{
    content: string
    page_number: string
  }>
}

export async function qaPaper(arxivId: string, question: string): Promise<QaResult> {
  return request.post(`/paper/${arxivId}/qa`, { question })
}