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
  RecommendationResult,
  PaperActionType,
  PaperNote,
  PaperNoteType,
  UserPaperAction,
  UserPaperActionMap,
  UserResearchProfile
} from '@/types/paper'
import { mockPapers, mockRecommendedPapers, mockLabeledPapers, mockStats } from '@/mock/papers'

const isMockMode = false
const DEFAULT_USER_ID = 'local_user'

export interface DashboardStats {
  totalPapers: number
  labeledPapers: number
  latestSyncNewPapers: number
  lastSyncedDate: string | null
  lastSyncRunAt: string | null
  lastSyncStatus: string
  lastSyncMode: string
  latestSyncMatchedPapers: number
  syncErrors: number
  syncErrorMessage: string | null
}

function normalizeDashboardStats(raw: any): DashboardStats {
  return {
    totalPapers: Number(raw?.totalPapers || 0),
    labeledPapers: Number(raw?.labeledPapers || 0),
    latestSyncNewPapers: Number(raw?.latestSyncNewPapers ?? raw?.todayNewPapers ?? 0),
    lastSyncedDate: raw?.lastSyncedDate ? String(raw.lastSyncedDate) : null,
    lastSyncRunAt: raw?.lastSyncRunAt ? String(raw.lastSyncRunAt) : null,
    lastSyncStatus: String(raw?.lastSyncStatus || 'unknown'),
    lastSyncMode: String(raw?.lastSyncMode || 'sync'),
    latestSyncMatchedPapers: Number(raw?.latestSyncMatchedPapers || 0),
    syncErrors: Number(raw?.syncErrors || 0),
    syncErrorMessage: raw?.syncErrorMessage ? String(raw.syncErrorMessage) : null
  }
}

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
    absUrl: raw.abs_url || raw.absUrl || raw.url || '',
    label: raw.label || null,
    paperActions: raw.paperActions || raw.paper_actions || undefined,
    query_match_score: raw.query_match_score,
    personalization_score: raw.personalization_score,
    final_score: raw.final_score,
    score_breakdown: raw.score_breakdown || null,
    matched_terms: Array.isArray(raw.matched_terms) ? raw.matched_terms : undefined,
    personalized_reason: raw.personalized_reason || null,
    match_reason: raw.match_reason || null,
    priority: typeof raw.priority === 'number' ? raw.priority : undefined
  }
}

function emptyResearchProfile(): UserResearchProfile {
  return {
    user_id: DEFAULT_USER_ID,
    positive_topics: [],
    negative_topics: [],
    recent_topics: [],
    preferred_categories: [],
    preferred_answer_style: '',
    common_question_types: [],
    representative_papers: []
  }
}

function normalizeResearchProfile(raw: any): UserResearchProfile {
  return {
    user_id: raw?.user_id || DEFAULT_USER_ID,
    positive_topics: Array.isArray(raw?.positive_topics) ? raw.positive_topics : [],
    negative_topics: Array.isArray(raw?.negative_topics) ? raw.negative_topics : [],
    recent_topics: Array.isArray(raw?.recent_topics) ? raw.recent_topics : [],
    preferred_categories: Array.isArray(raw?.preferred_categories) ? raw.preferred_categories : [],
    preferred_answer_style: String(raw?.preferred_answer_style || ''),
    common_question_types: Array.isArray(raw?.common_question_types) ? raw.common_question_types : [],
    representative_papers: Array.isArray(raw?.representative_papers) ? raw.representative_papers : [],
    created_at: raw?.created_at || null,
    updated_at: raw?.updated_at || null
  }
}

function normalizePaperActionMap(raw: any): UserPaperActionMap {
  if (!raw || typeof raw !== 'object') return {}
  return Object.fromEntries(
    Object.entries(raw)
      .filter(([, value]) => Array.isArray(value))
      .map(([key, value]) => [key, (value as any[]).map(item => String(item)).filter(Boolean)])
  ) as UserPaperActionMap
}

function normalizeUserPaperAction(raw: any): UserPaperAction {
  return {
    id: raw?.id,
    user_id: raw?.user_id || DEFAULT_USER_ID,
    arxiv_id: String(raw?.arxiv_id || ''),
    action_type: raw?.action_type,
    metadata: raw?.metadata || raw?.metadata_json || {},
    created_at: raw?.created_at,
    updated_at: raw?.updated_at
  }
}

function normalizePaperNote(raw: any): PaperNote {
  return {
    note_id: String(raw?.note_id || ''),
    user_id: raw?.user_id || DEFAULT_USER_ID,
    arxiv_id: String(raw?.arxiv_id || ''),
    session_id: raw?.session_id || null,
    source_message_id: raw?.source_message_id || null,
    source_turn_id: raw?.source_turn_id || null,
    title: String(raw?.title || ''),
    content: String(raw?.content || ''),
    note_type: (raw?.note_type || 'custom') as PaperNoteType,
    source_chunk_ids: Array.isArray(raw?.source_chunk_ids) ? raw.source_chunk_ids.map((item: any) => String(item)) : [],
    tags: Array.isArray(raw?.tags) ? raw.tags.map((item: any) => String(item)) : [],
    include_in_profile: Boolean(raw?.include_in_profile),
    created_at: raw?.created_at || null,
    updated_at: raw?.updated_at || null,
    sources: Array.isArray(raw?.sources) ? raw.sources : []
  }
}

export async function getUserPreferences(): Promise<{
  liked_papers: string[]
  disliked_papers: string[]
  paper_actions?: UserPaperActionMap
  research_profile?: UserResearchProfile | null
}> {
  try {
    const response = await request.get(`/user/preferences/${DEFAULT_USER_ID}`)
    return {
      liked_papers: Array.isArray(response?.liked_papers) ? response.liked_papers : [],
      disliked_papers: Array.isArray(response?.disliked_papers) ? response.disliked_papers : [],
      paper_actions: normalizePaperActionMap(response?.paper_actions),
      research_profile: response?.research_profile ? normalizeResearchProfile(response.research_profile) : null
    }
  } catch (error: any) {
    if (error?.response?.status === 404) {
      return { liked_papers: [], disliked_papers: [], paper_actions: {}, research_profile: emptyResearchProfile() }
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

export async function recordPaperAction(
  paper: Paper,
  actionType: PaperActionType,
  metadata?: Record<string, any>
): Promise<void> {
  await request.post('/user/paper-action', {
    user_id: DEFAULT_USER_ID,
    arxiv_id: getPaperArxivId(paper),
    action_type: actionType,
    paper: buildPaperMaterializationPayload(paper),
    metadata: metadata || null
  })
}

export async function removePaperAction(paper: Paper, actionType: PaperActionType): Promise<void> {
  await request.delete('/user/paper-action', {
    data: {
      user_id: DEFAULT_USER_ID,
      arxiv_id: getPaperArxivId(paper),
      action_type: actionType
    }
  })
}

export async function getUserPaperActions(actionType?: PaperActionType): Promise<{
  status: string
  user_id: string
  action_type?: PaperActionType
  actions: UserPaperAction[]
  action_map: UserPaperActionMap
}> {
  const response = await request.get(`/user/paper-actions/${DEFAULT_USER_ID}`, {
    params: actionType ? { action_type: actionType } : undefined
  })
  return {
    status: response?.status || 'success',
    user_id: response?.user_id || DEFAULT_USER_ID,
    action_type: response?.action_type,
    actions: Array.isArray(response?.actions) ? response.actions.map(normalizeUserPaperAction) : [],
    action_map: normalizePaperActionMap(response?.action_map)
  }
}

export async function getUserResearchProfile(): Promise<UserResearchProfile> {
  try {
    const response = await request.get(`/user/research-profile/${DEFAULT_USER_ID}`)
    return normalizeResearchProfile(response)
  } catch (error: any) {
    if (error?.response?.status === 404) {
      return emptyResearchProfile()
    }
    throw error
  }
}

export async function upsertUserResearchProfile(profile: Partial<UserResearchProfile>): Promise<UserResearchProfile> {
  const response = await request.put('/user/research-profile', {
    user_id: DEFAULT_USER_ID,
    ...profile
  })
  return normalizeResearchProfile(response?.profile || response)
}

export async function patchUserResearchProfile(profile: Partial<UserResearchProfile>): Promise<UserResearchProfile> {
  const response = await request.patch('/user/research-profile', {
    user_id: DEFAULT_USER_ID,
    ...profile
  })
  return normalizeResearchProfile(response?.profile || response)
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

export async function getStats(): Promise<DashboardStats> {
  if (isMockMode) {
    const { recommendedPapers, ...stats } = mockStats
    return normalizeDashboardStats(stats)
  }

  const response = await request.get('/stats', {
    params: {
      user_id: DEFAULT_USER_ID
    }
  })
  return normalizeDashboardStats(response)
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
  const response = await request.post('/user/recommend-papers', { top_n: topN, max_age_months: maxAgeMonths })
  return {
    ...response,
    research_profile: response?.research_profile ? normalizeResearchProfile(response.research_profile) : null,
    paper_actions: normalizePaperActionMap(response?.paper_actions),
    recommendations: Array.isArray(response?.recommendations)
      ? response.recommendations.map((item: any) => ({
          ...normalizePaper(item),
          similarityScore: typeof item.similarity_score === 'number'
            ? item.similarity_score
            : (typeof item.similarityScore === 'number' ? item.similarityScore : (typeof item.score === 'number' ? item.score : 0)),
          finalScore: typeof item.final_score === 'number' ? item.final_score : undefined,
          reason: item.reason,
          scoreBreakdown: item.score_breakdown || item.scoreBreakdown || undefined,
          recall_source: item.recall_source || item.recallSource || undefined,
          recall_cluster_id: item.recall_cluster_id || item.recallClusterId || null,
          recall_cluster_similarity: typeof item.recall_cluster_similarity === 'number'
            ? item.recall_cluster_similarity
            : (typeof item.recallClusterSimilarity === 'number' ? item.recallClusterSimilarity : null),
          recall_cluster_rank: typeof item.recall_cluster_rank === 'number'
            ? item.recall_cluster_rank
            : (typeof item.recallClusterRank === 'number' ? item.recallClusterRank : null),
          recall_cluster_hits: Array.isArray(item.recall_cluster_hits) ? item.recall_cluster_hits : [],
          best_matched_cluster_id: item.best_matched_cluster_id || item.bestMatchedClusterId || null,
          best_matched_cluster_similarity: typeof item.best_matched_cluster_similarity === 'number'
            ? item.best_matched_cluster_similarity
            : (typeof item.bestMatchedClusterSimilarity === 'number' ? item.bestMatchedClusterSimilarity : null),
          cluster_similarities: Array.isArray(item.cluster_similarities)
            ? item.cluster_similarities
            : (Array.isArray(item.clusterSimilarities) ? item.clusterSimilarities : undefined),
          diversityDebug: item.diversity_debug || item.diversityDebug || undefined
        }))
      : []
  }
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
  job_id?: string
  job_status?: string
  current_stage?: string
  progress?: number
  loading_method?: string
  pdf_path?: string
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

export interface QaIndexJobResult {
  job_id: string
  arxiv_id: string
  status: 'pending' | 'running' | 'success' | 'failed' | string
  current_stage?: string | null
  progress?: number | null
  error_message?: string | null
  loading_method?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export async function getLatestPaperQaIndexJob(arxivId: string): Promise<QaIndexJobResult> {
  return request.get(`/paper/${arxivId}/qa-index-jobs/latest`, {
    params: { arxiv_id: arxivId }
  })
}

export async function getPaperQaIndexJob(arxivId: string, jobId: string): Promise<QaIndexJobResult> {
  return request.get(`/paper/${arxivId}/qa-index-jobs/${jobId}`)
}

export interface QaResult {
  status: string
  arxiv_id: string
  question: string
  session_id?: string
  chat_session?: PaperChatSession | null
  turn_id?: string
  original_question?: string
  contextualized_question?: string
  used_short_term_memory?: boolean
  question_contextualization?: Record<string, any> | null
  answer: string
  retrieval_debug?: RetrievalDebug | null
  sources: Array<{
    content: string
    page_number: string
    source?: string
    subchunk_label?: string
    section_path?: string
    parent_chunk_id?: string | number
    chunk_type?: string
    asset_summary?: string
    asset_preview_text?: string
  }>
}

export interface PaperChatSession {
  session_id: string
  user_id: string
  arxiv_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
  status: string
}

export interface PaperChatMessage {
  message_id: string
  turn_id: string
  session_id: string
  role: 'user' | 'assistant'
  content: string
  sources: Array<{
    content?: string
    page_number?: string
    source?: string
    section_path?: string
    parent_chunk_id?: string | number
    chunk_type?: string
    asset_summary?: string
    asset_preview_text?: string
  }>
  retrieval_debug_snapshot?: RetrievalDebug | null
  contextualized_question?: string
  question_contextualization?: Record<string, any> | null
  status?: string
  created_at: string
}

export interface PaperNotePayload {
  user_id?: string
  session_id?: string
  source_message_id?: string
  source_turn_id?: string
  title?: string
  content: string
  note_type?: PaperNoteType
  source_chunk_ids?: string[]
  tags?: string[]
  include_in_profile?: boolean
}

export async function listPaperNotes(
  arxivId: string,
  params: { user_id?: string; note_type?: PaperNoteType } = {}
): Promise<{ items: PaperNote[] }> {
  const response = await request.get(`/paper/${arxivId}/notes`, { params })
  return {
    items: Array.isArray(response?.items) ? response.items.map(normalizePaperNote) : []
  }
}

export async function createPaperNote(
  arxivId: string,
  payload: PaperNotePayload
): Promise<{ item: PaperNote | null }> {
  const response = await request.post(`/paper/${arxivId}/notes`, payload)
  return {
    item: response?.item ? normalizePaperNote(response.item) : null
  }
}

export async function updatePaperNote(
  arxivId: string,
  noteId: string,
  payload: Partial<PaperNotePayload>
): Promise<{ item: PaperNote | null }> {
  const response = await request.patch(`/paper/${arxivId}/notes/${noteId}`, payload)
  return {
    item: response?.item ? normalizePaperNote(response.item) : null
  }
}

export async function deletePaperNote(
  arxivId: string,
  noteId: string,
  userId?: string
): Promise<{ status: string; deleted: boolean }> {
  return request.delete(`/paper/${arxivId}/notes/${noteId}`, {
    params: { user_id: userId }
  })
}

export function getPaperNotesExportUrl(arxivId: string, userId?: string): string {
  const params = new URLSearchParams()
  if (userId) params.set('user_id', userId)
  const query = params.toString()
  return `/api/paper/${arxivId}/notes/export${query ? `?${query}` : ''}`
}

export interface QaConversationContextTurn {
  turn_id: string
  question: string
  answer_summary: string
  created_at: string
  sources: Array<{
    source_id?: string | number
    content?: string
    page_number?: string
    source?: string
    section_path?: string
    chunk_type?: string
    asset_summary?: string
    asset_preview_text?: string
  }>
}

export interface QaRequestOptions {
  user_id?: string
  session_id?: string
  top_k?: number
  enable_query_rewrite?: boolean
  enable_hyde?: boolean
  enable_keyword_search?: boolean
  enable_llm_rerank?: boolean
  debug?: boolean
  conversation_context?: QaConversationContextTurn[]
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

export interface RetrievalDebugMemoryContext {
  enabled?: boolean
  reason?: string
  query_keywords?: string[]
  referenced_turn_ids?: string[]
  referenced_source_ids?: string[]
  candidates?: Array<Record<string, any>>
  fallback_reason?: string | null
}

export interface RetrievalDebugMemoryRuntime {
  enabled?: boolean
  config?: Record<string, any>
}

export interface RetrievalDebugMemoryModule {
  enabled?: boolean
  applied?: boolean
  reason?: string
  fallback_reason?: string | null
  session_id?: string | null
  provided_turn_count?: number
  used_turn_count?: number
}

export interface RetrievalDebug {
  original_query: string
  original_question?: string
  contextualized_question?: string
  rewritten_queries: string[]
  hyde_text: string
  query_plan?: Record<string, any>
  query_rewrite?: RetrievalDebugQueryRewrite
  hyde?: RetrievalDebugHyde
  keyword_search?: RetrievalDebugKeywordSearch
  question_contextualization?: Record<string, any>
  memory_context?: RetrievalDebugMemoryContext
  memory_runtime?: RetrievalDebugMemoryRuntime
  memory_modules?: {
    short_term_memory?: RetrievalDebugMemoryModule
    session?: RetrievalDebugMemoryModule
    memory_retrieval?: RetrievalDebugMemoryModule
    user_profile?: RetrievalDebugMemoryModule
  }
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

export async function listPaperChatSessions(
  arxivId: string,
  params: { user_id?: string; limit?: number } = {}
): Promise<{ items: PaperChatSession[] }> {
  return request.get(`/paper/${arxivId}/chat-sessions`, { params })
}

export async function getRecentPaperChatSession(
  arxivId: string,
  userId?: string
): Promise<{ item: PaperChatSession | null }> {
  return request.get(`/paper/${arxivId}/chat-sessions/recent`, {
    params: { user_id: userId }
  })
}

export async function createPaperChatSession(
  arxivId: string,
  payload: { user_id?: string; title?: string } = {}
): Promise<{ item: PaperChatSession | null }> {
  return request.post(`/paper/${arxivId}/chat-sessions`, payload)
}

export async function getPaperChatSession(
  arxivId: string,
  sessionId: string,
  userId?: string
): Promise<{ item: PaperChatSession | null }> {
  return request.get(`/paper/${arxivId}/chat-sessions/${sessionId}`, {
    params: { user_id: userId }
  })
}

export async function getPaperChatMessages(
  arxivId: string,
  sessionId: string,
  userId?: string
): Promise<{ session: PaperChatSession | null; items: PaperChatMessage[] }> {
  return request.get(`/paper/${arxivId}/chat-sessions/${sessionId}/messages`, {
    params: { user_id: userId }
  })
}

export async function clearPaperChatSession(
  arxivId: string,
  sessionId: string,
  userId?: string
): Promise<{ item: PaperChatSession | null }> {
  return request.post(`/paper/${arxivId}/chat-sessions/${sessionId}/clear`, {
    user_id: userId
  })
}

export async function deletePaperChatSession(
  arxivId: string,
  sessionId: string,
  userId?: string
): Promise<{ status: string; deleted: boolean }> {
  return request.delete(`/paper/${arxivId}/chat-sessions/${sessionId}`, {
    params: { user_id: userId }
  })
}

export interface QaStreamHandlers {
  onMeta?: (meta: {
    status: string
    arxiv_id: string
    question: string
    session_id?: string
    chat_session?: PaperChatSession | null
    original_question?: string
    contextualized_question?: string
    used_short_term_memory?: boolean
    question_contextualization?: Record<string, any> | null
    sources: Array<{
      content: string
      page_number: string
      source?: string
      section_path?: string
      parent_chunk_id?: string | number
      chunk_type?: string
      asset_summary?: string
      asset_preview_text?: string
    }>
    retrieval_debug?: RetrievalDebug | null
  }) => void
  onDelta?: (delta: string) => void
  onDone?: (payload: {
    status: string
    answer: string
    session_id?: string
    chat_session?: PaperChatSession | null
    turn_id?: string
    original_question?: string
    contextualized_question?: string
    used_short_term_memory?: boolean
    question_contextualization?: Record<string, any> | null
    sources: Array<{
      content: string
      page_number: string
      source?: string
      section_path?: string
      parent_chunk_id?: string | number
      chunk_type?: string
      asset_summary?: string
      asset_preview_text?: string
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
    section_path?: string
    parent_chunk_id?: string | number
    chunk_type?: string
    asset_summary?: string
    asset_preview_text?: string
  }> = []
  let finalRetrievalDebug: RetrievalDebug | null = null
  let finalSessionId = options.session_id || ''
  let finalChatSession: PaperChatSession | null = null
  let finalOriginalQuestion = question
  let finalContextualizedQuestion = question
  let finalUsedShortTermMemory = false
  let finalQuestionContextualization: Record<string, any> | null = null

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
        if (typeof parsed.data.session_id === 'string' && parsed.data.session_id) {
          finalSessionId = parsed.data.session_id
        }
        if (parsed.data.chat_session) {
          finalChatSession = parsed.data.chat_session
        }
        if (typeof parsed.data.original_question === 'string' && parsed.data.original_question) {
          finalOriginalQuestion = parsed.data.original_question
        }
        if (typeof parsed.data.contextualized_question === 'string' && parsed.data.contextualized_question) {
          finalContextualizedQuestion = parsed.data.contextualized_question
        }
        if (typeof parsed.data.used_short_term_memory === 'boolean') {
          finalUsedShortTermMemory = parsed.data.used_short_term_memory
        }
        if (parsed.data.question_contextualization) {
          finalQuestionContextualization = parsed.data.question_contextualization
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
        if (typeof parsed.data.session_id === 'string' && parsed.data.session_id) {
          finalSessionId = parsed.data.session_id
        }
        if (parsed.data.chat_session) {
          finalChatSession = parsed.data.chat_session
        }
        if (typeof parsed.data.original_question === 'string' && parsed.data.original_question) {
          finalOriginalQuestion = parsed.data.original_question
        }
        if (typeof parsed.data.contextualized_question === 'string' && parsed.data.contextualized_question) {
          finalContextualizedQuestion = parsed.data.contextualized_question
        }
        if (typeof parsed.data.used_short_term_memory === 'boolean') {
          finalUsedShortTermMemory = parsed.data.used_short_term_memory
        }
        if (parsed.data.question_contextualization) {
          finalQuestionContextualization = parsed.data.question_contextualization
        }
        handlers.onDone?.(parsed.data)
        return {
          status: parsed.data.status || 'success',
          arxiv_id: arxivId,
          question,
          session_id: finalSessionId || undefined,
          chat_session: finalChatSession,
          original_question: finalOriginalQuestion,
          contextualized_question: finalContextualizedQuestion,
          used_short_term_memory: finalUsedShortTermMemory,
          question_contextualization: finalQuestionContextualization,
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
    session_id: finalSessionId || undefined,
    chat_session: finalChatSession,
    original_question: finalOriginalQuestion,
    contextualized_question: finalContextualizedQuestion,
    used_short_term_memory: finalUsedShortTermMemory,
    question_contextualization: finalQuestionContextualization,
    answer: finalAnswer,
    sources: finalSources,
    retrieval_debug: finalRetrievalDebug
  }
}
