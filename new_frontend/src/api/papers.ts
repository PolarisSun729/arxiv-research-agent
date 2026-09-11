import request from './request'
import { apiFetch } from './auth'
import { ApiError, normalizeApiError, parseFetchErrorResponse } from './errors'
import type {
  Paper,
  RecommendedPaper,
  LabeledPaper,
  SearchParams,
  LabelParams,
  PaginatedResponse,
  ArxivSearchResult,
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
  UserResearchProfile,
  UserResearchProfileDetail,
  UserProfileBuildJob
} from '@/types/paper'
import { mockPapers, mockRecommendedPapers, mockLabeledPapers, mockStats } from '@/mock/papers'
import { getCurrentUserId } from '@/composables/useUserContext'

const isMockMode = false

function resolveUserId(userId?: string | null): string {
  const normalized = String(userId || '').trim()
  // API 层只负责解析当前身份，避免后续认证接入时出现多处默认用户兜底。
  return normalized || getCurrentUserId()
}

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
    profile_reasons: Array.isArray(raw.profile_reasons) ? raw.profile_reasons : undefined,
    profile_match_details: Array.isArray(raw.profile_match_details) ? raw.profile_match_details : undefined,
    profile_match_score: typeof raw.profile_match_score === 'number' ? raw.profile_match_score : undefined,
    priority: typeof raw.priority === 'number' ? raw.priority : undefined
  }
}

function normalizeRecommendedPaper(raw: any): RecommendedPaper {
  return {
    ...normalizePaper(raw),
    similarityScore: typeof raw?.similarity_score === 'number'
      ? raw.similarity_score
      : (typeof raw?.similarityScore === 'number' ? raw.similarityScore : (typeof raw?.score === 'number' ? raw.score : 0)),
    finalScore: typeof raw?.final_score === 'number'
      ? raw.final_score
      : (typeof raw?.finalScore === 'number' ? raw.finalScore : undefined),
    reason: raw?.reason || undefined,
    scoreBreakdown: raw?.score_breakdown || raw?.scoreBreakdown || undefined,
    recall_source: raw?.recall_source || raw?.recallSource || undefined,
    recall_cluster_id: raw?.recall_cluster_id || raw?.recallClusterId || null,
    recall_cluster_similarity: typeof raw?.recall_cluster_similarity === 'number'
      ? raw.recall_cluster_similarity
      : (typeof raw?.recallClusterSimilarity === 'number' ? raw.recallClusterSimilarity : null),
    recall_cluster_rank: typeof raw?.recall_cluster_rank === 'number'
      ? raw.recall_cluster_rank
      : (typeof raw?.recallClusterRank === 'number' ? raw.recallClusterRank : null),
    recall_cluster_hits: Array.isArray(raw?.recall_cluster_hits)
      ? raw.recall_cluster_hits
      : (Array.isArray(raw?.recallClusterHits) ? raw.recallClusterHits : []),
    best_matched_cluster_id: raw?.best_matched_cluster_id || raw?.bestMatchedClusterId || null,
    best_matched_cluster_similarity: typeof raw?.best_matched_cluster_similarity === 'number'
      ? raw.best_matched_cluster_similarity
      : (typeof raw?.bestMatchedClusterSimilarity === 'number' ? raw.bestMatchedClusterSimilarity : null),
    cluster_similarities: Array.isArray(raw?.cluster_similarities)
      ? raw.cluster_similarities
      : (Array.isArray(raw?.clusterSimilarities) ? raw.clusterSimilarities : undefined),
    diversityDebug: raw?.diversity_debug || raw?.diversityDebug || undefined
  }
}

function emptyResearchProfile(userId?: string): UserResearchProfile {
  return {
    user_id: resolveUserId(userId),
    positive_topics: [],
    negative_topics: [],
    recent_topics: [],
    pinned_topics: [],
    hidden_topics: [],
    preferred_categories: [],
    preferred_answer_style: '',
    common_question_types: [],
    representative_papers: [],
    canonical_topics: [],
    canonical_negative_topics: [],
    canonical_recent_topics: [],
    topic_evidence: {},
    aggregation_report: {},
    review_status: {},
    quality_report: {},
    snapshot_id: null
  }
}

function normalizeResearchProfile(raw: any, fallbackUserId?: string): UserResearchProfile {
  return {
    user_id: raw?.user_id || resolveUserId(fallbackUserId),
    positive_topics: Array.isArray(raw?.positive_topics) ? raw.positive_topics : [],
    negative_topics: Array.isArray(raw?.negative_topics) ? raw.negative_topics : [],
    recent_topics: Array.isArray(raw?.recent_topics) ? raw.recent_topics : [],
    pinned_topics: Array.isArray(raw?.pinned_topics) ? raw.pinned_topics : [],
    hidden_topics: Array.isArray(raw?.hidden_topics) ? raw.hidden_topics : [],
    preferred_categories: Array.isArray(raw?.preferred_categories) ? raw.preferred_categories : [],
    preferred_answer_style: String(raw?.preferred_answer_style || ''),
    common_question_types: Array.isArray(raw?.common_question_types) ? raw.common_question_types : [],
    representative_papers: Array.isArray(raw?.representative_papers) ? raw.representative_papers : [],
    canonical_topics: Array.isArray(raw?.canonical_topics) ? raw.canonical_topics : [],
    canonical_negative_topics: Array.isArray(raw?.canonical_negative_topics) ? raw.canonical_negative_topics : [],
    canonical_recent_topics: Array.isArray(raw?.canonical_recent_topics) ? raw.canonical_recent_topics : [],
    topic_evidence: raw?.topic_evidence && typeof raw.topic_evidence === 'object' ? raw.topic_evidence : {},
    aggregation_report: raw?.aggregation_report && typeof raw.aggregation_report === 'object' ? raw.aggregation_report : {},
    review_status: raw?.review_status && typeof raw.review_status === 'object' ? raw.review_status : {},
    quality_report: raw?.quality_report && typeof raw.quality_report === 'object' ? raw.quality_report : {},
    snapshot_id: raw?.snapshot_id || null,
    created_at: raw?.created_at || null,
    updated_at: raw?.updated_at || null
  }
}

function normalizeProfileBuildJob(raw: any): UserProfileBuildJob {
  const metrics = raw?.metrics && typeof raw.metrics === 'object' ? raw.metrics : {}
  const numericMetric = (key: string) => {
    const value = raw?.[key] ?? metrics?.[key]
    return typeof value === 'number' ? value : Number(value || 0)
  }
  return {
    job_id: String(raw?.job_id || ''),
    user_id: raw?.user_id,
    status: String(raw?.status || 'unknown'),
    snapshot_id: raw?.snapshot_id || null,
    current_stage: raw?.current_stage || null,
    progress: typeof raw?.progress === 'number' ? raw.progress : Number(raw?.progress || 0),
    error_message: raw?.error_message || null,
    metrics,
    build_mode: raw?.build_mode ?? metrics?.build_mode ?? null,
    paper_limit: numericMetric('paper_limit'),
    candidate_papers: numericMetric('candidate_papers'),
    total_papers: numericMetric('total_papers'),
    cached_papers: numericMetric('cached_papers'),
    uncached_papers: numericMetric('uncached_papers'),
    processed_papers: numericMetric('processed_papers'),
    failed_papers: numericMetric('failed_papers'),
    successful_papers: numericMetric('successful_papers'),
    cache_hit_count: numericMetric('cache_hit_count'),
    generated_count: numericMetric('generated_count'),
    failed_count: numericMetric('failed_count'),
    skipped_count: numericMetric('skipped_count'),
    average_seconds_per_paper: numericMetric('average_seconds_per_paper'),
    total_evidence_extraction_seconds: numericMetric('total_evidence_extraction_seconds'),
    evidence_concurrency: numericMetric('evidence_concurrency'),
    rate_limit_backoff_count: numericMetric('rate_limit_backoff_count'),
    skipped_paper_count: numericMetric('skipped_paper_count'),
    skipped_read_only_papers: numericMetric('skipped_read_only_papers'),
    skipped_failed_cache_papers: numericMetric('skipped_failed_cache_papers'),
    skipped_limit_papers: numericMetric('skipped_limit_papers'),
    repair_candidate_papers: numericMetric('repair_candidate_papers'),
    paper_evidence_failure_details: Array.isArray(raw?.paper_evidence_failure_details)
      ? raw.paper_evidence_failure_details
      : Array.isArray(metrics?.paper_evidence_failure_details) ? metrics.paper_evidence_failure_details : [],
    evidence_counts: raw?.evidence_counts && typeof raw.evidence_counts === 'object' ? raw.evidence_counts : metrics?.evidence_counts || {},
    current_arxiv_id: raw?.current_arxiv_id ?? metrics?.current_arxiv_id ?? null,
    stage_message: raw?.stage_message ?? metrics?.stage_message ?? null,
    recent_logs: Array.isArray(raw?.recent_logs) ? raw.recent_logs : Array.isArray(metrics?.recent_logs) ? metrics.recent_logs : [],
    created_at: raw?.created_at || null,
    updated_at: raw?.updated_at || null
  }
}

function normalizeResearchProfileDetail(raw: any, fallbackUserId?: string): UserResearchProfileDetail {
  const effectiveUserId = resolveUserId(fallbackUserId || raw?.user_id)
  return {
    user_id: effectiveUserId,
    manual_profile: normalizeResearchProfile(raw?.manual_profile, effectiveUserId),
    generated_profile: normalizeResearchProfile(raw?.generated_profile, effectiveUserId),
    effective_profile: normalizeResearchProfile(raw?.effective_profile, effectiveUserId),
    evidence_summary: raw?.evidence_summary || {},
    quality_report: raw?.quality_report || {},
    build_jobs: Array.isArray(raw?.build_jobs) ? raw.build_jobs.map(normalizeProfileBuildJob) : [],
    snapshots: Array.isArray(raw?.snapshots) ? raw.snapshots : [],
    latest_build_job: raw?.latest_build_job ? normalizeProfileBuildJob(raw.latest_build_job) : null
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

function normalizeUserPaperAction(raw: any, fallbackUserId?: string): UserPaperAction {
  return {
    id: raw?.id,
    user_id: raw?.user_id || resolveUserId(fallbackUserId),
    arxiv_id: String(raw?.arxiv_id || ''),
    action_type: raw?.action_type,
    metadata: raw?.metadata || raw?.metadata_json || {},
    created_at: raw?.created_at,
    updated_at: raw?.updated_at
  }
}

function normalizePaperNote(raw: any, fallbackUserId?: string): PaperNote {
  return {
    note_id: String(raw?.note_id || ''),
    user_id: raw?.user_id || resolveUserId(fallbackUserId),
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

export async function getUserPreferences(userId?: string): Promise<{
  liked_papers: string[]
  disliked_papers: string[]
  paper_actions?: UserPaperActionMap
  research_profile?: UserResearchProfile | null
}> {
  const effectiveUserId = resolveUserId(userId)
  try {
    const response: any = await request.get(`/user/preferences/${effectiveUserId}`)
    return {
      liked_papers: Array.isArray(response?.liked_papers) ? response.liked_papers : [],
      disliked_papers: Array.isArray(response?.disliked_papers) ? response.disliked_papers : [],
      paper_actions: normalizePaperActionMap(response?.paper_actions),
      research_profile: response?.research_profile ? normalizeResearchProfile(response.research_profile, effectiveUserId) : null
    }
  } catch (error: any) {
    if (error?.response?.status === 404) {
      return { liked_papers: [], disliked_papers: [], paper_actions: {}, research_profile: emptyResearchProfile(effectiveUserId) }
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

export async function likePaper(paper: Paper, userId?: string): Promise<void> {
  const effectiveUserId = resolveUserId(userId)
  const arxivId = getPaperArxivId(paper)
  const payload: PaperPreferenceRequest = {
    user_id: effectiveUserId,
    arxiv_id: arxivId,
    paper: buildPaperMaterializationPayload(paper)
  }
  await request.post('/user/like-paper', payload)
}

export async function dislikePaper(paper: Paper, userId?: string): Promise<void> {
  const effectiveUserId = resolveUserId(userId)
  const arxivId = getPaperArxivId(paper)
  const payload: PaperPreferenceRequest = {
    user_id: effectiveUserId,
    arxiv_id: arxivId,
    paper: buildPaperMaterializationPayload(paper)
  }
  await request.post('/user/dislike-paper', payload)
}

export async function removePaperPreference(paper: Paper, label: LabelParams['label'], userId?: string): Promise<void> {
  const effectiveUserId = resolveUserId(userId)
  const endpoint = label === 'liked' ? '/user/like-paper' : '/user/dislike-paper'
  await request.delete(endpoint, {
    data: { user_id: effectiveUserId, arxiv_id: getPaperArxivId(paper) }
  })
}

export async function recordPaperAction(
  paper: Paper,
  actionType: PaperActionType,
  metadata?: Record<string, any>,
  userId?: string
): Promise<void> {
  const effectiveUserId = resolveUserId(userId)
  await request.post('/user/paper-action', {
    user_id: effectiveUserId,
    arxiv_id: getPaperArxivId(paper),
    action_type: actionType,
    paper: buildPaperMaterializationPayload(paper),
    metadata: metadata || null
  })
}

export async function removePaperAction(paper: Paper, actionType: PaperActionType, userId?: string): Promise<void> {
  const effectiveUserId = resolveUserId(userId)
  await request.delete('/user/paper-action', {
    data: {
      user_id: effectiveUserId,
      arxiv_id: getPaperArxivId(paper),
      action_type: actionType
    }
  })
}

export async function getUserPaperActions(actionType?: PaperActionType, userId?: string): Promise<{
  status: string
  user_id: string
  action_type?: PaperActionType
  actions: UserPaperAction[]
  action_map: UserPaperActionMap
}> {
  const effectiveUserId = resolveUserId(userId)
  const response: any = await request.get(`/user/paper-actions/${effectiveUserId}`, {
    params: actionType ? { action_type: actionType } : undefined
  })
  return {
    status: response?.status || 'success',
    user_id: response?.user_id || effectiveUserId,
    action_type: response?.action_type,
    actions: Array.isArray(response?.actions) ? response.actions.map((item: any) => normalizeUserPaperAction(item, effectiveUserId)) : [],
    action_map: normalizePaperActionMap(response?.action_map)
  }
}

export async function getUserResearchProfile(userId?: string): Promise<UserResearchProfile> {
  const effectiveUserId = resolveUserId(userId)
  try {
    const response: any = await request.get(`/user/research-profile/${effectiveUserId}`)
    return normalizeResearchProfile(response, effectiveUserId)
  } catch (error: any) {
    if (error?.response?.status === 404) {
      return emptyResearchProfile(effectiveUserId)
    }
    throw error
  }
}

export async function getUserResearchProfileDetail(userId?: string): Promise<UserResearchProfileDetail> {
  const effectiveUserId = resolveUserId(userId)
  const response: any = await request.get(`/user/research-profile/${effectiveUserId}/detail`)
  return normalizeResearchProfileDetail(response?.detail || response, effectiveUserId)
}

export async function upsertUserResearchProfile(profile: Partial<UserResearchProfile>, userId?: string): Promise<UserResearchProfile> {
  const effectiveUserId = resolveUserId(userId || profile.user_id)
  const response: any = await request.put('/user/research-profile', {
    ...profile,
    user_id: effectiveUserId
  })
  return normalizeResearchProfile(response?.profile || response, effectiveUserId)
}

export async function patchUserResearchProfile(profile: Partial<UserResearchProfile>, userId?: string): Promise<UserResearchProfile> {
  const effectiveUserId = resolveUserId(userId || profile.user_id)
  const response: any = await request.patch('/user/research-profile', {
    ...profile,
    user_id: effectiveUserId
  })
  return normalizeResearchProfile(response?.profile || response, effectiveUserId)
}

export async function rebuildUserResearchProfile(
  userId?: string,
  asyncBuild: boolean = true,
  buildMode: 'incremental' | 'full' | 'repair' = 'incremental',
  maxPapers?: number
): Promise<{ profile: UserResearchProfile; job?: UserProfileBuildJob | null; status: string }> {
  const effectiveUserId = resolveUserId(userId)
  const response: any = await request.post('/user/research-profile/rebuild', {
    user_id: effectiveUserId,
    async_build: asyncBuild,
    build_mode: buildMode,
    max_papers: maxPapers
  })
  return {
    status: response?.status || 'success',
    profile: normalizeResearchProfile(response?.profile || response, effectiveUserId),
    job: response?.job ? normalizeProfileBuildJob(response.job) : null
  }
}

export async function getUserResearchProfileBuildJob(jobId: string): Promise<UserProfileBuildJob> {
  const response: any = await request.get(`/user/research-profile/build-jobs/${jobId}`)
  return normalizeProfileBuildJob(response?.job || response)
}

export async function getUserResearchProfileTopicEvidence(topic: string, userId?: string): Promise<{ found: boolean; evidence: Record<string, any>; topic: string }> {
  const effectiveUserId = resolveUserId(userId)
  const response: any = await request.get(`/user/research-profile/${effectiveUserId}/topic-evidence`, { params: { topic } })
  return {
    found: Boolean(response?.found),
    evidence: response?.evidence || {},
    topic: response?.topic || topic
  }
}

export async function activateUserResearchProfileSnapshot(snapshotId: string, userId?: string): Promise<UserResearchProfile> {
  const effectiveUserId = resolveUserId(userId)
  const response: any = await request.post('/user/research-profile/snapshots/activate', {
    user_id: effectiveUserId,
    snapshot_id: snapshotId
  })
  return normalizeResearchProfile(response?.profile || response, effectiveUserId)
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

  const response: any = await request.get('/papers/recommendations', { params })
  const items = Array.isArray(response?.items)
    ? response.items.map(normalizeRecommendedPaper)
    : []
  return {
    total: Number(response?.total ?? items.length),
    items
  }
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

export async function getStats(userId?: string): Promise<DashboardStats> {
  if (isMockMode) {
    const { recommendedPapers, ...stats } = mockStats
    return normalizeDashboardStats(stats)
  }

  const effectiveUserId = resolveUserId(userId)
  const response = await request.get('/stats', {
    params: {
      user_id: effectiveUserId
    }
  })
  return normalizeDashboardStats(response)
}

export async function searchArxiv(params: ArxivSearchParams): Promise<ArxivSearchResult> {
  const defaultParams: ArxivSearchParams = {
    submitted_days_ago: 30,
    ...params
  }
  const response: any = await request.post('/arxiv/search', defaultParams)
  const papers = response.papers || response.items || []
  const items = papers.map(normalizePaper)
  const maxResults = defaultParams.max_results || 10
  const actualTotal = Math.min(response.total_results || items.length, maxResults)
  // 本地 OAI 搜索会返回能力边界契约；API 层必须保留它，避免页面误把语法受限解释为“无结果”。
  return {
    total: actualTotal,
    items,
    source: response.source || null,
    queryCapability: response.query_capability || null,
    warnings: Array.isArray(response.warnings) ? response.warnings : []
  }
}

export async function generateInterestVector(userId?: string): Promise<InterestVectorResult> {
  return request.post('/user/generate-interest-vector', { user_id: resolveUserId(userId) })
}

export async function getInterestVector(userId?: string): Promise<InterestVector> {
  return request.get('/user/interest-vector', {
    params: { user_id: resolveUserId(userId) }
  })
}

export async function recommendPapers(topN: number = 10, maxAgeMonths: number = 6, userId?: string): Promise<RecommendationResult> {
  const effectiveUserId = resolveUserId(userId)
  const response: any = await request.post('/user/recommend-papers', {
    user_id: effectiveUserId,
    top_n: topN,
    max_age_months: maxAgeMonths
  })
  return {
    ...response,
    research_profile: response?.research_profile ? normalizeResearchProfile(response.research_profile, effectiveUserId) : null,
    paper_actions: normalizePaperActionMap(response?.paper_actions),
    recommendations: Array.isArray(response?.recommendations)
      ? response.recommendations.map(normalizeRecommendedPaper)
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
  stage_message?: string | null
  progress?: number
  attempt_no?: number | null
  max_attempts?: number | null
  failure_code?: string | null
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
  stage_message?: string | null
  progress?: number | null
  attempt_no?: number | null
  max_attempts?: number | null
  failure_code?: string | null
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

export type QaStreamPersistenceStatus = 'unknown' | 'saved' | 'failed' | 'not_saved'
export type ResearchOutcome = 'completed' | 'partial' | 'abstained'

export interface ResearchSummary {
  research_run_id: string
  outcome: ResearchOutcome
  termination_reason: string
  retrieval_count: number
  draft_attempt_count: number
  verification_count: number
  confirmed_need_count: number
  satisfied_need_count: number
  blocked_need_count: number
  supported_claim_count: number
  removed_claim_count?: number
  citation_repair_count: number
  unresolved_topics: string[]
}

export interface ResearchCitation {
  source_id: string
  content: string
  claim_ids: string[]
  chunk_type?: string
  section_path?: string
  page_number?: number | null
}

export interface QaUsage {
  llm_calls?: number | null
  input_tokens?: number | null
  output_tokens?: number | null
  total_tokens?: number | null
  usage_reported_calls?: number
  models?: string[]
  task_calls?: Record<string, number>
}

export type QaStreamClientStatus =
  | 'completed'
  | 'partial'
  | 'failed'
  | 'aborted'
  | 'interrupted'
  | 'persistence_failed'
  | string

export interface QaResult {
  status: string
  // status 表示执行状态，outcome 表示证据完整程度；正常拒答同样是一次成功执行。
  outcome?: ResearchOutcome
  research_summary?: ResearchSummary | null
  citations?: ResearchCitation[]
  usage?: QaUsage | null
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
  cited_source_ids?: string[]
  citation_debug?: Record<string, any> | null
  citation_warning?: string | null
  partial?: boolean
  completed_at?: string | null
  interrupted_reason?: string | null
  persistence_status?: QaStreamPersistenceStatus
  retrieval_debug?: RetrievalDebug | null
  qa_observation?: QaObservation | null
  sources: Array<{
    source_id: string
    content: string
    page_number: string
    source?: string
    subchunk_label?: string
    section_path?: string
    parent_chunk_id?: string | number
    chunk_type?: string
    context_role?: string
    context_budget_score?: number
    context_budget_reason?: string
    expansion_source_anchor_ids?: Array<string | number>
    relationship_types?: string[]
    expansion_reasons?: string[]
    final_context_reason?: string
    asset_summary?: string
    asset_preview_text?: string
    asset_url?: string
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
    source_id?: string | number
    content?: string
    page_number?: string
    source?: string
    section_path?: string
    parent_chunk_id?: string | number
    chunk_type?: string
    context_role?: string
    context_budget_score?: number
    context_budget_reason?: string
    expansion_source_anchor_ids?: Array<string | number>
    relationship_types?: string[]
    expansion_reasons?: string[]
    final_context_reason?: string
    asset_summary?: string
    asset_preview_text?: string
    asset_url?: string
  }>
  retrieval_debug_snapshot?: RetrievalDebug | null
  contextualized_question?: string
  question_contextualization?: Record<string, any> | null
  cited_source_ids?: string[]
  citation_warning?: string | null
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
  const effectiveUserId = resolveUserId(params.user_id)
  const response: any = await request.get(`/paper/${arxivId}/notes`, {
    params: {
      ...params,
      user_id: effectiveUserId
    }
  })
  return {
    items: Array.isArray(response?.items) ? response.items.map((item: any) => normalizePaperNote(item, effectiveUserId)) : []
  }
}

export async function createPaperNote(
  arxivId: string,
  payload: PaperNotePayload
): Promise<{ item: PaperNote | null }> {
  const effectiveUserId = resolveUserId(payload.user_id)
  const response: any = await request.post(`/paper/${arxivId}/notes`, {
    ...payload,
    user_id: effectiveUserId
  })
  return {
    item: response?.item ? normalizePaperNote(response.item, effectiveUserId) : null
  }
}

export async function updatePaperNote(
  arxivId: string,
  noteId: string,
  payload: Partial<PaperNotePayload>
): Promise<{ item: PaperNote | null }> {
  const effectiveUserId = resolveUserId(payload.user_id)
  const response: any = await request.patch(`/paper/${arxivId}/notes/${noteId}`, {
    ...payload,
    user_id: effectiveUserId
  })
  return {
    item: response?.item ? normalizePaperNote(response.item, effectiveUserId) : null
  }
}

export async function deletePaperNote(
  arxivId: string,
  noteId: string,
  userId?: string
): Promise<{ status: string; deleted: boolean }> {
  return request.delete(`/paper/${arxivId}/notes/${noteId}`, {
    params: { user_id: resolveUserId(userId) }
  })
}

export function getPaperNotesExportUrl(arxivId: string, userId?: string): string {
  const params = new URLSearchParams()
  params.set('user_id', resolveUserId(userId))
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
    context_role?: string
    context_budget_score?: number
    context_budget_reason?: string
    expansion_source_anchor_ids?: Array<string | number>
    relationship_types?: string[]
    expansion_reasons?: string[]
    final_context_reason?: string
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
  enable_context_expansion?: boolean
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
  qa_observation?: QaObservation
  trace_export?: Record<string, string>
}

export interface QaObservationStageStatus {
  enabled: boolean | 'unknown'
  status: string
  fallback: boolean | 'unknown'
  reason: string
}

export interface QaObservation {
  schema_version: string
  outcome?: ResearchOutcome
  research_summary?: ResearchSummary | null
  answer_quality?: string
  answer_quality_reason?: string
  retrieval_quality: 'good' | 'partial' | 'weak' | 'failed' | 'unknown' | string
  retrieval_stage_status: Record<string, QaObservationStageStatus>
  missing_evidence_type: string
  weak_source_reason: string
  rerank_failed_reason: string
  answer_insufficient_evidence: 'yes' | 'no' | 'unknown' | string
  recommended_repair_actions: string[]
  source_count?: number
  error_code?: string
  error_stage?: string
  observation_reason?: string
}

export async function qaPaper(arxivId: string, question: string, options: QaRequestOptions = {}): Promise<QaResult> {
  return request.post(`/paper/${arxivId}/qa`, {
    question,
    ...options,
    user_id: resolveUserId(options.user_id)
  })
}

export async function listPaperChatSessions(
  arxivId: string,
  params: { user_id?: string; limit?: number } = {}
): Promise<{ items: PaperChatSession[] }> {
  return request.get(`/paper/${arxivId}/chat-sessions`, {
    params: {
      ...params,
      user_id: resolveUserId(params.user_id)
    }
  })
}

export async function getRecentPaperChatSession(
  arxivId: string,
  userId?: string
): Promise<{ item: PaperChatSession | null }> {
  return request.get(`/paper/${arxivId}/chat-sessions/recent`, {
    params: { user_id: resolveUserId(userId) }
  })
}

export async function createPaperChatSession(
  arxivId: string,
  payload: { user_id?: string; title?: string } = {}
): Promise<{ item: PaperChatSession | null }> {
  return request.post(`/paper/${arxivId}/chat-sessions`, {
    ...payload,
    user_id: resolveUserId(payload.user_id)
  })
}

export async function getPaperChatSession(
  arxivId: string,
  sessionId: string,
  userId?: string
): Promise<{ item: PaperChatSession | null }> {
  return request.get(`/paper/${arxivId}/chat-sessions/${sessionId}`, {
    params: { user_id: resolveUserId(userId) }
  })
}

export async function getPaperChatMessages(
  arxivId: string,
  sessionId: string,
  userId?: string
): Promise<{ session: PaperChatSession | null; items: PaperChatMessage[] }> {
  return request.get(`/paper/${arxivId}/chat-sessions/${sessionId}/messages`, {
    params: { user_id: resolveUserId(userId) }
  })
}

export async function clearPaperChatSession(
  arxivId: string,
  sessionId: string,
  userId?: string
): Promise<{ item: PaperChatSession | null }> {
  return request.post(`/paper/${arxivId}/chat-sessions/${sessionId}/clear`, {
    user_id: resolveUserId(userId)
  })
}

export async function deletePaperChatSession(
  arxivId: string,
  sessionId: string,
  userId?: string
): Promise<{ status: string; deleted: boolean }> {
  return request.delete(`/paper/${arxivId}/chat-sessions/${sessionId}`, {
    params: { user_id: resolveUserId(userId) }
  })
}

type QaStreamSource = {
  source_id: string
  content: string
  page_number: string
  source?: string
  section_path?: string
  parent_chunk_id?: string | number
  chunk_type?: string
  asset_summary?: string
  asset_preview_text?: string
  asset_url?: string
}

type QaStreamMetaPayload = {
  status: string
  outcome?: ResearchOutcome
  research_summary?: ResearchSummary | null
  citations?: ResearchCitation[]
  arxiv_id: string
  question: string
  session_id?: string
  chat_session?: PaperChatSession | null
  original_question?: string
  contextualized_question?: string
  used_short_term_memory?: boolean
  question_contextualization?: Record<string, any> | null
  sources: QaStreamSource[]
  retrieval_debug?: RetrievalDebug | null
  qa_observation?: QaObservation | null
  cited_source_ids?: string[]
  citation_debug?: Record<string, any> | null
  citation_warning?: string | null
}

type QaStreamDonePayload = QaStreamMetaPayload & {
  answer: string
  turn_id?: string
  completed_at?: string | null
  completedAt?: string | null
  interrupted_reason?: string | null
  interruptedReason?: string | null
  persistence_status?: QaStreamPersistenceStatus
  persistenceStatus?: QaStreamPersistenceStatus
  usage?: QaUsage | null
}

export interface QaStreamProgressPayload {
  stage: 'retrieval' | 'draft' | 'verification' | 'completed'
  retrieval_count?: number
  new_candidate_count?: number
  draft_attempt?: number
  supported_count?: number
  outcome?: ResearchOutcome
}

export interface QaStreamHandlers {
  onMeta?: (meta: QaStreamMetaPayload) => void
  onProgress?: (progress: QaStreamProgressPayload) => void
  onDelta?: (delta: string) => void
  onDone?: (payload: QaStreamDonePayload) => void
  onError?: (detail: string) => void
  signal?: AbortSignal
}

function normalizeStreamPersistenceStatus(payload: Record<string, any> | null | undefined): QaStreamPersistenceStatus {
  const rawStatus = String(payload?.persistence_status || payload?.persistenceStatus || '').trim().toLowerCase()
  const doneStatus = String(payload?.status || '').trim().toLowerCase()

  if (rawStatus === 'failed' || rawStatus === 'not_saved' || rawStatus === 'saved' || rawStatus === 'unknown') {
    return rawStatus as QaStreamPersistenceStatus
  }
  if (['partial_success', 'persistence_failed', 'database_write_failed'].includes(doneStatus)) {
    return 'failed'
  }
  if (doneStatus === 'stream_interrupted') {
    return 'not_saved'
  }
  if (doneStatus === 'success' || doneStatus === 'completed') {
    return 'saved'
  }
  return 'unknown'
}

function normalizeStreamDoneStatus(payload: Record<string, any> | null | undefined): QaStreamClientStatus {
  const status = String(payload?.status || 'success').trim().toLowerCase()
  const persistenceStatus = normalizeStreamPersistenceStatus(payload)

  if (['partial_success', 'persistence_failed', 'database_write_failed'].includes(status) || persistenceStatus === 'failed') {
    return 'persistence_failed'
  }
  if (status === 'stream_interrupted') return 'partial'
  if (status === 'aborted') return 'aborted'
  if (status === 'failed') return 'failed'
  return 'completed'
}

function isAbortError(error: unknown, signal?: AbortSignal) {
  return Boolean(
    signal?.aborted ||
      (error instanceof DOMException && error.name === 'AbortError') ||
      (error && typeof error === 'object' && (error as { name?: unknown }).name === 'AbortError')
  )
}

function createQaStreamError(code: string, message: string, detail: string | null = null, recoverable = true) {
  return new ApiError({
    status: 'failed',
    code,
    message,
    detail,
    recoverable
  })
}

function applyQaStreamPayload(
  payload: Partial<QaStreamMetaPayload> & Record<string, any>,
  target: {
    sources: QaStreamSource[]
    retrievalDebug: RetrievalDebug | null
    qaObservation: QaObservation | null
    sessionId: string
    chatSession: PaperChatSession | null
    originalQuestion: string
    contextualizedQuestion: string
    usedShortTermMemory: boolean
    questionContextualization: Record<string, any> | null
    citedSourceIds: string[]
    citationWarning: string | null
  }
) {
  if (Array.isArray(payload.sources)) {
    target.sources = payload.sources
  }
  if (payload.retrieval_debug) {
    target.retrievalDebug = payload.retrieval_debug
  }
  if (payload.qa_observation) {
    target.qaObservation = payload.qa_observation
  }
  if (typeof payload.session_id === 'string' && payload.session_id) {
    target.sessionId = payload.session_id
  }
  if (payload.chat_session) {
    target.chatSession = payload.chat_session
  }
  if (typeof payload.original_question === 'string' && payload.original_question) {
    target.originalQuestion = payload.original_question
  }
  if (typeof payload.contextualized_question === 'string' && payload.contextualized_question) {
    target.contextualizedQuestion = payload.contextualized_question
  }
  if (typeof payload.used_short_term_memory === 'boolean') {
    target.usedShortTermMemory = payload.used_short_term_memory
  }
  if (payload.question_contextualization) {
    target.questionContextualization = payload.question_contextualization
  }
  if (Array.isArray(payload.cited_source_ids)) {
    target.citedSourceIds = payload.cited_source_ids.map(value => String(value))
  }
  if (typeof payload.citation_warning === 'string') {
    target.citationWarning = payload.citation_warning
  }
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
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let finalAnswer = ''
  let hasDone = false
  const finalState = {
    sources: [] as QaStreamSource[],
    retrievalDebug: null as RetrievalDebug | null,
    qaObservation: null as QaObservation | null,
    sessionId: options.session_id || '',
    chatSession: null as PaperChatSession | null,
    originalQuestion: question,
    contextualizedQuestion: question,
    usedShortTermMemory: false,
    questionContextualization: null as Record<string, any> | null,
    citedSourceIds: [] as string[],
    citationWarning: null as string | null
  }

  const buildResult = (payload: QaStreamDonePayload): QaResult => {
    const status = normalizeStreamDoneStatus(payload)
    const persistenceStatus = normalizeStreamPersistenceStatus(payload)
    const interruptedReason = payload.interrupted_reason || payload.interruptedReason || null
    const completedAt = payload.completed_at || payload.completedAt || new Date().toISOString()

    return {
      status,
      outcome: payload.outcome,
      research_summary: payload.research_summary,
      citations: payload.citations,
      usage: payload.usage,
      arxiv_id: arxivId,
      question,
      session_id: finalState.sessionId || undefined,
      chat_session: finalState.chatSession,
      turn_id: payload.turn_id,
      original_question: finalState.originalQuestion,
      contextualized_question: finalState.contextualizedQuestion,
      used_short_term_memory: finalState.usedShortTermMemory,
      question_contextualization: finalState.questionContextualization,
      answer: finalAnswer,
      partial: status !== 'completed',
      completed_at: status === 'completed' ? completedAt : null,
      interrupted_reason: interruptedReason,
      persistence_status: persistenceStatus,
      sources: finalState.sources,
      cited_source_ids: finalState.citedSourceIds,
      citation_warning: finalState.citationWarning,
      qa_observation: finalState.qaObservation,
      retrieval_debug: finalState.retrievalDebug
    }
  }

  const processEvent = (part: string): QaResult | null => {
    const parsed = parseSseEvent(part)
    if (!parsed) return null

    if (parsed.event === 'meta' && parsed.data && typeof parsed.data === 'object') {
      handlers.onMeta?.(parsed.data)
      applyQaStreamPayload(parsed.data, finalState)
      return null
    }

    if (parsed.event === 'delta') {
      const delta = typeof parsed.data?.delta === 'string' ? parsed.data.delta : ''
      if (!delta) return null
      finalAnswer += delta
      handlers.onDelta?.(delta)
      return null
    }

    if (parsed.event === 'progress' && parsed.data && typeof parsed.data === 'object') {
      // 研究引擎先完成校验，再通过 done 交付最终答案；阶段进度不应混入可见答案。
      handlers.onProgress?.(parsed.data as QaStreamProgressPayload)
      return null
    }

    if (parsed.event === 'error' && parsed.data) {
      const apiError = normalizeApiError(parsed.data, '问答失败，请稍后重试。')
      handlers.onError?.(apiError.message)
      // 后端 error event 是业务失败终态，必须立即抛出，避免 reader 结束被误判成 partial。
      throw new ApiError(apiError)
    }

    if (parsed.event === 'done' && parsed.data && typeof parsed.data === 'object') {
      hasDone = true
      const payload = parsed.data as QaStreamDonePayload
      if (typeof payload.answer === 'string') {
        finalAnswer = payload.answer
      }
      applyQaStreamPayload(payload, finalState)
      handlers.onDone?.(payload)
      return buildResult(payload)
    }

    // malformed 或未知 SSE 事件只跳过，避免调试噪声打断页面；关键 error/done 分支仍会被严格处理。
    return null
  }

  try {
    const response = await apiFetch(`/paper/${arxivId}/qa/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream'
      },
      body: JSON.stringify({
        question,
        ...options,
        user_id: resolveUserId(options.user_id)
      }),
      signal: handlers.signal
    })

    if (!response.ok) {
      throw await parseFetchErrorResponse(response, '问答失败，请稍后重试。')
    }

    if (!response.body) {
      throw createQaStreamError('stream_incomplete', '回答中断，请重试。', 'Streaming response body is empty')
    }

    reader = response.body.getReader()

    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      const parts = buffer.split(/\r?\n\r?\n/)
      buffer = parts.pop() || ''

      for (const part of parts) {
        const result = processEvent(part)
        if (result) return result
      }
    }

    const tail = `${buffer}${decoder.decode()}`
    if (tail.trim()) {
      const result = processEvent(tail)
      if (result) return result
    }

    // 自然结束但没有 done 只能说明流不完整，不能构造 success/completed。
    if (!hasDone) {
      throw createQaStreamError(
        'stream_incomplete',
        '回答中断，请重试。',
        finalAnswer ? 'Stream ended before done event after partial answer.' : 'Stream ended before done event.'
      )
    }
  } catch (error) {
    if (isAbortError(error, handlers.signal)) {
      const abortedError = createQaStreamError('aborted', '已取消生成', 'AbortController aborted the QA stream.', true)
      handlers.onError?.(abortedError.message)
      throw abortedError
    }
    throw error
  } finally {
    reader?.releaseLock()
  }

  throw createQaStreamError('stream_incomplete', '回答中断，请重试。', 'Stream finished without a completed result.')
}
