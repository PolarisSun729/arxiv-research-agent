import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import type {
  Paper,
  RecommendedPaper,
  LabeledPaper,
  ArxivSearchParams,
  InterestVector,
  InterestVectorResult,
  RecommendationResult,
  PaperActionType,
  PaperNote,
  PaperNoteType,
  UserPaperActionMap,
  UserResearchProfile,
  UserResearchProfileDetail,
  UserProfileBuildJob
} from '@/types/paper'
import type { NormalizedArxivQueryCapability } from '@/types/arxivCapability'
import {
  searchPapers,
  getPaperById,
  getRecommendations,
  getLabeledPapers,
  getStats,
  type DashboardStats,
  searchArxiv,
  getUserPreferences,
  likePaper,
  dislikePaper,
  removePaperPreference,
  generateInterestVector,
  getInterestVector,
  recommendPapers,
  getUserResearchProfile,
  getUserResearchProfileDetail,
  patchUserResearchProfile,
  rebuildUserResearchProfile,
  getUserResearchProfileBuildJob,
  activateUserResearchProfileSnapshot,
  getUserPaperActions,
  recordPaperAction,
  removePaperAction,
  listPaperNotes,
  createPaperNote,
  updatePaperNote,
  deletePaperNote,
  getPaperNotesExportUrl
} from '@/api/papers'
import { useUserContext } from '@/composables/useUserContext'
import {
  normalizeArxivQueryCapability,
  normalizeArxivQueryCapabilityFromError
} from '@/utils/arxivQueryCapability'

export const usePaperStore = defineStore('paper', () => {
  const userContext = useUserContext()
  const papers = ref<Paper[]>([])
  const allPapers = ref<Paper[]>([])
  const totalPapers = ref(0)
  const currentPaper = ref<Paper | null>(null)
  const recommendations = ref<RecommendedPaper[]>([])
  const totalRecommendations = ref(0)
  const labeledPapers = ref<LabeledPaper[]>([])
  const totalLabeledPapers = ref(0)
  const stats = ref<DashboardStats>({
    totalPapers: 0,
    labeledPapers: 0,
    latestSyncNewPapers: 0,
    lastSyncedDate: null,
    lastSyncRunAt: null,
    lastSyncStatus: 'unknown',
    lastSyncMode: 'sync',
    latestSyncMatchedPapers: 0,
    syncErrors: 0,
    syncErrorMessage: null
  })
  const loading = ref(false)
  const interestVectorGenerating = ref(false)
  const lastInterestVector = ref<InterestVector | null>(null)
  const researchProfileDetail = ref<UserResearchProfileDetail | null>(null)
  const latestProfileBuildJob = ref<UserProfileBuildJob | null>(null)
  const researchProfile = ref<UserResearchProfile | null>(null)
  const paperActionMap = ref<UserPaperActionMap>({})
  const paperNotes = ref<PaperNote[]>([])
  const arxivSearchCapability = ref<NormalizedArxivQueryCapability | null>(null)
  const arxivSearchError = ref<string | null>(null)

  function resetUserScopedState() {
    // userId 切换后清理本地用户态缓存，避免偏好、画像、笔记和推荐结果短暂串到新用户视图。
    recommendations.value = []
    totalRecommendations.value = 0
    labeledPapers.value = []
    totalLabeledPapers.value = 0
    researchProfile.value = null
    researchProfileDetail.value = null
    latestProfileBuildJob.value = null
    paperActionMap.value = {}
    paperNotes.value = []
    lastInterestVector.value = null
    arxivSearchCapability.value = null
    arxivSearchError.value = null
    if (currentPaper.value) {
      currentPaper.value.label = null
      currentPaper.value.paperActions = {}
    }
    syncPaperCollections()
  }

  watch(() => userContext.userId.value, resetUserScopedState)

  function buildRecommendationReason(item: any): string {
    const breakdown = item?.score_breakdown || item?.scoreBreakdown || {}
    if (Array.isArray(item?.profile_reasons) && item.profile_reasons.length > 0) {
      return item.profile_reasons.slice(0, 2).join(' · ')
    }
    const parts: string[] = []
    if (typeof breakdown.semantic_score === 'number' && breakdown.semantic_score > 0) {
      parts.push(`语义相似 ${Math.round(breakdown.semantic_score * 100)}%`)
    }
    if (typeof breakdown.category_score === 'number' && breakdown.category_score > 0) {
      parts.push(`分类匹配 ${Math.round(breakdown.category_score * 100)}%`)
    }
    if (typeof breakdown.recency_score === 'number' && breakdown.recency_score > 0) {
      parts.push(`较新论文 ${Math.round(breakdown.recency_score * 100)}%`)
    }
    return parts.length ? parts.join(' · ') : '基于用户兴趣画像生成'
  }

  function toRecallClusterHits(value: any): Array<{ cluster_id?: string | null; similarity?: number; rank?: number | null }> {
    if (!Array.isArray(value)) return []
    return value
      .map((item: any) => ({
        cluster_id: item?.cluster_id || item?.clusterId || null,
        similarity: typeof item?.similarity === 'number' ? item.similarity : undefined,
        rank: typeof item?.rank === 'number' ? item.rank : null
      }))
      .filter((item: any) => item.cluster_id || typeof item.similarity === 'number' || typeof item.rank === 'number')
  }

  function applyPreferenceLabels(targetPapers: Paper[], likedIds: string[], dislikedIds: string[]) {
    targetPapers.forEach(paper => {
      const arxivId = paper.arxivId || paper.id
      if (likedIds.includes(arxivId)) {
        paper.label = 'liked'
      } else if (dislikedIds.includes(arxivId)) {
        paper.label = 'disliked'
      } else {
        paper.label = null
      }
    })
  }

  function buildPaperActions(arxivId: string): Partial<Record<PaperActionType, boolean>> {
    const normalizedId = String(arxivId || '')
    const result: Partial<Record<PaperActionType, boolean>> = {}
    Object.entries(paperActionMap.value || {}).forEach(([actionType, ids]) => {
      if (Array.isArray(ids) && ids.includes(normalizedId)) {
        result[actionType as PaperActionType] = true
      }
    })
    return result
  }

  function applyPaperActions(targetPapers: Paper[]) {
    targetPapers.forEach(paper => {
      const arxivId = paper.arxivId || paper.id
      paper.paperActions = buildPaperActions(arxivId)
    })
  }

  function syncPaperCollections() {
    applyPaperActions(papers.value)
    applyPaperActions(allPapers.value)
    applyPaperActions(recommendations.value)
    applyPaperActions(labeledPapers.value)
    if (currentPaper.value) {
      currentPaper.value.paperActions = buildPaperActions(currentPaper.value.arxivId || currentPaper.value.id)
    }
  }

  async function fetchPapers(params: {
    keyword?: string
    category?: string
    page: number
    pageSize: number
    sortBy?: string
  }) {
    loading.value = true
    try {
      const result = await searchPapers(params)
      papers.value = result.items
      totalPapers.value = result.total
    } finally {
      loading.value = false
    }
  }

  async function fetchPaperById(id: string) {
    loading.value = true
    try {
      currentPaper.value = await getPaperById(id)
      if (currentPaper.value) {
        currentPaper.value.paperActions = buildPaperActions(currentPaper.value.arxivId || currentPaper.value.id)
      }
      return currentPaper.value
    } finally {
      loading.value = false
    }
  }

  async function fetchRecommendations(params: { page: number; pageSize: number }) {
    loading.value = true
    try {
      const result = await getRecommendations(params)
      recommendations.value = result.items.map((p: any) => ({
        ...p,
        summary: p.summary || '',
        label: p.label || null
      }))
      applyPaperActions(recommendations.value)
      totalRecommendations.value = result.total
    } finally {
      loading.value = false
    }
  }

  async function fetchLabeledPapers(params: {
    label?: 'liked' | 'disliked'
    page: number
    pageSize: number
  }) {
    loading.value = true
    try {
      const result = await getLabeledPapers(params)
      labeledPapers.value = result.items
      totalLabeledPapers.value = result.total
    } finally {
      loading.value = false
    }
  }

  async function updateLabel(id: string, label: 'liked' | 'disliked' | null) {
    const paper = [...papers.value, ...recommendations.value, ...allPapers.value]
      .find(item => item.id === id || item.arxivId === id)
    if (!paper) {
      throw new Error('Paper not found')
    }

    if (label === null) {
      if (paper.label) {
        await removePaperPreference(paper, paper.label)
      }
    } else if (paper.label === label) {
      await removePaperPreference(paper, label)
      label = null
    } else if (label === 'liked') {
      await likePaper(paper)
    } else {
      await dislikePaper(paper)
    }

    if (label === 'liked' || label === 'disliked') {
      const targetPaperId = paper.arxivId || paper.id
      // 显式偏好不再写入 paperActionMap；这里只同步清理与强偏好冲突的弱负反馈。
      paperActionMap.value.not_interested = (paperActionMap.value.not_interested || []).filter((item: string) => item !== targetPaperId)
    }

    if (currentPaper.value?.id === id) {
      currentPaper.value.label = label
    }
    const paperIndex = papers.value.findIndex(p => p.id === id)
    if (paperIndex !== -1) {
      papers.value[paperIndex].label = label
    }
    const recIndex = recommendations.value.findIndex(r => r.id === id)
    if (recIndex !== -1) {
      recommendations.value[recIndex].label = label
    }
    const allPaperIndex = allPapers.value.findIndex(p => p.id === id)
    if (allPaperIndex !== -1) {
      allPapers.value[allPaperIndex].label = label
    }
    syncPaperCollections()
  }

  async function fetchStats() {
    loading.value = true
    try {
      stats.value = await getStats()
    } finally {
      loading.value = false
    }
  }

  async function fetchArxivPapers(params: ArxivSearchParams, page: number = 1, pageSize: number = 10) {
    loading.value = true
    arxivSearchCapability.value = null
    arxivSearchError.value = null
    try {
      const result = await searchArxiv(params)
      arxivSearchCapability.value = normalizeArxivQueryCapability({
        capability: result.queryCapability,
        warnings: result.warnings
      })
      allPapers.value = [...result.items]
      const preferences = await getUserPreferences()
      paperActionMap.value = preferences.paper_actions || {}
      researchProfile.value = preferences.research_profile || researchProfile.value
      applyPreferenceLabels(allPapers.value, preferences.liked_papers || [], preferences.disliked_papers || [])
      applyPaperActions(allPapers.value)
      totalPapers.value = result.total
      const start = (page - 1) * pageSize
      const end = start + pageSize
      papers.value = allPapers.value.slice(start, end)
    } catch (error) {
      const capabilityFromError = normalizeArxivQueryCapabilityFromError(error)
      if (capabilityFromError) {
        // 本地查询契约错误是可解释的业务失败：页面展示能力边界，不再伪装成普通“无结果”。
        arxivSearchCapability.value = capabilityFromError
        arxivSearchError.value = capabilityFromError.errorMessage || capabilityFromError.summary
        allPapers.value = []
        papers.value = []
        totalPapers.value = 0
        return
      }
      throw error
    } finally {
      loading.value = false
    }
  }

  function paginatePapers(page: number, pageSize: number) {
    const start = (page - 1) * pageSize
    const end = start + pageSize
    papers.value = allPapers.value.slice(start, end)
  }

  async function generateUserInterestVector(): Promise<InterestVectorResult> {
    interestVectorGenerating.value = true
    try {
      const result = await generateInterestVector()
      if (result.status === 'success') {
        await fetchUserInterestVector()
      }
      return result
    } finally {
      interestVectorGenerating.value = false
    }
  }

  async function fetchUserInterestVector() {
    try {
      lastInterestVector.value = await getInterestVector()
    } catch (error) {
      lastInterestVector.value = null
    }
  }

  const recommendationsGenerating = ref(false)

  async function fetchResearchProfile() {
    researchProfile.value = await getUserResearchProfile()
    return researchProfile.value
  }

  async function fetchResearchProfileDetail() {
    researchProfileDetail.value = await getUserResearchProfileDetail()
    researchProfile.value = researchProfileDetail.value.effective_profile || researchProfile.value
    latestProfileBuildJob.value = researchProfileDetail.value.latest_build_job || null
    return researchProfileDetail.value
  }

  async function saveResearchProfile(profile: Partial<UserResearchProfile>) {
    researchProfile.value = await patchUserResearchProfile(profile)
    await fetchResearchProfileDetail()
    return researchProfile.value
  }

  async function rebuildResearchProfile() {
    const result = await rebuildUserResearchProfile(undefined, true)
    researchProfile.value = result.profile || researchProfile.value
    latestProfileBuildJob.value = result.job || null
    return result
  }

  async function pollProfileBuildJob(jobId: string) {
    latestProfileBuildJob.value = await getUserResearchProfileBuildJob(jobId)
    if (['completed', 'needs_review', 'failed'].includes(latestProfileBuildJob.value.status)) {
      await fetchResearchProfileDetail()
    }
    return latestProfileBuildJob.value
  }

  async function activateProfileSnapshot(snapshotId: string) {
    researchProfile.value = await activateUserResearchProfileSnapshot(snapshotId)
    await fetchResearchProfileDetail()
    return researchProfile.value
  }

  async function fetchPaperActions() {
    const result = await getUserPaperActions()
    paperActionMap.value = result.action_map || {}
    syncPaperCollections()
    return result
  }

  async function togglePaperAction(paper: Paper, actionType: PaperActionType, enabled?: boolean) {
    const arxivId = paper.arxivId || paper.id
    const exists = (paperActionMap.value[actionType] || []).includes(arxivId)
    const nextEnabled = typeof enabled === 'boolean' ? enabled : !exists
    if (nextEnabled) {
      await recordPaperAction(paper, actionType)
      paperActionMap.value[actionType] = Array.from(new Set([...(paperActionMap.value[actionType] || []), arxivId]))
    } else {
      await removePaperAction(paper, actionType)
      paperActionMap.value[actionType] = (paperActionMap.value[actionType] || []).filter(item => item !== arxivId)
    }
    syncPaperCollections()
  }

  async function generateRecommendations(topN: number = 10, maxAgeMonths: number = 6): Promise<RecommendationResult> {
    recommendationsGenerating.value = true
    try {
      const result = await recommendPapers(topN, maxAgeMonths)
      if (result.status === 'success') {
        researchProfile.value = result.research_profile || researchProfile.value
        paperActionMap.value = result.paper_actions || paperActionMap.value
        recommendations.value = result.recommendations.map((p: any) => ({
          ...p,
          summary: p.summary || '',
          reason: p.reason || buildRecommendationReason(p),
          recall_cluster_hits: toRecallClusterHits(p.recall_cluster_hits),
          label: null
        }))
        applyPaperActions(recommendations.value)
        totalRecommendations.value = result.total_found
      }
      return result
    } finally {
      recommendationsGenerating.value = false
    }
  }

  async function fetchPaperNotes(arxivId: string, noteType?: PaperNoteType) {
    const result = await listPaperNotes(arxivId, noteType ? { note_type: noteType } : {})
    paperNotes.value = result.items
    return result.items
  }

  async function savePaperNote(
    arxivId: string,
    payload: {
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
  ) {
    const result = await createPaperNote(arxivId, payload)
    if (result.item) {
      const next = [result.item, ...paperNotes.value.filter(item => item.note_id !== result.item?.note_id)]
      paperNotes.value = next
    }
    return result.item
  }

  async function editPaperNote(
    arxivId: string,
    noteId: string,
    payload: {
      title?: string
      content?: string
      note_type?: PaperNoteType
      source_chunk_ids?: string[]
      tags?: string[]
      include_in_profile?: boolean
    }
  ) {
    const result = await updatePaperNote(arxivId, noteId, payload)
    if (result.item) {
      paperNotes.value = paperNotes.value.map(item => item.note_id === noteId ? result.item as PaperNote : item)
    }
    return result.item
  }

  async function removeExistingPaperNote(arxivId: string, noteId: string) {
    const result = await deletePaperNote(arxivId, noteId)
    if (result.deleted) {
      paperNotes.value = paperNotes.value.filter(item => item.note_id !== noteId)
    }
    return result.deleted
  }

  function getPaperNotesExportLink(arxivId: string) {
    return getPaperNotesExportUrl(arxivId)
  }

  return {
    papers,
    allPapers,
    totalPapers,
    currentPaper,
    recommendations,
    totalRecommendations,
    labeledPapers,
    totalLabeledPapers,
    stats,
    loading,
    interestVectorGenerating,
    lastInterestVector,
    researchProfile,
    researchProfileDetail,
    latestProfileBuildJob,
    paperActionMap,
    paperNotes,
    arxivSearchCapability,
    arxivSearchError,
    recommendationsGenerating,
    fetchPapers,
    fetchPaperById,
    fetchRecommendations,
    fetchLabeledPapers,
    updateLabel,
    fetchStats,
    fetchArxivPapers,
    paginatePapers,
    generateUserInterestVector,
    fetchUserInterestVector,
    fetchResearchProfile,
    fetchResearchProfileDetail,
    saveResearchProfile,
    rebuildResearchProfile,
    pollProfileBuildJob,
    activateProfileSnapshot,
    fetchPaperActions,
    togglePaperAction,
    fetchPaperNotes,
    savePaperNote,
    editPaperNote,
    removeExistingPaperNote,
    getPaperNotesExportLink,
    generateRecommendations
  }
})
