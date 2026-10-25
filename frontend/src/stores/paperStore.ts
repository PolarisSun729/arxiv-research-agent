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
  UserPaperActionMap,
  UserResearchProfile,
  UserResearchProfileDetail,
  UserProfileBuildJob
} from '@/types/paper'
import type { NormalizedArxivQueryCapability } from '@/types/arxivCapability'
import {
  getPaperById,
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
  removePaperAction
} from '@/api/papers'
import { useUserContext } from '@/composables/useUserContext'
import {
  normalizeArxivQueryCapability,
  normalizeArxivQueryCapabilityFromError
} from '@/utils/arxivQueryCapability'
import { resolveSearchFetchSize } from '@/utils/searchResults'

export const usePaperStore = defineStore('paper', () => {
  const userContext = useUserContext()
  // 搜索页拿到的完整结果（可能为过滤已标记论文而多拉），展示哪些、怎么分页由页面决定。
  const allPapers = ref<Paper[]>([])
  // 搜索发起时已标记（感兴趣 / 不感兴趣 / 已读）的论文快照；搜索后新标记的不算，避免卡片点完就消失。
  const searchMarkedIds = ref<string[]>([])
  const currentPaper = ref<Paper | null>(null)
  const recommendations = ref<RecommendedPaper[]>([])
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
  const arxivSearchCapability = ref<NormalizedArxivQueryCapability | null>(null)
  const arxivSearchError = ref<string | null>(null)

  function resetUserScopedState() {
    // userId 切换后清理本地用户态缓存，避免偏好、画像和推荐结果短暂串到新用户视图。
    recommendations.value = []
    labeledPapers.value = []
    totalLabeledPapers.value = 0
    researchProfile.value = null
    researchProfileDetail.value = null
    latestProfileBuildJob.value = null
    paperActionMap.value = {}
    // 搜索结果上的标记和过滤快照都属于上一个用户，切换后需重新搜索。
    allPapers.value = []
    searchMarkedIds.value = []
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
    applyPaperActions(allPapers.value)
    applyPaperActions(recommendations.value)
    applyPaperActions(labeledPapers.value)
    if (currentPaper.value) {
      currentPaper.value.paperActions = buildPaperActions(currentPaper.value.arxivId || currentPaper.value.id)
    }
  }

  async function fetchPaperById(id: string) {
    loading.value = true
    try {
      // 详情接口不带用户偏好，这里同时取偏好表，让详情页的感兴趣/不感兴趣与列表卡片显示一致。
      const [paper, preferences] = await Promise.all([getPaperById(id), getUserPreferences().catch(() => null)])
      currentPaper.value = paper
      if (preferences) {
        // 偏好接口已带当前研究画像，详情页据此读取回答偏好，不再单独请求画像接口。
        researchProfile.value = preferences.research_profile || researchProfile.value
      }
      if (currentPaper.value) {
        if (preferences) {
          applyPreferenceLabels([currentPaper.value], preferences.liked_papers || [], preferences.disliked_papers || [])
        }
        currentPaper.value.paperActions = buildPaperActions(currentPaper.value.arxivId || currentPaper.value.id)
      }
      return currentPaper.value
    } finally {
      loading.value = false
    }
  }

  async function fetchLabeledPapers(params: {
    label?: 'liked' | 'disliked'
    cluster?: string
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
    // 详情页的论文只在 currentPaper 里，也要能被找到，否则详情页无法标记偏好。
    const paper = [...recommendations.value, ...allPapers.value, ...(currentPaper.value ? [currentPaper.value] : [])]
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

    if (currentPaper.value?.id === id) {
      currentPaper.value.label = label
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

  async function fetchArxivPapers(params: ArxivSearchParams) {
    loading.value = true
    arxivSearchCapability.value = null
    arxivSearchError.value = null
    try {
      // 先取偏好：要根据已标记数量决定多拉多少。偏好失败不影响搜索本身，只是不做多拉和过滤。
      const preferences = await getUserPreferences().catch(() => null)
      const likedIds = preferences?.liked_papers || []
      const dislikedIds = preferences?.disliked_papers || []
      if (preferences) {
        paperActionMap.value = preferences.paper_actions || {}
        researchProfile.value = preferences.research_profile || researchProfile.value
      }
      const markedIds = Array.from(new Set([...likedIds, ...dislikedIds, ...(paperActionMap.value.read || [])]))
      const maxResults = params.max_results || 10
      const result = await searchArxiv({
        ...params,
        max_results: resolveSearchFetchSize(maxResults, markedIds.length)
      })
      arxivSearchCapability.value = normalizeArxivQueryCapability({
        capability: result.queryCapability,
        warnings: result.warnings
      })
      allPapers.value = [...result.items]
      searchMarkedIds.value = markedIds
      applyPreferenceLabels(allPapers.value, likedIds, dislikedIds)
      applyPaperActions(allPapers.value)
    } catch (error) {
      const capabilityFromError = normalizeArxivQueryCapabilityFromError(error)
      if (capabilityFromError) {
        // 本地查询契约错误是可解释的业务失败：页面展示能力边界，不再伪装成普通“无结果”。
        arxivSearchCapability.value = capabilityFromError
        arxivSearchError.value = capabilityFromError.errorMessage || capabilityFromError.summary
        allPapers.value = []
        searchMarkedIds.value = []
        return
      }
      throw error
    } finally {
      loading.value = false
    }
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
      }
      return result
    } finally {
      recommendationsGenerating.value = false
    }
  }

  return {
    allPapers,
    searchMarkedIds,
    currentPaper,
    recommendations,
    labeledPapers,
    totalLabeledPapers,
    stats,
    loading,
    interestVectorGenerating,
    lastInterestVector,
    researchProfile,
    researchProfileDetail,
    latestProfileBuildJob,
    arxivSearchCapability,
    arxivSearchError,
    recommendationsGenerating,
    paperActionMap,
    fetchPaperById,
    fetchLabeledPapers,
    updateLabel,
    fetchStats,
    fetchArxivPapers,
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
    generateRecommendations
  }
})
