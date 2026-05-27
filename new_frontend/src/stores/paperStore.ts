import { defineStore } from 'pinia'
import { ref } from 'vue'
import type {
  Paper,
  RecommendedPaper,
  LabeledPaper,
  ArxivSearchParams,
  InterestVector,
  InterestVectorResult,
  RecommendationResult
} from '@/types/paper'
import {
  searchPapers,
  getPaperById,
  getRecommendations,
  getLabeledPapers,
  getStats,
  searchArxiv,
  getUserPreferences,
  likePaper,
  dislikePaper,
  removePaperPreference,
  generateInterestVector,
  getInterestVector,
  recommendPapers
} from '@/api/papers'

export const usePaperStore = defineStore('paper', () => {
  const papers = ref<Paper[]>([])
  const allPapers = ref<Paper[]>([])
  const totalPapers = ref(0)
  const currentPaper = ref<Paper | null>(null)
  const recommendations = ref<RecommendedPaper[]>([])
  const totalRecommendations = ref(0)
  const labeledPapers = ref<LabeledPaper[]>([])
  const totalLabeledPapers = ref(0)
  const stats = ref({
    totalPapers: 0,
    labeledPapers: 0,
    todayNewPapers: 0,
    recommendedPapers: 0
  })
  const loading = ref(false)
  const interestVectorGenerating = ref(false)
  const lastInterestVector = ref<InterestVector | null>(null)

  function toStringArray(value: any): string[] {
    if (Array.isArray(value)) {
      return value.map((item: any) => String(item).trim()).filter((item: string) => item)
    }
    if (!value) return []
    return String(value)
      .replace(/[\[\]\(\)]/g, '')
      .split(/[,;]/)
      .map((item: string) => item.trim())
      .filter((item: string) => item)
  }

  function buildRecommendationReason(item: any): string {
    const breakdown = item?.score_breakdown || item?.scoreBreakdown || {}
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
        id: p.arxiv_id || p.id,
        arxivId: p.arxiv_id || p.id,
        title: p.title,
        authors: Array.isArray(p.authors) ? p.authors : (p.authors ? p.authors.split(',').map((a: string) => a.trim()).filter((a: string) => a) : []),
        summary: p.abstract || p.summary,
        publishedAt: p.published_date || p.publishedAt,
        categories: Array.isArray(p.categories) ? p.categories : (p.categories ? p.categories.split(',').map((c: string) => c.trim()).filter((c: string) => c) : []),
        pdfUrl: p.url || p.pdfUrl,
        absUrl: p.url || p.absUrl,
        similarityScore: typeof p.similarity === 'number' ? p.similarity : (p.similarityScore || 0),
        label: p.label || null
      }))
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
    try {
      const result = await searchArxiv(params)
      allPapers.value = [...result.items]
      const preferences = await getUserPreferences()
      applyPreferenceLabels(allPapers.value, preferences.liked_papers || [], preferences.disliked_papers || [])
      totalPapers.value = result.total
      const start = (page - 1) * pageSize
      const end = start + pageSize
      papers.value = allPapers.value.slice(start, end)
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

  async function generateRecommendations(topN: number = 10, maxAgeMonths: number = 6): Promise<RecommendationResult> {
    recommendationsGenerating.value = true
    try {
      const result = await recommendPapers(topN, maxAgeMonths)
      if (result.status === 'success') {
        recommendations.value = result.recommendations.map((p: any) => ({
          id: p.arxiv_id,
          arxivId: p.arxiv_id,
          title: p.title,
          authors: toStringArray(p.authors),
          summary: p.abstract,
          publishedAt: p.published_date,
          categories: toStringArray(p.categories),
          pdfUrl: p.url,
          absUrl: p.url,
          similarityScore: typeof p.similarity_score === 'number'
            ? p.similarity_score
            : (typeof p.similarityScore === 'number' ? p.similarityScore : (typeof p.score === 'number' ? p.score : 0)),
          finalScore: typeof p.final_score === 'number' ? p.final_score : undefined,
          reason: buildRecommendationReason(p),
          scoreBreakdown: p.score_breakdown || p.scoreBreakdown || undefined,
          recall_source: p.recall_source || p.recallSource || undefined,
          recall_cluster_id: p.recall_cluster_id || p.recallClusterId || null,
          recall_cluster_similarity: typeof p.recall_cluster_similarity === 'number'
            ? p.recall_cluster_similarity
            : (typeof p.recallClusterSimilarity === 'number' ? p.recallClusterSimilarity : null),
          recall_cluster_rank: typeof p.recall_cluster_rank === 'number'
            ? p.recall_cluster_rank
            : (typeof p.recallClusterRank === 'number' ? p.recallClusterRank : null),
          recall_cluster_hits: toRecallClusterHits(p.recall_cluster_hits || p.recallClusterHits),
          best_matched_cluster_id: p.best_matched_cluster_id || p.bestMatchedClusterId || null,
          best_matched_cluster_similarity: typeof p.best_matched_cluster_similarity === 'number'
            ? p.best_matched_cluster_similarity
            : (typeof p.bestMatchedClusterSimilarity === 'number' ? p.bestMatchedClusterSimilarity : null),
          cluster_similarities: Array.isArray(p.cluster_similarities)
            ? p.cluster_similarities
            : (Array.isArray(p.clusterSimilarities) ? p.clusterSimilarities : undefined),
          label: null
        }))
        totalRecommendations.value = result.total_found
      }
      return result
    } finally {
      recommendationsGenerating.value = false
    }
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
    generateRecommendations
  }
})
