<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import type { Paper, RecommendedPaper } from '@/types/paper'
import { ElMessage } from 'element-plus'
import { ArrowRight, CircleCheck, CircleClose, Document, Reading } from '@element-plus/icons-vue'
import SimilarityTag from './SimilarityTag.vue'
import { toPercent } from '@/utils/format'
import { getLabelText } from '@/utils/paperLabel'
import { usePaperStore } from '@/stores/paperStore'
import { useAbstractTranslation } from '@/composables/useAbstractTranslation'

const props = defineProps<{
  paper: Paper | RecommendedPaper
  showActions?: boolean
  isRecommendation?: boolean
}>()

const emit = defineEmits<{
  (e: 'view-detail', id: string): void
  (e: 'label', id: string, label: 'liked' | 'disliked' | null): void
}>()

const selectedLabel = ref(props.paper.label === 'liked' ? 'liked' : props.paper.label === 'disliked' ? 'disliked' : '')

watch(() => props.paper.label, (newLabel) => {
  selectedLabel.value = newLabel === 'liked' ? 'liked' : newLabel === 'disliked' ? 'disliked' : ''
})

const store = usePaperStore()
// 已读是弱行为，直接读 store 里的行为表，保证卡片与详情页显示同一份状态。
const isRead = computed(() => (store.paperActionMap.read || []).includes(props.paper.arxivId || props.paper.id))

const isRecommended = (p: Paper | RecommendedPaper): p is RecommendedPaper => {
  return 'similarityScore' in p
}

const recommendationReason = computed(() => {
  return isRecommended(props.paper) ? props.paper.reason || '' : ''
})

const recommendationScoreBreakdown = computed(() => {
  return isRecommended(props.paper) ? props.paper.scoreBreakdown || null : null
})

const showRecommendationMetrics = computed(() => {
  return Boolean(props.isRecommendation || isRecommended(props.paper))
})

const diversityDebug = computed(() => {
  return isRecommended(props.paper) ? props.paper.diversityDebug || null : null
})

const diversityMatchScore = computed(() => {
  if (diversityDebug.value && typeof diversityDebug.value.diversity_penalty_value === 'number') {
    return Math.max(0, Math.min(1, 1 - diversityDebug.value.diversity_penalty_value))
  }
  if (recommendationScoreBreakdown.value && typeof recommendationScoreBreakdown.value.diversity_score === 'number') {
    return recommendationScoreBreakdown.value.diversity_score
  }
  return 0
})

const diversityReason = computed(() => {
  const reason = diversityDebug.value?.diversity_reason || ''
  const reasonMap: Record<string, string> = {
    seed: '种子项',
    semantic_repeat: '语义相似',
    cluster_repeat: '簇相似',
    category_repeat: '类别相似',
    semantic_guidance: '语义补充',
    cluster_guidance: '簇补充',
    semantic_cluster_mix: '语义+簇综合',
    no_previous_selection: '首篇',
  }
  return reasonMap[reason] || reason
})

const bestMatchedClusterId = computed(() => {
  if (!isRecommended(props.paper)) return ''
  return props.paper.best_matched_cluster_id || ''
})

const recallClusterId = computed(() => {
  if (!isRecommended(props.paper)) return ''
  return props.paper.recall_cluster_id || ''
})

const recallClusterSimilarity = computed(() => {
  if (!isRecommended(props.paper) || typeof props.paper.recall_cluster_similarity !== 'number') return 0
  return props.paper.recall_cluster_similarity
})

const recallClusterHitsCount = computed(() => {
  if (!isRecommended(props.paper) || !Array.isArray(props.paper.recall_cluster_hits)) return 0
  return props.paper.recall_cluster_hits.length
})

const finalScoreText = computed(() => {
  if (isRecommended(props.paper) && typeof props.paper.finalScore === 'number') {
    return `${Math.round(props.paper.finalScore * 100)}%`
  }
  if (agentFinalScore.value > 0) {
    return `${Math.round(agentFinalScore.value * 100)}%`
  }
  if (typeof (props.paper as any).final_score === 'number' && (props.paper as any).final_score > 0) {
    return `${Math.round((props.paper as any).final_score * 100)}%`
  }
  if (typeof (props.paper as any).finalScore !== 'number') {
    return ''
  }
  return `${Math.round((props.paper as any).finalScore * 100)}%`
})

function readPaperNumber(keys: string[]) {
  for (const key of keys) {
    const value = (props.paper as any)[key]
    if (typeof value === 'number' && !Number.isNaN(value)) {
      return value
    }
  }
  return 0
}

function readPaperString(keys: string[]) {
  for (const key of keys) {
    const value = (props.paper as any)[key]
    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }
  return ''
}

function readPaperStringArray(keys: string[]) {
  for (const key of keys) {
    const value = (props.paper as any)[key]
    if (Array.isArray(value)) {
      return value.map((item: unknown) => String(item).trim()).filter(Boolean)
    }
  }
  return []
}

const agentQueryMatchScore = computed(() => readPaperNumber(['queryMatchScore', 'query_match_score']))
const agentPersonalizationScore = computed(() => readPaperNumber(['personalizationScore', 'personalization_score']))
const agentFinalScore = computed(() => readPaperNumber(['finalScore', 'final_score']))
const agentPriority = computed(() => readPaperNumber(['priority']))
const agentMatchedTerms = computed(() => readPaperStringArray(['matchedTerms', 'matched_terms']))
const agentMatchReason = computed(() => readPaperString(['matchReason', 'match_reason']))
const agentPersonalizedReason = computed(() => readPaperString(['personalizedReason', 'personalized_reason']))
const agentScoreBreakdown = computed(() => {
  const breakdown = (props.paper as any).scoreBreakdown || (props.paper as any).score_breakdown || null
  if (!breakdown || typeof breakdown !== 'object') return null

  const hasAgentScores = [breakdown.query_match_score, breakdown.personalization_score, breakdown.final_score]
    .some((value) => typeof value === 'number' && !Number.isNaN(value))

  return hasAgentScores ? breakdown : null
})
const similarityDisplayScore = computed(() => {
  if (isRecommended(props.paper)) {
    return (props.paper as any).similarityScore || (props.paper as any).similarity || 0
  }
  return agentQueryMatchScore.value || 0
})

const hasAgentRerankData = computed(() => {
  return Boolean(
    agentQueryMatchScore.value ||
    agentPersonalizationScore.value ||
    agentFinalScore.value ||
    agentPriority.value ||
    agentMatchedTerms.value.length ||
    agentMatchReason.value ||
    agentPersonalizedReason.value
  )
})

const showAgentMetrics = computed(() => {
  if (props.isRecommendation) return false
  return Boolean(hasAgentRerankData.value || agentScoreBreakdown.value)
})

const authorsDisplay = computed(() => {
  if (Array.isArray(props.paper.authors)) {
    return props.paper.authors.join(', ')
  }
  return props.paper.authors || ''
})

function handleViewDetail() {
  emit('view-detail', props.paper.id)
}

const summaryExpanded = ref(false)
const { displaySummary, isTranslating, translateButtonText, toggleTranslation } = useAbstractTranslation()
const translationId = computed(() => props.paper.arxivId || props.paper.id)
const detailsOpen = ref(false)

// 再次点击已选中的标记即取消标记。
function toggleLabel(value: 'liked' | 'disliked') {
  const next = selectedLabel.value === value ? '' : value
  selectedLabel.value = next
  emit('label', props.paper.id, next || null)
}

async function toggleRead() {
  const enabled = !isRead.value
  try {
    await store.togglePaperAction(props.paper, 'read', enabled)
    ElMessage.success(enabled ? '已标记已读' : '已取消已读')
  } catch {
    ElMessage.error('操作失败')
  }
}

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('zh-CN')
}
</script>

<template>
  <article :class="['surface', 'paper-card', { 'is-liked': paper.label === 'liked', 'is-disliked': paper.label === 'disliked' }]">
    <header class="paper-head">
      <h3 class="paper-title" @click="handleViewDetail">{{ paper.title }}</h3>
      <span v-if="paper.label" :class="['chip', paper.label === 'liked' ? 'chip--success' : 'chip--danger']">
        {{ getLabelText(paper.label) }}
      </span>
    </header>

    <p v-if="authorsDisplay" class="paper-authors" :title="authorsDisplay">{{ authorsDisplay }}</p>

    <div class="paper-meta">
      <span class="meta-date">{{ formatDate(paper.publishedAt) }}</span>
      <span v-for="cat in paper.categories" :key="cat" class="chip">{{ cat }}</span>
    </div>

    <p :class="['paper-summary', { 'clamp-3': !summaryExpanded }]">{{ displaySummary(translationId, paper.summary) }}</p>
    <div v-if="paper.summary" class="summary-actions">
      <button v-if="paper.summary.length > 240" type="button" class="link-btn" @click="summaryExpanded = !summaryExpanded">
        {{ summaryExpanded ? '收起摘要' : '展开摘要' }}
      </button>
      <button type="button" class="link-btn" :disabled="isTranslating(translationId)" @click="toggleTranslation(translationId)">
        {{ translateButtonText(translationId) }}
      </button>
    </div>

    <section v-if="showRecommendationMetrics || showAgentMetrics" class="match-panel">
      <div class="match-top">
        <span class="match-label">相似度</span>
        <SimilarityTag :score="similarityDisplayScore" class="match-bar" />
        <span v-if="finalScoreText" class="match-final">综合 <strong>{{ finalScoreText }}</strong></span>
      </div>

      <template v-if="showRecommendationMetrics">
        <p v-if="recommendationReason" class="match-reason">{{ recommendationReason }}</p>
      </template>
      <template v-if="showAgentMetrics">
        <p v-if="agentMatchReason" class="match-reason">{{ agentMatchReason }}</p>
        <p v-if="agentPersonalizedReason" class="match-reason">{{ agentPersonalizedReason }}</p>
      </template>

      <button type="button" class="link-btn" @click="detailsOpen = !detailsOpen">
        {{ detailsOpen ? '收起评分明细' : '查看评分明细' }}
      </button>

      <div v-if="detailsOpen" class="match-details">
        <div class="chip-row">
          <template v-if="showRecommendationMetrics">
            <span v-if="bestMatchedClusterId" class="chip chip--success">命中簇 {{ bestMatchedClusterId }}</span>
            <span v-if="recallClusterId" class="chip chip--warning">召回来源 {{ recallClusterId }}</span>
            <span v-if="recallClusterId" class="chip">簇召回相似度 {{ toPercent(recallClusterSimilarity) }}%</span>
            <span v-if="recallClusterHitsCount > 1" class="chip">多簇命中 {{ recallClusterHitsCount }}</span>
            <span class="chip">多样性 {{ toPercent(diversityMatchScore) }}%</span>
            <span v-if="diversityReason" class="chip">{{ diversityReason }}</span>
          </template>
          <template v-if="showAgentMetrics">
            <span v-if="agentPriority" class="chip chip--primary">优先级 #{{ agentPriority }}</span>
            <span v-if="agentQueryMatchScore" class="chip">查询 {{ toPercent(agentQueryMatchScore) }}%</span>
            <span v-if="agentPersonalizationScore" class="chip">兴趣 {{ toPercent(agentPersonalizationScore) }}%</span>
            <span v-if="agentFinalScore" class="chip">综合 {{ toPercent(agentFinalScore) }}%</span>
            <span v-if="agentMatchedTerms.length" class="chip">命中词 {{ agentMatchedTerms.slice(0, 3).join(' / ') }}</span>
          </template>
        </div>

        <div v-if="showRecommendationMetrics && recommendationScoreBreakdown" class="score-bars">
          <div class="score-bar">
            <span>语义</span>
            <el-progress :percentage="toPercent(recommendationScoreBreakdown.semantic_score)" :show-text="false" :stroke-width="6" color="var(--color-chart-1)" />
            <strong>{{ toPercent(recommendationScoreBreakdown.semantic_score) }}%</strong>
          </div>
          <div class="score-bar">
            <span>分类</span>
            <el-progress :percentage="toPercent(recommendationScoreBreakdown.category_score)" :show-text="false" :stroke-width="6" color="var(--color-chart-2)" />
            <strong>{{ toPercent(recommendationScoreBreakdown.category_score) }}%</strong>
          </div>
          <div class="score-bar">
            <span>新鲜度</span>
            <el-progress :percentage="toPercent(recommendationScoreBreakdown.recency_score)" :show-text="false" :stroke-width="6" color="var(--color-chart-3)" />
            <strong>{{ toPercent(recommendationScoreBreakdown.recency_score) }}%</strong>
          </div>
          <div v-if="typeof recommendationScoreBreakdown.diversity_score === 'number'" class="score-bar">
            <span>多样性</span>
            <el-progress :percentage="toPercent(recommendationScoreBreakdown.diversity_score)" :show-text="false" :stroke-width="6" color="var(--color-chart-4)" />
            <strong>{{ toPercent(recommendationScoreBreakdown.diversity_score) }}%</strong>
          </div>
        </div>

        <div v-if="showAgentMetrics && agentScoreBreakdown" class="score-bars">
          <div class="score-bar">
            <span>查询</span>
            <el-progress :percentage="toPercent(agentScoreBreakdown.query_match_score)" :show-text="false" :stroke-width="6" color="var(--color-chart-2)" />
            <strong>{{ toPercent(agentScoreBreakdown.query_match_score) }}%</strong>
          </div>
          <div class="score-bar">
            <span>兴趣</span>
            <el-progress :percentage="toPercent(agentScoreBreakdown.personalization_score)" :show-text="false" :stroke-width="6" color="var(--color-chart-3)" />
            <strong>{{ toPercent(agentScoreBreakdown.personalization_score) }}%</strong>
          </div>
          <div class="score-bar">
            <span>综合</span>
            <el-progress :percentage="toPercent(agentScoreBreakdown.final_score)" :show-text="false" :stroke-width="6" color="var(--color-chart-4)" />
            <strong>{{ toPercent(agentScoreBreakdown.final_score) }}%</strong>
          </div>
        </div>
      </div>
    </section>

    <footer class="paper-actions">
      <div class="label-toggle" role="group" aria-label="兴趣标记">
        <button
          type="button"
          :class="['label-btn', { 'is-on': selectedLabel === 'liked' }]"
          :aria-pressed="selectedLabel === 'liked'"
          @click="toggleLabel('liked')"
        >
          <el-icon><CircleCheck /></el-icon>感兴趣
        </button>
        <button
          type="button"
          :class="['label-btn', 'label-btn--read', { 'is-on': isRead }]"
          :aria-pressed="isRead"
          @click="toggleRead"
        >
          <el-icon><Reading /></el-icon>{{ isRead ? '已读' : '标记已读' }}
        </button>
        <button
          type="button"
          :class="['label-btn', 'label-btn--neg', { 'is-on': selectedLabel === 'disliked' }]"
          :aria-pressed="selectedLabel === 'disliked'"
          @click="toggleLabel('disliked')"
        >
          <el-icon><CircleClose /></el-icon>不感兴趣
        </button>
      </div>
      <div class="action-links">
        <a :href="paper.pdfUrl" target="_blank" rel="noopener" class="pdf-link">
          <el-button size="small" :icon="Document">PDF</el-button>
        </a>
        <el-button size="small" type="primary" plain @click="handleViewDetail">
          查看详情
          <el-icon class="el-icon--right"><ArrowRight /></el-icon>
        </el-button>
      </div>
    </footer>
  </article>
</template>

<style scoped>
.paper-card {
  display: flex;
  flex-direction: column;
  /* 卡片放在 grid/flex 列表里时，默认最小宽度是内容宽度；作者行不换行会把卡片撑出容器。 */
  min-width: 0;
  padding: 18px 20px 14px;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.paper-card:hover {
  border-color: var(--color-border-strong);
  box-shadow: var(--shadow-sm);
}

.paper-card.is-liked {
  border-left: 3px solid var(--color-success);
}

.paper-card.is-disliked {
  border-left: 3px solid var(--color-border-strong);
}

.paper-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.paper-title {
  font-size: 16px;
  font-weight: 600;
  line-height: 1.45;
  color: var(--color-text);
  cursor: pointer;
  word-break: break-word;
}

.paper-title:hover {
  color: var(--color-primary);
}

.paper-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
  font-size: 13px;
  color: var(--color-text-3);
}

/* 作者单独一行、正常换行；作者特别多时最多两行，完整列表看悬浮提示。 */
.paper-authors {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  overflow: hidden;
  margin-top: 6px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-text-2);
  overflow-wrap: anywhere;
}

.meta-date {
  margin-right: 4px;
  white-space: nowrap;
}

.paper-summary {
  margin-top: 10px;
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--color-text-2);
}

.link-btn {
  align-self: flex-start;
  margin-top: 4px;
  padding: 0;
  border: none;
  background: none;
  color: var(--color-primary);
  font: inherit;
  font-size: 12.5px;
  cursor: pointer;
}

.link-btn:hover {
  text-decoration: underline;
}

.link-btn:disabled {
  color: var(--color-text-3);
  cursor: default;
  text-decoration: none;
}

.summary-actions {
  display: flex;
  gap: 14px;
}

.summary-actions .link-btn {
  align-self: auto;
}

.match-panel {
  display: flex;
  flex-direction: column;
  margin-top: 12px;
  padding: 12px 14px;
  border-radius: var(--radius);
  background: var(--color-surface-muted);
  border: 1px solid var(--color-border);
}

.match-top {
  display: flex;
  align-items: center;
  gap: 12px;
}

.match-label {
  font-size: 12.5px;
  color: var(--color-text-3);
  white-space: nowrap;
}

.match-bar {
  flex: 1;
  min-width: 0;
}

.match-final {
  font-size: 12.5px;
  color: var(--color-text-3);
  white-space: nowrap;
}

.match-final strong {
  color: var(--color-text);
  font-variant-numeric: tabular-nums;
}

.match-reason {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-text-2);
}

.match-details {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.score-bars {
  display: grid;
  gap: 6px;
}

.score-bar {
  display: grid;
  grid-template-columns: 48px 1fr 40px;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: var(--color-text-3);
}

.score-bar strong {
  text-align: right;
  font-weight: 600;
  color: var(--color-text-2);
  font-variant-numeric: tabular-nums;
}

.paper-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border);
}

.label-toggle {
  display: inline-flex;
  gap: 6px;
}

.label-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  height: 28px;
  padding: 0 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-pill);
  background: var(--color-surface);
  color: var(--color-text-3);
  font: inherit;
  font-size: 12.5px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.label-btn:hover {
  color: var(--color-success);
  border-color: var(--color-success-border);
}

.label-btn.is-on {
  background: var(--color-success-soft);
  border-color: var(--color-success-border);
  color: var(--color-success);
}

.label-btn--read:hover {
  color: var(--color-primary);
  border-color: var(--color-primary-border);
}

.label-btn--read.is-on {
  background: var(--color-primary-soft);
  border-color: var(--color-primary-border);
  color: var(--color-primary);
}

.label-btn--neg:hover {
  color: var(--color-danger);
  border-color: var(--color-danger-border);
}

.label-btn--neg.is-on {
  background: var(--color-danger-soft);
  border-color: var(--color-danger-border);
  color: var(--color-danger);
}

.action-links {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.pdf-link {
  text-decoration: none;
}
</style>
