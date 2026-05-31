<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import type { Paper, RecommendedPaper } from '@/types/paper'
import SimilarityTag from './SimilarityTag.vue'

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

const isRecommended = (p: Paper | RecommendedPaper): p is RecommendedPaper => {
  return 'similarityScore' in p
}

const recommendationReason = computed(() => {
  return isRecommended(props.paper) ? props.paper.reason || '' : ''
})

const recommendationScoreBreakdown = computed(() => {
  return isRecommended(props.paper) ? props.paper.scoreBreakdown || null : null
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
const agentScoreBreakdown = computed(() => (props.paper as any).scoreBreakdown || (props.paper as any).score_breakdown || null)
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

function toPercent(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 0
  return Math.max(0, Math.min(100, Math.round(value * 100)))
}

const authorsDisplay = computed(() => {
  if (Array.isArray(props.paper.authors)) {
    return props.paper.authors.join(', ')
  }
  return props.paper.authors || ''
})

function handleViewDetail() {
  emit('view-detail', props.paper.id)
}

function handleLabelChange(value: string) {
  const label = value === 'liked' ? 'liked' : value === 'disliked' ? 'disliked' : null
  emit('label', props.paper.id, label)
}

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('zh-CN')
}

function getLabelClass(label?: 'liked' | 'disliked' | null) {
  if (label === 'liked') return 'el-tag--success'
  if (label === 'disliked') return 'el-tag--danger'
  return ''
}

function getLabelText(label?: 'liked' | 'disliked' | null) {
  if (label === 'liked') return '感兴趣'
  if (label === 'disliked') return '不感兴趣'
  return ''
}

const labelOptions = [
  { label: '未标记', value: '', emoji: '🫧' },
  { label: '感兴趣', value: 'liked', emoji: '💚' },
  { label: '不感兴趣', value: 'disliked', emoji: '💔' }
]
</script>

<template>
  <el-card class="paper-card">
    <div class="paper-main">
      <div class="paper-header">
        <div class="paper-title">
          <el-tooltip :content="paper.title" placement="top">
            <h3 class="title-text">{{ paper.title }}</h3>
          </el-tooltip>
        </div>
        <div class="paper-labels">
          <el-tag
            v-if="paper.label"
            :class="getLabelClass(paper.label)"
            size="small"
          >
            {{ getLabelText(paper.label) }}
          </el-tag>
        </div>
      </div>

      <div class="paper-authors">
        <span class="authors-label">作者：</span>
        <span class="authors-list">{{ authorsDisplay }}</span>
      </div>

      <div class="paper-summary">
        <p class="summary-expanded">
          {{ paper.summary }}
        </p>
      </div>
    </div>

      <div class="paper-footer">
        <div class="paper-meta">
        <div class="meta-item">
          <el-tag size="small">{{ formatDate(paper.publishedAt) }}</el-tag>
        </div>
        <div class="meta-item">
          <el-tag
            v-for="cat in paper.categories"
            :key="cat"
            size="small"
            type="info"
          >
            {{ cat }}
          </el-tag>
        </div>
      </div>

      <div v-if="isRecommendation || isRecommended(paper) || hasAgentRerankData" class="paper-similarity">
        <span class="similarity-label">相似度：</span>
        <SimilarityTag :score="similarityDisplayScore" />
        <span v-if="finalScoreText" class="final-score-label">最终 {{ finalScoreText }}</span>
        <el-tag v-if="agentPriority" size="small" type="success" effect="plain">
          优先级 #{{ agentPriority }}
        </el-tag>
        <el-tag v-if="agentQueryMatchScore" size="small" type="info" effect="plain">
          查询 {{ toPercent(agentQueryMatchScore) }}%
        </el-tag>
        <el-tag v-if="agentPersonalizationScore" size="small" type="warning" effect="plain">
          兴趣 {{ toPercent(agentPersonalizationScore) }}%
        </el-tag>
        <el-tag v-if="agentFinalScore" size="small" type="success" effect="plain">
          综合 {{ toPercent(agentFinalScore) }}%
        </el-tag>
        <el-tag v-if="bestMatchedClusterId" size="small" type="success" effect="plain">
          命中簇 {{ bestMatchedClusterId }}
        </el-tag>
        <el-tag v-if="recallClusterId" size="small" type="warning" effect="plain">
          召回来源 {{ recallClusterId }}
        </el-tag>
        <el-tag v-if="recallClusterId" size="small" type="info" effect="plain">
          簇召回相似度 {{ toPercent(recallClusterSimilarity) }}%
        </el-tag>
        <el-tag v-if="recallClusterHitsCount > 1" size="small" type="info" effect="plain">
          多簇命中 {{ recallClusterHitsCount }}
        </el-tag>
        <el-tag size="small" type="success" effect="plain">
          多样性得分 {{ toPercent(diversityMatchScore) }}%
        </el-tag>
        <el-tag v-if="agentMatchedTerms.length" size="small" type="info" effect="plain">
          命中词 {{ agentMatchedTerms.slice(0, 3).join(' / ') }}
        </el-tag>
        <el-tag v-if="diversityReason" size="small" type="warning" effect="plain">
          {{ diversityReason }}
        </el-tag>
        <p v-if="recommendationReason" class="recommendation-reason">
          {{ recommendationReason }}
        </p>
        <p v-if="agentMatchReason" class="recommendation-reason">
          {{ agentMatchReason }}
        </p>
        <p v-if="agentPersonalizedReason" class="recommendation-reason">
          {{ agentPersonalizedReason }}
        </p>
        <div v-if="recommendationScoreBreakdown" class="score-breakdown">
          <div class="score-breakdown-row">
            <div class="score-breakdown-meta">
              <span class="breakdown-label">语义</span>
              <strong>{{ toPercent(recommendationScoreBreakdown.semantic_score) }}%</strong>
            </div>
            <el-progress :percentage="toPercent(recommendationScoreBreakdown.semantic_score)" :show-text="false" />
          </div>
          <div class="score-breakdown-row">
            <div class="score-breakdown-meta">
              <span class="breakdown-label">分类</span>
              <strong>{{ toPercent(recommendationScoreBreakdown.category_score) }}%</strong>
            </div>
            <el-progress :percentage="toPercent(recommendationScoreBreakdown.category_score)" :show-text="false" color="#8b5cf6" />
          </div>
          <div class="score-breakdown-row">
            <div class="score-breakdown-meta">
              <span class="breakdown-label">新鲜度</span>
              <strong>{{ toPercent(recommendationScoreBreakdown.recency_score) }}%</strong>
            </div>
            <el-progress :percentage="toPercent(recommendationScoreBreakdown.recency_score)" :show-text="false" color="#0ea5e9" />
          </div>
          <div class="score-breakdown-row" v-if="typeof recommendationScoreBreakdown.diversity_score === 'number'">
            <div class="score-breakdown-meta">
              <span class="breakdown-label">多样性</span>
              <strong>{{ toPercent(recommendationScoreBreakdown.diversity_score) }}%</strong>
            </div>
            <el-progress :percentage="toPercent(recommendationScoreBreakdown.diversity_score)" :show-text="false" color="#22c55e" />
          </div>
        </div>
        <div v-if="agentScoreBreakdown" class="score-breakdown">
          <div class="score-breakdown-row">
            <div class="score-breakdown-meta">
              <span class="breakdown-label">查询</span>
              <strong>{{ toPercent(agentScoreBreakdown.query_match_score) }}%</strong>
            </div>
            <el-progress :percentage="toPercent(agentScoreBreakdown.query_match_score)" :show-text="false" color="#0ea5e9" />
          </div>
          <div class="score-breakdown-row">
            <div class="score-breakdown-meta">
              <span class="breakdown-label">兴趣</span>
              <strong>{{ toPercent(agentScoreBreakdown.personalization_score) }}%</strong>
            </div>
            <el-progress :percentage="toPercent(agentScoreBreakdown.personalization_score)" :show-text="false" color="#f59e0b" />
          </div>
          <div class="score-breakdown-row">
            <div class="score-breakdown-meta">
              <span class="breakdown-label">综合</span>
              <strong>{{ toPercent(agentScoreBreakdown.final_score) }}%</strong>
            </div>
            <el-progress :percentage="toPercent(agentScoreBreakdown.final_score)" :show-text="false" color="#22c55e" />
          </div>
        </div>
      </div>

      <div class="paper-actions">
        <a
          :href="paper.pdfUrl"
          target="_blank"
          class="pdf-link"
        >
          <el-button size="small" type="default">
            📘 打开 PDF
          </el-button>
        </a>
        <el-button
          size="small"
          type="primary"
          @click="handleViewDetail"
        >
          🧾 查看详情
        </el-button>
        <el-select
          v-model="selectedLabel"
          placeholder="选择兴趣"
          size="small"
          :style="{ width: '120px' }"
          @change="handleLabelChange"
        >
          <el-option
            v-for="option in labelOptions"
            :key="option.value"
            :label="option.label"
            :value="option.value"
          >
            <span class="option-row">
              <span class="option-emoji">{{ option.emoji }}</span>
              <span>{{ option.label }}</span>
            </span>
          </el-option>
        </el-select>
      </div>
    </div>
  </el-card>
</template>

<style scoped>
.paper-card {
  height: auto;
  overflow: hidden;
  margin-bottom: 0;
}

.paper-card :deep(.el-card__body) {
  height: auto;
  display: flex;
  flex-direction: column;
}

.paper-main {
  height: 320px;
  overflow: hidden;
}

/* 上半部分固定，方便多张卡片对齐；内容超出时只在自身区域里截断。 */
.paper-main :deep(.el-tooltip__trigger) {
  display: block;
}

.paper-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}

.paper-title {
  flex: 1;
  margin-right: 12px;
}

.title-text {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
  line-height: 1.4;
  word-break: break-word;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.paper-authors {
  margin-bottom: 12px;
  font-size: 14px;
  color: #6b7280;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.authors-label {
  font-weight: 500;
}

.authors-list {
  color: #4b5563;
}

.paper-summary {
  margin-bottom: 12px;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.paper-summary p {
  font-size: 14px;
  color: #4b5563;
  line-height: 1.6;
  margin: 0;
  min-height: 0;
}

.summary-expanded {
  max-height: 180px;
  overflow-y: auto;
  padding-right: 6px;
}

.paper-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}

.meta-item {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.paper-similarity {
  margin-bottom: 12px;
  padding: 12px;
  background: #f8fafc;
  border-radius: 8px;
}

.final-score-label {
  display: inline-block;
  margin-left: 8px;
  font-size: 12px;
  color: #111827;
  font-weight: 600;
}

.similarity-label {
  font-size: 13px;
  color: #6b7280;
  margin-right: 8px;
}

.recommendation-reason {
  font-size: 13px;
  color: #059669;
  margin: 8px 0 0 0;
  font-style: italic;
}

.score-breakdown {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  margin-top: 8px;
}

.breakdown-label {
  font-weight: 600;
  color: #374151;
}

.score-breakdown-row {
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.65);
  border-radius: 12px;
}

.score-breakdown-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 12px;
  color: #4b5563;
}

.paper-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #e5e7eb;
}

.pdf-link {
  text-decoration: none;
}

.option-row {
  display: flex;
  align-items: center;
  gap: 6px;
}

.option-emoji {
  width: 18px;
  text-align: center;
}

@media (max-width: 900px) {
  .paper-card {
    height: auto;
    min-height: 0;
    overflow: visible;
  }

  .paper-card :deep(.el-card__body) {
    height: auto;
  }

  .paper-main {
    height: auto;
    overflow: visible;
  }
}
</style>


