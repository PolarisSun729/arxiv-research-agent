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

const showFullSummary = ref(false)
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

const bestMatchedClusterId = computed(() => {
  if (!isRecommended(props.paper)) return ''
  return props.paper.best_matched_cluster_id || ''
})

const finalScoreText = computed(() => {
  if (!isRecommended(props.paper) || typeof props.paper.finalScore !== 'number') {
    return ''
  }
  return `${Math.round(props.paper.finalScore * 100)}%`
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
      <p :class="{ 'summary-collapsed': !showFullSummary && paper.summary.length > 150 }">
        {{ showFullSummary ? paper.summary : paper.summary.slice(0, 150) + '...' }}
      </p>
      <button
        v-if="paper.summary.length > 150"
        class="toggle-summary"
        @click="showFullSummary = !showFullSummary"
      >
        {{ showFullSummary ? '收起' : '展开' }}
      </button>
    </div>

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

    <div v-if="isRecommendation || isRecommended(paper)" class="paper-similarity">
      <span class="similarity-label">相似度：</span>
      <SimilarityTag :score="(paper as any).similarityScore || (paper as any).similarity || 0" />
      <span v-if="finalScoreText" class="final-score-label">最终 {{ finalScoreText }}</span>
      <el-tag v-if="bestMatchedClusterId" size="small" type="success" effect="plain">
        命中 {{ bestMatchedClusterId }}
      </el-tag>
      <p v-if="recommendationReason" class="recommendation-reason">
        {{ recommendationReason }}
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
  </el-card>
</template>

<style scoped>
.paper-card {
  margin-bottom: 16px;
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
}

.paper-authors {
  margin-bottom: 12px;
  font-size: 14px;
  color: #6b7280;
}

.authors-label {
  font-weight: 500;
}

.authors-list {
  color: #4b5563;
}

.paper-summary {
  margin-bottom: 12px;
}

.paper-summary p {
  font-size: 14px;
  color: #4b5563;
  line-height: 1.6;
  margin: 0;
}

.summary-collapsed {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.toggle-summary {
  background: none;
  border: none;
  color: #1890ff;
  font-size: 13px;
  cursor: pointer;
  margin-top: 8px;
  padding: 0;
}

.toggle-summary:hover {
  text-decoration: underline;
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
</style>
