<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { usePaperStore } from '@/stores/paperStore'
import PaperCard from '@/components/PaperCard.vue'
import SimilarityTag from '@/components/SimilarityTag.vue'

const router = useRouter()
const store = usePaperStore()

const recommendationCount = ref(10)
const recommendationAgeMonths = ref(6)
const topRecommendation = computed(() => store.recommendations[0] || null)
const topBreakdown = computed(() => topRecommendation.value?.scoreBreakdown || null)
const topDiversityDebug = computed(() => topRecommendation.value?.diversityDebug || null)
const topDiversityMatch = computed(() => {
  if (topDiversityDebug.value && typeof topDiversityDebug.value.diversity_penalty_value === 'number') {
    return Math.max(0, Math.min(1, 1 - topDiversityDebug.value.diversity_penalty_value))
  }
  if (topBreakdown.value && typeof topBreakdown.value.diversity_score === 'number') {
    return topBreakdown.value.diversity_score
  }
  return 0
})
const interestVector = computed(() => store.lastInterestVector)
const interestClusterCount = computed(() => interestVector.value?.cluster_count || 0)
const interestProfileMode = computed(() => interestVector.value?.profile_mode || 'mean')
const topFinalScore = computed(() => {
  const paper = topRecommendation.value
  if (!paper) return 0
  return typeof paper.finalScore === 'number' ? paper.finalScore : paper.similarityScore || 0
})

onMounted(() => {
  store.fetchUserInterestVector()
})

const countOptions = [
  { value: 5, label: '5篇' },
  { value: 10, label: '10篇' },
  { value: 20, label: '20篇' },
  { value: 50, label: '50篇' }
]

const ageOptions = [
  { value: 3, label: '最近3个月' },
  { value: 6, label: '最近6个月' },
  { value: 12, label: '最近12个月' }
]

async function handleGenerateRecommendations() {
  try {
    await store.generateRecommendations(recommendationCount.value, recommendationAgeMonths.value)
    ElMessage.success(`已生成 ${recommendationCount.value} 篇最近${recommendationAgeMonths.value}个月内的 CS.AI 推荐论文`)
  } catch (error: any) {
    if (error?.response?.data?.detail) {
      ElMessage.error(error.response.data.detail)
    } else {
      ElMessage.error('生成推荐失败')
    }
  }
}

function handleViewDetail(id: string) {
  router.push(`/paper/${id}`)
}

function toPercent(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 0
  return Math.max(0, Math.min(100, Math.round(value * 100)))
}

async function handleLabel(id: string, label: 'liked' | 'disliked' | null) {
  try {
    await store.updateLabel(id, label)
    if (label === null) {
      ElMessage.success('已取消标记')
    } else {
      ElMessage.success(`已标记为${label === 'liked' ? '感兴趣' : '不感兴趣'}`)
    }
  } catch {
    ElMessage.error('偏好保存失败')
  }
}
</script>

<template>
  <div class="recommendations-page">
    <div class="page-header">
      <div>
        <h1>推荐论文</h1>
        <p class="description">
          根据你已经标记过的论文，系统会从 CS.AI 论文中生成一批相似度较高的推荐结果。
        </p>
      </div>

      <div class="action-bar">
        <el-select
          v-model="recommendationCount"
          placeholder="推荐数量"
          style="width: 120px; margin-right: 12px"
        >
          <el-option
            v-for="option in countOptions"
            :key="option.value"
            :value="option.value"
            :label="option.label"
          />
        </el-select>
        <el-select
          v-model="recommendationAgeMonths"
          placeholder="时间范围"
          style="width: 140px; margin-right: 12px"
        >
          <el-option
            v-for="option in ageOptions"
            :key="option.value"
            :value="option.value"
            :label="option.label"
          />
        </el-select>
        <el-button
          type="primary"
          :loading="store.recommendationsGenerating"
          @click="handleGenerateRecommendations"
        >
          生成推荐
        </el-button>
      </div>
    </div>

    <div v-if="store.recommendations.length > 0" class="summary-panel">
      <div class="summary-panel__lead">
        <div class="summary-copy-wrap">
          <div class="summary-kicker">为什么推荐这篇</div>
          <h2 class="summary-title">{{ topRecommendation?.title }}</h2>
          <p class="summary-copy">
            这篇 CS.AI 论文在语义相似度、主题匹配上更符合你当前的兴趣画像，且限定在最近 {{ recommendationAgeMonths }} 个月内。
          </p>
        </div>

        <div class="summary-score">
          <SimilarityTag :score="topRecommendation?.similarityScore || 0" />
          <div class="summary-score__value">最终 {{ Math.round(topFinalScore * 100) }}%</div>
        </div>
      </div>

      <div v-if="interestVector" class="summary-interest">
        <span>兴趣画像模式：{{ interestProfileMode }}</span>
        <span>兴趣簇数量：{{ interestClusterCount }}</span>
      </div>

      <div v-if="topBreakdown" class="summary-breakdown">
        <div class="breakdown-row">
          <div class="breakdown-meta">
            <span>语义相似</span>
            <strong>{{ toPercent(topBreakdown.semantic_score) }}%</strong>
          </div>
          <el-progress :percentage="toPercent(topBreakdown.semantic_score)" :show-text="false" />
        </div>
        <div class="breakdown-row">
          <div class="breakdown-meta">
            <span>分类匹配</span>
            <strong>{{ toPercent(topBreakdown.category_score) }}%</strong>
          </div>
          <el-progress :percentage="toPercent(topBreakdown.category_score)" :show-text="false" color="#8b5cf6" />
        </div>
        <div class="breakdown-row">
          <div class="breakdown-meta">
            <span>新鲜度</span>
            <strong>{{ toPercent(topBreakdown.recency_score) }}%</strong>
          </div>
          <el-progress :percentage="toPercent(topBreakdown.recency_score)" :show-text="false" color="#0ea5e9" />
        </div>
        <div class="breakdown-row">
          <div class="breakdown-meta">
            <span>多样性</span>
            <strong>{{ toPercent(topDiversityMatch) }}%</strong>
          </div>
          <el-progress :percentage="toPercent(topDiversityMatch)" :show-text="false" color="#22c55e" />
        </div>
      </div>

      <div v-if="topRecommendation?.reason" class="summary-reason">
        {{ topRecommendation.reason }}
      </div>
    </div>

    <div v-if="store.recommendationsGenerating" class="loading">
      <el-skeleton :rows="4" animated />
    </div>

    <div v-else-if="store.recommendations.length === 0" class="empty-results">
      <el-empty description="暂无推荐论文，请先生成推荐结果" />
    </div>

    <div v-else class="recommendations-list">
      <div class="sort-info">按相似度排序，候选范围仅限 CS.AI</div>
      <div class="paper-grid">
        <PaperCard
          v-for="paper in store.recommendations"
          :key="paper.id"
          :paper="paper"
          :is-recommendation="true"
          @view-detail="handleViewDetail"
          @label="handleLabel"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.recommendations-page {
  padding: 20px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
}

.page-header h1 {
  font-size: 24px;
  font-weight: 700;
  color: #1f2937;
  margin: 0 0 8px 0;
}

.description {
  font-size: 14px;
  color: #6b7280;
  margin: 0;
}

.action-bar {
  display: flex;
  align-items: center;
}

.summary-panel {
  margin-bottom: 24px;
  padding: 20px;
  border-radius: 18px;
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.92)),
    radial-gradient(circle at top right, rgba(56, 189, 248, 0.18), transparent 38%),
    radial-gradient(circle at bottom left, rgba(16, 185, 129, 0.14), transparent 34%);
  color: #f8fafc;
  box-shadow: 0 18px 48px rgba(15, 23, 42, 0.18);
}

.summary-panel__lead {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: flex-start;
}

.summary-copy-wrap {
  min-width: 0;
}

.summary-kicker {
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #7dd3fc;
  margin-bottom: 8px;
}

.summary-title {
  margin: 0;
  font-size: 22px;
  line-height: 1.35;
  color: #fff;
}

.summary-copy {
  margin: 10px 0 0;
  max-width: 760px;
  color: rgba(226, 232, 240, 0.88);
  line-height: 1.65;
}

.summary-score {
  min-width: 180px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: flex-end;
}

.summary-score__value {
  font-size: 18px;
  font-weight: 700;
  color: #fff;
}

.summary-interest {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
  margin-top: 14px;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.36);
  color: rgba(226, 232, 240, 0.9);
  font-size: 13px;
}

.summary-breakdown {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px 18px;
  margin-top: 18px;
}

.breakdown-row {
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(15, 23, 42, 0.34);
  border: 1px solid rgba(148, 163, 184, 0.16);
}

.breakdown-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 13px;
  color: rgba(226, 232, 240, 0.88);
}

.summary-reason {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid rgba(148, 163, 184, 0.16);
  color: #e2e8f0;
  line-height: 1.65;
  font-size: 14px;
}

.loading {
  padding: 24px 0;
}

.empty-results {
  padding: 40px 0;
}

.sort-info {
  margin-bottom: 16px;
  font-size: 14px;
  color: #6b7280;
}

.paper-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 16px;
}

@media (max-width: 900px) {
  .page-header {
    flex-direction: column;
    gap: 16px;
  }

  .summary-panel__lead {
    flex-direction: column;
  }

  .summary-score {
    align-items: flex-start;
  }

  .summary-breakdown {
    grid-template-columns: 1fr;
  }

  .paper-grid {
    grid-template-columns: 1fr;
  }
}
</style>


