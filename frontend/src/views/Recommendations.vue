<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { MagicStick } from '@element-plus/icons-vue'
import { usePaperStore } from '@/stores/paperStore'
import PaperCard from '@/components/PaperCard.vue'
import SimilarityTag from '@/components/SimilarityTag.vue'
import { toPercent } from '@/utils/format'

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
const researchProfile = computed(() => store.researchProfile)
const profileTopicPreview = computed(() => (researchProfile.value?.positive_topics || []).slice(0, 4))
const profileNegativePreview = computed(() => (researchProfile.value?.negative_topics || []).slice(0, 3))
const profileCategoryPreview = computed(() => (researchProfile.value?.preferred_categories || []).slice(0, 4))
const interestClusterCount = computed(() => interestVector.value?.cluster_count || 0)
const interestProfileMode = computed(() => interestVector.value?.profile_mode || 'mean')
const topFinalScore = computed(() => {
  const paper = topRecommendation.value
  if (!paper) return 0
  return typeof paper.finalScore === 'number' ? paper.finalScore : paper.similarityScore || 0
})

onMounted(() => {
  store.fetchUserInterestVector()
  store.fetchResearchProfile()
  store.fetchPaperActions()
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

function goToProfile() {
  router.push('/profile')
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
  <div class="page recs-page">
    <header class="page-header">
      <div class="page-header__text">
        <h1 class="page-title">推荐论文</h1>
        <p class="page-desc">根据你标记过的论文和研究画像，从 CS.AI 论文中挑选最相关的一批。</p>
      </div>
      <div class="page-actions">
        <el-select v-model="recommendationCount" style="width: 100px" aria-label="推荐数量">
          <el-option v-for="option in countOptions" :key="option.value" :value="option.value" :label="option.label" />
        </el-select>
        <el-select v-model="recommendationAgeMonths" style="width: 130px" aria-label="时间范围">
          <el-option v-for="option in ageOptions" :key="option.value" :value="option.value" :label="option.label" />
        </el-select>
        <el-button type="primary" :icon="MagicStick" :loading="store.recommendationsGenerating" @click="handleGenerateRecommendations">
          生成推荐
        </el-button>
      </div>
    </header>

    <section v-if="researchProfile" class="surface profile-strip">
      <div class="profile-strip__head">
        <div>
          <div class="section-title">参考的研究画像</div>
          <div class="section-desc">推荐会轻量参考你的长期画像</div>
        </div>
        <el-button link type="primary" @click="goToProfile">编辑画像</el-button>
      </div>
      <div class="profile-strip__body">
        <div v-if="profileTopicPreview.length" class="profile-line">
          <span class="profile-label">关注主题</span>
          <div class="chip-row">
            <span v-for="item in profileTopicPreview" :key="item" class="chip chip--primary">{{ item }}</span>
          </div>
        </div>
        <div v-if="profileCategoryPreview.length" class="profile-line">
          <span class="profile-label">偏好分类</span>
          <div class="chip-row">
            <span v-for="item in profileCategoryPreview" :key="item" class="chip mono">{{ item }}</span>
          </div>
        </div>
        <div v-if="profileNegativePreview.length" class="profile-line">
          <span class="profile-label">排除主题</span>
          <div class="chip-row">
            <span v-for="item in profileNegativePreview" :key="item" class="chip chip--danger">{{ item }}</span>
          </div>
        </div>
        <div v-if="researchProfile.preferred_answer_style" class="profile-line">
          <span class="profile-label">回答风格</span>
          <span class="profile-note">{{ researchProfile.preferred_answer_style }}</span>
        </div>
      </div>
    </section>

    <div v-if="store.recommendationsGenerating" class="list">
      <div v-for="n in 3" :key="n" class="surface skeleton-card">
        <el-skeleton :rows="3" animated />
      </div>
    </div>

    <div v-else-if="store.recommendations.length === 0" class="surface empty">
      <el-empty description="还没有推荐结果。选择数量和时间范围后点击「生成推荐」" :image-size="96" />
    </div>

    <template v-else>
      <section class="surface top-pick">
        <div class="top-pick__main">
          <span class="chip chip--primary">首推</span>
          <h2 class="top-pick__title" @click="topRecommendation && handleViewDetail(topRecommendation.id)">
            {{ topRecommendation?.title }}
          </h2>
          <p class="top-pick__copy">
            在语义相似度和主题匹配上最符合你当前的兴趣画像，发表于最近 {{ recommendationAgeMonths }} 个月内。
          </p>
          <p v-if="topRecommendation?.reason" class="top-pick__reason">{{ topRecommendation.reason }}</p>
          <div v-if="interestVector" class="chip-row top-pick__meta">
            <span class="chip">画像模式 {{ interestProfileMode }}</span>
            <span class="chip">兴趣簇 {{ interestClusterCount }}</span>
          </div>
        </div>

        <div class="top-pick__score">
          <div class="score-hero">
            <span class="score-hero__value">{{ Math.round(topFinalScore * 100) }}<small>%</small></span>
            <span class="score-hero__label">综合得分</span>
          </div>
          <SimilarityTag :score="topRecommendation?.similarityScore || 0" />
          <div v-if="topBreakdown" class="score-bars">
            <div class="score-bar">
              <span>语义相似</span>
              <el-progress :percentage="toPercent(topBreakdown.semantic_score)" :show-text="false" :stroke-width="6" color="var(--color-chart-1)" />
              <strong>{{ toPercent(topBreakdown.semantic_score) }}%</strong>
            </div>
            <div class="score-bar">
              <span>分类匹配</span>
              <el-progress :percentage="toPercent(topBreakdown.category_score)" :show-text="false" :stroke-width="6" color="var(--color-chart-2)" />
              <strong>{{ toPercent(topBreakdown.category_score) }}%</strong>
            </div>
            <div class="score-bar">
              <span>新鲜度</span>
              <el-progress :percentage="toPercent(topBreakdown.recency_score)" :show-text="false" :stroke-width="6" color="var(--color-chart-3)" />
              <strong>{{ toPercent(topBreakdown.recency_score) }}%</strong>
            </div>
            <div class="score-bar">
              <span>多样性</span>
              <el-progress :percentage="toPercent(topDiversityMatch)" :show-text="false" :stroke-width="6" color="var(--color-chart-4)" />
              <strong>{{ toPercent(topDiversityMatch) }}%</strong>
            </div>
          </div>
        </div>
      </section>

      <div class="list-head">
        <h2 class="section-title">全部推荐 <span class="muted">{{ store.recommendations.length }}</span></h2>
        <span class="muted">按综合得分排序 · 候选范围仅限 CS.AI</span>
      </div>
      <div class="list">
        <PaperCard
          v-for="paper in store.recommendations"
          :key="paper.id"
          :paper="paper"
          :is-recommendation="true"
          @view-detail="handleViewDetail"
          @label="handleLabel"
        />
      </div>
    </template>
  </div>
</template>

<style scoped>
.recs-page {
  max-width: 1080px;
}

.profile-strip {
  padding: 16px 20px;
  margin-bottom: 20px;
}

.profile-strip__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.profile-strip__body {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.profile-line {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.profile-label {
  flex-shrink: 0;
  width: 64px;
  font-size: 12.5px;
  line-height: 22px;
  color: var(--color-text-3);
}

.profile-note {
  font-size: 13px;
  line-height: 22px;
  color: var(--color-text-2);
}

.top-pick {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: 28px;
  padding: 22px 24px;
  margin-bottom: 28px;
  border-color: var(--color-primary-border);
  background: linear-gradient(180deg, var(--color-primary-soft) 0%, var(--color-surface) 70%);
}

.top-pick__title {
  margin-top: 10px;
  font-size: 19px;
  font-weight: 650;
  line-height: 1.4;
  color: var(--color-text);
  cursor: pointer;
}

.top-pick__title:hover {
  color: var(--color-primary);
}

.top-pick__copy {
  margin-top: 8px;
  font-size: 13.5px;
  color: var(--color-text-3);
}

.top-pick__reason {
  margin-top: 12px;
  padding: 10px 12px;
  border-radius: var(--radius);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--color-text-2);
}

.top-pick__meta {
  margin-top: 12px;
}

.top-pick__score {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-left: 24px;
  border-left: 1px solid var(--color-border);
}

.score-hero {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.score-hero__value {
  font-size: 40px;
  font-weight: 700;
  line-height: 1;
  letter-spacing: -0.03em;
  color: var(--color-primary);
  font-variant-numeric: tabular-nums;
}

.score-hero__value small {
  font-size: 18px;
  font-weight: 600;
}

.score-hero__label {
  font-size: 13px;
  color: var(--color-text-3);
}

.score-bars {
  display: grid;
  gap: 8px;
}

.score-bar {
  display: grid;
  grid-template-columns: 60px 1fr 40px;
  align-items: center;
  gap: 10px;
  font-size: 12.5px;
  color: var(--color-text-3);
}

.score-bar strong {
  text-align: right;
  font-weight: 600;
  color: var(--color-text-2);
  font-variant-numeric: tabular-nums;
}

.list-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
  font-size: 12.5px;
}

.list-head .section-title .muted {
  font-weight: 500;
  margin-left: 4px;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.skeleton-card {
  padding: 20px;
}

.empty {
  padding: 24px;
}

@media (max-width: 860px) {
  .top-pick {
    grid-template-columns: 1fr;
  }

  .top-pick__score {
    padding-left: 0;
    padding-top: 16px;
    border-left: none;
    border-top: 1px solid var(--color-border);
  }
}
</style>
