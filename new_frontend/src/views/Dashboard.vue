<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { usePaperStore } from '@/stores/paperStore'

const store = usePaperStore()
const loading = ref(true)
type StatKey = keyof typeof store.stats

const statsConfig = [
  { key: 'totalPapers', label: '已抓取论文', color: '#1890ff' },
  { key: 'labeledPapers', label: '已标记论文', color: '#52c41a' },
  { key: 'todayNewPapers', label: '今日新增', color: '#faad14' },
  { key: 'recommendedPapers', label: '推荐论文', color: '#f5222d' }
] satisfies Array<{ key: StatKey; label: string; color: string }>

onMounted(async () => {
  try {
    await store.fetchStats()
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div class="dashboard">
    <div class="dashboard-header">
      <h1>arXiv 计算机论文推荐系统</h1>
      <p class="description">
        基于向量相似度的智能论文推荐平台，帮助您发现最新、最相关的计算机领域研究成果。
      </p>
    </div>

    <div class="feature-card">
      <h2 class="section-title">系统功能</h2>
      <div class="features">
        <div class="feature-item">
          <div class="feature-content">
            <h3>论文搜索</h3>
            <p>支持关键词搜索、分类筛选、时间排序</p>
          </div>
        </div>
        <div class="feature-item">
          <div class="feature-content">
            <h3>论文详情</h3>
            <p>查看论文完整信息，包括标题、作者、摘要等</p>
          </div>
        </div>
        <div class="feature-item">
          <div class="feature-content">
            <h3>智能推荐</h3>
            <p>基于历史标记论文进行向量相似度匹配</p>
          </div>
        </div>
        <div class="feature-item">
          <div class="feature-content">
            <h3>论文标记</h3>
            <p>标记论文为相关或不相关，助力推荐系统学习</p>
          </div>
        </div>
      </div>
    </div>

    <div class="stats-row">
      <div v-for="stat in statsConfig" :key="stat.key" class="stat-card">
        <div v-if="loading" class="skeleton"></div>
        <div v-else class="stat-content">
          <div class="stat-label">{{ stat.label }}</div>
          <div class="stat-value" :style="{ color: stat.color }">
            {{ store.stats[stat.key] }}
          </div>
        </div>
      </div>
    </div>

    <div class="guide-card">
      <h2 class="section-title">使用指南</h2>
      <div class="guide-steps">
        <div class="guide-step">
          <div class="step-number">1</div>
          <div class="step-content">
            <h3>搜索论文</h3>
            <p>在论文搜索页输入关键词，筛选感兴趣的分类</p>
          </div>
        </div>
        <div class="guide-step">
          <div class="step-number">2</div>
          <div class="step-content">
            <h3>标记论文</h3>
            <p>将感兴趣的论文标记为"相关"，不感兴趣的标记为"不相关"</p>
          </div>
        </div>
        <div class="guide-step">
          <div class="step-number">3</div>
          <div class="step-content">
            <h3>查看推荐</h3>
            <p>系统会根据您的标记自动推荐相似论文</p>
          </div>
        </div>
      </div>
    </div>

    <div v-if="loading" class="loading">加载中...</div>
  </div>
</template>

<style scoped>
.dashboard {
  padding: 20px;
}

.dashboard-header {
  margin-bottom: 24px;
}

.dashboard-header h1 {
  font-size: 28px;
  font-weight: 700;
  color: #1f2937;
  margin: 0 0 12px 0;
}

.description {
  font-size: 14px;
  color: #6b7280;
  margin: 0;
}

.feature-card {
  margin-bottom: 24px;
}

.section-title {
  font-size: 18px;
  font-weight: 600;
  color: #374151;
  margin: 0 0 16px 0;
}

.features {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
}

.feature-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 16px;
  background: #f9fafb;
  border-radius: 8px;
}

.feature-content h3 {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 4px 0;
}

.feature-content p {
  font-size: 13px;
  color: #6b7280;
  margin: 0;
}

.stat-card {
  text-align: center;
}

.stat-content {
  padding: 8px;
}

.guide-card {
  margin-top: 24px;
}

.guide-steps {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
}

.guide-step {
  display: flex;
  gap: 16px;
}

.step-number {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  font-weight: 600;
  flex-shrink: 0;
}

.step-content h3 {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 4px 0 4px 0;
}

.step-content p {
  font-size: 14px;
  color: #6b7280;
  margin: 0;
}

@media (max-width: 1200px) {
  .features {
    grid-template-columns: repeat(2, 1fr);
  }
  .guide-steps {
    grid-template-columns: repeat(1, 1fr);
  }
}

.stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.stat-card {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.stat-label {
  font-size: 14px;
  color: #6b7280;
  margin-bottom: 8px;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
}

.skeleton {
  height: 60px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: loading 1.5s infinite;
  border-radius: 4px;
}

.loading {
  text-align: center;
  padding: 20px;
  color: #6b7280;
}

@keyframes loading {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}
</style>
