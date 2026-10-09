<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import { usePaperStore } from '@/stores/paperStore'

const store = usePaperStore()
const loading = ref(true)
const statsError = ref('')
type StatKey = keyof typeof store.stats

const statsConfig = [
  { key: 'totalPapers', label: '论文库总量', color: '#1890ff' },
  { key: 'labeledPapers', label: '我的标记论文', color: '#52c41a' },
  { key: 'latestSyncNewPapers', label: '最近同步新增论文数', color: '#faad14' }
] satisfies Array<{ key: StatKey; label: string; color: string }>

const syncStatusText = computed(() => {
  switch (store.stats.lastSyncStatus) {
    case 'success':
      return '同步成功'
    case 'failed':
      return '最近同步失败'
    default:
      return '暂无同步记录'
  }
})

const syncHintText = computed(() => {
  const parts: string[] = []
  if (store.stats.lastSyncedDate) {
    parts.push(`已同步到 ${store.stats.lastSyncedDate}`)
  }
  parts.push(syncStatusText.value)
  return parts.join(' · ')
})

const syncErrorText = computed(() => {
  if (store.stats.lastSyncStatus !== 'failed') return ''
  return store.stats.syncErrorMessage || '最近一次同步失败，请检查同步脚本日志。'
})

onMounted(async () => {
  try {
    await store.fetchStats()
  } catch (error) {
    statsError.value = '统计数据加载失败，请检查后端服务是否正常运行。'
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
        首页展示系统真实统计数据，可快速了解论文库规模、个人标记进度，以及最近一次完整日增量同步的入库结果。
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
            <h3>个性化发现</h3>
            <p>结合您的标记记录，辅助发现更相关的论文内容</p>
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
          <div v-if="stat.key === 'latestSyncNewPapers'" class="stat-hint">
            {{ syncHintText }}
          </div>
        </div>
      </div>
    </div>

    <div v-if="syncErrorText" class="sync-warning">
      {{ syncErrorText }}
    </div>

    <div v-if="statsError" class="stats-error">
      {{ statsError }}
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
            <h3>继续浏览</h3>
            <p>结合标记结果继续筛选论文，逐步沉淀更清晰的兴趣方向</p>
          </div>
        </div>
      </div>
    </div>
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
  grid-template-columns: repeat(3, 1fr);
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

.stat-hint {
  margin-top: 8px;
  font-size: 12px;
  color: #6b7280;
}

.skeleton {
  height: 60px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: loading 1.5s infinite;
  border-radius: 4px;
}

.stats-error {
  margin-bottom: 24px;
  padding: 12px 16px;
  border-radius: 8px;
  color: #b91c1c;
  background: #fef2f2;
  border: 1px solid #fecaca;
}

.sync-warning {
  margin-bottom: 24px;
  padding: 12px 16px;
  border-radius: 8px;
  color: #92400e;
  background: #fff7ed;
  border: 1px solid #fed7aa;
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
