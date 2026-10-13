<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import type { Component } from 'vue'
import { Collection, CollectionTag, Refresh, Search, MagicStick, Star, Compass, ArrowRight } from '@element-plus/icons-vue'
import { usePaperStore } from '@/stores/paperStore'
import { useUserContext } from '@/composables/useUserContext'

const store = usePaperStore()
const userContext = useUserContext()
const loading = ref(true)
const statsError = ref('')
type StatKey = keyof typeof store.stats

const statsConfig = [
  { key: 'totalPapers', label: '论文库总量', icon: Collection, tone: 'primary' },
  { key: 'labeledPapers', label: '我的标记论文', icon: CollectionTag, tone: 'success' },
  { key: 'latestSyncNewPapers', label: '最近同步新增', icon: Refresh, tone: 'warning' }
] satisfies Array<{ key: StatKey; label: string; icon: Component; tone: string }>

interface QuickEntry {
  path: string
  title: string
  desc: string
  icon: Component
}

const quickEntries = computed<QuickEntry[]>(() => {
  const entries: QuickEntry[] = [
    { path: '/search', title: '论文搜索', desc: '按关键词、作者、分类检索本地 arXiv 镜像库', icon: Search },
    { path: '/recommendations', title: '推荐论文', desc: '基于你的标记与研究画像生成个性化推荐', icon: Star },
    { path: '/profile', title: '研究画像', desc: '查看并调整系统对你研究方向的理解', icon: Compass },
    { path: '/labeled', title: '已标记论文', desc: '回顾标记过的论文，整理阅读清单', icon: CollectionTag }
  ]
  if (userContext.canResearch.value) {
    entries.splice(1, 0, { path: '/agent-search', title: 'Agent 搜索', desc: '用自然语言描述需求，由 Agent 规划检索并总结', icon: MagicStick })
  }
  return entries
})

const greeting = computed(() => {
  const name = userContext.displayName.value
  return name && name !== '未登录' ? `欢迎回来，${name}` : '欢迎回来'
})

const syncStatusText = computed(() => {
  switch (store.stats.lastSyncStatus) {
    case 'success':
      return '同步成功'
    case 'failed':
      return '最近同步失败'
    // 同步脚本按天推进，运行中会持续刷新摘要；“已同步到”显示的是已完成的最后一天。
    case 'running':
      return '同步进行中'
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

// 总数读取失败时后端返回 null；同步写库期间最常见，提示稍后刷新，而不是显示 0 篇误导用户。
const totalPapersUnavailableText = computed(() => {
  if (store.stats.totalPapers !== null) return ''
  return store.stats.lastSyncStatus === 'running'
    ? '同步写入中，暂时无法统计，请稍后刷新'
    : '暂时无法统计，请稍后刷新'
})

function formatStatValue(key: StatKey) {
  const value = store.stats[key]
  if (value === null || value === undefined) return '—'
  return typeof value === 'number' ? value.toLocaleString('zh-CN') : value
}

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
  <div class="page dashboard">
    <header class="page-header">
      <div class="page-header__text">
        <h1 class="page-title">{{ greeting }}</h1>
        <p class="page-desc">这里是论文库的整体情况和常用入口。</p>
      </div>
    </header>

    <section class="stats-row">
      <div v-for="stat in statsConfig" :key="stat.key" class="surface stat-card">
        <el-skeleton v-if="loading" :rows="1" animated />
        <template v-else>
          <div class="stat-top">
            <span class="stat-label">{{ stat.label }}</span>
            <span :class="['stat-icon', `stat-icon--${stat.tone}`]">
              <el-icon><component :is="stat.icon" /></el-icon>
            </span>
          </div>
          <div class="stat-value">{{ formatStatValue(stat.key) }}</div>
          <template v-if="stat.key === 'totalPapers'">
            <div class="stat-hint">{{ totalPapersUnavailableText || '篇论文已入库' }}</div>
          </template>
          <div v-else-if="stat.key === 'labeledPapers'" class="stat-hint">篇论文已标记偏好</div>
          <div v-else class="stat-hint">
            <span :class="['sync-dot', `sync-dot--${store.stats.lastSyncStatus || 'none'}`]" />
            {{ syncHintText }}
          </div>
        </template>
      </div>
    </section>

    <div v-if="syncErrorText" class="notice notice--warning section">
      {{ syncErrorText }}
    </div>

    <div v-if="statsError" class="notice notice--danger section">
      {{ statsError }}
    </div>

    <section class="section">
      <div class="section-head">
        <h2 class="section-title">快速开始</h2>
      </div>
      <div class="entry-grid">
        <router-link v-for="entry in quickEntries" :key="entry.path" :to="entry.path" class="surface entry-card">
          <span class="entry-icon">
            <el-icon><component :is="entry.icon" /></el-icon>
          </span>
          <span class="entry-body">
            <span class="entry-title">{{ entry.title }}</span>
            <span class="entry-desc">{{ entry.desc }}</span>
          </span>
          <el-icon class="entry-arrow"><ArrowRight /></el-icon>
        </router-link>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2 class="section-title">推荐使用流程</h2>
      </div>
      <ol class="surface steps">
        <li class="step">
          <span class="step-number">1</span>
          <div>
            <div class="step-title">搜索论文</div>
            <p class="step-desc">在论文搜索页输入关键词，筛选感兴趣的分类</p>
          </div>
        </li>
        <li class="step">
          <span class="step-number">2</span>
          <div>
            <div class="step-title">标记偏好</div>
            <p class="step-desc">把相关论文标记为“感兴趣”，无关的标记为“不感兴趣”</p>
          </div>
        </li>
        <li class="step">
          <span class="step-number">3</span>
          <div>
            <div class="step-title">获取推荐与问答</div>
            <p class="step-desc">标记积累后生成推荐，并在详情页对单篇论文提问</p>
          </div>
        </li>
      </ol>
    </section>
  </div>
</template>

<style scoped>
.stats-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.stat-card {
  padding: 18px 20px;
}

.stat-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.stat-label {
  font-size: 13px;
  color: var(--color-text-3);
}

.stat-icon {
  width: 30px;
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  font-size: 16px;
}

.stat-icon--primary {
  background: var(--color-primary-soft);
  color: var(--color-primary);
}

.stat-icon--success {
  background: var(--color-success-soft);
  color: var(--color-success);
}

.stat-icon--warning {
  background: var(--color-warning-soft);
  color: var(--color-warning);
}

.stat-value {
  margin-top: 8px;
  font-size: 30px;
  font-weight: 650;
  line-height: 1.2;
  letter-spacing: -0.02em;
  color: var(--color-text);
  font-variant-numeric: tabular-nums;
}

.stat-hint {
  margin-top: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--color-text-3);
}

.sync-dot {
  width: 7px;
  height: 7px;
  flex-shrink: 0;
  border-radius: 50%;
  background: var(--color-text-4);
}

.sync-dot--success {
  background: var(--color-success);
}

.sync-dot--running {
  background: var(--color-primary);
  animation: pulse 1.4s ease-in-out infinite;
}

.sync-dot--failed {
  background: var(--color-danger);
}

.entry-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 12px;
}

.entry-card {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px;
  color: inherit;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.entry-card:hover {
  color: inherit;
  border-color: var(--color-primary-border);
  box-shadow: var(--shadow-sm);
}

.entry-icon {
  width: 38px;
  height: 38px;
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  background: var(--color-primary-soft);
  color: var(--color-primary);
  font-size: 18px;
}

.entry-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.entry-title {
  font-weight: 600;
  color: var(--color-text);
}

.entry-desc {
  font-size: 12.5px;
  color: var(--color-text-3);
  line-height: 1.5;
}

.entry-arrow {
  color: var(--color-text-4);
  transition: transform 0.15s ease, color 0.15s ease;
}

.entry-card:hover .entry-arrow {
  color: var(--color-primary);
  transform: translateX(2px);
}

.steps {
  list-style: none;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.step {
  display: flex;
  gap: 12px;
  padding: 18px 20px;
}

.step + .step {
  border-left: 1px solid var(--color-border);
}

.step-number {
  width: 24px;
  height: 24px;
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  border: 1px solid var(--color-primary-border);
  color: var(--color-primary);
  font-size: 12px;
  font-weight: 650;
}

.step-title {
  font-weight: 600;
  color: var(--color-text);
}

.step-desc {
  margin-top: 2px;
  font-size: 13px;
  color: var(--color-text-3);
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}

@media (max-width: 960px) {
  .stats-row,
  .steps {
    grid-template-columns: 1fr;
  }

  .step + .step {
    border-left: none;
    border-top: 1px solid var(--color-border);
  }
}
</style>
