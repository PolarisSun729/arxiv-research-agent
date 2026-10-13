<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue';
import { useRouter } from 'vue-router';
import { usePaperStore } from '@/stores/paperStore';
import { ElMessage } from 'element-plus';
import { MagicStick } from '@element-plus/icons-vue';
import { getLabelText } from '@/utils/paperLabel';

const router = useRouter();
const store = usePaperStore();

const filterLabel = ref<string>('');
const currentPage = ref(1);
const pageSize = ref(10);

const filterOptions = [
  { value: '', label: '全部' },
  { value: 'liked', label: '感兴趣' },
  { value: 'disliked', label: '不感兴趣' }
];
const expandedIds = ref(new Set<string>());

function toggleExpanded(id: string) {
  const next = new Set(expandedIds.value);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  expandedIds.value = next;
}

const interestVector = computed(() => store.lastInterestVector);
const interestClusters = computed(() => interestVector.value?.interest_clusters || []);
const clusterModeText = computed(() => interestVector.value?.profile_mode || 'mean');

onMounted(() => {
  fetchLabeledPapers();
  store.fetchUserInterestVector();
});

watch(filterLabel, () => {
  // 切换筛选后从第一页开始；页码本就是 1 时由下方监听负责重新加载。
  currentPage.value = 1;
});

watch([filterLabel, currentPage], () => {
  fetchLabeledPapers();
});

async function fetchLabeledPapers() {
  await store.fetchLabeledPapers({
    label: filterLabel.value ? (filterLabel.value as 'liked' | 'disliked') : undefined,
    page: currentPage.value,
    pageSize: pageSize.value
  });
}

function handleViewDetail(id: string) {
  router.push(`/paper/${id}`);
}

function handlePageChange(page: number) {
  currentPage.value = page;
}

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleString('zh-CN');
}

function formatPaperTitle(title: string, id: string) {
  const normalizedTitle = String(title || '').trim()
  // 回源失败时仍保留可识别的 arXiv ID，避免空标题让用户误以为卡片渲染坏了。
  return normalizedTitle || `论文 ${id}`
}

function formatPaperSummary(summary: string) {
  const normalizedSummary = String(summary || '').trim()
  if (!normalizedSummary) {
    return '摘要暂不可用'
  }
  // 摘要是用户判断论文是否值得继续阅读的核心信息，列表页不再静默截断原文。
  return normalizedSummary
}

async function handleGenerateInterestVector() {
  try {
    const result = await store.generateUserInterestVector();
    const milvusUsedCount = result.milvus_used_count ?? 0;
    const fallbackUsedCount = result.fallback_used_count ?? 0;
    const unresolvedCount = result.unresolved_count ?? 0;
    const totalUsedCount = result.used_count ?? result.paper_count ?? (milvusUsedCount + fallbackUsedCount);
    const parts = [
      `使用了 ${milvusUsedCount} 条已存 Milvus 向量`,
      fallbackUsedCount > 0 ? `并对 ${fallbackUsedCount} 条缺失向量做了 fallback 重算` : ''
    ].filter(Boolean);
    const warning = unresolvedCount > 0
      ? `，另有 ${unresolvedCount} 条论文仍缺少可用向量`
      : '';

    ElMessage.success(
      `用户兴趣向量生成成功！${parts.join('，')}，共计 ${totalUsedCount} 条，生成 ${result.vector_dimension} 维向量${warning}`
    );
  } catch (error: any) {
    if (error?.response?.data?.detail) {
      ElMessage.error(error.response.data.detail);
    } else {
      ElMessage.error('生成用户兴趣向量失败');
    }
  }
}
</script>

<template>
  <div class="page labeled-page">
    <header class="page-header">
      <div class="page-header__text">
        <h1 class="page-title">已标记论文</h1>
        <p class="page-desc">回顾标记过的论文。标记积累后可生成偏好向量，推荐会据此寻找相似论文。</p>
      </div>
      <div class="page-actions">
        <el-button
          type="primary"
          :icon="MagicStick"
          :loading="store.interestVectorGenerating"
          :disabled="store.labeledPapers.length === 0"
          @click="handleGenerateInterestVector"
        >
          生成偏好向量
        </el-button>
      </div>
    </header>

    <section v-if="interestVector" class="surface cluster-panel">
      <div class="section-head">
        <div>
          <h2 class="section-title">兴趣簇概览</h2>
          <p class="section-desc">
            画像模式 <span class="mono">{{ clusterModeText }}</span> ·
            {{ interestVector.cluster_count || interestClusters.length || 0 }} 个簇
          </p>
        </div>
        <span :class="['chip', interestClusters.length > 0 ? 'chip--success' : '']">
          {{ interestClusters.length > 0 ? '已聚类' : '均值回退' }}
        </span>
      </div>

      <div v-if="interestClusters.length > 0" class="cluster-grid">
        <div v-for="cluster in interestClusters" :key="cluster.cluster_id" class="surface-muted cluster-card">
          <div class="cluster-top">
            <strong class="mono">{{ cluster.cluster_id }}</strong>
            <span class="muted">{{ cluster.paper_count }} 篇</span>
          </div>
          <div class="chip-row">
            <span v-for="paperId in cluster.paper_ids.slice(0, 4)" :key="paperId" class="chip mono">{{ paperId }}</span>
            <span v-if="cluster.paper_ids.length > 4" class="muted more">+{{ cluster.paper_ids.length - 4 }}</span>
          </div>
        </div>
      </div>
      <p v-else class="muted cluster-empty">感兴趣的论文数量不足，暂时回退为单一均值向量。</p>
    </section>

    <div class="toolbar">
      <el-segmented v-model="filterLabel" :options="filterOptions" />
      <span v-if="store.totalLabeledPapers > 0" class="muted">共 {{ store.totalLabeledPapers }} 篇</span>
    </div>

    <div v-if="store.loading" class="labeled-list">
      <div v-for="n in 3" :key="n" class="surface skeleton-card">
        <el-skeleton :rows="2" animated />
      </div>
    </div>

    <div v-else-if="store.labeledPapers.length === 0" class="surface empty">
      <el-empty description="还没有标记过论文，去搜索页标记几篇感兴趣的论文吧" :image-size="96">
        <el-button type="primary" plain @click="router.push('/search')">去搜索论文</el-button>
      </el-empty>
    </div>

    <div v-else class="labeled-list">
      <article
        v-for="paper in store.labeledPapers"
        :key="paper.id"
        :class="['surface', 'labeled-card', paper.label === 'liked' ? 'is-liked' : 'is-disliked']"
      >
        <div class="card-head">
          <h3 class="paper-title" @click="handleViewDetail(paper.id)">{{ formatPaperTitle(paper.title, paper.id) }}</h3>
          <span :class="['chip', paper.label === 'liked' ? 'chip--success' : 'chip--danger']">
            {{ getLabelText(paper.label) }}
          </span>
        </div>

        <p :class="['paper-summary', { 'clamp-3': !expandedIds.has(paper.id) }]">{{ formatPaperSummary(paper.summary) }}</p>

        <div class="card-foot">
          <span class="muted">标记于 {{ formatDate(paper.labeledAt) }}</span>
          <div class="card-actions">
            <el-button link type="primary" @click="toggleExpanded(paper.id)">
              {{ expandedIds.has(paper.id) ? '收起摘要' : '展开摘要' }}
            </el-button>
            <el-button size="small" type="primary" plain @click="handleViewDetail(paper.id)">查看详情</el-button>
          </div>
        </div>
      </article>
    </div>

    <div v-if="store.totalLabeledPapers > pageSize" class="pagination">
      <el-pagination
        :current-page="currentPage"
        :page-size="pageSize"
        :total="store.totalLabeledPapers"
        background
        layout="prev, pager, next, jumper"
        @current-change="handlePageChange"
      />
    </div>
  </div>
</template>

<style scoped>
.labeled-page {
  max-width: 1080px;
}

.cluster-panel {
  padding: 18px 20px;
  margin-bottom: 20px;
}

.cluster-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 10px;
}

.cluster-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px 14px;
}

.cluster-top {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
}

.more {
  font-size: 12px;
  align-self: center;
}

.cluster-empty {
  font-size: 13px;
}

.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
  font-size: 13px;
}

.labeled-list {
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

.labeled-card {
  padding: 16px 20px 12px;
}

.labeled-card.is-liked {
  border-left: 3px solid var(--color-success);
}

.labeled-card.is-disliked {
  border-left: 3px solid var(--color-border-strong);
}

.card-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.paper-title {
  font-size: 15.5px;
  font-weight: 600;
  line-height: 1.45;
  color: var(--color-text);
  cursor: pointer;
}

.paper-title:hover {
  color: var(--color-primary);
}

.paper-summary {
  margin-top: 8px;
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--color-text-2);
}

.card-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--color-border);
  font-size: 12.5px;
}

.card-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.pagination {
  display: flex;
  justify-content: center;
  margin-top: 24px;
}
</style>
