<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue';
import { useRouter } from 'vue-router';
import { usePaperStore } from '@/stores/paperStore';
import { ElMessage } from 'element-plus';
import { MagicStick } from '@element-plus/icons-vue';
import { getLabelText } from '@/utils/paperLabel';
import { useAbstractTranslation } from '@/composables/useAbstractTranslation';

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
const { displaySummary, isTranslating, translateButtonText, toggleTranslation } = useAbstractTranslation();

function toggleExpanded(id: string) {
  const next = new Set(expandedIds.value);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  expandedIds.value = next;
}

const interestVector = computed(() => store.lastInterestVector);
const interestClusters = computed(() => interestVector.value?.interest_clusters || []);
const weakInterestPool = computed(() => {
  const pool = interestVector.value?.weak_interest_pool;
  return pool && pool.paper_ids?.length ? pool : null;
});
const clusterModeText = computed(() => interestVector.value?.profile_mode || 'mean');

// 簇筛选值与后端 cluster 参数一致：cluster_N / weak_interest_pool / unclustered。
const filterCluster = ref('');

interface ClusterFilterCard {
  value: string;
  title: string;
  hint: string;
  paperCount: number | null;
  duplicateIds: string[];
}

function stripArxivVersion(id: string) {
  return id.replace(/v\d+$/i, '');
}

// 同一篇论文带不带版本号会被当成两篇参与聚类，单独提示出来，便于发现虚高的小簇。
function findVersionDuplicates(paperIds: string[]) {
  const groups = new Map<string, string[]>();
  for (const id of paperIds) {
    const base = stripArxivVersion(id);
    groups.set(base, [...(groups.get(base) || []), id]);
  }
  return [...groups.values()].filter(ids => ids.length > 1).map(ids => ids.join(' / '));
}

const clusterCards = computed<ClusterFilterCard[]>(() => {
  const cards: ClusterFilterCard[] = interestClusters.value.map((cluster, index) => ({
    value: cluster.cluster_id,
    title: `兴趣簇 ${index + 1}`,
    hint: cluster.cluster_id,
    paperCount: cluster.paper_count ?? cluster.paper_ids.length,
    duplicateIds: findVersionDuplicates(cluster.paper_ids)
  }));
  if (weakInterestPool.value) {
    cards.push({
      value: 'weak_interest_pool',
      title: '弱兴趣池',
      hint: '没有归入任何簇的零散兴趣',
      paperCount: weakInterestPool.value.paper_count ?? weakInterestPool.value.paper_ids.length,
      duplicateIds: findVersionDuplicates(weakInterestPool.value.paper_ids)
    });
  }
  if (interestVector.value) {
    cards.push({
      value: 'unclustered',
      title: '未参与聚类',
      hint: '点赞了但没进簇，多为缺少向量或在生成偏好向量之后才点赞',
      paperCount: null,
      duplicateIds: []
    });
  }
  return cards;
});

const activeClusterCard = computed(() => clusterCards.value.find(card => card.value === filterCluster.value) || null);

function toggleClusterFilter(value: string) {
  filterCluster.value = filterCluster.value === value ? '' : value;
}

onMounted(() => {
  fetchLabeledPapers();
  store.fetchUserInterestVector();
});

watch(filterCluster, value => {
  // 簇里只有点赞论文，选簇时清掉标签筛选，避免「不感兴趣 + 某簇」出现必然为空的组合。
  if (value) filterLabel.value = '';
});

watch([filterLabel, filterCluster], () => {
  // 切换筛选后从第一页开始；页码本就是 1 时由下方监听负责重新加载。
  currentPage.value = 1;
});

watch([filterLabel, filterCluster, currentPage], () => {
  fetchLabeledPapers();
});

async function fetchLabeledPapers() {
  try {
    await store.fetchLabeledPapers({
      label: filterLabel.value ? (filterLabel.value as 'liked' | 'disliked') : undefined,
      cluster: filterCluster.value || undefined,
      page: currentPage.value,
      pageSize: pageSize.value
    });
  } catch {
    // 偏好向量重新生成后旧的簇 ID 可能已不存在，回到全部论文而不是停在报错状态。
    if (filterCluster.value) {
      ElMessage.warning('该兴趣簇已不存在，可能是偏好向量刚重新生成');
      filterCluster.value = '';
    } else {
      ElMessage.error('加载已标记论文失败');
    }
  }
}

function handleViewDetail(id: string) {
  router.push(`/paper/${id}`);
}

function handlePageChange(page: number) {
  currentPage.value = page;
}

function formatLabeledAt(value: string | null) {
  if (!value) return '';
  // SQLite CURRENT_TIMESTAMP 是不带时区的 UTC 字符串，补上 Z 再按本地时区展示。
  const normalized = /[zZ]|[+-]\d{2}:?\d{2}$/.test(value) ? value : `${value.replace(' ', 'T')}Z`;
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString('zh-CN', { hour12: false });
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
    // 重新聚类后簇编号会变，旧筛选不再对应原来那批论文。
    filterCluster.value = '';
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

      <p v-if="interestClusters.length === 0" class="muted cluster-empty">感兴趣的论文数量不足，暂时回退为单一均值向量。</p>
      <p v-else class="muted cluster-tip">点击某个簇，下方只列出这个簇里的论文；再点一次取消筛选。</p>

      <div v-if="clusterCards.length > 0" class="cluster-grid">
        <button
          v-for="card in clusterCards"
          :key="card.value"
          type="button"
          :class="['surface-muted', 'cluster-card', { 'is-selected': filterCluster === card.value, 'is-secondary': card.value === 'unclustered' || card.value === 'weak_interest_pool' }]"
          :aria-pressed="filterCluster === card.value"
          @click="toggleClusterFilter(card.value)"
        >
          <div class="cluster-top">
            <strong>{{ card.title }}</strong>
            <span v-if="card.paperCount !== null" class="muted">{{ card.paperCount }} 篇</span>
          </div>
          <span :class="['cluster-hint', { mono: card.hint.startsWith('cluster_') }]">{{ card.hint }}</span>
          <span v-if="card.duplicateIds.length" class="cluster-warning" :title="card.duplicateIds.join('\n')">
            含 {{ card.duplicateIds.length }} 组同一论文的不同版本 ID
          </span>
        </button>
      </div>
    </section>

    <div class="toolbar">
      <div class="toolbar-filters">
        <el-segmented v-model="filterLabel" :options="filterOptions" :disabled="!!filterCluster" />
        <span v-if="activeClusterCard" class="cluster-filter-tag">
          {{ activeClusterCard.title }}
          <button type="button" aria-label="取消兴趣簇筛选" @click="filterCluster = ''">×</button>
        </span>
      </div>
      <span v-if="store.totalLabeledPapers > 0" class="muted">共 {{ store.totalLabeledPapers }} 篇</span>
    </div>

    <div v-if="store.loading" class="labeled-list">
      <div v-for="n in 3" :key="n" class="surface skeleton-card">
        <el-skeleton :rows="2" animated />
      </div>
    </div>

    <div v-else-if="store.labeledPapers.length === 0 && filterCluster" class="surface empty">
      <el-empty :description="`「${activeClusterCard?.title || '该兴趣簇'}」里没有论文`" :image-size="96">
        <el-button plain @click="filterCluster = ''">查看全部论文</el-button>
      </el-empty>
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

        <p :class="['paper-summary', { 'clamp-3': !expandedIds.has(paper.id) }]">{{ displaySummary(paper.arxivId || paper.id, formatPaperSummary(paper.summary)) }}</p>

        <div class="card-foot">
          <span class="muted">{{ paper.labeledAt ? `标记于 ${formatLabeledAt(paper.labeledAt)}` : '标记时间未知' }}</span>
          <div class="card-actions">
            <el-button link type="primary" @click="toggleExpanded(paper.id)">
              {{ expandedIds.has(paper.id) ? '收起摘要' : '展开摘要' }}
            </el-button>
            <el-button
              v-if="paper.summary"
              link
              type="primary"
              :disabled="isTranslating(paper.arxivId || paper.id)"
              @click="toggleTranslation(paper.arxivId || paper.id)"
            >
              {{ translateButtonText(paper.arxivId || paper.id) }}
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
  gap: 6px;
  padding: 12px 14px;
  border: 1px solid transparent;
  font: inherit;
  text-align: left;
  color: inherit;
  cursor: pointer;
  transition: border-color 0.15s ease, background-color 0.15s ease;
}

.cluster-card:hover {
  border-color: var(--color-border-strong);
}

.cluster-card.is-secondary {
  background: transparent;
  border: 1px dashed var(--color-border);
}

.cluster-card.is-selected {
  border: 1px solid var(--color-primary);
  background: var(--color-primary-soft);
}

.cluster-card.is-selected .cluster-top strong {
  color: var(--color-primary);
}

.cluster-top {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
}

.cluster-hint {
  font-size: 12px;
  color: var(--color-text-4);
}

.cluster-warning {
  font-size: 12px;
  color: var(--color-warning);
}

.cluster-empty,
.cluster-tip {
  margin-bottom: 10px;
  font-size: 13px;
}

.toolbar-filters {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}

.cluster-filter-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  height: 28px;
  padding: 0 4px 0 12px;
  border-radius: var(--radius-pill);
  background: var(--color-primary-soft);
  color: var(--color-primary);
  font-size: 13px;
}

.cluster-filter-tag button {
  width: 22px;
  height: 22px;
  border: none;
  border-radius: 50%;
  background: none;
  color: inherit;
  font-size: 15px;
  line-height: 1;
  cursor: pointer;
}

.cluster-filter-tag button:hover {
  background: var(--color-surface);
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
