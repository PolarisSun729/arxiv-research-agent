<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue';
import { useRouter } from 'vue-router';
import { usePaperStore } from '@/stores/paperStore';
import { ElMessage } from 'element-plus';

const router = useRouter();
const store = usePaperStore();

const filterLabel = ref<string>('');
const currentPage = ref(1);
const pageSize = ref(10);

const filterOptions = [
  { value: '', label: '全部' },
  { value: 'liked', label: '喜欢' },
  { value: 'disliked', label: '不喜欢' }
];

const interestVector = computed(() => store.lastInterestVector);
const interestClusters = computed(() => interestVector.value?.interest_clusters || []);
const clusterModeText = computed(() => interestVector.value?.profile_mode || 'mean');

onMounted(() => {
  fetchLabeledPapers();
  store.fetchUserInterestVector();
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

function getLabelClass(label: string) {
  return label === 'liked' ? 'el-tag--success' : 'el-tag--danger';
}

function getLabelText(label: string) {
  return label === 'liked' ? '喜欢' : '不喜欢';
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
  <div class="labeled-papers-page">
    <div class="page-header">
      <div class="header-content">
        <div>
          <h1>已标记论文</h1>
          <p class="description">查看和管理您标记过的论文</p>
        </div>
        <el-button
          type="primary"
          :loading="store.interestVectorGenerating"
          :disabled="store.labeledPapers.length === 0"
          @click="handleGenerateInterestVector"
        >
          生成用户偏好向量
        </el-button>
      </div>
    </div>

    <div class="filter-bar">
      <el-select
        v-model="filterLabel"
        placeholder="筛选标记状态"
        style="width: 180px"
      >
        <el-option
          v-for="option in filterOptions"
          :key="option.value"
          :value="option.value"
        >
          {{ option.label }}
        </el-option>
      </el-select>
    </div>

    <div v-if="interestVector" class="interest-cluster-panel">
      <div class="interest-cluster-panel__header">
        <div>
          <h2>兴趣簇概览</h2>
          <p>
            当前画像模式：{{ clusterModeText }}，
            聚类数：{{ interestVector.cluster_count || interestClusters.length || 0 }}
          </p>
        </div>
        <el-tag type="success" effect="dark">
          {{ interestClusters.length > 0 ? 'clustered' : 'mean fallback' }}
        </el-tag>
      </div>

      <div v-if="interestClusters.length > 0" class="interest-cluster-grid">
        <el-card v-for="cluster in interestClusters" :key="cluster.cluster_id" class="interest-cluster-card">
          <div class="cluster-card__top">
            <strong>{{ cluster.cluster_id }}</strong>
            <span>{{ cluster.paper_count }} papers</span>
          </div>
          <div class="cluster-paper-ids">
            <el-tag
              v-for="paperId in cluster.paper_ids.slice(0, 4)"
              :key="paperId"
              size="small"
              type="info"
            >
              {{ paperId }}
            </el-tag>
            <span v-if="cluster.paper_ids.length > 4" class="more-ids">
              +{{ cluster.paper_ids.length - 4 }} more
            </span>
          </div>
        </el-card>
      </div>
      <div v-else class="interest-cluster-empty">
        当前 liked papers 数量不足或聚类回退到了单一均值向量。
      </div>
    </div>

    <div v-if="store.loading" class="loading">
      <div class="el-loading-spinner"></div>
      <p>加载中...</p>
    </div>

    <div v-else-if="store.labeledPapers.length === 0" class="empty-results">
      <el-empty description="暂无已标记论文" />
    </div>

    <div v-else class="labeled-list">
      <el-card
        v-for="paper in store.labeledPapers"
        :key="paper.id"
        class="labeled-card"
      >
        <div class="card-header">
          <h3 class="paper-title">{{ paper.title }}</h3>
          <el-tag :class="getLabelClass(paper.label)" size="small">
            {{ getLabelText(paper.label) }}
          </el-tag>
        </div>

        <p class="paper-summary">{{ paper.summary.slice(0, 100) }}...</p>

        <div class="card-footer">
          <span class="labeled-time">标记时间：{{ formatDate(paper.labeledAt) }}</span>
          <el-button size="small" type="primary" @click="handleViewDetail(paper.id)">
            查看详情
          </el-button>
        </div>
      </el-card>
    </div>

    <div v-if="store.totalLabeledPapers > 0" class="pagination-wrapper">
      <el-pagination
        :current-page="currentPage"
        :page-size="pageSize"
        :total="store.totalLabeledPapers"
        @current-change="handlePageChange"
        layout="total, prev, pager, next, jumper"
      />
    </div>
  </div>
</template>

<style scoped>
.labeled-papers-page {
  padding: 20px;
}

.page-header {
  margin-bottom: 20px;
}

.page-header h1 {
  font-size: 24px;
  font-weight: 700;
  color: #1f2937;
  margin: 0 0 8px 0;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.description {
  font-size: 14px;
  color: #6b7280;
  margin: 0;
}

.filter-bar {
  margin-bottom: 20px;
}

.interest-cluster-panel {
  margin-bottom: 20px;
  padding: 18px 20px;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(31, 41, 55, 0.92));
  color: #f8fafc;
  box-shadow: 0 16px 32px rgba(15, 23, 42, 0.18);
}

.interest-cluster-panel__header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 16px;
}

.interest-cluster-panel__header h2 {
  margin: 0 0 6px 0;
  font-size: 18px;
  color: #fff;
}

.interest-cluster-panel__header p {
  margin: 0;
  color: rgba(226, 232, 240, 0.84);
  font-size: 13px;
}

.interest-cluster-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
}

.interest-cluster-card {
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(15, 23, 42, 0.45);
}

.cluster-card__top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  color: #e2e8f0;
}

.cluster-paper-ids {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.more-ids {
  font-size: 12px;
  color: rgba(226, 232, 240, 0.76);
}

.interest-cluster-empty {
  color: rgba(226, 232, 240, 0.84);
  font-size: 13px;
}

.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px;
}

.empty-results {
  padding: 40px;
}

.labeled-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 16px;
}

.labeled-card {
  padding: 16px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}

.paper-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
  flex: 1;
  margin-right: 12px;
  line-height: 1.4;
}

.paper-summary {
  font-size: 14px;
  color: #6b7280;
  line-height: 1.6;
  margin: 0 0 12px 0;
}

.card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 12px;
  border-top: 1px solid #e5e7eb;
}

.labeled-time {
  font-size: 13px;
  color: #9ca3af;
}

.pagination-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 24px;
}

@media (max-width: 900px) {
  .labeled-list {
    grid-template-columns: 1fr;
  }
}
</style>
