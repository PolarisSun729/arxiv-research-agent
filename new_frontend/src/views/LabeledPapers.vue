<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
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

onMounted(() => {
  fetchLabeledPapers();
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
