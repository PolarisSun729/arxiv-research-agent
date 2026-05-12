<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { usePaperStore } from '@/stores/paperStore'
import PaperCard from '@/components/PaperCard.vue'

const router = useRouter()
const store = usePaperStore()

const recommendationCount = ref(10)

const countOptions = [
  { value: 5, label: '5篇' },
  { value: 10, label: '10篇' },
  { value: 20, label: '20篇' },
  { value: 50, label: '50篇' }
]

async function handleGenerateRecommendations() {
  try {
    await store.generateRecommendations(recommendationCount.value)
    ElMessage.success(`已生成 ${recommendationCount.value} 篇推荐论文`)
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
          根据你已经标记过的论文，系统会生成一批相似度较高的推荐结果。
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
        <el-button
          type="primary"
          :loading="store.recommendationsGenerating"
          @click="handleGenerateRecommendations"
        >
          生成推荐
        </el-button>
      </div>
    </div>

    <div v-if="store.recommendationsGenerating" class="loading">
      <el-skeleton :rows="4" animated />
    </div>

    <div v-else-if="store.recommendations.length === 0" class="empty-results">
      <el-empty description="暂无推荐论文，请先生成推荐结果" />
    </div>

    <div v-else class="recommendations-list">
      <div class="sort-info">按相似度排序</div>
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

  .paper-grid {
    grid-template-columns: 1fr;
  }
}
</style>
