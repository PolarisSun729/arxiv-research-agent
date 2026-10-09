<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  score: number
  showLabel?: boolean
  showProgress?: boolean
}>()

const safeScore = computed(() => {
  const s = Number(props.score)
  if (isNaN(s)) return 0
  return Math.max(0, Math.min(1, s))
})

const scorePercent = computed(() => Math.round(safeScore.value * 100))

const scoreColor = computed(() => {
  if (safeScore.value >= 0.7) return 'success'
  if (safeScore.value >= 0.4) return 'warning'
  return 'danger'
})

const scoreText = computed(() => {
  if (safeScore.value >= 0.7) return '高相似度'
  if (safeScore.value >= 0.4) return '中等相似度'
  return '低相似度'
})
</script>

<template>
  <div class="similarity-tag">
    <el-tag
      v-if="showLabel !== false"
      :type="scoreColor"
      size="small"
    >
      {{ scoreText }}
    </el-tag>
    <div v-if="showProgress !== false" class="similarity-progress">
      <el-progress
        :percentage="scorePercent"
        :stroke-color="{
          '0%': '#10b981',
          '70%': '#f59e0b',
          '100%': '#ef4444'
        }"
        :show-text="false"
        height="6"
      />
      <span class="score-text">{{ scorePercent }}%</span>
    </div>
  </div>
</template>

<style scoped>
.similarity-tag {
  display: flex;
  align-items: center;
  gap: 8px;
}

.similarity-progress {
  flex: 1;
  min-width: 120px;
}

.score-text {
  font-size: 12px;
  color: #666;
  min-width: 36px;
  text-align: right;
}
</style>