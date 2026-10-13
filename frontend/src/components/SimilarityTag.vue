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
  <div :class="['similarity-tag', `is-${scoreColor}`]">
    <div v-if="showProgress !== false" class="similarity-track">
      <div class="similarity-fill" :style="{ width: `${scorePercent}%` }" />
    </div>
    <span class="score-text">{{ scorePercent }}%</span>
    <span v-if="showLabel !== false" class="score-level">{{ scoreText }}</span>
  </div>
</template>

<style scoped>
.similarity-tag {
  --level-color: var(--color-text-4);
  display: flex;
  align-items: center;
  gap: 8px;
}

.is-success {
  --level-color: var(--color-success);
}

.is-warning {
  --level-color: #d97706;
}

.is-danger {
  --level-color: var(--color-text-4);
}

.similarity-track {
  flex: 1;
  min-width: 80px;
  height: 6px;
  border-radius: 3px;
  background: var(--color-surface-sunken);
  overflow: hidden;
}

.similarity-fill {
  height: 100%;
  border-radius: 3px;
  background: var(--level-color);
  transition: width 0.3s ease;
}

.score-text {
  min-width: 34px;
  text-align: right;
  font-size: 12.5px;
  font-weight: 600;
  color: var(--color-text);
  font-variant-numeric: tabular-nums;
}

.score-level {
  font-size: 12px;
  color: var(--level-color);
  white-space: nowrap;
}
</style>
