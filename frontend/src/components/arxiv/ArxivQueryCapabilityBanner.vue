<script setup lang="ts">
import { computed } from 'vue'
import type { NormalizedArxivQueryCapability } from '@/types/arxivCapability'

const props = withDefaults(defineProps<{
  capability: NormalizedArxivQueryCapability
  defaultOpen?: boolean
  compact?: boolean
}>(), {
  compact: false
})

const shouldOpen = computed(() =>
  props.defaultOpen ?? (props.capability.status !== 'info' || props.capability.warnings.length > 0)
)

const statusLabel = computed(() => {
  if (props.capability.status === 'error') return '查询未执行'
  if (props.capability.status === 'warning') return '能力边界'
  return '数据源提示'
})
</script>

<template>
  <section
    class="arxiv-query-capability-banner"
    :class="[
      `arxiv-query-capability-banner--${capability.status}`,
      { 'arxiv-query-capability-banner--compact': compact }
    ]"
    :role="capability.status === 'error' ? 'alert' : 'status'"
  >
    <div class="capability-main">
      <div class="capability-title-row">
        <span class="capability-badge">{{ statusLabel }}</span>
        <span class="capability-source">{{ capability.sourceLabel }}</span>
      </div>
      <div class="capability-summary">{{ capability.summary }}</div>
      <div class="capability-meta">
        <span>{{ capability.modeLabel }}</span>
        <span>完整 arXiv API：{{ capability.fullArxivSyntaxSupported ? '支持' : '不支持' }}</span>
        <span v-if="capability.errorCode">错误码：{{ capability.errorCode }}</span>
      </div>
    </div>

    <details class="capability-details" :open="shouldOpen">
      <summary>查看支持语法和边界</summary>

      <div class="capability-detail-body">
        <div v-if="capability.warnings.length" class="capability-warning-list">
          <div class="capability-detail-label">提示</div>
          <p v-for="warning in capability.warnings" :key="warning">{{ warning }}</p>
        </div>

        <div class="capability-detail-lines">
          <p v-for="line in capability.detailLines" :key="line">{{ line }}</p>
        </div>

        <div class="capability-grid">
          <div class="capability-group">
            <div class="capability-detail-label">支持字段</div>
            <div class="capability-chip-list">
              <span v-for="field in capability.supportedFields" :key="field" class="capability-chip">
                {{ field }}
              </span>
            </div>
          </div>

          <div class="capability-group">
            <div class="capability-detail-label">支持布尔与短语</div>
            <div class="capability-chip-list">
              <span v-for="operator in capability.supportedOperators" :key="operator" class="capability-chip">
                {{ operator }}
              </span>
            </div>
          </div>

          <div class="capability-group">
            <div class="capability-detail-label">不支持</div>
            <div class="capability-chip-list">
              <span
                v-for="syntax in capability.unsupportedSyntax"
                :key="syntax"
                class="capability-chip capability-chip--muted"
              >
                {{ syntax }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </details>
  </section>
</template>

<style scoped>
.arxiv-query-capability-banner {
  --tone: var(--color-primary);
  --tone-soft: var(--color-primary-soft);
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 14px 16px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  background: var(--color-surface);
  color: var(--color-text);
}

.arxiv-query-capability-banner--warning {
  --tone: var(--color-warning);
  --tone-soft: var(--color-warning-soft);
  border-color: var(--color-warning-border);
  background: var(--color-warning-soft);
}

.arxiv-query-capability-banner--error {
  --tone: var(--color-danger);
  --tone-soft: var(--color-danger-soft);
  border-color: var(--color-danger-border);
  background: var(--color-danger-soft);
}

.arxiv-query-capability-banner--compact {
  padding: 12px 14px;
}

.capability-main {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.capability-title-row,
.capability-meta,
.capability-chip-list {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}

.capability-badge,
.capability-source,
.capability-chip {
  display: inline-flex;
  align-items: center;
  height: 22px;
  padding: 0 8px;
  border-radius: var(--radius-sm);
  font-size: 12px;
  font-weight: 550;
}

.capability-badge {
  background: var(--tone);
  color: #fff;
}

.capability-source,
.capability-chip {
  background: rgba(17, 24, 39, 0.06);
  color: var(--color-text-2);
}

.capability-summary {
  font-size: 14px;
  font-weight: 550;
  line-height: 1.6;
}

.capability-meta {
  gap: 4px 14px;
  color: var(--color-text-3);
  font-size: 12px;
}

.capability-details summary {
  cursor: pointer;
  color: var(--tone);
  font-size: 12.5px;
  font-weight: 550;
}

.capability-detail-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed rgba(17, 24, 39, 0.12);
}

.capability-warning-list,
.capability-detail-lines {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.capability-warning-list p,
.capability-detail-lines p {
  margin: 0;
  color: var(--color-text-2);
  font-size: 13px;
  line-height: 1.6;
}

.capability-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.capability-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.capability-detail-label {
  color: var(--color-text-3);
  font-size: 12px;
  font-weight: 600;
}

.capability-chip--muted {
  color: var(--color-text-4);
  text-decoration: line-through;
}

@media (max-width: 760px) {
  .capability-grid {
    grid-template-columns: 1fr;
  }
}
</style>
