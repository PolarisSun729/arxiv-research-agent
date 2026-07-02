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
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-left: 5px solid #0f766e;
  border-radius: 18px;
  background:
    radial-gradient(circle at top left, rgba(20, 184, 166, 0.16), transparent 34%),
    linear-gradient(135deg, #f8fffd 0%, #ffffff 58%, #f8fafc 100%);
  color: #0f172a;
}

.arxiv-query-capability-banner--warning {
  border-left-color: #d97706;
  background:
    radial-gradient(circle at top left, rgba(245, 158, 11, 0.18), transparent 34%),
    linear-gradient(135deg, #fffbeb 0%, #ffffff 58%, #f8fafc 100%);
}

.arxiv-query-capability-banner--error {
  border-left-color: #dc2626;
  background:
    radial-gradient(circle at top left, rgba(239, 68, 68, 0.16), transparent 34%),
    linear-gradient(135deg, #fff7f7 0%, #ffffff 58%, #f8fafc 100%);
}

.arxiv-query-capability-banner--compact {
  padding: 14px;
}

.capability-main {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.capability-title-row,
.capability-meta,
.capability-chip-list {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.capability-badge,
.capability-source,
.capability-chip {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
}

.capability-badge {
  background: rgba(15, 118, 110, 0.12);
  color: #0f766e;
}

.arxiv-query-capability-banner--warning .capability-badge {
  background: rgba(217, 119, 6, 0.14);
  color: #92400e;
}

.arxiv-query-capability-banner--error .capability-badge {
  background: rgba(220, 38, 38, 0.12);
  color: #991b1b;
}

.capability-source,
.capability-chip {
  background: rgba(15, 23, 42, 0.06);
  color: #334155;
}

.capability-summary {
  font-size: 15px;
  font-weight: 700;
  line-height: 1.55;
}

.capability-meta {
  color: #64748b;
  font-size: 12px;
}

.capability-details {
  border-top: 1px solid rgba(15, 23, 42, 0.08);
  padding-top: 10px;
}

.capability-details summary {
  cursor: pointer;
  color: #0f172a;
  font-size: 13px;
  font-weight: 700;
}

.capability-detail-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 12px;
}

.capability-warning-list,
.capability-detail-lines {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.capability-warning-list p,
.capability-detail-lines p {
  margin: 0;
  color: #475569;
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
  gap: 8px;
}

.capability-detail-label {
  color: #64748b;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.capability-chip--muted {
  color: #64748b;
  background: rgba(100, 116, 139, 0.1);
}

@media (max-width: 760px) {
  .capability-grid {
    grid-template-columns: 1fr;
  }
}
</style>
