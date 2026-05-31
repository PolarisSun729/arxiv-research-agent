<script setup lang="ts">
import type { RagChatSource } from '@/types/ragChat'

defineProps<{
  sources: RagChatSource[]
  emptyText?: string
}>()

function getSourceLabel(source: RagChatSource, index: number) {
  const pageNumber = (source.page_number || '').trim()
  if (pageNumber && pageNumber.toUpperCase() !== 'N/A') {
    return `Page ${pageNumber}`
  }
  return `Source ${index + 1}`
}

function getContentPreview(content: string) {
  const normalized = (content || '').replace(/\s+/g, ' ').trim()
  if (normalized.length <= 220) {
    return normalized || '-'
  }
  return `${normalized.slice(0, 220)}...`
}
</script>

<template>
  <div v-if="sources.length" class="rag-citation-list">
    <details v-for="(source, index) in sources" :key="`${source.page_number}-${index}`" class="rag-citation-list__item" open>
      <summary class="rag-citation-list__summary">
        <div class="rag-citation-list__summary-main">
          <span class="rag-citation-list__index">{{ index + 1 }}</span>
          <div class="rag-citation-list__meta">
            <div class="rag-citation-list__title-row">
              <span class="rag-citation-list__title">{{ source.source || '论文片段' }}</span>
              <span class="rag-citation-list__badge">{{ getSourceLabel(source, index) }}</span>
            </div>
            <div class="rag-citation-list__preview">{{ getContentPreview(source.content) }}</div>
          </div>
        </div>
      </summary>
      <div class="rag-citation-list__body">
        <p>{{ source.content || '-' }}</p>
      </div>
    </details>
  </div>
  <div v-else class="rag-citation-list__empty">
    {{ emptyText || '这次回答没有返回结构化来源。' }}
  </div>
</template>

<style scoped>
.rag-citation-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.rag-citation-list__item {
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: linear-gradient(180deg, #f8fbff, #ffffff);
  padding: 12px 14px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.rag-citation-list__item[open] {
  border-color: rgba(96, 165, 250, 0.38);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

.rag-citation-list__summary {
  cursor: pointer;
  list-style: none;
}

.rag-citation-list__summary::-webkit-details-marker {
  display: none;
}

.rag-citation-list__summary-main {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.rag-citation-list__index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.1);
  color: #2563eb;
  font-size: 12px;
  font-weight: 700;
  flex: none;
}

.rag-citation-list__meta {
  min-width: 0;
  flex: 1;
}

.rag-citation-list__title-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.rag-citation-list__title {
  color: #0f172a;
  font-size: 14px;
  font-weight: 700;
}

.rag-citation-list__badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.08);
  color: #2563eb;
  font-size: 12px;
}

.rag-citation-list__preview {
  margin-top: 8px;
  color: #64748b;
  font-size: 13px;
  line-height: 1.6;
}

.rag-citation-list__body {
  margin-top: 12px;
  color: #334155;
  font-size: 13px;
  line-height: 1.75;
  white-space: pre-wrap;
}

.rag-citation-list__body p {
  margin: 0;
}

.rag-citation-list__empty {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(148, 163, 184, 0.08);
  color: #64748b;
  font-size: 13px;
  line-height: 1.6;
}
</style>
