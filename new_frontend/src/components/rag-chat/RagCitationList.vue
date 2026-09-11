<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { RagChatSource } from '@/types/ragChat'
import AuthenticatedImage from '@/components/AuthenticatedImage.vue'

const props = withDefaults(defineProps<{
  sources: RagChatSource[]
  citedSourceIds?: string[]
  highlightedSourceId?: string | null
  emptyText?: string
}>(), {
  citedSourceIds: () => [],
  highlightedSourceId: null
})

const emit = defineEmits<{
  (event: 'select-source', sourceId: string): void
}>()

const sourceElements = ref<Record<string, HTMLElement | null>>({})
const citedSources = computed(() => props.sources.filter(source => props.citedSourceIds.includes(source.source_id)))
const otherSources = computed(() => props.sources.filter(source => !props.citedSourceIds.includes(source.source_id)))

function getSourceLabel(source: RagChatSource, index: number) {
  const pageNumber = (source.page_number || '').trim()
  if (pageNumber && pageNumber.toUpperCase() !== 'N/A') return `Page ${pageNumber}`
  return `Evidence ${index + 1}`
}

function getContentPreview(content: string) {
  const normalized = (content || '').replace(/\s+/g, ' ').trim()
  if (normalized.length <= 220) return normalized || '-'
  return `${normalized.slice(0, 220)}...`
}

function setSourceElement(sourceId: string, element: unknown) {
  sourceElements.value[sourceId] = element instanceof HTMLElement ? element : null
}

function scrollToHighlightedSource() {
  if (!props.highlightedSourceId) return
  nextTick(() => sourceElements.value[props.highlightedSourceId as string]?.scrollIntoView({ behavior: 'smooth', block: 'center' }))
}

watch(() => props.highlightedSourceId, scrollToHighlightedSource, { immediate: true })
</script>

<template>
  <div v-if="sources.length" class="rag-citation-list">
    <div v-if="citedSources.length" class="rag-citation-list__group-title">回答引用</div>
    <details
      v-for="(source, index) in citedSources"
      :key="source.source_id"
      :ref="element => setSourceElement(source.source_id, element)"
      class="rag-citation-list__item"
      :class="{ 'rag-citation-list__item--highlighted': highlightedSourceId === source.source_id }"
      open
      @click="emit('select-source', source.source_id)"
    >
      <summary class="rag-citation-list__summary">
        <div class="rag-citation-list__summary-main">
          <span class="rag-citation-list__index">{{ index + 1 }}</span>
          <div class="rag-citation-list__meta">
            <div class="rag-citation-list__title-row">
              <span class="rag-citation-list__title">{{ source.source || '论文证据' }}</span>
              <span class="rag-citation-list__badge">{{ getSourceLabel(source, index) }}</span>
            </div>
            <div class="rag-citation-list__preview">{{ getContentPreview(source.content) }}</div>
          </div>
        </div>
      </summary>
      <div class="rag-citation-list__body">
        <p>{{ source.content || '-' }}</p>
        <AuthenticatedImage
          v-if="source.chunk_type === 'figure' && source.asset_url"
          class="rag-citation-list__asset"
          :src="source.asset_url"
          fit="contain"
        >
          <template #error><div class="rag-citation-list__asset-missing">图片暂不可用，仍保留文字证据。</div></template>
        </AuthenticatedImage>
        <div v-if="source.chunk_type === 'figure' && !source.asset_url" class="rag-citation-list__asset-missing">
          图片暂不可用，仍保留文字证据。
        </div>
      </div>
    </details>

    <details v-if="otherSources.length" class="rag-citation-list__others">
      <summary class="rag-citation-list__others-summary">其他检索候选（{{ otherSources.length }}）</summary>
      <details
        v-for="(source, index) in otherSources"
        :key="source.source_id"
        :ref="element => setSourceElement(source.source_id, element)"
        class="rag-citation-list__item"
        :class="{ 'rag-citation-list__item--highlighted': highlightedSourceId === source.source_id }"
        @click="emit('select-source', source.source_id)"
      >
        <summary class="rag-citation-list__summary">
          <div class="rag-citation-list__summary-main">
            <span class="rag-citation-list__index">{{ citedSources.length + index + 1 }}</span>
            <div class="rag-citation-list__meta">
              <div class="rag-citation-list__title-row">
                <span class="rag-citation-list__title">{{ source.source || '论文证据' }}</span>
                <span class="rag-citation-list__badge">{{ getSourceLabel(source, citedSources.length + index) }}</span>
              </div>
              <div class="rag-citation-list__preview">{{ getContentPreview(source.content) }}</div>
            </div>
          </div>
        </summary>
        <div class="rag-citation-list__body">
          <p>{{ source.content || '-' }}</p>
          <AuthenticatedImage
            v-if="source.chunk_type === 'figure' && source.asset_url"
            class="rag-citation-list__asset"
            :src="source.asset_url"
            fit="contain"
          >
            <template #error><div class="rag-citation-list__asset-missing">图片暂不可用，仍保留文字证据。</div></template>
          </AuthenticatedImage>
        </div>
      </details>
    </details>
  </div>
  <div v-else class="rag-citation-list__empty">
    {{ emptyText || '这次回答没有可追踪的证据。' }}
  </div>
</template>

<style scoped>
.rag-citation-list { display: flex; flex-direction: column; gap: 12px; }
.rag-citation-list__group-title { color: #0f172a; font-size: 14px; font-weight: 800; }
.rag-citation-list__item { border: 1px solid rgba(148, 163, 184, .18); border-radius: 18px; padding: 12px 14px; background: linear-gradient(180deg, #f8fbff, #fff); transition: border-color .2s, box-shadow .2s; }
.rag-citation-list__item[open] { border-color: rgba(96, 165, 250, .38); box-shadow: 0 10px 24px rgba(15, 23, 42, .06); }
.rag-citation-list__item--highlighted { border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37, 99, 235, .12) !important; }
.rag-citation-list__summary, .rag-citation-list__others-summary { cursor: pointer; list-style: none; }
.rag-citation-list__summary::-webkit-details-marker, .rag-citation-list__others-summary::-webkit-details-marker { display: none; }
.rag-citation-list__summary-main { display: flex; align-items: flex-start; gap: 12px; }
.rag-citation-list__index { display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px; flex: none; border-radius: 999px; background: rgba(37, 99, 235, .1); color: #2563eb; font-size: 12px; font-weight: 700; }
.rag-citation-list__meta { min-width: 0; flex: 1; }
.rag-citation-list__title-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.rag-citation-list__title { color: #0f172a; font-size: 14px; font-weight: 700; }
.rag-citation-list__badge { display: inline-flex; padding: 3px 8px; border-radius: 999px; background: rgba(37, 99, 235, .08); color: #2563eb; font-size: 12px; }
.rag-citation-list__preview { margin-top: 6px; color: #64748b; font-size: 12px; line-height: 1.6; }
.rag-citation-list__body { margin-top: 12px; color: #334155; font-size: 13px; line-height: 1.75; white-space: pre-wrap; }
.rag-citation-list__body p { margin: 0; }
.rag-citation-list__asset { display: block; width: min(100%, 360px); max-height: 220px; margin-top: 12px; border-radius: 12px; background: #f8fafc; }
.rag-citation-list__asset-missing { margin-top: 8px; color: #94a3b8; font-size: 12px; }
.rag-citation-list__others { display: flex; flex-direction: column; gap: 10px; padding: 10px; border-radius: 16px; background: rgba(148, 163, 184, .08); }
.rag-citation-list__others-summary { color: #475569; font-size: 13px; font-weight: 700; }
.rag-citation-list__empty { padding: 14px 16px; border-radius: 16px; background: rgba(148, 163, 184, .08); color: #64748b; font-size: 13px; line-height: 1.6; }
</style>
