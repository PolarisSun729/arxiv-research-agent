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
// 默认只展示章节、页码和三行摘要；全文按需展开，避免一次铺开上万像素。
const expandedIds = ref<Set<string>>(new Set())
const othersOpen = ref(false)
// 列表内点开的来源也会回传为高亮，此时它就在用户眼前，不需要再滚动定位。
let locallySelectedId: string | null = null

// 按 citedSourceIds 的顺序排列，编号与回答正文里的引用角标一一对应。
const citedSources = computed(() =>
  props.citedSourceIds
    .map(sourceId => props.sources.find(source => source.source_id === sourceId))
    .filter((source): source is RagChatSource => Boolean(source))
)
const otherSources = computed(() => props.sources.filter(source => !props.citedSourceIds.includes(source.source_id)))

const CHUNK_TYPE_LABELS: Record<string, string> = { text: '正文', figure: '图表', table: '表格' }

function getSourceTitle(source: RagChatSource) {
  return (source.section_path || source.section_title || source.source || '').trim() || '论文证据'
}

function getSourceMeta(source: RagChatSource) {
  const pageNumber = (source.page_number || '').trim()
  return [
    pageNumber && pageNumber.toUpperCase() !== 'N/A' ? `第 ${pageNumber} 页` : '',
    CHUNK_TYPE_LABELS[(source.chunk_type || '').toLowerCase()] || ''
  ].filter(Boolean)
}

// chunk 正文第一行通常就是章节标题，标题栏已经展示过，正文里去掉以免重复。
function getSourceBody(source: RagChatSource) {
  const content = (source.content || '').trim()
  const [firstLine, ...rest] = content.split('\n')
  if (rest.length && firstLine.trim() === getSourceTitle(source)) return rest.join('\n').trim()
  return content || '-'
}

function getContentPreview(source: RagChatSource) {
  return getSourceBody(source).replace(/\s+/g, ' ').trim()
}

function isExpanded(sourceId: string) {
  return expandedIds.value.has(sourceId)
}

function toggleSource(sourceId: string) {
  const next = new Set(expandedIds.value)
  if (next.has(sourceId)) {
    next.delete(sourceId)
    expandedIds.value = next
    return
  }
  next.add(sourceId)
  expandedIds.value = next
  locallySelectedId = sourceId
  emit('select-source', sourceId)
}

function setSourceElement(sourceId: string, element: unknown) {
  sourceElements.value[sourceId] = element instanceof HTMLElement ? element : null
}

function findScrollParent(element: HTMLElement) {
  for (let node = element.parentElement; node; node = node.parentElement) {
    const { overflowY } = getComputedStyle(node)
    if ((overflowY === 'auto' || overflowY === 'scroll') && node.scrollHeight > node.clientHeight) return node
  }
  return null
}

// 只滚动来源列表所在的容器，把被点的来源顶到容器顶部；用 scrollIntoView 会连带滚动整页，把正在看的回答滚走。
function scrollToHighlightedSource() {
  const sourceId = props.highlightedSourceId
  const fromList = sourceId !== null && sourceId === locallySelectedId
  locallySelectedId = null
  if (!sourceId || fromList) return
  if (!isExpanded(sourceId)) expandedIds.value = new Set(expandedIds.value).add(sourceId)
  if (otherSources.value.some(source => source.source_id === sourceId)) othersOpen.value = true
  nextTick(() => {
    const element = sourceElements.value[sourceId]
    if (!element) return
    const scroller = findScrollParent(element)
    if (!scroller) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
      return
    }
    // scroll-margin-top 让出面板顶部吸附的分页栏。
    const margin = parseFloat(getComputedStyle(element).scrollMarginTop) || 12
    const offset = element.getBoundingClientRect().top - scroller.getBoundingClientRect().top
    // 直接跳转而不是平滑滚动：平滑滚动会被同时发生的展开、页面滚动打断，停在半路；定位后由高亮边框提示。
    scroller.scrollTo({ top: scroller.scrollTop + offset - margin })
    scroller.scrollIntoView({ block: 'nearest' })
  })
}

watch(
  () => props.sources,
  () => {
    expandedIds.value = new Set()
    othersOpen.value = false
  }
)
watch(() => props.highlightedSourceId, scrollToHighlightedSource, { immediate: true })

const groups = computed(() => [
  { key: 'cited', title: '回答引用', items: citedSources.value, offset: 0, collapsible: false },
  { key: 'others', title: '其他检索候选', items: otherSources.value, offset: citedSources.value.length, collapsible: true }
].filter(group => group.items.length))
</script>

<template>
  <div v-if="sources.length" class="cite-list">
    <section v-for="group in groups" :key="group.key" class="cite-group">
      <button
        v-if="group.collapsible"
        type="button"
        class="cite-group__title cite-group__title--toggle"
        :aria-expanded="othersOpen"
        @click="othersOpen = !othersOpen"
      >
        <span>{{ group.title }}</span>
        <span class="cite-group__count">{{ group.items.length }}</span>
        <span class="cite-chevron" :class="{ 'cite-chevron--open': othersOpen }" aria-hidden="true" />
      </button>
      <div v-else class="cite-group__title">
        <span>{{ group.title }}</span>
        <span class="cite-group__count">{{ group.items.length }}</span>
      </div>

      <div v-show="!group.collapsible || othersOpen" class="cite-group__items">
        <article
          v-for="(source, index) in group.items"
          :key="source.source_id"
          :ref="element => setSourceElement(source.source_id, element)"
          class="cite-item"
          :class="{
            'cite-item--open': isExpanded(source.source_id),
            'cite-item--highlighted': highlightedSourceId === source.source_id,
            'cite-item--muted': group.collapsible
          }"
        >
          <button
            type="button"
            class="cite-item__head"
            :aria-expanded="isExpanded(source.source_id)"
            @click="toggleSource(source.source_id)"
          >
            <span class="cite-item__index">{{ group.offset + index + 1 }}</span>
            <span class="cite-item__heading">
              <span class="cite-item__title" :title="getSourceTitle(source)">{{ getSourceTitle(source) }}</span>
              <span v-if="getSourceMeta(source).length" class="cite-item__meta">
                <span v-for="meta in getSourceMeta(source)" :key="meta">{{ meta }}</span>
              </span>
            </span>
            <span class="cite-chevron" :class="{ 'cite-chevron--open': isExpanded(source.source_id) }" aria-hidden="true" />
          </button>

          <p v-if="!isExpanded(source.source_id)" class="cite-item__preview">{{ getContentPreview(source) }}</p>
          <div v-else class="cite-item__body">
            <p>{{ getSourceBody(source) }}</p>
            <template v-if="source.chunk_type === 'figure'">
              <AuthenticatedImage v-if="source.asset_url" class="cite-item__asset" :src="source.asset_url" fit="contain">
                <template #error><div class="cite-item__asset-missing">图片暂不可用，仍保留文字证据。</div></template>
              </AuthenticatedImage>
              <div v-else class="cite-item__asset-missing">图片暂不可用，仍保留文字证据。</div>
            </template>
          </div>
        </article>
      </div>
    </section>
  </div>
  <div v-else class="cite-empty">
    {{ emptyText || '这次回答没有可追踪的证据。' }}
  </div>
</template>

<style scoped>
.cite-list {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.cite-group {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.cite-group__title {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 2px;
  color: var(--color-text-3);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
}

.cite-group__title--toggle {
  width: 100%;
  border: 0;
  background: none;
  cursor: pointer;
  font-family: inherit;
}

.cite-group__title--toggle:hover {
  color: var(--color-text-2);
}

.cite-group__count {
  padding: 1px 7px;
  border-radius: 999px;
  background: var(--color-surface-sunken);
  color: var(--color-text-3);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0;
}

.cite-group__title--toggle .cite-chevron {
  margin-left: auto;
}

.cite-group__items {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.cite-item {
  /* 让出面板顶部吸附的分页栏，跳转定位时标题不被挡住。 */
  scroll-margin-top: 72px;
  border: 1px solid var(--color-border);
  border-radius: 14px;
  background: var(--color-surface);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.cite-item:hover {
  border-color: var(--color-border-strong);
}

.cite-item--open {
  border-color: var(--color-border-strong);
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
}

.cite-item--highlighted,
.cite-item--highlighted:hover {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px var(--color-primary-soft);
}

.cite-item__head {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 12px 14px 0;
  border: 0;
  background: none;
  cursor: pointer;
  text-align: left;
  font-family: inherit;
}

.cite-item__index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: none;
  min-width: 22px;
  height: 22px;
  padding: 0 6px;
  border-radius: 7px;
  background: var(--color-primary-soft);
  color: var(--color-primary);
  font-size: 12px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.cite-item--muted .cite-item__index {
  background: var(--color-surface-sunken);
  color: var(--color-text-3);
}

.cite-item__heading {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  flex: 1;
}

.cite-item__title {
  overflow: hidden;
  color: var(--color-text);
  font-size: 13.5px;
  font-weight: 600;
  line-height: 1.4;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cite-item__meta {
  display: flex;
  gap: 6px;
  color: var(--color-text-4);
  font-size: 12px;
}

.cite-item__meta span + span::before {
  content: '·';
  margin-right: 6px;
}

.cite-chevron {
  flex: none;
  width: 7px;
  height: 7px;
  margin: 0 4px;
  border-right: 1.5px solid var(--color-text-4);
  border-bottom: 1.5px solid var(--color-text-4);
  transform: rotate(-45deg);
  transition: transform 0.2s;
}

.cite-chevron--open {
  transform: rotate(45deg);
}

.cite-item__preview {
  display: -webkit-box;
  margin: 8px 14px 12px 46px;
  overflow: hidden;
  color: var(--color-text-3);
  font-size: 12.5px;
  line-height: 1.65;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
}

.cite-item__body {
  margin: 10px 14px 14px;
  padding-top: 10px;
  border-top: 1px dashed var(--color-border);
  color: var(--color-text-2);
  font-size: 13px;
  line-height: 1.75;
  white-space: pre-wrap;
  word-break: break-word;
}

.cite-item__body p {
  margin: 0;
}

.cite-item__asset {
  display: block;
  width: 100%;
  max-height: 260px;
  margin-top: 12px;
  border-radius: 10px;
  border: 1px solid var(--color-surface-sunken);
  background: var(--color-surface-muted);
}

.cite-item__asset-missing {
  margin-top: 8px;
  color: var(--color-text-4);
  font-size: 12px;
}

.cite-empty {
  padding: 14px 16px;
  border-radius: 12px;
  background: var(--color-surface-muted);
  color: var(--color-text-3);
  font-size: 13px;
  line-height: 1.6;
}
</style>
