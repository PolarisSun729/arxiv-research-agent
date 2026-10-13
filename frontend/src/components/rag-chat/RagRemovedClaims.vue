<script setup lang="ts">
import { computed } from 'vue'
import type { RetrievalDebug } from '@/api/papers'
import { buildRemovedClaimsView } from '@/utils/researchProcess'

const props = defineProps<{
  debug?: RetrievalDebug | null
}>()

// 部分回答删掉了哪些句子、为什么删：排查"回答为什么变短"时直接看这里，不用翻研究轨迹。
const view = computed(() => buildRemovedClaimsView(props.debug))
const totalCount = computed(() => view.value.items.length + view.value.omittedCount)
</script>

<template>
  <details v-if="totalCount > 0" class="rag-removed-claims">
    <summary class="rag-removed-claims__summary">查看被移除的内容（{{ totalCount }} 条）</summary>
    <p v-if="view.joinedFallback" class="rag-removed-claims__note">
      原草稿的排版无法原位保留，正文改为拼接有证据支持的句子。
    </p>
    <ol class="rag-removed-claims__list">
      <li v-for="item in view.items" :key="item.key" class="rag-removed-claims__item">
        <div class="rag-removed-claims__text">{{ item.text }}</div>
        <div class="rag-removed-claims__reason">{{ item.reason }}</div>
        <div v-if="item.meta.length" class="rag-removed-claims__meta">
          <span v-for="entry in item.meta" :key="entry">{{ entry }}</span>
        </div>
      </li>
    </ol>
    <p v-if="view.omittedCount" class="rag-removed-claims__note">
      另有 {{ view.omittedCount }} 条未保存在历史记录中。
    </p>
  </details>
</template>

<style scoped>
.rag-removed-claims { margin-top: 8px; padding: 10px 12px; border: 1px solid rgba(251, 191, 36, .35); border-radius: 12px; background: rgba(255, 251, 235, .7); color: #78350f; font-size: 13px; }
.rag-removed-claims__summary { cursor: pointer; font-weight: 700; }
.rag-removed-claims__list { margin: 10px 0 0; padding-left: 20px; display: flex; flex-direction: column; gap: 10px; }
.rag-removed-claims__text { color: #1e293b; line-height: 1.7; white-space: pre-wrap; }
.rag-removed-claims__reason { margin-top: 2px; color: #b45309; font-weight: 600; }
.rag-removed-claims__meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.rag-removed-claims__meta span { padding: 1px 8px; border-radius: 999px; background: rgba(148, 163, 184, .16); color: #475569; font-size: 12px; }
.rag-removed-claims__note { margin: 8px 0 0; color: #92400e; font-size: 12px; }
</style>
