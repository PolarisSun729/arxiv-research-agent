<script setup lang="ts">
import { computed } from 'vue'
import type { DebugChunkRow, RetrievalDebugRun } from '@/api/adminRetrievalDebug'
import { chunkOutcome, routeLabel, voteShares } from '@/utils/retrievalDebug'

// 检索调试页的 chunk 详情：逐路投票拆解、命中入口和原文，解释这个 chunk 为什么排在这里。
const props = defineProps<{ row: DebugChunkRow; run: RetrievalDebugRun }>()

const outcome = computed(() => chunkOutcome(props.row, props.run))
const shares = computed(() => voteShares(props.row))
const rrfK = computed(() => props.run.fusion.rrf_k ?? 0)

function fixed(value: number | null | undefined, digits = 3) {
  return value === null || value === undefined ? '—' : value.toFixed(digits)
}
</script>

<template>
  <div class="chunk-detail">
    <section class="chunk-detail__head">
      <div class="chip-row">
        <span class="chip">#{{ row.chunk_id }}</span>
        <span class="chip">{{ row.chunk_type }}</span>
        <span v-if="row.page" class="chip">第 {{ row.page }} 页</span>
        <span :class="['chip', `outcome--${outcome.code}`]">{{ outcome.label }}</span>
      </div>
      <div v-if="row.section" class="chunk-detail__section">{{ row.section }}</div>
      <dl class="ranks">
        <div><dt>融合</dt><dd>{{ row.fused_rank ?? '—' }}<small v-if="row.fused_score !== null"> · {{ fixed(row.fused_score, 4) }}</small></dd></div>
        <div><dt>rerank</dt><dd>{{ row.rerank_rank ?? '—' }}<small v-if="row.rerank_score !== null"> · {{ fixed(row.rerank_score) }}</small></dd></div>
        <div><dt>输出</dt><dd>{{ row.final_rank ?? '—' }}<small v-if="row.context_role"> · {{ row.context_role }}</small></dd></div>
      </dl>
      <p v-if="row.final_reason" class="muted chunk-detail__reason">{{ row.final_reason }}</p>
    </section>

    <section v-if="shares.length">
      <h3 class="block-title">融合投票拆解</h3>
      <p class="muted formula">票 = 权重 × 1/({{ rrfK }} + 名次) × 表格加权</p>
      <table class="votes">
        <thead>
          <tr><th>路由</th><th>名次</th><th>权重</th><th>加权</th><th>票</th><th class="votes__share">占比</th></tr>
        </thead>
        <tbody>
          <tr v-for="item in shares" :key="item.route">
            <td>{{ routeLabel(item.route) }}</td>
            <td>{{ item.rank ?? '—' }}</td>
            <td>{{ fixed(item.weight, 2) }}</td>
            <td>{{ fixed(item.multiplier, 2) }}</td>
            <td class="mono">{{ fixed(item.vote, 5) }}</td>
            <td class="votes__share">
              <span class="share-bar"><span :style="{ width: `${Math.round(item.share * 100)}%` }" /></span>
              {{ Math.round(item.share * 100) }}%
            </td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-if="Object.keys(row.route_scores).length">
      <h3 class="block-title">各路原始分</h3>
      <div class="chip-row">
        <span v-for="(score, route) in row.route_scores" :key="route" class="chip">
          {{ routeLabel(String(route)) }} 第 {{ row.route_ranks[route] }} 名 · {{ fixed(score) }}
        </span>
      </div>
    </section>

    <section v-if="row.matched_indexes.length">
      <h3 class="block-title">命中入口</h3>
      <ul class="indexes">
        <li v-for="(entry, index) in row.matched_indexes" :key="index">
          <span class="chip">{{ entry.type || 'body' }}</span>
          <span class="muted">{{ entry.routes.map(routeLabel).join('、') }}<template v-if="entry.score !== null"> · {{ fixed(entry.score) }}</template></span>
          <div class="indexes__text">{{ entry.text }}</div>
        </li>
      </ul>
    </section>

    <section v-if="row.keyword_terms.length || row.source_queries.length">
      <h3 class="block-title">命中查询</h3>
      <div v-if="row.keyword_terms.length" class="chip-row">
        <span class="muted">BM25 命中词：</span>
        <span v-for="term in row.keyword_terms" :key="term" class="chip chip--primary">{{ term }}</span>
      </div>
      <ul v-if="row.source_queries.length" class="queries">
        <li v-for="query in row.source_queries" :key="query">{{ query }}</li>
      </ul>
    </section>

    <section>
      <h3 class="block-title">原文</h3>
      <pre class="content">{{ row.content }}</pre>
    </section>
  </div>
</template>

<style scoped>
.chunk-detail {
  display: grid;
  gap: 20px;
}

.chunk-detail__section {
  margin-top: 8px;
  font-weight: 600;
  color: var(--color-text);
}

.chunk-detail__reason {
  margin: 8px 0 0;
  font-size: 12px;
}

.ranks {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin: 12px 0 0;
}

.ranks div {
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  background: var(--color-surface-muted);
}

.ranks dt {
  font-size: 12px;
  color: var(--color-text-3);
}

.ranks dd {
  margin: 2px 0 0;
  font-size: 18px;
  font-weight: 650;
  color: var(--color-text);
}

.ranks small {
  font-size: 12px;
  font-weight: 400;
  color: var(--color-text-3);
}

.block-title {
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: 650;
  color: var(--color-text);
}

.formula {
  margin: -4px 0 8px;
  font-size: 12px;
}

.votes {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}

.votes th,
.votes td {
  padding: 6px 8px;
  border-bottom: 1px solid var(--color-border);
  text-align: right;
  white-space: nowrap;
}

.votes th:first-child,
.votes td:first-child {
  text-align: left;
}

.votes th {
  font-weight: 500;
  color: var(--color-text-3);
}

.votes__share {
  width: 120px;
}

.share-bar {
  display: inline-block;
  width: 56px;
  height: 6px;
  margin-right: 6px;
  border-radius: var(--radius-pill);
  background: var(--color-surface-sunken);
  vertical-align: middle;
  overflow: hidden;
}

.share-bar span {
  display: block;
  height: 100%;
  background: var(--color-primary);
}

.indexes,
.queries {
  display: grid;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.queries {
  margin-top: 8px;
  font-size: 13px;
  color: var(--color-text-2);
}

.indexes__text {
  margin-top: 4px;
  font-size: 13px;
  color: var(--color-text-2);
}

.content {
  margin: 0;
  padding: 12px;
  border-radius: var(--radius-sm);
  background: var(--color-surface-muted);
  font-family: var(--font-sans);
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--color-text-2);
}

.outcome--final { background: var(--color-success-soft); border-color: var(--color-success-border); color: var(--color-success); }
.outcome--expansion { background: var(--color-primary-soft); border-color: var(--color-primary-border); color: var(--color-primary); }
.outcome--fusion_cut { background: var(--color-danger-soft); border-color: var(--color-danger-border); color: var(--color-danger); }
.outcome--rerank_dropped { background: var(--color-warning-soft); border-color: var(--color-warning-border); color: var(--color-warning); }
</style>
