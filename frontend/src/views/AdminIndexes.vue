<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Search } from '@element-plus/icons-vue'
import AuthenticatedImage from '@/components/AuthenticatedImage.vue'
import { figureAssetUrl, getPaperIndex, getParsedDocument, listPaperIndexes } from '@/api/adminIndexes'
import type { IndexedChunk, LengthBucket, PaperIndexBuild, PaperIndexDetail, PaperIndexSummary, ParsedDocument } from '@/api/adminIndexes'
import { getErrorMessage } from '@/api/errors'
import { normalizeServerTimestamp } from '@/utils/format'

const LIST_PAGE_SIZE = 50
const CHUNK_PAGE_SIZE = 20

// first_build_failed 由列表接口派生：主表仍是 not_indexed，但最近一个版本已构建失败。
const INDEX_STATUS_LABELS: Record<string, string> = {
  indexed: '已索引',
  first_build_failed: '首次构建失败',
  failed: '失败',
  not_indexed: '未索引',
  building: '构建中',
  deleted: '已删除',
}
const FAILED_INDEX_STATUSES = new Set(['failed', 'first_build_failed'])
const BUILD_STATUS_LABELS: Record<string, string> = { active: '生效中', build_success: '已构建', build_failed: '构建失败', building: '构建中', orphaned: '已废弃' }
const CHUNK_TYPE_LABELS: Record<string, string> = { text: '正文', table: '表格', figure: '图片' }
const UNAVAILABLE_REASONS: Record<string, string> = {
  not_recorded: '这个版本没有记录 chunk 文件（早期的索引只写入了向量库），无法查看切分结果。',
  missing: 'chunk 文件已被清理，或不在当前服务器上。',
  invalid: 'chunk 文件内容无法解析。'
}

const route = useRoute()
const router = useRouter()

// ---- 左侧论文列表 ----
const papers = ref<PaperIndexSummary[]>([])
const total = ref(0)
const listPage = ref(1)
const keyword = ref('')
const statusFilter = ref('')
const loadingList = ref(false)

async function loadList() {
  loadingList.value = true
  try {
    const result = await listPaperIndexes({ q: keyword.value.trim(), status: statusFilter.value, offset: (listPage.value - 1) * LIST_PAGE_SIZE, limit: LIST_PAGE_SIZE })
    papers.value = result.items
    total.value = result.total
  } catch (error) { ElMessage.error(getErrorMessage(error, '索引列表读取失败')) }
  finally { loadingList.value = false }
}

function searchList() {
  listPage.value = 1
  void loadList()
}

// ---- 右侧详情：选中的论文与版本同步到地址栏，刷新后仍停在原处 ----
const selectedId = computed(() => String(route.query.paper || ''))
const selectedBuildId = computed(() => String(route.query.build || ''))
const selectedSummary = computed(() => papers.value.find(item => item.arxiv_id === selectedId.value) || null)
const detail = ref<PaperIndexDetail | null>(null)
const loadingDetail = ref(false)

function selectPaper(arxivId: string) {
  void router.replace({ query: { paper: arxivId } })
}

function selectBuild(buildId: string) {
  void router.replace({ query: { paper: selectedId.value, build: buildId } })
}

async function loadDetail() {
  resetChunkFilters()
  parsed.value = null
  activeTab.value = 'chunks'
  if (!selectedId.value) {
    detail.value = null
    return
  }
  const requestedId = selectedId.value
  loadingDetail.value = true
  try {
    const result = await getPaperIndex(requestedId, selectedBuildId.value)
    // 快速切换论文时只采用最后一次选择的结果。
    if (requestedId === selectedId.value) detail.value = result
  } catch (error) {
    detail.value = null
    ElMessage.error(getErrorMessage(error, '索引详情读取失败'))
  } finally { loadingDetail.value = false }
}

watch(() => [selectedId.value, selectedBuildId.value], loadDetail)

const artifact = computed(() => detail.value?.artifact.available ? detail.value.artifact : null)
const unavailableReason = computed(() => detail.value && !detail.value.artifact.available ? UNAVAILABLE_REASONS[detail.value.artifact.reason] : '')

function buildLabel(build: PaperIndexBuild) {
  const status = build.is_active ? '生效中' : (BUILD_STATUS_LABELS[build.status] || build.status)
  return `${status} · ${formatTime(build.created_at)} · ${build.chunk_count ?? 0} chunks`
}

function formatTime(value: string | null | undefined) {
  if (!value) return '—'
  const date = new Date(normalizeServerTimestamp(value))
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

// ---- 切分结果：统计、直方图与 chunk 过滤 ----
const typeFilter = ref('all')
const flagFilter = ref('all')
const chunkKeyword = ref('')
const bucketFilter = ref<number | null>(null)
const chunkPage = ref(1)
const expanded = ref(new Set<number>())

function resetChunkFilters() {
  typeFilter.value = 'all'
  flagFilter.value = 'all'
  chunkKeyword.value = ''
  bucketFilter.value = null
  chunkPage.value = 1
  expanded.value = new Set()
}

const shortThreshold = computed(() => artifact.value?.stats.short_text_threshold ?? 0)
const histogramMax = computed(() => Math.max(1, ...(artifact.value?.stats.length_histogram.map(bucket => bucket.count) || [])))

function bucketLabel(bucket: LengthBucket) {
  if (bucket.max === null) return `≥${bucket.min}`
  return bucket.min === 0 ? `<${bucket.max}` : `${bucket.min}–${bucket.max}`
}

function toggleBucket(index: number) {
  bucketFilter.value = bucketFilter.value === index ? null : index
  chunkPage.value = 1
}

function chunkType(chunk: IndexedChunk) {
  return String(chunk.metadata.chunk_type || 'text')
}

function isShort(chunk: IndexedChunk) {
  return chunkType(chunk) === 'text' && chunk.char_count < shortThreshold.value
}

function isSplitBySize(chunk: IndexedChunk) {
  return chunkType(chunk) === 'text' && Number(chunk.metadata.section_part_count || 1) > 1
}

function isOverlong(chunk: IndexedChunk) {
  return Number(chunk.metadata.subchunk_count || 1) > 1
}

const filteredChunks = computed(() => {
  if (!artifact.value) return []
  const bucket = bucketFilter.value === null ? null : artifact.value.stats.length_histogram[bucketFilter.value]
  const needle = chunkKeyword.value.trim().toLowerCase()
  return artifact.value.chunks.filter(chunk => {
    if (typeFilter.value !== 'all' && chunkType(chunk) !== typeFilter.value) return false
    if (flagFilter.value === 'short' && !isShort(chunk)) return false
    if (flagFilter.value === 'split' && !isSplitBySize(chunk)) return false
    if (flagFilter.value === 'overlong' && !isOverlong(chunk)) return false
    // 直方图只统计正文，按长度区间筛选时同样只看正文。
    if (bucket && (chunkType(chunk) !== 'text' || chunk.char_count < bucket.min || (bucket.max !== null && chunk.char_count >= bucket.max))) return false
    if (needle && !chunk.content.toLowerCase().includes(needle) && !String(chunk.metadata.section_path || '').toLowerCase().includes(needle)) return false
    return true
  })
})

const pagedChunks = computed(() => filteredChunks.value.slice((chunkPage.value - 1) * CHUNK_PAGE_SIZE, chunkPage.value * CHUNK_PAGE_SIZE))

watch([typeFilter, flagFilter, chunkKeyword], () => { chunkPage.value = 1 })

// 折叠后约 10 行高；字数多或换行多（参考文献、表格）才折叠，短内容完整显示、不加渐隐遮罩。
function isCollapsible(chunk: IndexedChunk) {
  return chunk.char_count > 600 || chunk.content.split('\n').length > 10
}

function toggleExpanded(chunkId: number) {
  const next = new Set(expanded.value)
  if (next.has(chunkId)) next.delete(chunkId)
  else next.add(chunkId)
  expanded.value = next
}

function pageRange(chunk: IndexedChunk) {
  const start = chunk.metadata.page_start
  const end = chunk.metadata.page_end
  if (!start) return ''
  return end && end !== start ? `p.${start}–${end}` : `p.${start}`
}

function imageUrl(chunk: IndexedChunk) {
  return figureAssetUrl(selectedId.value, chunk.metadata.chunk_id, detail.value?.selected_build_id || '')
}

// ---- 解析结果：切到该标签时才加载 ----
const activeTab = ref('chunks')
const parsed = ref<ParsedDocument | null>(null)
const parsedView = ref<'pages' | 'markdown'>('pages')
const loadingParsed = ref(false)

watch(activeTab, async tab => {
  if (tab !== 'parsed' || parsed.value || !detail.value) return
  const requestedId = selectedId.value
  loadingParsed.value = true
  try {
    const result = await getParsedDocument(requestedId, detail.value.selected_build_id)
    if (requestedId === selectedId.value) {
      parsed.value = result
      parsedView.value = result.markdown ? parsedView.value : 'pages'
    }
  } catch (error) { ElMessage.error(getErrorMessage(error, '解析结果读取失败')) }
  finally { loadingParsed.value = false }
})

onMounted(() => {
  void loadList()
  void loadDetail()
})
</script>

<template>
  <div class="page page--wide">
    <header class="page-header">
      <div class="page-header__text">
        <h1 class="page-title">索引切分</h1>
        <p class="page-desc">查看已建索引论文的 chunk 切分结果和 PDF 解析结果，用来判断切分参数是否合适。数据来自索引构建时保存的 chunk 文件，只读。</p>
      </div>
    </header>

    <div class="layout">
      <aside class="surface paper-list">
        <form class="paper-list__filters" @submit.prevent="searchList">
          <el-input v-model="keyword" placeholder="arXiv ID 或标题" :prefix-icon="Search" clearable aria-label="搜索论文" @clear="searchList" />
          <el-select v-model="statusFilter" placeholder="全部状态" clearable aria-label="索引状态" @change="searchList">
            <el-option v-for="(label, value) in INDEX_STATUS_LABELS" :key="value" :label="label" :value="value" />
          </el-select>
        </form>
        <div v-loading="loadingList" class="paper-list__items">
          <button
            v-for="item in papers"
            :key="item.arxiv_id"
            type="button"
            :class="['paper-item', { 'paper-item--active': item.arxiv_id === selectedId }]"
            @click="selectPaper(item.arxiv_id)"
          >
            <span class="paper-item__head">
              <span class="paper-item__id">{{ item.arxiv_id }}</span>
              <span :class="['chip', item.status === 'indexed' ? 'chip--success' : FAILED_INDEX_STATUSES.has(item.status) ? 'chip--danger' : '']">
                {{ INDEX_STATUS_LABELS[item.status] || item.status }}
              </span>
            </span>
            <span class="paper-item__title">{{ item.title || '（论文库中没有标题）' }}</span>
            <span class="paper-item__meta">{{ item.chunk_count ?? 0 }} chunks · {{ item.build_count }} 个版本 · {{ item.chunking_strategy || '策略未记录' }}</span>
          </button>
          <el-empty v-if="!loadingList && papers.length === 0" description="没有符合条件的索引" :image-size="80" />
        </div>
        <el-pagination
          v-if="total > LIST_PAGE_SIZE"
          v-model:current-page="listPage"
          class="paper-list__pager"
          size="small"
          :page-size="LIST_PAGE_SIZE"
          :total="total"
          layout="prev, pager, next"
          @current-change="loadList"
        />
      </aside>

      <section v-loading="loadingDetail" class="detail">
        <div v-if="!selectedId" class="surface surface-pad">
          <el-empty description="从左侧选择一篇论文" :image-size="100" />
        </div>

        <template v-else-if="detail">
          <div class="surface surface-pad detail-head">
            <div class="detail-head__text">
              <div class="detail-head__id">{{ detail.arxiv_id }}</div>
              <div class="detail-head__title">{{ selectedSummary?.title || '' }}</div>
            </div>
            <el-select
              v-if="detail.builds.length"
              :model-value="detail.selected_build_id"
              class="detail-head__build"
              aria-label="索引版本"
              @change="selectBuild"
            >
              <el-option v-for="build in detail.builds" :key="build.build_id" :label="buildLabel(build)" :value="build.build_id" :disabled="!build.has_chunk_file" />
            </el-select>
          </div>

          <div v-if="unavailableReason" class="surface surface-pad">
            <el-empty :description="unavailableReason" :image-size="90" />
          </div>

          <el-tabs v-else-if="artifact" v-model="activeTab" class="detail-tabs">
            <el-tab-pane label="切分结果" name="chunks">
              <div class="overview">
                <section class="surface surface-pad">
                  <h2 class="block-title">切分参数</h2>
                  <dl class="params">
                    <div><dt>加载方式</dt><dd>{{ artifact.overview.loading_method || '—' }}</dd></div>
                    <div><dt>切分策略</dt><dd>{{ artifact.overview.chunking_strategy || '—' }}</dd></div>
                    <template v-if="artifact.overview.chunking_params">
                      <div><dt>chunk_size</dt><dd>{{ artifact.overview.chunking_params.chunk_size }} 词</dd></div>
                      <div><dt>超长硬切阈值</dt><dd>{{ artifact.overview.chunking_params.max_chunk_content_length }} 字符</dd></div>
                      <div><dt>硬切重叠</dt><dd>{{ artifact.overview.chunking_params.chunk_overlap_length }} 字符</dd></div>
                      <div><dt>断点搜索窗口</dt><dd>{{ artifact.overview.chunking_params.split_search_window }} 字符</dd></div>
                    </template>
                    <div><dt>页数</dt><dd>{{ artifact.overview.total_pages ?? '—' }}</dd></div>
                    <div><dt>生成时间</dt><dd>{{ formatTime(artifact.overview.saved_at) }}</dd></div>
                  </dl>
                  <p v-if="!artifact.overview.chunking_params" class="help">这个版本生成时还没有记录切分参数，当时用的参数不一定和现在的配置相同。</p>
                </section>

                <section class="surface surface-pad">
                  <h2 class="block-title">切分统计</h2>
                  <div class="stats">
                    <div class="stat">
                      <span class="stat__value">{{ artifact.stats.total }}</span>
                      <span class="stat__label">chunk 总数</span>
                      <span class="stat__sub">
                        <template v-for="(count, type) in artifact.stats.by_type" :key="type">{{ CHUNK_TYPE_LABELS[type] || type }} {{ count }}　</template>
                      </span>
                    </div>
                    <div class="stat">
                      <span class="stat__value">{{ artifact.stats.text_chars.p50 }}</span>
                      <span class="stat__label">正文长度中位数（字符）</span>
                      <span class="stat__sub">P90 {{ artifact.stats.text_chars.p90 }} · 最长 {{ artifact.stats.text_chars.max }}</span>
                    </div>
                    <div class="stat">
                      <span class="stat__value">{{ artifact.stats.short_text_count }}</span>
                      <span class="stat__label">过短正文</span>
                      <span class="stat__sub">少于 {{ artifact.stats.short_text_threshold }} 字符</span>
                    </div>
                    <div class="stat">
                      <span class="stat__value">{{ artifact.stats.split_by_size_count }}</span>
                      <span class="stat__label">按 chunk_size 拆分</span>
                      <span class="stat__sub">共 {{ artifact.stats.section_count }} 个章节</span>
                    </div>
                    <div class="stat">
                      <span class="stat__value">{{ artifact.stats.split_overlong_count }}</span>
                      <span class="stat__label">超长硬切</span>
                      <span class="stat__sub">超过硬切阈值后按字符切开</span>
                    </div>
                  </div>
                </section>
              </div>

              <section class="surface surface-pad histogram-card">
                <h2 class="block-title">正文 chunk 字符数分布 <span class="muted">点击柱子筛选下方列表</span></h2>
                <div class="histogram" role="group" aria-label="正文 chunk 字符数分布">
                  <button
                    v-for="(bucket, index) in artifact.stats.length_histogram"
                    :key="index"
                    type="button"
                    :class="['histogram__col', { 'histogram__col--active': bucketFilter === index, 'histogram__col--dim': bucketFilter !== null && bucketFilter !== index }]"
                    :title="`${bucketLabel(bucket)} 字符：${bucket.count} 个正文 chunk`"
                    :aria-pressed="bucketFilter === index"
                    @click="toggleBucket(index)"
                  >
                    <span class="histogram__count">{{ bucket.count }}</span>
                    <span class="histogram__track"><span class="histogram__bar" :style="{ height: `${(bucket.count / histogramMax) * 100}%` }" /></span>
                    <span class="histogram__label">{{ bucketLabel(bucket) }}</span>
                  </button>
                </div>
              </section>

              <div class="chunk-filters">
                <el-radio-group v-model="typeFilter" size="small" aria-label="chunk 类型">
                  <el-radio-button value="all">全部</el-radio-button>
                  <el-radio-button value="text">正文</el-radio-button>
                  <el-radio-button value="table">表格</el-radio-button>
                  <el-radio-button value="figure">图片</el-radio-button>
                </el-radio-group>
                <el-select v-model="flagFilter" size="small" class="chunk-filters__flag" aria-label="问题筛选">
                  <el-option label="不限" value="all" />
                  <el-option label="只看过短正文" value="short" />
                  <el-option label="只看按 chunk_size 拆分" value="split" />
                  <el-option label="只看超长硬切" value="overlong" />
                </el-select>
                <el-input v-model="chunkKeyword" size="small" class="chunk-filters__search" placeholder="搜索正文或章节" clearable aria-label="搜索 chunk" />
                <span class="muted">{{ filteredChunks.length }} / {{ artifact.chunks.length }}</span>
              </div>

              <article v-for="chunk in pagedChunks" :key="chunk.metadata.chunk_id" class="surface chunk">
                <header class="chunk__head">
                  <span class="chunk__id">#{{ chunk.metadata.chunk_id }}</span>
                  <span class="chip">{{ CHUNK_TYPE_LABELS[chunkType(chunk)] || chunkType(chunk) }}</span>
                  <span v-if="pageRange(chunk)" class="muted">{{ pageRange(chunk) }}</span>
                  <span class="chunk__section" :title="String(chunk.metadata.section_path || '')">{{ chunk.metadata.section_path || '（无章节）' }}</span>
                  <span class="chunk__size">{{ chunk.metadata.word_count ?? '—' }} 词 · {{ chunk.char_count }} 字符</span>
                </header>
                <div v-if="isShort(chunk) || isSplitBySize(chunk) || isOverlong(chunk)" class="chip-row chunk__flags">
                  <span v-if="isShort(chunk)" class="chip chip--warning">过短</span>
                  <span v-if="isSplitBySize(chunk)" class="chip chip--primary">章节拆分 {{ chunk.metadata.section_part_index }}/{{ chunk.metadata.section_part_count }}</span>
                  <span v-if="isOverlong(chunk)" class="chip chip--danger">超长硬切 {{ chunk.metadata.subchunk_index }}/{{ chunk.metadata.subchunk_count }}</span>
                </div>
                <AuthenticatedImage v-if="chunk.has_image" :src="imageUrl(chunk)" class="chunk__image">
                  <template #error><span class="muted">原图已不可用</span></template>
                </AuthenticatedImage>
                <pre :class="['chunk__content', { 'chunk__content--clamped': isCollapsible(chunk) && !expanded.has(chunk.metadata.chunk_id) }]">{{ chunk.content }}</pre>
                <footer class="chunk__foot">
                  <el-button v-if="isCollapsible(chunk)" link type="primary" @click="toggleExpanded(chunk.metadata.chunk_id)">
                    {{ expanded.has(chunk.metadata.chunk_id) ? '收起' : '展开全文' }}
                  </el-button>
                  <details class="chunk__meta">
                    <summary>元数据</summary>
                    <pre>{{ JSON.stringify(chunk.metadata, null, 2) }}</pre>
                  </details>
                </footer>
              </article>
              <el-empty v-if="filteredChunks.length === 0" description="没有符合条件的 chunk" :image-size="80" />
              <el-pagination
                v-if="filteredChunks.length > CHUNK_PAGE_SIZE"
                v-model:current-page="chunkPage"
                class="chunk-pager"
                :page-size="CHUNK_PAGE_SIZE"
                :total="filteredChunks.length"
                layout="total, prev, pager, next"
              />
            </el-tab-pane>

            <el-tab-pane label="解析结果" name="parsed">
              <div v-loading="loadingParsed" class="parsed">
                <template v-if="parsed">
                  <div class="parsed__toolbar">
                    <el-radio-group v-model="parsedView" size="small" aria-label="解析结果视图">
                      <el-radio-button value="pages">逐页文本（{{ parsed.pages.length }} 页）</el-radio-button>
                      <el-radio-button value="markdown" :disabled="!parsed.markdown">Docling Markdown</el-radio-button>
                    </el-radio-group>
                    <span v-if="artifact.overview.parsing.docling_item_count != null" class="muted">
                      Docling 元素 {{ artifact.overview.parsing.docling_item_count }} · 图片 {{ artifact.overview.parsing.docling_picture_count ?? 0 }} · 表格 {{ artifact.overview.parsing.docling_table_count ?? 0 }}
                    </span>
                  </div>
                  <template v-if="parsedView === 'pages'">
                    <section v-for="page in parsed.pages" :key="page.page_number" class="surface parsed__page">
                      <h3 class="parsed__page-title">第 {{ page.page_number }} 页 <span class="muted">{{ page.text.length }} 字符</span></h3>
                      <pre class="parsed__text">{{ page.text || '（本页没有解析出文本）' }}</pre>
                    </section>
                  </template>
                  <pre v-else class="surface parsed__text parsed__markdown">{{ parsed.markdown }}</pre>
                </template>
              </div>
            </el-tab-pane>
          </el-tabs>
        </template>
      </section>
    </div>
  </div>
</template>

<style scoped>
.layout {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
  gap: 20px;
  align-items: start;
}

.paper-list {
  position: sticky;
  top: 16px;
  display: flex;
  flex-direction: column;
  max-height: calc(100vh - 32px);
  overflow: hidden;
}

.paper-list__filters {
  display: grid;
  gap: 8px;
  padding: 12px;
  border-bottom: 1px solid var(--color-border);
}

.paper-list__items {
  flex: 1;
  min-height: 120px;
  overflow-y: auto;
  padding: 6px;
}

.paper-list__pager {
  padding: 8px;
  border-top: 1px solid var(--color-border);
  justify-content: center;
}

.paper-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 100%;
  padding: 10px;
  border: 1px solid transparent;
  border-radius: var(--radius);
  background: none;
  text-align: left;
  cursor: pointer;
  font: inherit;
  color: inherit;
}

.paper-item:hover {
  background: var(--color-surface-muted);
}

.paper-item--active {
  background: var(--color-primary-soft);
  border-color: var(--color-primary-border);
}

.paper-item__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.paper-item__id {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text);
}

.paper-item__title {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  font-size: 13px;
  line-height: 1.45;
  color: var(--color-text-2);
}

.paper-item__meta {
  font-size: 12px;
  color: var(--color-text-3);
}

.detail {
  display: grid;
  gap: 16px;
  min-width: 0;
  min-height: 240px;
}

.detail-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.detail-head__text {
  min-width: 0;
}

.detail-head__id {
  font-family: var(--font-mono);
  font-weight: 650;
  color: var(--color-text);
}

.detail-head__title {
  margin-top: 4px;
  color: var(--color-text-2);
}

.detail-head__build {
  width: 340px;
  max-width: 100%;
}

.detail-tabs :deep(.el-tabs__content) {
  display: grid;
  gap: 16px;
}

.detail-tabs :deep(.el-tab-pane) {
  display: grid;
  gap: 14px;
}

.overview {
  display: grid;
  grid-template-columns: minmax(260px, 1fr) minmax(0, 2fr);
  gap: 14px;
}

.block-title {
  margin-bottom: 12px;
  font-size: 14px;
  font-weight: 650;
  color: var(--color-text);
}

.block-title .muted {
  margin-left: 8px;
  font-size: 12px;
  font-weight: 400;
}

.params {
  display: grid;
  gap: 8px;
  font-size: 13px;
}

.params div {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.params dt {
  color: var(--color-text-3);
}

.params dd {
  font-weight: 550;
  color: var(--color-text);
  text-align: right;
}

.help {
  margin-top: 10px;
  font-size: 12.5px;
  line-height: 1.6;
  color: var(--color-warning);
}

.stats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
}

.stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 12px;
  border-radius: var(--radius);
  background: var(--color-surface-muted);
}

.stat__value {
  font-size: 22px;
  font-weight: 650;
  font-variant-numeric: tabular-nums;
  color: var(--color-text);
}

.stat__label {
  font-size: 12.5px;
  color: var(--color-text-2);
}

.stat__sub {
  font-size: 12px;
  color: var(--color-text-3);
}

.histogram {
  display: grid;
  grid-auto-flow: column;
  grid-auto-columns: minmax(0, 1fr);
  gap: 2px;
  height: 180px;
}

.histogram__col {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 4px;
  padding: 0 2px;
  border: none;
  border-radius: var(--radius-sm);
  background: none;
  cursor: pointer;
  font: inherit;
}

.histogram__col:hover,
.histogram__col--active {
  background: var(--color-surface-muted);
}

.histogram__col--dim .histogram__bar {
  opacity: 0.35;
}

.histogram__count {
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--color-text-2);
  text-align: center;
}

.histogram__track {
  display: flex;
  align-items: flex-end;
  flex: 1;
  border-bottom: 1px solid var(--color-border-strong);
}

.histogram__bar {
  width: 100%;
  min-height: 0;
  border-radius: 4px 4px 0 0;
  background: var(--color-chart-1);
}

.histogram__label {
  font-size: 11.5px;
  color: var(--color-text-3);
  text-align: center;
  white-space: nowrap;
}

.chunk-filters {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.chunk-filters__flag {
  width: 190px;
}

.chunk-filters__search {
  width: 220px;
}

.chunk {
  padding: 14px 16px;
}

.chunk__head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 13px;
}

.chunk__id {
  font-family: var(--font-mono);
  font-weight: 650;
  color: var(--color-text);
}

.chunk__section {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--color-text-2);
}

.chunk__size {
  font-variant-numeric: tabular-nums;
  color: var(--color-text-3);
}

.chunk__flags {
  margin-top: 8px;
}

.chunk__image {
  display: block;
  max-width: 480px;
  margin-top: 10px;
}

.chunk__content {
  margin-top: 10px;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--font-sans);
  font-size: 13px;
  line-height: 1.65;
  color: var(--color-text-2);
}

.chunk__content--clamped {
  max-height: 220px;
  overflow: hidden;
  mask-image: linear-gradient(to bottom, #000 75%, transparent);
}

.chunk__foot {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  margin-top: 6px;
}

.chunk__meta summary {
  font-size: 12.5px;
  color: var(--color-text-3);
  cursor: pointer;
}

.chunk__meta pre {
  margin-top: 6px;
  padding: 10px;
  max-height: 320px;
  overflow: auto;
  border-radius: var(--radius-sm);
  background: var(--color-surface-sunken);
  font-family: var(--font-mono);
  font-size: 12px;
}

.chunk-pager {
  justify-content: center;
}

.parsed {
  display: grid;
  gap: 12px;
  min-height: 160px;
}

.parsed__toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.parsed__page {
  padding: 14px 16px;
}

.parsed__page-title {
  margin-bottom: 8px;
  font-size: 13.5px;
  font-weight: 650;
  color: var(--color-text);
}

.parsed__page-title .muted {
  margin-left: 6px;
  font-size: 12px;
  font-weight: 400;
}

.parsed__text {
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.6;
  color: var(--color-text-2);
}

.parsed__markdown {
  padding: 16px;
}

@media (max-width: 1100px) {
  .layout,
  .overview {
    grid-template-columns: minmax(0, 1fr);
  }

  .paper-list {
    position: static;
    max-height: 420px;
  }
}
</style>
