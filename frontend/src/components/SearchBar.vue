<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { ArrowDown, Filter, Search } from '@element-plus/icons-vue'
import { arxivCategories as categories } from '@/constants/arxivCategories'

const props = defineProps<{
  modelValue?: {
    searchQuery: string
    title: string
    author: string
    abstract: string
    category: string
    comment: string
    journalRef: string
    reportNumber: string
    operator: '' | 'AND' | 'OR'
    sortBy: string
    sortOrder: 'ascending' | 'descending'
    maxResults: number
    submittedDateBefore: string
  }
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: {
    searchQuery: string
    title: string
    author: string
    abstract: string
    category: string
    comment: string
    journalRef: string
    reportNumber: string
    operator: '' | 'AND' | 'OR'
    sortBy: string
    sortOrder: 'ascending' | 'descending'
    maxResults: number
    submittedDateBefore: string
  }): void
  (e: 'search'): void
}>()

const searchQuery = ref(props.modelValue?.searchQuery || '')
const title = ref(props.modelValue?.title || '')
const author = ref(props.modelValue?.author || '')
const abstract = ref(props.modelValue?.abstract || '')
const category = ref(props.modelValue?.category || '')
const comment = ref(props.modelValue?.comment || '')
const journalRef = ref(props.modelValue?.journalRef || '')
const reportNumber = ref(props.modelValue?.reportNumber || '')
const operator = ref<('' | 'AND' | 'OR')>(props.modelValue?.operator || '')
const sortBy = ref(props.modelValue?.sortBy || 'relevance')
const sortOrder = ref<('ascending' | 'descending')>(props.modelValue?.sortOrder || 'descending')
const maxResults = ref(props.modelValue?.maxResults || 10)
const submittedDateBefore = ref(props.modelValue?.submittedDateBefore || '')
const showAdvanced = ref(false)
const advancedCount = computed(() => [title, author, abstract, category, comment, journalRef, reportNumber, operator]
  .filter(field => Boolean(field.value)).length)

const sortOptions = [
  { value: 'relevance', label: '相关性' },
  { value: 'submittedDate', label: '提交日期' },
  { value: 'lastUpdatedDate', label: '更新日期' }
]

const sortOrderOptions = [
  { value: 'descending', label: '降序' },
  { value: 'ascending', label: '升序' }
]

const operatorOptions = [
  { value: 'AND', label: 'AND' },
  { value: 'OR', label: 'OR' }
]

watch(() => props.modelValue, (val) => {
  if (val) {
    searchQuery.value = val.searchQuery
    title.value = val.title
    author.value = val.author
    abstract.value = val.abstract
    category.value = val.category
    comment.value = val.comment
    journalRef.value = val.journalRef
    reportNumber.value = val.reportNumber
    operator.value = val.operator
    sortBy.value = val.sortBy
    sortOrder.value = val.sortOrder
    maxResults.value = val.maxResults
    submittedDateBefore.value = val.submittedDateBefore
  }
}, { deep: true })

function emitChange() {
  emit('update:modelValue', {
    searchQuery: searchQuery.value,
    title: title.value,
    author: author.value,
    abstract: abstract.value,
    category: category.value,
    comment: comment.value,
    journalRef: journalRef.value,
    reportNumber: reportNumber.value,
    operator: operator.value,
    sortBy: sortBy.value,
    sortOrder: sortOrder.value,
    maxResults: maxResults.value,
    submittedDateBefore: submittedDateBefore.value
  })
}

function handleSearch() {
  emitChange()
  emit('search')
}

function clearAdvanced() {
  title.value = ''
  author.value = ''
  abstract.value = ''
  category.value = ''
  comment.value = ''
  journalRef.value = ''
  reportNumber.value = ''
  emitChange()
}
</script>

<template>
  <form class="surface search-bar" @submit.prevent="handleSearch">
    <div class="search-main">
      <el-input
        v-model="searchQuery"
        class="search-input"
        size="large"
        placeholder="搜索论文，支持字段前缀语法，如 ti:diffusion AND au:hinton"
        clearable
        :prefix-icon="Search"
      />
      <el-button type="primary" size="large" native-type="submit" class="search-btn">搜索</el-button>
    </div>

    <div class="filter-row">
      <div class="filter">
        <span class="filter-label">排序</span>
        <el-select v-model="sortBy" style="width: 120px" @change="handleSearch">
          <el-option v-for="option in sortOptions" :key="option.value" :label="option.label" :value="option.value" />
        </el-select>
        <el-select v-model="sortOrder" style="width: 84px" @change="handleSearch">
          <el-option v-for="option in sortOrderOptions" :key="option.value" :label="option.label" :value="option.value" />
        </el-select>
      </div>

      <label class="filter">
        <span class="filter-label">最晚提交</span>
        <el-date-picker
          v-model="submittedDateBefore"
          type="date"
          placeholder="默认近 30 天"
          style="width: 160px"
          :disabled-date="(time: Date) => time.getTime() > Date.now()"
        />
      </label>

      <label class="filter">
        <span class="filter-label">结果数</span>
        <el-input-number v-model="maxResults" :min="1" :max="3000" controls-position="right" style="width: 110px" />
      </label>

      <button type="button" :class="['advanced-toggle', { 'is-open': showAdvanced }]" @click="showAdvanced = !showAdvanced">
        <el-icon><Filter /></el-icon>
        高级筛选
        <span v-if="advancedCount" class="advanced-count">{{ advancedCount }}</span>
        <el-icon class="toggle-arrow"><ArrowDown /></el-icon>
      </button>
    </div>

    <div v-if="showAdvanced" class="advanced">
      <div class="advanced-grid">
        <label class="field">
          <span class="filter-label">标题</span>
          <el-input v-model="title" placeholder="标题关键词" clearable />
        </label>
        <label class="field">
          <span class="filter-label">作者</span>
          <el-input v-model="author" placeholder="作者姓名" clearable />
        </label>
        <label class="field">
          <span class="filter-label">摘要</span>
          <el-input v-model="abstract" placeholder="摘要关键词" clearable />
        </label>
        <label class="field">
          <span class="filter-label">分类</span>
          <el-select v-model="category" placeholder="全部分类" clearable filterable>
            <el-option v-for="cat in categories" :key="cat.value" :label="cat.label" :value="cat.value" />
          </el-select>
        </label>
        <label class="field">
          <span class="filter-label">评论</span>
          <el-input v-model="comment" placeholder="评论关键词" clearable />
        </label>
        <label class="field">
          <span class="filter-label">期刊引用</span>
          <el-input v-model="journalRef" placeholder="期刊引用关键词" clearable />
        </label>
        <label class="field">
          <span class="filter-label">报告编号</span>
          <el-input v-model="reportNumber" placeholder="报告编号关键词" clearable />
        </label>
        <label class="field">
          <span class="filter-label">字段组合</span>
          <el-select v-model="operator" placeholder="默认" clearable>
            <el-option v-for="opt in operatorOptions" :key="opt.value" :label="opt.label" :value="opt.value" />
          </el-select>
        </label>
      </div>
      <div class="advanced-actions">
        <el-button text @click="clearAdvanced">清除高级条件</el-button>
      </div>
    </div>
  </form>
</template>

<style scoped>
.search-bar {
  padding: 16px;
}

.search-main {
  display: flex;
  gap: 10px;
}

.search-input {
  flex: 1;
}

.search-btn {
  min-width: 96px;
}

.filter-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px 20px;
  margin-top: 12px;
}

.filter {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.filter-label {
  font-size: 13px;
  color: var(--color-text-3);
  white-space: nowrap;
}

.advanced-toggle {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 32px;
  padding: 0 10px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: var(--color-surface);
  color: var(--color-text-2);
  font: inherit;
  font-size: 13px;
  cursor: pointer;
}

.advanced-toggle:hover,
.advanced-toggle.is-open {
  border-color: var(--color-primary-border);
  color: var(--color-primary);
}

.advanced-count {
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  border-radius: 9px;
  background: var(--color-primary);
  color: #fff;
  font-size: 11px;
  line-height: 18px;
  text-align: center;
}

.toggle-arrow {
  transition: transform 0.15s ease;
}

.is-open .toggle-arrow {
  transform: rotate(180deg);
}

.advanced {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid var(--color-border);
}

.advanced-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 12px 16px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.advanced-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 8px;
}

@media (max-width: 640px) {
  .advanced-toggle {
    margin-left: 0;
  }
}
</style>
