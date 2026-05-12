<script setup lang="ts">
import { ref, watch } from 'vue'
import { categories } from '@/mock/papers'

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
  <div class="search-bar">
    <el-form class="search-form" @submit.prevent="handleSearch">
      <div class="basic-search">
        <el-form-item label="搜索查询">
          <el-input
            v-model="searchQuery"
            placeholder="输入搜索查询，支持字段前缀语法如 ti:deep learning"
            clearable
            style="width: 400px"
          />
        </el-form-item>
        
        <el-form-item label="逻辑操作符">
          <el-select
            v-model="operator"
            placeholder="选择操作符"
            style="width: 100px"
          >
            <el-option
              v-for="opt in operatorOptions"
              :key="opt.value"
              :value="opt.value"
            >
              {{ opt.label }}
            </el-option>
          </el-select>
        </el-form-item>

        <el-form-item label="最晚提交日期">
          <el-date-picker
            v-model="submittedDateBefore"
            type="date"
            placeholder="选择日期（不选则默认30天内）"
            style="width: 220px"
            :disabled-date="(time: Date) => time.getTime() > Date.now()"
          />
        </el-form-item>

        <el-form-item label="结果数量">
          <el-input-number
            v-model="maxResults"
            :min="1"
            :max="3000"
            style="width: 120px"
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSearch">
            搜索
          </el-button>
          <el-button @click="showAdvanced = !showAdvanced">
            {{ showAdvanced ? '隐藏高级选项' : '高级搜索' }}
          </el-button>
        </el-form-item>
      </div>

      <div v-if="showAdvanced" class="advanced-search">
        <div class="advanced-row">
          <el-form-item label="标题">
            <el-input
              v-model="title"
              placeholder="标题关键词"
              clearable
              style="width: 200px"
            />
          </el-form-item>

          <el-form-item label="作者">
            <el-input
              v-model="author"
              placeholder="作者姓名"
              clearable
              style="width: 200px"
            />
          </el-form-item>

          <el-form-item label="摘要">
            <el-input
              v-model="abstract"
              placeholder="摘要关键词"
              clearable
              style="width: 200px"
            />
          </el-form-item>

          <el-form-item label="分类">
            <el-select
              v-model="category"
              placeholder="选择分类"
              style="width: 180px"
            >
              <el-option
                v-for="cat in categories"
                :key="cat.value"
                :value="cat.value"
              >
                {{ cat.label }}
              </el-option>
            </el-select>
          </el-form-item>
        </div>

        <div class="advanced-row">
          <el-form-item label="评论">
            <el-input
              v-model="comment"
              placeholder="评论关键词"
              clearable
              style="width: 200px"
            />
          </el-form-item>

          <el-form-item label="期刊引用">
            <el-input
              v-model="journalRef"
              placeholder="期刊引用关键词"
              clearable
              style="width: 200px"
            />
          </el-form-item>

          <el-form-item label="报告编号">
            <el-input
              v-model="reportNumber"
              placeholder="报告编号关键词"
              clearable
              style="width: 200px"
            />
          </el-form-item>

          <el-form-item label="排序方式">
            <el-select
              v-model="sortBy"
              placeholder="选择排序方式"
              style="width: 140px"
              @change="handleSearch"
            >
              <el-option
                v-for="option in sortOptions"
                :key="option.value"
                :value="option.value"
              >
                {{ option.label }}
              </el-option>
            </el-select>
          </el-form-item>

          <el-form-item label="排序顺序">
            <el-select
              v-model="sortOrder"
              placeholder="选择顺序"
              style="width: 100px"
              @change="handleSearch"
            >
              <el-option
                v-for="option in sortOrderOptions"
                :key="option.value"
                :value="option.value"
              >
                {{ option.label }}
              </el-option>
            </el-select>
          </el-form-item>

          <el-form-item>
            <el-button @click="clearAdvanced">
              清除高级选项
            </el-button>
          </el-form-item>
        </div>
      </div>
    </el-form>
  </div>
</template>

<style scoped>
.search-bar {
  padding: 16px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.search-form {
  width: 100%;
}

.basic-search {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
}

.advanced-search {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #eee;
}

.advanced-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.advanced-row:last-child {
  margin-bottom: 0;
}
</style>
