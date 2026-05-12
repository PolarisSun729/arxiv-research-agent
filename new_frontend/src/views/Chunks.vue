<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import request from '@/api/request'

interface ChunkFile {
  filename: string
  size: number
  modified_time: number
}

interface Chunk {
  content: string
  metadata: {
    chunk_id: number
    page_number?: number
    page_range?: string
    word_count?: number
  }
}

interface ChunkData {
  filename: string
  total_chunks?: number
  total_pages?: number
  loading_method?: string
  chunking_method?: string
  timestamp?: string
  chunks?: Chunk[]
  embeddings?: Array<{
    embedding: number[]
    metadata: {
      chunk_id: number
      page_number?: number
      page_range?: string
      content?: string
      word_count?: number
      total_chunks?: number
      embedding_provider?: string
      embedding_model?: string
      embedding_timestamp?: string
      vector_dimension?: number
      filename?: string
    }
  }>
}

const files = ref<ChunkFile[]>([])
const selectedFile = ref('')
const chunkData = ref<ChunkData | null>(null)
const loadingFiles = ref(false)
const loadingDetail = ref(false)
const errorMessage = ref('')

const totalChunks = computed(() => {
  if (!chunkData.value) return 0
  return chunkData.value.total_chunks
    ?? chunkData.value.embeddings?.length
    ?? chunkData.value.chunks?.length
    ?? 0
})

const totalPages = computed(() => {
  if (!chunkData.value) return 0
  return chunkData.value.total_pages ?? 0
})

const detailTags = computed(() => {
  if (!chunkData.value) return []
  return [
    chunkData.value.loading_method ? `加载方式: ${chunkData.value.loading_method}` : '',
    chunkData.value.chunking_method ? `切分方式: ${chunkData.value.chunking_method}` : ''
  ].filter(Boolean)
})

const contentItems = computed(() => {
  if (!chunkData.value) return []
  if (Array.isArray(chunkData.value.embeddings) && chunkData.value.embeddings.length > 0) {
    return chunkData.value.embeddings.map(item => ({
      chunk_id: item.metadata.chunk_id,
      page_number: item.metadata.page_number,
      page_range: item.metadata.page_range,
      word_count: item.metadata.word_count,
      content: item.metadata.content || '',
      raw: item
    }))
  }
  if (Array.isArray(chunkData.value.chunks) && chunkData.value.chunks.length > 0) {
    return chunkData.value.chunks.map(item => ({
      chunk_id: item.metadata.chunk_id,
      page_number: item.metadata.page_number,
      page_range: item.metadata.page_range,
      word_count: item.metadata.word_count,
      content: item.content || '',
      raw: item
    }))
  }
  return []
})

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function formatTime(unixTime: number): string {
  return new Date(unixTime * 1000).toLocaleString('zh-CN')
}

async function loadFiles() {
  loadingFiles.value = true
  errorMessage.value = ''
  try {
    const response = await request.get('/chunks/files') as { files?: ChunkFile[] }
    files.value = response.files || []
  } catch (error) {
    console.error('Failed to load chunk files:', error)
    errorMessage.value = '加载分块文件失败，请确认后端服务已启动。'
    files.value = []
  } finally {
    loadingFiles.value = false
  }
}

async function selectFile(filename: string) {
  if (selectedFile.value === filename) return
  selectedFile.value = filename
  chunkData.value = null
  loadingDetail.value = true
  errorMessage.value = ''

  try {
    const response = await request.get(`/chunks/file/${encodeURIComponent(filename)}`) as { data?: ChunkData }
    if (!response.data) {
      throw new Error('invalid chunk file data')
    }
    chunkData.value = response.data
  } catch (error) {
    console.error('Failed to load chunk detail:', error)
    errorMessage.value = '加载分块详情失败，请检查文件内容是否为合法 JSON。'
    selectedFile.value = ''
  } finally {
    loadingDetail.value = false
  }
}

onMounted(() => {
  loadFiles()
})
</script>

<template>
  <div class="chunks-page">
    <div class="page-header">
      <div class="title-row">
        <span class="title-icon">📄</span>
        <h1>文档分块查看</h1>
      </div>
      <el-button type="primary" :loading="loadingFiles" @click="loadFiles">
        <span class="btn-emoji">🔄</span>
        刷新文件
      </el-button>
    </div>

    <el-alert
      v-if="errorMessage"
      :title="errorMessage"
      type="error"
      show-icon
      class="error-alert"
    />

    <div class="content-grid">
      <div class="left-panel">
        <div class="panel-title">已处理文件（{{ files.length }}）</div>
        <el-skeleton v-if="loadingFiles" :rows="6" animated />
        <div v-else-if="files.length === 0" class="empty-tip">暂无分块文件</div>
        <div v-else class="file-list">
          <button
            v-for="file in files"
            :key="file.filename"
            type="button"
            class="file-item"
            :class="{ active: file.filename === selectedFile }"
            @click="selectFile(file.filename)"
          >
            <div class="file-icon">🗂️</div>
            <div class="file-body">
              <div class="file-name">{{ file.filename }}</div>
              <div class="file-meta">{{ formatSize(file.size) }} | {{ formatTime(file.modified_time) }}</div>
            </div>
          </button>
        </div>
      </div>

      <div class="right-panel">
        <div v-if="loadingDetail" class="detail-loading">
          <el-skeleton :rows="8" animated />
        </div>

        <div v-else-if="!chunkData" class="empty-tip">
          请选择左侧文件查看分块详情
        </div>

        <div v-else class="detail-content">
          <div class="detail-header">
            <div class="detail-title">{{ chunkData.filename }}</div>
            <div class="detail-stats">
              <el-tag type="info">chunks: {{ totalChunks }}</el-tag>
              <el-tag type="success">pages: {{ totalPages }}</el-tag>
              <el-tag v-for="tag in detailTags" :key="tag" effect="plain">{{ tag }}</el-tag>
            </div>
          </div>

          <el-collapse>
            <el-collapse-item
              v-for="item in contentItems"
              :key="item.chunk_id"
              :name="item.chunk_id"
            >
              <template #title>
                <div class="chunk-title">
                  <span class="chunk-badge">🧩</span>
                  <span class="chunk-id">Chunk #{{ item.chunk_id }}</span>
                  <span class="chunk-meta">
                    page: {{ item.page_number ?? '-' }}
                    | range: {{ item.page_range ?? '-' }}
                    | words: {{ item.word_count ?? '-' }}
                  </span>
                </div>
              </template>
              <pre class="chunk-content">{{ item.content }}</pre>
            </el-collapse-item>
          </el-collapse>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chunks-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.title-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.title-icon {
  font-size: 24px;
}

.btn-emoji {
  margin-right: 6px;
}

h1 {
  margin: 0;
  font-size: 22px;
}

.error-alert {
  margin-bottom: 4px;
}

.content-grid {
  display: grid;
  grid-template-columns: 360px minmax(0, 1fr);
  gap: 16px;
  min-height: 580px;
}

.left-panel,
.right-panel {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}

.panel-title {
  font-size: 14px;
  font-weight: 700;
  color: #4b5563;
  margin-bottom: 12px;
}

.file-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 680px;
  overflow: auto;
}

.file-item {
  text-align: left;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 10px 12px;
  background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  transition: all 0.22s ease;
}

.file-item:hover {
  border-color: #93c5fd;
  background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
  transform: translateY(-1px);
}

.file-item.active {
  border-color: #409eff;
  background: linear-gradient(135deg, #eef6ff 0%, #e0efff 100%);
  box-shadow: 0 10px 20px rgba(64, 158, 255, 0.12);
}

.file-icon {
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  background: linear-gradient(135deg, #c7b5ff 0%, #8b7cff 100%);
  color: #fff;
  font-size: 18px;
  box-shadow: 0 10px 16px rgba(139, 124, 255, 0.18);
  flex-shrink: 0;
}

.file-body {
  min-width: 0;
}

.file-name {
  color: #111827;
  font-weight: 700;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-meta {
  margin-top: 5px;
  color: #6b7280;
  font-size: 12px;
}

.detail-loading {
  padding: 8px;
}

.empty-tip {
  color: #6b7280;
  height: 100%;
  min-height: 260px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}

.detail-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.detail-header {
  border-bottom: 1px solid #f0f2f5;
  padding-bottom: 10px;
}

.detail-title {
  font-size: 16px;
  font-weight: 800;
  color: #1f2937;
  margin-bottom: 8px;
}

.detail-stats {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.chunk-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding-right: 8px;
  gap: 8px;
}

.chunk-badge {
  width: 26px;
  height: 26px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #ffd976 0%, #ffb84d 100%);
  box-shadow: 0 8px 14px rgba(255, 184, 77, 0.18);
  flex-shrink: 0;
}

.chunk-id {
  color: #1f2937;
  font-weight: 700;
  flex-shrink: 0;
}

.chunk-meta {
  color: #6b7280;
  font-size: 12px;
  margin-left: auto;
}

.chunk-content {
  white-space: pre-wrap;
  word-break: break-word;
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 12px;
  margin: 0;
  line-height: 1.6;
  color: #374151;
  max-height: 320px;
  overflow: auto;
}

@media (max-width: 1024px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
}
</style>
