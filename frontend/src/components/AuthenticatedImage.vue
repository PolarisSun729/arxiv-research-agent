<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import { fetchApiBlob } from '@/api/auth'

const props = defineProps<{ src: string }>()
const imageUrl = ref('')
const failed = ref(false)
let controller: AbortController | null = null

function releaseImage() {
  controller?.abort()
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
  imageUrl.value = ''
}

watch(() => props.src, async src => {
  releaseImage()
  failed.value = false
  if (!src) {
    failed.value = true
    return
  }
  const requestController = new AbortController()
  controller = requestController
  try {
    // img 标签无法携带认证头，先通过受保护请求加载，再用 Blob URL 展示与预览。
    const blob = await fetchApiBlob(src, requestController.signal)
    if (!requestController.signal.aborted) imageUrl.value = URL.createObjectURL(blob)
  } catch {
    // 认证失败由统一客户端处理；图片缺失保留文字证据，不把内部错误暴露在页面上。
    if (!requestController.signal.aborted) failed.value = true
  }
}, { immediate: true })

onBeforeUnmount(releaseImage)
</script>

<template>
  <el-image v-if="imageUrl" :src="imageUrl" :preview-src-list="[imageUrl]" preview-teleported>
    <template #error><slot name="error">图片暂不可用，仍保留文字证据。</slot></template>
  </el-image>
  <!-- 请求还没回来时显示占位，不能先闪出“图片暂不可用”。 -->
  <div v-else-if="failed"><slot name="error">图片暂不可用，仍保留文字证据。</slot></div>
  <div v-else class="authenticated-image__loading" aria-busy="true" />
</template>

<style scoped>
.authenticated-image__loading {
  min-height: 120px;
  background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%) 0 0 / 200% 100%;
  animation: authenticated-image-shimmer 1.2s ease-in-out infinite;
}

@keyframes authenticated-image-shimmer {
  to { background-position: -200% 0; }
}
</style>
