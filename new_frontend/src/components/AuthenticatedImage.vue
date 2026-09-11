<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import { fetchApiBlob } from '@/api/auth'

const props = defineProps<{ src: string }>()
const imageUrl = ref('')
let controller: AbortController | null = null

function releaseImage() {
  controller?.abort()
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
  imageUrl.value = ''
}

watch(() => props.src, async src => {
  releaseImage()
  if (!src) return
  const requestController = new AbortController()
  controller = requestController
  try {
    // img 标签无法携带认证头，先通过受保护请求加载，再用 Blob URL 展示与预览。
    const blob = await fetchApiBlob(src, requestController.signal)
    if (!requestController.signal.aborted) imageUrl.value = URL.createObjectURL(blob)
  } catch {
    // 认证失败由统一客户端处理；图片缺失保留文字证据，不把内部错误暴露在页面上。
  }
}, { immediate: true })

onBeforeUnmount(releaseImage)
</script>

<template>
  <el-image :src="imageUrl" :preview-src-list="imageUrl ? [imageUrl] : []" preview-teleported>
    <template #error><slot name="error">图片暂不可用，仍保留文字证据。</slot></template>
  </el-image>
</template>
