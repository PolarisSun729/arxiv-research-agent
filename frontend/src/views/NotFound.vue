<script setup lang="ts">
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
</script>

<template>
  <div class="page not-found">
    <div class="code">404</div>
    <h1 class="page-title">页面不存在</h1>
    <p class="muted">找不到 <span class="mono">{{ route.path }}</span>，它可能已被移动或删除。</p>
    <el-button type="primary" @click="router.push('/dashboard')">返回首页</el-button>
  </div>
</template>

<style scoped>
.not-found {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  min-height: 70vh;
  text-align: center;
}

.code {
  font-size: 64px;
  font-weight: 700;
  line-height: 1;
  letter-spacing: -0.04em;
  color: var(--color-primary-border);
}

.not-found .el-button {
  margin-top: 12px;
}
</style>
