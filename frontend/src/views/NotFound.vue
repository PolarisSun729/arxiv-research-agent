<script setup lang="ts">
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
</script>

<template>
  <div class="not-found">
    <el-result icon="warning" title="404" :sub-title="`页面不存在：${route.path}`">
      <template #extra>
        <el-button type="primary" @click="router.push('/dashboard')">返回首页</el-button>
      </template>
    </el-result>
  </div>
</template>

<style scoped>
.not-found {
  display: flex;
  justify-content: center;
  padding: 80px 16px;
}
</style>
