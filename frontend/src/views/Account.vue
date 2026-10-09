<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { changePassword, currentUser, QUOTA_LABELS, refreshCurrentUser, ROLE_LABELS } from '@/api/auth'
import { getErrorMessage } from '@/api/errors'

const router = useRouter()
const currentPassword = ref('')
const newPassword = ref('')
const confirmation = ref('')
const saving = ref(false)

async function refresh() {
  try { await refreshCurrentUser() } catch (error) { ElMessage.error(getErrorMessage(error, '账号信息读取失败')) }
}
onMounted(refresh)

async function savePassword() {
  if (newPassword.value !== confirmation.value) { ElMessage.error('两次新密码不一致'); return }
  saving.value = true
  try {
    await changePassword(currentPassword.value, newPassword.value)
    ElMessage.success('密码已更新，请重新登录。其他登录会话也已失效。')
    await router.replace('/login')
  } catch (error) {
    ElMessage.error(getErrorMessage(error, '密码修改失败'))
  } finally {
    saving.value = false
    currentPassword.value = newPassword.value = confirmation.value = ''
  }
}
</script>

<template>
  <div class="account-page" v-if="currentUser">
    <el-card>
      <template #header><strong>账号与配额</strong><el-button text @click="refresh">刷新</el-button></template>
      <p>{{ currentUser.username }} · {{ ROLE_LABELS[currentUser.role] }}</p>
      <p>{{ currentUser.email }}</p>
      <el-table :data="currentUser.quotas">
        <el-table-column label="项目"><template #default="{ row }">{{ QUOTA_LABELS[row.quota_type as keyof typeof QUOTA_LABELS] }}</template></el-table-column>
        <el-table-column prop="used_today" label="今日已用" />
        <el-table-column label="每日限额"><template #default="{ row }">{{ row.daily_limit === -1 ? '不限' : row.daily_limit }}</template></el-table-column>
        <el-table-column label="剩余"><template #default="{ row }">{{ row.remaining === -1 ? '不限' : row.remaining }}</template></el-table-column>
      </el-table>
      <p class="help">配额每天 UTC 00:00 重置。论文搜索、详情、推荐和索引等按请求计数；Agent 内部调用也计入对应的论文与问答额度。</p>
    </el-card>
    <el-card>
      <template #header><strong>修改密码</strong></template>
      <form class="password-form" @submit.prevent="savePassword">
        <el-input v-model="currentPassword" type="password" placeholder="当前密码" aria-label="当前密码" autocomplete="current-password" :maxlength="72" />
        <el-input v-model="newPassword" type="password" placeholder="新密码" aria-label="新密码" autocomplete="new-password" :maxlength="72" />
        <el-input v-model="confirmation" type="password" placeholder="确认新密码" aria-label="确认新密码" autocomplete="new-password" :maxlength="72" />
        <p class="help">至少 8 个字符，包含大小写字母和数字，最多 72 个 UTF-8 字节。修改后所有旧登录会话失效。</p>
        <el-button native-type="submit" type="primary" :loading="saving" :disabled="!currentPassword || !newPassword || !confirmation">更新密码</el-button>
      </form>
    </el-card>
  </div>
</template>

<style scoped>
.account-page { display: grid; gap: 20px; max-width: 960px; margin: 0 auto; }
.password-form { display: grid; gap: 14px; max-width: 440px; }
.help { color: #64748b; font-size: 13px; line-height: 1.7; }
</style>
