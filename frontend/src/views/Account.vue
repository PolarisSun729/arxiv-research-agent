<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
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
  <div v-if="currentUser" class="page account-page">
    <header class="page-header">
      <div class="page-header__text">
        <h1 class="page-title">账号设置</h1>
        <p class="page-desc">查看账号信息和每日配额，修改登录密码。</p>
      </div>
      <div class="page-actions">
        <el-button :icon="Refresh" @click="refresh">刷新</el-button>
      </div>
    </header>

    <section class="surface profile-card">
      <span class="avatar">{{ currentUser.username.slice(0, 1).toUpperCase() }}</span>
      <div class="profile-meta">
        <div class="profile-name">
          {{ currentUser.username }}
          <span class="chip chip--primary">{{ ROLE_LABELS[currentUser.role] }}</span>
        </div>
        <div class="muted">{{ currentUser.email }}</div>
      </div>
    </section>

    <section class="surface section-card">
      <div class="section-head">
        <div>
          <h2 class="section-title">今日配额</h2>
          <p class="section-desc">配额每天 UTC 00:00 重置。论文搜索、详情、推荐和索引等按请求计数；Agent 内部调用也计入对应额度。</p>
        </div>
      </div>
      <div class="quota-grid">
        <div v-for="quota in currentUser.quotas" :key="quota.quota_type" class="surface-muted quota-item">
          <div class="quota-label">{{ QUOTA_LABELS[quota.quota_type as keyof typeof QUOTA_LABELS] }}</div>
          <div class="quota-value">
            {{ quota.used_today }}
            <span class="quota-limit">/ {{ quota.daily_limit === -1 ? '不限' : quota.daily_limit }}</span>
          </div>
          <el-progress
            v-if="quota.daily_limit > 0"
            :percentage="Math.min(100, Math.round((quota.used_today / quota.daily_limit) * 100))"
            :show-text="false"
            :stroke-width="6"
            :status="quota.remaining === 0 ? 'exception' : undefined"
          />
          <div class="quota-remaining muted">剩余 {{ quota.remaining === -1 ? '不限' : quota.remaining }}</div>
        </div>
      </div>
    </section>

    <section class="surface section-card">
      <div class="section-head">
        <div>
          <h2 class="section-title">修改密码</h2>
          <p class="section-desc">修改后所有旧登录会话都会失效。</p>
        </div>
      </div>
      <form class="password-form" @submit.prevent="savePassword">
        <el-input v-model="currentPassword" type="password" placeholder="当前密码" aria-label="当前密码" autocomplete="current-password" :maxlength="72" show-password />
        <el-input v-model="newPassword" type="password" placeholder="新密码" aria-label="新密码" autocomplete="new-password" :maxlength="72" show-password />
        <el-input v-model="confirmation" type="password" placeholder="确认新密码" aria-label="确认新密码" autocomplete="new-password" :maxlength="72" show-password />
        <p class="help">至少 8 个字符，包含大小写字母和数字，最多 72 个 UTF-8 字节。</p>
        <div>
          <el-button native-type="submit" type="primary" :loading="saving" :disabled="!currentPassword || !newPassword || !confirmation">更新密码</el-button>
        </div>
      </form>
    </section>
  </div>
</template>

<style scoped>
.account-page {
  max-width: 880px;
}

.profile-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  margin-bottom: 16px;
}

.avatar {
  width: 48px;
  height: 48px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: var(--color-primary-soft);
  color: var(--color-primary);
  font-size: 20px;
  font-weight: 650;
}

.profile-name {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 17px;
  font-weight: 600;
}

.section-card {
  padding: 20px;
  margin-bottom: 16px;
}

.quota-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}

.quota-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 14px 16px;
}

.quota-label {
  font-size: 13px;
  color: var(--color-text-3);
}

.quota-value {
  font-size: 22px;
  font-weight: 650;
  font-variant-numeric: tabular-nums;
}

.quota-limit {
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-3);
}

.quota-remaining {
  font-size: 12px;
}

.password-form {
  display: grid;
  gap: 12px;
  max-width: 420px;
}

.help {
  color: var(--color-text-3);
  font-size: 12.5px;
}
</style>
