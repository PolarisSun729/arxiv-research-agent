<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Check, Key, SwitchButton } from '@element-plus/icons-vue'
import { DEMO_USER_OPTIONS, useUserContext } from '@/composables/useUserContext'

const router = useRouter()
const route = useRoute()
const userContext = useUserContext()
const userIdInput = ref(userContext.userId.value)

const normalizedUserId = computed(() => userIdInput.value.trim())
const canSubmit = computed(() => Boolean(normalizedUserId.value))

function applyUserId(userId: string) {
  userIdInput.value = userId
}

function handleSubmit() {
  if (!canSubmit.value) {
    ElMessage.warning('请输入用户 ID')
    return
  }

  // 当前阶段没有真实认证服务，登录页只负责确定本地演示身份；
  // 后续接入 token/session 时，替换 useUserContext 的来源即可保留调用方契约。
  userContext.setUserId(normalizedUserId.value)
  ElMessage.success(`已切换到用户 ${normalizedUserId.value}`)

  const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/dashboard'
  const safeRedirect = redirect.startsWith('/') && !['/', '/login'].includes(redirect) ? redirect : '/dashboard'
  router.push(safeRedirect)
}

function handleReset() {
  userContext.resetUserId()
  userIdInput.value = userContext.userId.value
  ElMessage.success(`已重置为 ${userContext.userId.value}`)
}
</script>

<template>
  <div class="login-page">
    <section class="login-panel">
      <div class="login-copy">
        <p class="kicker">用户身份</p>
        <h1>选择本轮使用的用户 ID</h1>
        <p class="description">
          偏好、兴趣向量、研究画像和 Agent 会话都按用户 ID 隔离。请选择已有演示账号，或输入一个新的本地账号。
        </p>
      </div>

      <div class="login-form">
        <el-input
          v-model="userIdInput"
          size="large"
          placeholder="请输入用户 ID"
          clearable
          @keyup.enter="handleSubmit"
        >
          <template #prefix>
            <el-icon><Key /></el-icon>
          </template>
        </el-input>

        <div class="quick-users">
          <button
            v-for="option in DEMO_USER_OPTIONS"
            :key="option.id"
            type="button"
            :class="['quick-user', { active: normalizedUserId === option.id }]"
            @click="applyUserId(option.id)"
          >
            <span class="quick-user__title">{{ option.label }}</span>
            <span class="quick-user__id">{{ option.id }}</span>
            <span class="quick-user__desc">{{ option.description }}</span>
          </button>
        </div>

        <div class="form-actions">
          <el-button :icon="SwitchButton" @click="handleReset">重置</el-button>
          <el-button type="primary" :icon="Check" :disabled="!canSubmit" @click="handleSubmit">
            使用该用户
          </el-button>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.login-page {
  min-height: calc(100vh - 100px);
  display: grid;
  place-items: center;
  padding: 24px;
}

.login-panel {
  width: min(920px, 100%);
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
  gap: 28px;
  align-items: stretch;
  padding: 28px;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 16px 42px rgba(15, 23, 42, 0.08);
}

.login-copy {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.kicker {
  margin: 0 0 10px;
  color: #2563eb;
  font-size: 13px;
  font-weight: 700;
}

.login-copy h1 {
  margin: 0;
  color: #111827;
  font-size: 28px;
  line-height: 1.25;
}

.description {
  margin: 14px 0 0;
  color: #475569;
  line-height: 1.7;
}

.login-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.quick-users {
  display: grid;
  gap: 10px;
}

.quick-user {
  display: grid;
  gap: 5px;
  width: 100%;
  padding: 14px 16px;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #f8fafc;
  color: #334155;
  cursor: pointer;
  text-align: left;
  transition: border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
}

.quick-user:hover,
.quick-user.active {
  border-color: #2563eb;
  background: #eff6ff;
  box-shadow: 0 8px 20px rgba(37, 99, 235, 0.12);
}

.quick-user__title {
  font-weight: 700;
  color: #0f172a;
}

.quick-user__id {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  color: #2563eb;
}

.quick-user__desc {
  color: #64748b;
  line-height: 1.5;
}

.form-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

@media (max-width: 900px) {
  .login-panel {
    grid-template-columns: 1fr;
  }
}
</style>
