<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useUserContext } from '@/composables/useUserContext'
import { loadAuthConfiguration, loginWithPassword, registerAccount, setApiKey, verifyAccessKey } from '@/api/auth'
import type { AuthMode } from '@/api/auth'
import { getErrorMessage } from '@/api/errors'

const router = useRouter()
const route = useRoute()
const userContext = useUserContext()
const mode = ref<AuthMode>('jwt')
const configured = ref(false)
const configError = ref('')
const registrationEnabled = ref(false)
const registering = ref(false)
const username = ref('')
const email = ref('')
const password = ref('')
const accessKey = ref('')
const compatUserId = ref(userContext.userId.value)
const submitting = ref(false)
const canSubmit = computed(() => configured.value && !submitting.value && (mode.value === 'api_key'
  ? Boolean(accessKey.value.trim()) : Boolean(username.value.trim() && password.value && (!registering.value || email.value.trim()))))

async function loadConfig() {
  configured.value = false
  configError.value = ''
  try {
    const config = await loadAuthConfiguration()
    mode.value = config.mode
    registrationEnabled.value = config.registration_enabled
    configured.value = true
  } catch (error) {
    configError.value = getErrorMessage(error, '无法读取登录配置，请重试')
  }
}
onMounted(loadConfig)

async function handleSubmit() {
  if (!canSubmit.value) return
  submitting.value = true
  try {
    if (mode.value === 'api_key') {
      const key = accessKey.value.trim()
      await verifyAccessKey(key)
      setApiKey(key)
      userContext.setUserId(compatUserId.value)
    } else {
      if (registering.value) await registerAccount(username.value.trim(), email.value.trim(), password.value)
      await loginWithPassword(username.value.trim(), password.value)
    }
    ElMessage.success('登录成功')
    const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/dashboard'
    // 回跳只能指向本站路由，不能让登录表单成为开放重定向入口。
    const safeRedirect = redirect.startsWith('/') && !redirect.startsWith('//') && !redirect.includes(String.fromCharCode(92)) && !['/', '/login'].includes(redirect) ? redirect : '/dashboard'
    await router.replace(safeRedirect)
  } catch (error) {
    ElMessage.error(getErrorMessage(error, '登录失败'))
  } finally {
    submitting.value = false
    password.value = ''
    accessKey.value = ''
  }
}
</script>

<template>
  <div class="login-page">
    <section class="login-panel">
      <div class="login-copy">
        <p class="kicker">arXiv 研究助手</p>
        <h1>{{ mode === 'api_key' ? '访问验证' : registering ? '创建账号' : '登录账号' }}</h1>
        <p class="description" v-if="mode === 'jwt'">使用个人账号管理阅读笔记、问答历史和研究画像。登录仅保留在当前标签页。</p>
        <p class="description" v-else>此服务使用团队访问密钥。用户 ID 用于选择团队内的数据，不提供个人账号隔离。</p>
        <p class="description" v-if="registering">新注册账号为访客。研究者权限由管理员分配。</p>
      </div>
      <form class="login-form" @submit.prevent="handleSubmit">
        <template v-if="mode === 'jwt'">
          <el-input v-model="username" size="large" placeholder="用户名" aria-label="用户名" autocomplete="username" :maxlength="50" />
          <el-input v-if="registering" v-model="email" size="large" type="email" placeholder="邮箱" aria-label="邮箱" autocomplete="email" :maxlength="254" />
          <el-input v-model="password" size="large" type="password" placeholder="密码" aria-label="密码" :autocomplete="registering ? 'new-password' : 'current-password'" :maxlength="72" show-password />
          <p v-if="registering" class="description">密码至少 8 个字符，包含大小写字母和数字，UTF-8 编码后最多 72 字节。</p>
        </template>
        <template v-else>
          <el-input v-model="accessKey" size="large" type="password" placeholder="管理员提供的访问密钥" aria-label="访问密钥" autocomplete="off" :maxlength="256" />
          <el-input v-model="compatUserId" size="large" placeholder="数据命名空间（可留空）" aria-label="数据命名空间" />
        </template>
        <el-alert v-if="configError" :title="configError" type="error" :closable="false" />
        <el-button v-if="configError" @click="loadConfig">重试连接</el-button>
        <el-button type="primary" native-type="submit" size="large" :disabled="!canSubmit" :loading="submitting">{{ registering ? '注册并登录' : '登录' }}</el-button>
        <el-button v-if="mode === 'jwt' && registrationEnabled" text :disabled="submitting" @click="registering = !registering">{{ registering ? '已有账号，返回登录' : '创建访客账号' }}</el-button>
        <p v-if="mode === 'jwt' && configured && !registrationEnabled" class="description">暂未开放注册，请联系管理员创建账号。</p>
      </form>
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
