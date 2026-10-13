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
    <section class="login-brand">
      <div class="brand-row">
        <span class="brand-mark">A</span>
        <span class="brand-name">arXiv 研究助手</span>
      </div>
      <div class="brand-copy">
        <h2>从海量论文里，找到真正与你相关的那几篇</h2>
        <ul class="brand-points">
          <li><span class="dot" />本地 OAI 镜像库检索，支持字段前缀与布尔语法</li>
          <li><span class="dot" />根据你的标记与研究画像生成个性化推荐</li>
          <li><span class="dot" />针对单篇论文的检索增强问答，回答附带原文证据</li>
        </ul>
      </div>
      <p class="brand-foot">登录状态仅保留在当前标签页</p>
    </section>

    <section class="login-main">
      <div class="login-card">
        <div class="login-head">
          <h1>{{ mode === 'api_key' ? '访问验证' : registering ? '创建账号' : '登录' }}</h1>
          <p v-if="mode === 'jwt' && !registering" class="muted">使用个人账号管理阅读笔记、问答历史和研究画像。</p>
          <p v-else-if="mode === 'jwt'" class="muted">新注册账号为访客，研究者权限由管理员分配。</p>
          <p v-else class="muted">此服务使用团队访问密钥。数据命名空间用于选择团队内的数据，不提供个人账号隔离。</p>
        </div>

        <form class="login-form" @submit.prevent="handleSubmit">
          <template v-if="mode === 'jwt'">
            <label class="field">
              <span class="field-label">用户名</span>
              <el-input v-model="username" size="large" placeholder="请输入用户名" aria-label="用户名" autocomplete="username" :maxlength="50" />
            </label>
            <label v-if="registering" class="field">
              <span class="field-label">邮箱</span>
              <el-input v-model="email" size="large" type="email" placeholder="name@example.com" aria-label="邮箱" autocomplete="email" :maxlength="254" />
            </label>
            <label class="field">
              <span class="field-label">密码</span>
              <el-input v-model="password" size="large" type="password" placeholder="请输入密码" aria-label="密码" :autocomplete="registering ? 'new-password' : 'current-password'" :maxlength="72" show-password />
              <span v-if="registering" class="field-hint">至少 8 个字符，包含大小写字母和数字，UTF-8 编码后最多 72 字节。</span>
            </label>
          </template>
          <template v-else>
            <label class="field">
              <span class="field-label">访问密钥</span>
              <el-input v-model="accessKey" size="large" type="password" placeholder="管理员提供的访问密钥" aria-label="访问密钥" autocomplete="off" :maxlength="256" />
            </label>
            <label class="field">
              <span class="field-label">数据命名空间 <span class="optional">可选</span></span>
              <el-input v-model="compatUserId" size="large" placeholder="留空使用默认命名空间" aria-label="数据命名空间" />
            </label>
          </template>

          <div v-if="configError" class="notice notice--danger config-error">
            <span>{{ configError }}</span>
            <el-button size="small" @click="loadConfig">重试连接</el-button>
          </div>

          <el-button class="submit-btn" type="primary" native-type="submit" size="large" :disabled="!canSubmit" :loading="submitting">
            {{ registering ? '注册并登录' : '登录' }}
          </el-button>

          <div class="login-alt">
            <el-button v-if="mode === 'jwt' && registrationEnabled" link type="primary" :disabled="submitting" @click="registering = !registering">
              {{ registering ? '已有账号，返回登录' : '没有账号？创建访客账号' }}
            </el-button>
            <span v-else-if="mode === 'jwt' && configured" class="muted">暂未开放注册，请联系管理员创建账号</span>
          </div>
        </form>
      </div>
    </section>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(360px, 5fr) 7fr;
  background: var(--color-bg);
}

.login-brand {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 40px 48px;
  background: #1e1b4b;
  background-image:
    radial-gradient(circle at 20% 0%, rgba(129, 140, 248, 0.35), transparent 50%),
    radial-gradient(circle at 100% 100%, rgba(79, 70, 229, 0.4), transparent 45%);
  color: #e0e7ff;
}

.brand-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.brand-mark {
  width: 34px;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 9px;
  background: #fff;
  color: var(--color-primary);
  font-weight: 700;
  font-size: 17px;
}

.brand-name {
  font-size: 16px;
  font-weight: 650;
  color: #fff;
}

.brand-copy h2 {
  max-width: 420px;
  font-size: 28px;
  line-height: 1.35;
  font-weight: 650;
  color: #fff;
  letter-spacing: -0.01em;
}

.brand-points {
  list-style: none;
  margin-top: 28px;
  display: grid;
  gap: 14px;
  font-size: 14px;
  color: #c7d2fe;
}

.brand-points li {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.dot {
  width: 6px;
  height: 6px;
  margin-top: 8px;
  flex-shrink: 0;
  border-radius: 50%;
  background: #a5b4fc;
}

.brand-foot {
  font-size: 12px;
  color: #a5b4fc;
}

.login-main {
  display: grid;
  place-items: center;
  padding: 40px 24px;
}

.login-card {
  width: min(400px, 100%);
}

.login-head h1 {
  font-size: 24px;
  font-weight: 650;
  color: var(--color-text);
}

.login-head p {
  margin-top: 8px;
  line-height: 1.7;
}

.login-form {
  margin-top: 28px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field-label {
  font-size: 13px;
  font-weight: 550;
  color: var(--color-text-2);
}

.optional {
  font-weight: 400;
  color: var(--color-text-4);
}

.field-hint {
  font-size: 12px;
  color: var(--color-text-3);
}

.config-error {
  align-items: center;
  justify-content: space-between;
}

.submit-btn {
  width: 100%;
  margin-top: 4px;
}

.login-alt {
  display: flex;
  justify-content: center;
  font-size: 13px;
}

@media (max-width: 900px) {
  .login-page {
    grid-template-columns: 1fr;
  }

  .login-brand {
    display: none;
  }
}
</style>
