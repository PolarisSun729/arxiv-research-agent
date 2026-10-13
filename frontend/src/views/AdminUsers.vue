<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import { createUser, currentUser, getUser, listUsers, QUOTA_LABELS, refreshCurrentUser, ROLE_LABELS, updateQuota, updateUser } from '@/api/auth'
import type { AuthUser, UserRole } from '@/api/auth'
import { getErrorMessage } from '@/api/errors'

const users = ref<AuthUser[]>([])
const total = ref(0)
const page = ref(1)
const loading = ref(false)
const creating = ref(false)
const createDialog = ref(false)
const quotaUser = ref<AuthUser | null>(null)
const saving = ref(false)
const newUser = reactive({ username: '', email: '', password: '', role: 'researcher' as UserRole })

async function load() {
  loading.value = true
  try {
    const result = await listUsers((page.value - 1) * 50)
    users.value = result.items
    total.value = result.total
  } catch (error) { ElMessage.error(getErrorMessage(error, '账号列表读取失败')) }
  finally { loading.value = false }
}
onMounted(load)

async function saveNewUser() {
  creating.value = true
  try {
    await createUser({ ...newUser })
    createDialog.value = false
    newUser.username = newUser.email = ''
    ElMessage.success('账号已创建')
    await load()
  } catch (error) { ElMessage.error(getErrorMessage(error, '账号创建失败')) }
  finally { creating.value = false; newUser.password = '' }
}

async function changeUser(user: AuthUser, change: { role?: UserRole; is_active?: boolean }) {
  saving.value = true
  try {
    await updateUser(user.user_id, change)
    ElMessage.success('账号已更新，原登录会话已失效')
    // 管理员修改自己的权限后同样需要重新认证，不能继续保留旧管理界面。
    if (user.user_id === currentUser.value?.user_id) await refreshCurrentUser()
    await load()
  } catch (error) { ElMessage.error(getErrorMessage(error, '账号更新失败')) }
  finally { saving.value = false }
}

async function editQuotas(user: AuthUser) {
  try { quotaUser.value = await getUser(user.user_id) }
  catch (error) { ElMessage.error(getErrorMessage(error, '配额读取失败')) }
}

async function saveQuotas() {
  if (!quotaUser.value) return
  saving.value = true
  try {
    for (const quota of quotaUser.value.quotas) await updateQuota(quotaUser.value.user_id, quota.quota_type, quota.daily_limit)
    ElMessage.success('配额已更新，今日已用量保留')
    quotaUser.value = null
  } catch (error) { ElMessage.error(getErrorMessage(error, '部分配额可能未保存，请刷新后重试')) }
  finally { saving.value = false }
}
</script>

<template>
  <div class="page">
    <header class="page-header">
      <div class="page-header__text">
        <h1 class="page-title">账号管理</h1>
        <p class="page-desc">停用或修改角色会撤销该账号的所有登录会话。角色变更恢复该角色的默认限额并保留今日已用量；系统至少保留一名启用的管理员。</p>
      </div>
      <div class="page-actions">
        <el-button type="primary" :icon="Plus" @click="createDialog = true">创建账号</el-button>
      </div>
    </header>

    <section class="surface table-card">
      <el-table :data="users" v-loading="loading">
        <el-table-column label="用户" min-width="200">
          <template #default="{ row }">
            <div class="user-cell">
              <span class="avatar">{{ row.username.slice(0, 1).toUpperCase() }}</span>
              <div class="user-cell__meta">
                <span class="user-cell__name">{{ row.username }}</span>
                <span class="muted">{{ row.email }}</span>
              </div>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="角色" min-width="145">
          <template #default="{ row }">
            <el-select :model-value="row.role" :disabled="saving" @change="(value: UserRole) => changeUser(row, { role: value })" aria-label="账号角色">
              <el-option v-for="(label, value) in ROLE_LABELS" :key="value" :label="label" :value="value" />
            </el-select>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="100">
          <template #default="{ row }">
            <span :class="['chip', row.is_active ? 'chip--success' : '']">{{ row.is_active ? '启用' : '停用' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="160" align="right">
          <template #default="{ row }">
            <el-button link type="primary" :disabled="saving" @click="editQuotas(row)">配额</el-button>
            <el-button link :type="row.is_active ? 'danger' : 'primary'" :disabled="saving" @click="changeUser(row, { is_active: !row.is_active })">{{ row.is_active ? '停用' : '启用' }}</el-button>
          </template>
        </el-table-column>
      </el-table>
      <div class="table-footer">
        <el-pagination v-model:current-page="page" :page-size="50" :total="total" layout="total, prev, pager, next" @current-change="load" />
      </div>
    </section>

    <el-dialog v-model="createDialog" title="创建账号" width="440px" @closed="newUser.password = ''">
      <form class="account-form" @submit.prevent="saveNewUser">
        <el-input v-model="newUser.username" placeholder="用户名：字母、数字、下划线或短横线" aria-label="新账号用户名" autocomplete="off" :maxlength="50" />
        <el-input v-model="newUser.email" type="email" placeholder="邮箱" aria-label="新账号邮箱" autocomplete="off" :maxlength="254" />
        <el-input v-model="newUser.password" type="password" placeholder="初始密码" aria-label="初始密码" autocomplete="new-password" :maxlength="72" show-password />
        <el-select v-model="newUser.role" aria-label="新账号角色"><el-option v-for="(label, value) in ROLE_LABELS" :key="value" :label="label" :value="value" /></el-select>
        <p class="help">密码至少 8 个字符，包含大小写字母和数字，UTF-8 编码后最多 72 字节。</p>
        <el-button type="primary" native-type="submit" :loading="creating" :disabled="!newUser.username || !newUser.email || !newUser.password">创建</el-button>
      </form>
    </el-dialog>

    <el-dialog :model-value="Boolean(quotaUser)" :title="`${quotaUser?.username || ''} 的配额`" width="460px" @close="quotaUser = null">
      <div class="quota-row" v-for="quota in quotaUser?.quotas" :key="quota.quota_type">
        <span>{{ QUOTA_LABELS[quota.quota_type] }}<span class="muted">（已用 {{ quota.used_today }}）</span></span>
        <el-input-number v-model="quota.daily_limit" :min="-1" :max="1000000" :precision="0" :aria-label="QUOTA_LABELS[quota.quota_type]" />
      </div>
      <p class="help">-1 表示不限，0 表示停止该类调用；配额不授予角色之外的权限。</p>
      <template #footer><el-button type="primary" :loading="saving" @click="saveQuotas">保存</el-button></template>
    </el-dialog>
  </div>
</template>

<style scoped>
.table-card {
  overflow: hidden;
}

.table-card :deep(.el-table th.el-table__cell) {
  background: var(--color-surface-muted);
  color: var(--color-text-3);
  font-weight: 550;
  font-size: 12.5px;
}

.table-footer {
  padding: 12px;
  border-top: 1px solid var(--color-border);
}

.user-cell {
  display: flex;
  align-items: center;
  gap: 10px;
}

.user-cell__meta {
  display: flex;
  flex-direction: column;
  line-height: 1.35;
  min-width: 0;
}

.user-cell__name {
  font-weight: 600;
  color: var(--color-text);
}

.user-cell__meta .muted {
  font-size: 12.5px;
}

.avatar {
  width: 30px;
  height: 30px;
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: var(--color-primary-soft);
  color: var(--color-primary);
  font-weight: 650;
  font-size: 13px;
}

.account-form {
  display: grid;
  gap: 14px;
}

.quota-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 14px;
}

.help {
  color: var(--color-text-3);
  font-size: 12.5px;
  line-height: 1.7;
}
</style>
