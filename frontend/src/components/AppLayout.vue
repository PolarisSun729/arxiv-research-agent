<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { Component } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import {
  House,
  Search,
  MagicStick,
  Star,
  Compass,
  CollectionTag,
  UserFilled,
  Files,
  Fold,
  Expand,
  Setting,
  SwitchButton
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { useUserContext } from '@/composables/useUserContext'
import { currentUser, logout, ROLE_LABELS } from '@/api/auth'
import { getErrorMessage } from '@/api/errors'

interface NavItem {
  path: string
  label: string
  icon: Component
  visible?: () => boolean
}

interface NavGroup {
  title: string
  items: NavItem[]
}

const router = useRouter()
const route = useRoute()
const userContext = useUserContext()
const COLLAPSE_KEY = 'arxiv_sidebar_collapsed'

function readCollapsed() {
  try {
    return window.localStorage.getItem(COLLAPSE_KEY) === '1'
  } catch {
    return false
  }
}

const collapsedPreference = ref(readCollapsed())
const narrowScreen = ref(false)
// 窄屏强制收起，不覆盖用户在宽屏下的偏好。
const collapsed = computed(() => narrowScreen.value || collapsedPreference.value)

function toggleCollapsed() {
  collapsedPreference.value = !collapsedPreference.value
  try {
    window.localStorage.setItem(COLLAPSE_KEY, collapsedPreference.value ? '1' : '0')
  } catch {
    // 偏好只是便利项，写不进去时保持当前会话内的状态即可。
  }
}

let mediaQuery: MediaQueryList | null = null
function syncNarrow() {
  narrowScreen.value = Boolean(mediaQuery?.matches)
}
onMounted(() => {
  mediaQuery = window.matchMedia('(max-width: 900px)')
  syncNarrow()
  mediaQuery.addEventListener('change', syncNarrow)
})
onBeforeUnmount(() => mediaQuery?.removeEventListener('change', syncNarrow))

const navGroups = computed<NavGroup[]>(() => {
  const groups: NavGroup[] = [
    {
      title: '发现',
      items: [
        { path: '/dashboard', label: '首页', icon: House },
        { path: '/search', label: '论文搜索', icon: Search },
        { path: '/agent-search', label: 'Agent 搜索', icon: MagicStick, visible: () => userContext.canResearch.value },
        { path: '/recommendations', label: '推荐论文', icon: Star }
      ]
    },
    {
      title: '我的',
      items: [
        { path: '/profile', label: '研究画像', icon: Compass },
        { path: '/labeled', label: '已标记论文', icon: CollectionTag }
      ]
    },
    {
      title: '管理',
      items: [
        { path: '/admin/users', label: '账号管理', icon: UserFilled, visible: () => currentUser.value?.role === 'admin' },
        { path: '/admin/indexes', label: '索引切分', icon: Files, visible: () => currentUser.value?.role === 'admin' }
      ]
    }
  ]
  return groups
    .map(group => ({ ...group, items: group.items.filter(item => !item.visible || item.visible()) }))
    .filter(group => group.items.length > 0)
})

function isActive(path: string) {
  return route.path === path || route.path.startsWith(`${path}/`)
}

const isBare = computed(() => route.meta.bare === true)
const displayName = computed(() => currentUser.value?.username || userContext.displayName.value)
const roleLabel = computed(() => {
  if (currentUser.value) return ROLE_LABELS[currentUser.value.role]
  return userContext.isDemoMode.value ? '团队密钥' : ''
})
const avatarText = computed(() => (displayName.value || '?').slice(0, 1).toUpperCase())
const canOpenAccount = computed(() => !userContext.isDemoMode.value && Boolean(currentUser.value))

async function handleLogout() {
  try {
    await logout()
    // 撤销成功后重新加载，释放已有会话和页面缓存，后退也会重新经过登录守卫。
    window.location.assign('/login')
  } catch (error) {
    ElMessage.error(getErrorMessage(error, '退出失败，请重试'))
  }
}

function handleUserCommand(command: string) {
  if (command === 'account') router.push('/account')
  if (command === 'login') router.push('/login')
  if (command === 'logout') void handleLogout()
}
</script>

<template>
  <router-view v-if="isBare" />

  <div v-else :class="['app-shell', { 'is-collapsed': collapsed }]">
    <aside class="sidebar">
      <router-link to="/dashboard" class="brand">
        <span class="brand-mark">A</span>
        <span v-if="!collapsed" class="brand-text">
          <span class="brand-name">arXiv 研究助手</span>
          <span class="brand-sub">论文发现 · 问答 · 推荐</span>
        </span>
      </router-link>

      <nav class="nav">
        <div v-for="group in navGroups" :key="group.title" class="nav-group">
          <div v-if="!collapsed" class="nav-group-title">{{ group.title }}</div>
          <div v-else class="nav-group-divider" />
          <el-tooltip
            v-for="item in group.items"
            :key="item.path"
            :content="item.label"
            placement="right"
            :disabled="!collapsed"
          >
            <router-link :to="item.path" :class="['nav-item', { 'is-active': isActive(item.path) }]">
              <el-icon class="nav-icon"><component :is="item.icon" /></el-icon>
              <span v-if="!collapsed" class="nav-label">{{ item.label }}</span>
            </router-link>
          </el-tooltip>
        </div>
      </nav>

      <div class="sidebar-footer">
        <el-dropdown trigger="click" placement="top-start" @command="handleUserCommand">
          <button type="button" class="user-card">
            <span class="avatar">{{ avatarText }}</span>
            <span v-if="!collapsed" class="user-meta">
              <span class="user-name">{{ displayName }}</span>
              <span v-if="roleLabel" class="user-role">{{ roleLabel }}</span>
            </span>
          </button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item v-if="canOpenAccount" command="account" :icon="Setting">账号设置</el-dropdown-item>
              <el-dropdown-item v-else command="login" :icon="Setting">切换身份</el-dropdown-item>
              <el-dropdown-item command="logout" :icon="SwitchButton" divided>退出登录</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>

        <button
          v-if="!narrowScreen"
          type="button"
          class="collapse-btn"
          :aria-label="collapsed ? '展开侧边栏' : '收起侧边栏'"
          @click="toggleCollapsed"
        >
          <el-icon><component :is="collapsed ? Expand : Fold" /></el-icon>
        </button>
      </div>
    </aside>

    <main class="main">
      <router-view :key="userContext.userId.value" />
    </main>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  height: 100vh;
  background: var(--color-bg);
}

.sidebar {
  flex-shrink: 0;
  width: var(--sidebar-width);
  display: flex;
  flex-direction: column;
  background: var(--color-surface);
  border-right: 1px solid var(--color-border);
  transition: width 0.2s ease;
}

.is-collapsed .sidebar {
  width: var(--sidebar-width-collapsed);
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 64px;
  padding: 0 16px;
  color: var(--color-text);
  border-bottom: 1px solid var(--color-border);
}

.is-collapsed .brand {
  justify-content: center;
  padding: 0;
}

.brand:hover {
  color: var(--color-text);
}

.brand-mark {
  width: 32px;
  height: 32px;
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  background: var(--color-primary);
  color: #fff;
  font-weight: 700;
  font-size: 16px;
}

.brand-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
  line-height: 1.25;
}

.brand-name {
  font-size: 15px;
  font-weight: 650;
  white-space: nowrap;
}

.brand-sub {
  font-size: 11px;
  color: var(--color-text-4);
  white-space: nowrap;
}

.nav {
  flex: 1;
  overflow-y: auto;
  padding: 12px 10px;
}

.nav-group + .nav-group {
  margin-top: 16px;
}

.nav-group-title {
  padding: 0 10px 6px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  color: var(--color-text-4);
}

.nav-group-divider {
  height: 1px;
  margin: 0 8px 8px;
  background: var(--color-border);
}

.nav-group:first-child .nav-group-divider {
  display: none;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 36px;
  padding: 0 10px;
  margin-bottom: 2px;
  border-radius: 8px;
  color: var(--color-text-2);
  font-size: 14px;
  transition: background 0.15s ease, color 0.15s ease;
}

.is-collapsed .nav-item {
  justify-content: center;
  padding: 0;
}

.nav-item:hover {
  background: var(--color-surface-sunken);
  color: var(--color-text);
}

.nav-item.is-active {
  background: var(--color-primary-soft);
  color: var(--color-primary);
  font-weight: 550;
}

.nav-icon {
  font-size: 17px;
  flex-shrink: 0;
}

.nav-label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.sidebar-footer {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px;
  border-top: 1px solid var(--color-border);
}

.sidebar-footer :deep(.el-dropdown) {
  display: block;
  width: 100%;
}

.user-card {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px;
  border: none;
  border-radius: 8px;
  background: transparent;
  color: inherit;
  cursor: pointer;
  text-align: left;
  font: inherit;
}

.is-collapsed .user-card {
  justify-content: center;
}

.user-card:hover {
  background: var(--color-surface-sunken);
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

.user-meta {
  display: flex;
  flex-direction: column;
  min-width: 0;
  line-height: 1.3;
}

.user-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-role {
  font-size: 12px;
  color: var(--color-text-3);
}

.collapse-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 30px;
  border: none;
  border-radius: 8px;
  background: transparent;
  color: var(--color-text-4);
  cursor: pointer;
  font-size: 16px;
}

.collapse-btn:hover {
  background: var(--color-surface-sunken);
  color: var(--color-text-2);
}

.main {
  flex: 1;
  min-width: 0;
  height: 100vh;
  overflow-y: auto;
}
</style>
