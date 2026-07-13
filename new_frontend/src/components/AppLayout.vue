<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { User } from '@element-plus/icons-vue'
import { useUserContext } from '@/composables/useUserContext'

const router = useRouter()
const route = useRoute()
const userContext = useUserContext()
const collapsed = ref(false)
const debugRoutesEnabled = import.meta.env.VITE_ENABLE_DEBUG_ROUTES === 'true'
const currentUserId = computed(() => userContext.userId.value)

function handleMenuClick(path: string) {
  router.push(path)
}

function handleUserClick() {
  // 保留当前路径作为切换身份后的回跳地址，避免用户在推荐或画像页切换账号后丢失上下文。
  router.push({
    path: '/login',
    query: route.path === '/login' ? {} : { redirect: route.fullPath }
  })
}
</script>

<template>
  <el-container class="app-layout">
    <el-aside :width="collapsed ? '72px' : '220px'" class="app-sidebar">
      <div class="sidebar-glow sidebar-glow-a"></div>
      <div class="sidebar-glow sidebar-glow-b"></div>

      <div class="logo">
        <span class="logo-badge">📚</span>
        <span v-if="!collapsed" class="logo-text">arXiv 推荐</span>
      </div>

      <el-menu
        mode="vertical"
        :default-active="route.path"
        class="app-menu"
        background-color="transparent"
        text-color="#bfc9d4"
        active-text-color="#fff"
        :collapse="collapsed"
      >
        <el-menu-item index="/dashboard" class="menu-item-home" @click="handleMenuClick('/dashboard')">
          <span class="menu-icon-badge badge-home">🏠</span>
          <span>首页</span>
        </el-menu-item>

        <el-menu-item index="/search" class="menu-item-search" @click="handleMenuClick('/search')">
          <span class="menu-icon-badge badge-search">🔎</span>
          <span>论文搜索</span>
        </el-menu-item>

        <el-menu-item index="/agent-search" class="menu-item-agent" @click="handleMenuClick('/agent-search')">
          <span class="menu-icon-badge badge-agent">AI</span>
          <span>Agent 搜索</span>
        </el-menu-item>

        <el-menu-item index="/recommendations" class="menu-item-star" @click="handleMenuClick('/recommendations')">
          <span class="menu-icon-badge badge-star">⭐</span>
          <span>推荐论文</span>
        </el-menu-item>

        <el-menu-item index="/profile" class="menu-item-profile" @click="handleMenuClick('/profile')">
          <span class="menu-icon-badge badge-profile">🧭</span>
          <span>研究画像</span>
        </el-menu-item>

        <el-menu-item index="/labeled" class="menu-item-folder" @click="handleMenuClick('/labeled')">
          <span class="menu-icon-badge badge-folder">📁</span>
          <span>已标记论文</span>
        </el-menu-item>

        <el-menu-item v-if="debugRoutesEnabled" index="/chunks" class="menu-item-doc" @click="handleMenuClick('/chunks')">
          <span class="menu-icon-badge badge-doc">📄</span>
          <span>调试切片</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container :class="['app-main', { 'sidebar-collapsed': collapsed }]">
      <el-header class="app-header">
        <button class="collapse-btn" @click="collapsed = !collapsed">
          <span :size="20">{{ collapsed ? '▶' : '◀' }}</span>
        </button>
        <span class="header-title">arXiv 计算机论文推荐系统</span>
        <button class="user-switch" type="button" @click="handleUserClick">
          <el-icon><User /></el-icon>
          <span class="user-switch__label">当前用户</span>
          <span class="user-switch__id">{{ currentUserId }}</span>
        </button>
      </el-header>

      <el-main class="app-content">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.app-layout {
  height: 100vh;
  background:
    radial-gradient(circle at top left, rgba(88, 133, 255, 0.18), transparent 35%),
    radial-gradient(circle at bottom right, rgba(255, 176, 84, 0.12), transparent 28%),
    #f3f6fb;
}

.app-sidebar {
  position: fixed;
  left: 0;
  top: 0;
  height: 100vh;
  z-index: 100;
  overflow: hidden;
  background:
    linear-gradient(180deg, rgba(4, 12, 26, 0.96) 0%, rgba(10, 17, 35, 0.96) 100%),
    linear-gradient(135deg, #12213a 0%, #1b2b52 55%, #0d1428 100%);
  box-shadow: 16px 0 40px rgba(6, 12, 24, 0.25);
}

.app-sidebar::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    linear-gradient(120deg, rgba(255, 255, 255, 0.06), transparent 35%),
    linear-gradient(200deg, transparent 65%, rgba(255, 255, 255, 0.05));
  pointer-events: none;
}

.sidebar-glow {
  position: absolute;
  border-radius: 999px;
  filter: blur(16px);
  opacity: 0.9;
  pointer-events: none;
  animation: floatGlow 10s ease-in-out infinite;
}

.sidebar-glow-a {
  width: 120px;
  height: 120px;
  left: -40px;
  top: 88px;
  background: rgba(84, 160, 255, 0.22);
}

.sidebar-glow-b {
  width: 140px;
  height: 140px;
  right: -48px;
  bottom: 140px;
  background: rgba(255, 175, 94, 0.16);
  animation-delay: -4s;
}

.logo {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 24px 16px 20px;
  color: #fff;
  font-size: 18px;
  font-weight: 800;
  letter-spacing: 0.3px;
}

.logo-badge {
  width: 42px;
  height: 42px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 14px;
  font-size: 22px;
  background: linear-gradient(135deg, #ffce6a 0%, #ff8d4d 100%);
  box-shadow: 0 10px 20px rgba(255, 146, 68, 0.24);
  transform: rotate(-6deg);
}

.logo-text {
  white-space: nowrap;
  text-shadow: 0 2px 16px rgba(0, 0, 0, 0.22);
}

.app-menu {
  position: relative;
  margin-top: 10px;
  background: transparent !important;
  border-right: none !important;
}

.menu-icon-badge {
  width: 34px;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-right: 12px;
  border-radius: 12px;
  font-size: 18px;
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.14);
  transform: translateY(-0.5px);
  transition: transform 0.25s ease, box-shadow 0.25s ease, filter 0.25s ease;
}

.badge-home {
  background: linear-gradient(135deg, #ffb86b 0%, #ff7a59 100%);
}

.badge-search {
  background: linear-gradient(135deg, #6ad1ff 0%, #3f8cff 100%);
}

.badge-agent {
  background: linear-gradient(135deg, #8b7cff 0%, #4f46e5 100%);
}

.badge-graph {
  background: linear-gradient(135deg, #34d399 0%, #0f766e 100%);
}

.badge-star {
  background: linear-gradient(135deg, #ffd976 0%, #ffb84d 100%);
}

.badge-profile {
  background: linear-gradient(135deg, #7dd3fc 0%, #38bdf8 100%);
}

.badge-folder {
  background: linear-gradient(135deg, #a8ffcb 0%, #4cd3a2 100%);
}

.badge-doc {
  background: linear-gradient(135deg, #c7b5ff 0%, #8b7cff 100%);
}

.el-menu-item {
  position: relative;
  margin: 8px 12px !important;
  border-radius: 14px;
  overflow: hidden;
  color: #dbe4f0 !important;
  transition:
    transform 0.24s ease,
    background 0.24s ease,
    box-shadow 0.24s ease;
}

.el-menu-item::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.12), transparent 65%);
  opacity: 0;
  transition: opacity 0.24s ease;
}

.el-menu-item:hover {
  transform: translateX(4px);
  background: rgba(255, 255, 255, 0.08) !important;
}

.el-menu-item:hover::before {
  opacity: 1;
}

.el-menu-item:hover .menu-icon-badge {
  transform: scale(1.06) rotate(-2deg);
  box-shadow: 0 10px 22px rgba(0, 0, 0, 0.18);
}

.el-menu-item.is-active {
  background: linear-gradient(135deg, rgba(75, 121, 255, 0.28), rgba(30, 194, 255, 0.18)) !important;
  box-shadow: 0 14px 26px rgba(33, 71, 156, 0.22);
  transform: translateX(6px);
}

.el-menu-item.is-active .menu-icon-badge {
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
  filter: saturate(1.08);
}

.el-menu-item.is-active::before {
  opacity: 1;
}

.app-main {
  margin-left: 220px;
  transition: margin-left 0.3s ease;
}

.app-main.sidebar-collapsed {
  margin-left: 72px;
}

.app-header {
  position: relative;
  background: rgba(255, 255, 255, 0.78);
  backdrop-filter: blur(14px);
  padding: 0 20px;
  display: flex;
  align-items: center;
  border-bottom: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

.collapse-btn {
  background: linear-gradient(135deg, #ffffff, #eef3ff);
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 12px;
  cursor: pointer;
  padding: 8px 10px;
  margin-right: 16px;
  color: #475569;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
  transition: transform 0.22s ease, box-shadow 0.22s ease;
}

.collapse-btn:hover {
  color: #2563eb;
  transform: translateY(-1px);
  box-shadow: 0 10px 18px rgba(37, 99, 235, 0.16);
}

.header-title {
  font-size: 16px;
  font-weight: 700;
  color: #1f2937;
}

.user-switch {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  max-width: min(360px, 52vw);
  padding: 8px 12px;
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 8px;
  background: #fff;
  color: #334155;
  cursor: pointer;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
}

.user-switch:hover {
  color: #2563eb;
  border-color: rgba(37, 99, 235, 0.35);
}

.user-switch__label {
  color: #64748b;
  font-size: 13px;
}

.user-switch__id {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-weight: 700;
}

.app-content {
  padding: 20px;
  overflow-y: auto;
  height: calc(100vh - 60px);
}

@keyframes floatGlow {
  0%, 100% {
    transform: translateY(0) scale(1);
  }
  50% {
    transform: translateY(8px) scale(1.05);
  }
}

@media (max-width: 900px) {
  .app-main {
    margin-left: 72px;
  }

  .logo-text {
    display: none;
  }
}
</style>
