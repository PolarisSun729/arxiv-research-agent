import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import { AUTH_REQUIRED_EVENT, currentUser, getAuthMode, restoreSession } from '@/api/auth'

const debugRoutesEnabled = import.meta.env.VITE_ENABLE_DEBUG_ROUTES === 'true'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/login'
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('@/views/Dashboard.vue')
  },
  {
    path: '/search',
    name: 'Search',
    component: () => import('@/views/Search.vue')
  },
  {
    path: '/paper/:id',
    name: 'PaperDetail',
    component: () => import('@/views/PaperDetail.vue')
  },
  {
    path: '/recommendations',
    name: 'Recommendations',
    component: () => import('@/views/Recommendations.vue')
  },
  {
    path: '/profile',
    name: 'ResearchProfile',
    component: () => import('@/views/ResearchProfile.vue')
  },
  {
    path: '/agent-search',
    name: 'AgentSearch',
    meta: { research: true },
    component: () => import('@/views/AgentSearch.vue')
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue')
  },
  {
    path: '/account',
    name: 'Account',
    meta: { jwt: true },
    component: () => import('@/views/Account.vue')
  },
  {
    path: '/admin/users',
    name: 'AdminUsers',
    meta: { admin: true, jwt: true },
    component: () => import('@/views/AdminUsers.vue')
  },
  {
    path: '/agent-graph',
    name: 'AgentGraph',
    redirect: '/'
  },
  {
    path: '/labeled',
    name: 'LabeledPapers',
    component: () => import('@/views/LabeledPapers.vue')
  }
]

if (debugRoutesEnabled) {
  // chunk 页面只服务本地排查，默认不进入正式前端路由表。
  routes.push({
    path: '/chunks',
    name: 'Chunks',
    meta: { admin: true },
    component: () => import('@/views/Chunks.vue')
  })
}

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach(async to => {
  if (to.path === '/login') return
  // 刷新页面后先从 /me 恢复身份，业务组件不能用空身份或演示 ID 提前发出请求。
  try {
    if (!await restoreSession()) return { path: '/login', query: { redirect: to.fullPath } }
  } catch {
    return { path: '/login', query: { redirect: to.fullPath } }
  }
  const mode = getAuthMode()
  if (to.meta.jwt && mode !== 'jwt') return '/dashboard'
  if (mode === 'jwt' && ((to.meta.admin && currentUser.value?.role !== 'admin')
      || (to.meta.research && !['admin', 'researcher'].includes(currentUser.value?.role || '')))) return '/dashboard'
})

window.addEventListener(AUTH_REQUIRED_EVENT, () => {
  if (router.currentRoute.value.path !== '/login') {
    void router.replace({ path: '/login', query: { redirect: router.currentRoute.value.fullPath } })
  }
})

export default router
