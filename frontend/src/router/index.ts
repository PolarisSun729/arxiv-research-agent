import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import { ElMessage } from 'element-plus'
import { AUTH_REQUIRED_EVENT, currentUser, getAuthMode, restoreSession } from '@/api/auth'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/login'
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    meta: { title: '首页' },
    component: () => import('@/views/Dashboard.vue')
  },
  {
    path: '/search',
    name: 'Search',
    meta: { title: '论文搜索' },
    component: () => import('@/views/Search.vue')
  },
  {
    path: '/paper/:id',
    name: 'PaperDetail',
    meta: { title: '论文详情' },
    component: () => import('@/views/PaperDetail.vue')
  },
  {
    path: '/recommendations',
    name: 'Recommendations',
    meta: { title: '推荐论文' },
    component: () => import('@/views/Recommendations.vue')
  },
  {
    path: '/profile',
    name: 'ResearchProfile',
    meta: { title: '研究画像' },
    component: () => import('@/views/ResearchProfile.vue')
  },
  {
    path: '/agent-search',
    name: 'AgentSearch',
    meta: { title: 'Agent 搜索', research: true },
    component: () => import('@/views/AgentSearch.vue')
  },
  {
    path: '/login',
    name: 'Login',
    meta: { title: '登录', bare: true },
    component: () => import('@/views/Login.vue')
  },
  {
    path: '/account',
    name: 'Account',
    meta: { title: '账号设置', jwt: true },
    component: () => import('@/views/Account.vue')
  },
  {
    path: '/admin/users',
    name: 'AdminUsers',
    meta: { title: '账号管理', admin: true, jwt: true },
    component: () => import('@/views/AdminUsers.vue')
  },
  {
    path: '/admin/indexes',
    name: 'AdminIndexes',
    meta: { title: '索引切分', admin: true, jwt: true },
    component: () => import('@/views/AdminIndexes.vue')
  },
  {
    path: '/admin/retrieval-debug',
    name: 'AdminRetrievalDebug',
    meta: { title: '检索调试', admin: true, jwt: true },
    component: () => import('@/views/AdminRetrievalDebug.vue')
  },
  {
    path: '/labeled',
    name: 'LabeledPapers',
    meta: { title: '已标记论文' },
    component: () => import('@/views/LabeledPapers.vue')
  }
]

// 兜底路由必须最后注册；它同样经过登录守卫，未登录访问未知地址会先去登录页。
routes.push({
  path: '/:pathMatch(.*)*',
  name: 'NotFound',
  meta: { title: '页面不存在' },
  component: () => import('@/views/NotFound.vue')
})

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.afterEach(to => {
  const title = typeof to.meta.title === 'string' ? to.meta.title : ''
  document.title = title ? `${title} · arXiv 研究助手` : 'arXiv 研究助手'
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
  const role = currentUser.value?.role || ''
  let deniedReason = ''
  if (to.meta.jwt && mode !== 'jwt') deniedReason = '该页面仅在账号登录模式下可用'
  else if (mode === 'jwt' && to.meta.admin && role !== 'admin') deniedReason = '该页面需要管理员权限'
  else if (mode === 'jwt' && to.meta.research && !['admin', 'researcher'].includes(role)) deniedReason = '该页面需要研究员及以上权限'
  if (deniedReason) {
    // 只是界面提示；真正的权限校验在后端 ROUTE_POLICIES。
    ElMessage.warning(deniedReason)
    return '/dashboard'
  }
})

window.addEventListener(AUTH_REQUIRED_EVENT, () => {
  if (router.currentRoute.value.path !== '/login') {
    void router.replace({ path: '/login', query: { redirect: router.currentRoute.value.fullPath } })
  }
})

export default router
