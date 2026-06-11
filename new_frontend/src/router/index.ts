import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const debugRoutesEnabled = import.meta.env.VITE_ENABLE_DEBUG_ROUTES === 'true'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
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
    component: () => import('@/views/AgentSearch.vue')
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
    component: () => import('@/views/Chunks.vue')
  })
}

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
