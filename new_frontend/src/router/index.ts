import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
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
      path: '/labeled',
      name: 'LabeledPapers',
      component: () => import('@/views/LabeledPapers.vue')
    },
    {
      path: '/chunks',
      name: 'Chunks',
      component: () => import('@/views/Chunks.vue')
    }
  ]
})

export default router