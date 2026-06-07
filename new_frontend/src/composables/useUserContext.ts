import { computed, ref, watch } from 'vue'

const USER_ID_STORAGE_KEY = 'rag_demo_user_id'

export const DEFAULT_DEMO_USER_ID = 'local_user'

function normalizeUserId(value?: string | null): string {
  const normalized = String(value || '').trim()
  return normalized || DEFAULT_DEMO_USER_ID
}

function readStoredUserId(): string {
  if (typeof window === 'undefined') {
    return DEFAULT_DEMO_USER_ID
  }

  try {
    return normalizeUserId(window.localStorage.getItem(USER_ID_STORAGE_KEY))
  } catch {
    // localStorage 可能在隐私模式或服务端渲染场景不可用，保留 demo 用户兜底让单用户演示不中断。
    return DEFAULT_DEMO_USER_ID
  }
}

const currentUserIdRef = ref(readStoredUserId())

watch(currentUserIdRef, userId => {
  if (typeof window === 'undefined') return

  try {
    // 统一持久化 demo 用户，后续接入 /me 或 token 时只需要替换这里的初始化来源。
    window.localStorage.setItem(USER_ID_STORAGE_KEY, normalizeUserId(userId))
  } catch {
    // 持久化失败不应影响业务请求，内存态 userId 仍然可以继续支撑当前会话。
  }
})

export function getCurrentUserId(): string {
  return normalizeUserId(currentUserIdRef.value)
}

export function setCurrentUserId(userId: string): void {
  currentUserIdRef.value = normalizeUserId(userId)
}

export function resetCurrentUserId(): void {
  currentUserIdRef.value = DEFAULT_DEMO_USER_ID
}

export function isCurrentDemoUser(): boolean {
  return getCurrentUserId() === DEFAULT_DEMO_USER_ID
}

export function useUserContext() {
  const userId = computed({
    get: getCurrentUserId,
    set: setCurrentUserId
  })
  const displayName = computed(() => (isCurrentDemoUser() ? 'Demo User' : `User ${userId.value}`))
  const isDemoUser = computed(isCurrentDemoUser)
  const isDemoMode = computed(() => true)
  const isLoggedIn = computed(() => !isDemoUser.value)

  return {
    defaultUserId: DEFAULT_DEMO_USER_ID,
    userId,
    currentUserId: userId,
    displayName,
    isDemoUser,
    isDemoMode,
    isLoggedIn,
    getUserId: getCurrentUserId,
    setUserId: setCurrentUserId,
    resetUserId: resetCurrentUserId
  }
}
