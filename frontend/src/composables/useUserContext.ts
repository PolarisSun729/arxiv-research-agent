import { computed, ref } from 'vue'
import { currentUser, getAuthMode, getCredential } from '../api/auth'

const COMPAT_USER_KEY = 'arxiv_compat_user_id'
const DEFAULT_COMPAT_USER_ID = 'local_user'

function readCompatUserId(): string {
  try {
    return typeof window === 'undefined' ? DEFAULT_COMPAT_USER_ID : window.sessionStorage.getItem(COMPAT_USER_KEY) || DEFAULT_COMPAT_USER_ID
  } catch {
    return DEFAULT_COMPAT_USER_ID
  }
}

const compatUserId = ref(readCompatUserId())

export function getCurrentUserId(): string {
  // JWT 身份只来自服务端 /me；旧 localStorage 演示用户绝不能自动认领真实账号的数据。
  return getAuthMode() === 'api_key' ? compatUserId.value : currentUser.value?.user_id || ''
}

export function setCurrentUserId(userId: string): void {
  if (getAuthMode() !== 'api_key') throw new Error('登录账号的身份不能手动修改')
  compatUserId.value = userId.trim() || DEFAULT_COMPAT_USER_ID
  try {
    if (typeof window !== 'undefined') window.sessionStorage.setItem(COMPAT_USER_KEY, compatUserId.value)
  } catch {
    // 显式兼容模式也只保留当前标签页内的命名空间，不写入持久 localStorage。
  }
}

export function resetCurrentUserId(): void { setCurrentUserId(DEFAULT_COMPAT_USER_ID) }
export function isCurrentDemoUser(): boolean { return getAuthMode() === 'api_key' && compatUserId.value === DEFAULT_COMPAT_USER_ID }

export function useUserContext() {
  const userId = computed(getCurrentUserId)
  const isDemoMode = computed(() => getAuthMode() === 'api_key')
  const isAdmin = computed(() => isDemoMode.value || currentUser.value?.role === 'admin')
  const canResearch = computed(() => isAdmin.value || currentUser.value?.role === 'researcher')
  return {
    get defaultUserId() { return getAuthMode() === 'api_key' ? DEFAULT_COMPAT_USER_ID : '' },
    userId, currentUserId: userId, user: currentUser,
    displayName: computed(() => currentUser.value?.username || (isDemoMode.value ? userId.value : '未登录')),
    isDemoMode, isDemoUser: computed(isCurrentDemoUser), isAdmin, canResearch,
    isLoggedIn: computed(() => Boolean(getCredential() && (isDemoMode.value || currentUser.value))),
    getUserId: getCurrentUserId, setUserId: setCurrentUserId, resetUserId: resetCurrentUserId
  }
}
