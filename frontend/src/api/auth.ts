import { API_BASE_URL } from './config'
import { parseFetchErrorResponse } from './errors'
import { shallowReadonly, shallowRef } from 'vue'

const STORAGE_KEY = 'arxiv_session_credential'
export const AUTH_REQUIRED_EVENT = 'backend-auth-required'
export type AuthMode = 'jwt' | 'api_key'
export type UserRole = 'admin' | 'researcher' | 'viewer' | 'guest'
export type QuotaType = 'papers' | 'qa_queries' | 'agent_runs'
export interface Credential { mode: AuthMode; value: string }
export interface UserQuota { quota_type: QuotaType; daily_limit: number; used_today: number; remaining: number; reset_at: string }
export interface AuthUser {
  user_id: string; username: string; email: string; role: UserRole; is_active: boolean
  created_at: string; last_login: string | null; quotas: UserQuota[]
}
export interface AuthConfiguration { mode: AuthMode; registration_enabled: boolean }
const credentialRef = shallowRef<Credential | null | undefined>(undefined)
const userRef = shallowRef<AuthUser | null>(null)
const configurationRef = shallowRef<AuthConfiguration | null>(null)
export const currentUser = shallowReadonly(userRef)
export const authConfiguration = shallowReadonly(configurationRef)
export const ROLE_LABELS: Record<UserRole, string> = { admin: '管理员', researcher: '研究者', viewer: '阅读者', guest: '访客' }
export const QUOTA_LABELS: Record<QuotaType, string> = { papers: '论文操作', qa_queries: '论文问答', agent_runs: 'Agent 运行' }
let sessionRevision = 0

export function getCredential(): Credential | null {
  if (credentialRef.value !== undefined) return credentialRef.value
  try {
    const stored = typeof window === 'undefined' ? null : JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) || 'null')
    credentialRef.value = stored && ['jwt', 'api_key'].includes(stored.mode) && typeof stored.value === 'string' && stored.value
      ? { mode: stored.mode, value: stored.value } : null
  } catch {
    // 隐私模式禁用存储时仍可使用当前标签页的内存凭据。
    credentialRef.value = null
  }
  return credentialRef.value || null
}

function setCredential(credential: Credential | null): void {
  sessionRevision += 1
  credentialRef.value = credential
  userRef.value = null
  try {
    if (typeof window === 'undefined') return
    if (credential) window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(credential))
    else window.sessionStorage.removeItem(STORAGE_KEY)
    window.sessionStorage.removeItem('arxiv_backend_api_key')
  } catch {
    // 不降级到长期 localStorage，避免浏览器关闭后继续保留访问凭据。
  }
}

export function getAuthMode(): AuthMode {
  return configurationRef.value?.mode || getCredential()?.mode || 'jwt'
}

export function getApiKey(): string { return getCredential()?.mode === 'api_key' ? getCredential()!.value : '' }
export function setApiKey(key: string): void { setCredential(key.trim() ? { mode: 'api_key', value: key.trim() } : null) }
export function clearApiKey(): void { clearCredentials() }
export function clearCredentials(): void { setCredential(null) }

function sameCredential(left: Credential | null, right: Credential | null): boolean {
  return left?.mode === right?.mode && left?.value === right?.value
}

export function getSessionRevision(): number { return sessionRevision }

export function requireUnchangedSession(credential: Credential | null, revision: number): void {
  if (revision !== sessionRevision || !sameCredential(credential, getCredential())) {
    // Pinia 等共享缓存可能晚于页面卸载收到结果，丢弃旧账号响应以免重新污染新账号视图。
    throw new DOMException('账号已切换，已丢弃旧请求结果', 'AbortError')
  }
}

export function handleAuthFailure(code: string, sent: Credential | null): void {
  if (!['missing_api_key', 'invalid_api_key', 'api_key_disabled', 'api_key_expired', 'missing_token', 'invalid_token'].includes(code)
      || !sent || !sameCredential(sent, getCredential())) return
  // 只撤销请求发出时的凭据；旧 SSE/下载的迟到 401 不能清掉刚登录的新账号。
  clearCredentials()
  if (typeof window !== 'undefined') window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT))
}

export function resolveApiUrl(endpoint: string): string {
  const pageOrigin = typeof window !== 'undefined' && window.location?.origin ? window.location.origin : 'http://localhost'
  const base = new URL(API_BASE_URL, pageOrigin)
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash) {
    throw new Error('后端 API 地址配置不合法')
  }
  const basePath = base.pathname.replace(/\/+$/, '')
  let target: URL
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(endpoint)) {
    target = new URL(endpoint, base)
  } else {
    // 后端证据 URL 带 /api 前缀，业务客户端用相对路径，两者统一映射到配置的 API 根地址。
    const suffix = endpoint === '/api' ? '' : endpoint.replace(/^\/api\//, '').replace(/^\/+/, '')
    target = new URL(`${basePath}/${suffix}`, base)
  }
  if (target.origin !== base.origin || !target.pathname.startsWith(`${basePath}/`) || target.username || target.password || target.hash) {
    throw new Error('不能向后端 API 以外的地址发送登录凭据')
  }
  return API_BASE_URL.startsWith('/') && target.origin === pageOrigin ? `${target.pathname}${target.search}` : target.href
}

export async function apiFetch(endpoint: string, options: RequestInit = {}, credential = getCredential()): Promise<Response> {
  const revision = sessionRevision
  const belongsToCurrentSession = sameCredential(credential, getCredential())
  const headers = new Headers(options.headers)
  headers.delete('X-API-Key')
  headers.delete('Authorization')
  if (credential) headers.set(credential.mode === 'jwt' ? 'Authorization' : 'X-API-Key', credential.mode === 'jwt' ? `Bearer ${credential.value}` : credential.value)
  // 普通请求、SSE 与 Blob 共用 Bearer；拒绝重定向和 Cookie，凭据不能转发到第三方。
  const response = await fetch(resolveApiUrl(endpoint), { ...options, headers, credentials: 'omit', redirect: 'error' })
  if (response.status === 401 || response.status === 403) {
    const payload = await response.clone().json().catch(() => null)
    handleAuthFailure(String(payload?.code || ''), credential)
  }
  if (response.ok && belongsToCurrentSession) requireUnchangedSession(credential, revision)
  return response
}

export async function verifyAccessKey(key: string): Promise<void> {
  if (!/^[\x21-\x7e]{32,256}$/.test(key)) {
    throw new Error('请输入管理员提供的完整访问密钥（32 至 256 个字符）')
  }
  // 先校验再保存，错误输入不应替换当前仍可用的凭据。
  const response = await apiFetch('/auth/check', {}, { mode: 'api_key', value: key })
  if (!response.ok) throw await parseFetchErrorResponse(response, '访问密钥验证失败')
  const payload = await response.json()
  if (payload?.status !== 'authenticated') throw new Error('后端未返回有效的认证结果')
}

async function authJson(endpoint: string, options: RequestInit = {}, credential = getCredential()): Promise<any> {
  const response = await apiFetch(endpoint, options, credential)
  if (!response.ok) throw await parseFetchErrorResponse(response, '账号操作失败')
  return response.json()
}

function jsonBody(value: unknown, method = 'POST'): RequestInit {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) }
}

function validateUser(value: any): AuthUser {
  if (!value?.user_id || !value?.username || !Object.prototype.hasOwnProperty.call(ROLE_LABELS, value.role) || value.is_active !== true || !Array.isArray(value.quotas)) {
    throw new Error('后端未返回有效的账号信息')
  }
  return value as AuthUser
}

export async function loadAuthConfiguration(): Promise<AuthConfiguration> {
  const value = await authJson('/auth/config', {}, null)
  if (!['jwt', 'api_key'].includes(value?.mode) || typeof value?.registration_enabled !== 'boolean') throw new Error('认证配置无效')
  configurationRef.value = value
  return value
}

export async function loginWithPassword(username: string, password: string): Promise<AuthUser> {
  const revision = ++sessionRevision
  const result = await authJson('/auth/login', jsonBody({ username, password }), null)
  if (result?.token_type !== 'bearer' || typeof result.access_token !== 'string' || !result.access_token.startsWith('eyJ')) throw new Error('登录响应无效')
  const candidate: Credential = { mode: 'jwt', value: result.access_token }
  const user = validateUser(await authJson('/auth/me', {}, candidate))
  if (revision !== sessionRevision) throw new Error('登录状态已改变，请重试')
  // /me 验证成功后一起提交身份与凭据，错误登录不能覆盖一个仍然有效的账号。
  setCredential(candidate)
  userRef.value = user
  return user
}

export async function registerAccount(username: string, email: string, password: string): Promise<void> {
  await authJson('/auth/register', jsonBody({ username, email, password }), null)
}

export async function refreshCurrentUser(): Promise<AuthUser | null> {
  const credential = getCredential()
  if (!credential || credential.mode !== 'jwt') return null
  const revision = sessionRevision
  const user = validateUser(await authJson('/auth/me', {}, credential))
  if (revision === sessionRevision && sameCredential(credential, getCredential())) userRef.value = user
  return userRef.value
}

export async function restoreSession(): Promise<boolean> {
  const credential = getCredential()
  if (!credential) return false
  const config = configurationRef.value || await loadAuthConfiguration()
  if (config.mode !== credential.mode) {
    clearCredentials()
    return false
  }
  if (credential.mode === 'jwt') return Boolean(userRef.value || await refreshCurrentUser())
  await verifyAccessKey(credential.value)
  return true
}

export async function logout(): Promise<void> {
  const credential = getCredential()
  if (credential?.mode === 'jwt') {
    // 服务端撤销失败时保留账号状态供用户重试，不能声称已经使丢失的令牌失效。
    const response = await apiFetch('/auth/logout', { method: 'POST' }, credential)
    if (!response.ok && response.status !== 401) throw await parseFetchErrorResponse(response, '退出失败，请重试')
  }
  if (sameCredential(credential, getCredential())) clearCredentials()
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  const credential = getCredential()
  await authJson('/auth/password', jsonBody({ current_password: currentPassword, new_password: newPassword }), credential)
  if (sameCredential(credential, getCredential())) clearCredentials()
}

export async function listUsers(offset = 0): Promise<{ items: AuthUser[]; total: number }> {
  return authJson(`/auth/users?offset=${offset}&limit=50`)
}
export async function createUser(value: { username: string; email: string; password: string; role: UserRole }): Promise<AuthUser> {
  return authJson('/auth/users', jsonBody(value))
}
export async function getUser(userId: string): Promise<AuthUser> { return authJson(`/auth/users/${encodeURIComponent(userId)}`) }
export async function updateUser(userId: string, value: { role?: UserRole; is_active?: boolean }): Promise<AuthUser> {
  return authJson(`/auth/users/${encodeURIComponent(userId)}`, jsonBody(value, 'PATCH'))
}
export async function updateQuota(userId: string, quotaType: QuotaType, dailyLimit: number): Promise<{ quotas: UserQuota[] }> {
  return authJson(`/auth/users/${encodeURIComponent(userId)}/quotas`, jsonBody({ quota_type: quotaType, daily_limit: dailyLimit }, 'PUT'))
}

export async function fetchApiBlob(endpoint: string, signal?: AbortSignal): Promise<Blob> {
  const credential = getCredential()
  const revision = sessionRevision
  const response = await apiFetch(endpoint, { signal })
  if (!response.ok) throw await parseFetchErrorResponse(response, '文件加载失败')
  const blob = await response.blob()
  requireUnchangedSession(credential, revision)
  return blob
}

export async function downloadApiFile(endpoint: string, filename: string): Promise<void> {
  const blob = await fetchApiBlob(endpoint)
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename.replace(/[\\/:\x00-\x1f]/g, '_')
  anchor.click()
  // 下载只使用临时 Blob URL，密钥不进入地址栏、浏览器历史或 Referer。
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
