import axios from 'axios'
import { ApiError, normalizeApiError } from './errors'
import { getCredential, getSessionRevision, handleAuthFailure, requireUnchangedSession, resolveApiUrl } from './auth'
import type { Credential } from './auth'
import { API_BASE_URL } from './config'

const request = axios.create({
  baseURL: API_BASE_URL,
  // Fetch adapter 可以拒绝重定向；XHR 会自动跟随，无法保证自定义密钥头不被带到第三方。
  adapter: 'fetch',
  fetchOptions: { redirect: 'error', credentials: 'omit' },
  withCredentials: false,
  timeout: 600000
})
const requestSessions = new WeakMap<object, { credential: Credential | null; revision: number }>()

request.interceptors.request.use(config => {
  // 每次请求读取当前凭据，让登录、退出和轮换即时生效；只向受信任的 API 根路径发送。
  config.url = resolveApiUrl(config.url || '')
  config.baseURL = undefined
  const credential = getCredential()
  requestSessions.set(config, { credential, revision: getSessionRevision() })
  config.headers.delete('X-API-Key')
  config.headers.delete('Authorization')
  if (credential) config.headers.set(credential.mode === 'jwt' ? 'Authorization' : 'X-API-Key', credential.mode === 'jwt' ? `Bearer ${credential.value}` : credential.value)
  return config
})

request.interceptors.response.use(
  response => {
    const session = requestSessions.get(response.config)
    if (session) requireUnchangedSession(session.credential, session.revision)
    return response.data
  },
  error => {
    // AxiosError.config 含完整请求头，不能把原始异常对象写入浏览器控制台。
    const payload = normalizeApiError(error, '请求失败，请稍后重试。')
    const bearer = String(error.config?.headers?.get?.('Authorization') || '')
    const key = String(error.config?.headers?.get?.('X-API-Key') || '')
    handleAuthFailure(payload.code, bearer.startsWith('Bearer ') ? { mode: 'jwt', value: bearer.slice(7) } : key ? { mode: 'api_key', value: key } : null)
    // API 层统一把后端错误契约转成 ApiError，页面只需要按 code 展示提示。
    return Promise.reject(new ApiError(payload))
  }
)

export default request
