import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const outfile = join(tmpdir(), `frontend-security-${Date.now()}.mjs`)
await build({
  stdin: {
    contents: `
      export * from './src/api/auth.ts'
      export { default as request } from './src/api/request.ts'
      export { streamAgentChat, streamAgentWorkContinuationResume } from './src/api/agent.ts'
      export { qaPaperStream } from './src/api/papers.ts'
    `,
    resolveDir: fileURLToPath(new URL('..', import.meta.url)),
    loader: 'ts'
  },
  bundle: true,
  format: 'esm',
  platform: 'browser',
  define: { 'import.meta.env.VITE_API_BASE_URL': '"https://backend.example/api"' },
  outfile,
  plugins: [{
    name: 'security-user-context',
    setup(builder) {
      builder.onResolve({ filter: /^@\/composables\/useUserContext$/ }, () => ({ path: 'context', namespace: 'security' }))
      builder.onLoad({ filter: /.*/, namespace: 'security' }, () => ({ contents: 'export function getCurrentUserId() { return "security-user"; }', loader: 'js' }))
    }
  }]
})

const saved = new Map()
const browser = new EventTarget()
browser.location = { origin: 'https://frontend.example' }
browser.sessionStorage = {
  getItem: key => saved.get(key) || null,
  setItem: (key, value) => saved.set(key, value),
  removeItem: key => saved.delete(key)
}
browser.localStorage = { setItem: () => { throw new Error('凭据不能写入 localStorage') } }
globalThis.window = browser

const api = await import(pathToFileURL(outfile))
const firstKey = 'frontend-test-' + 'a'.repeat(40)
const nextKey = 'frontend-test-' + 'b'.repeat(40)
assert.equal(api.getApiKey(), '')
api.setApiKey(firstKey)
assert.equal(api.getApiKey(), firstKey)
assert.equal(saved.size, 1)

const calls = []
globalThis.fetch = async (url, options) => {
  calls.push({ url, options })
  return Response.json({ status: 'authenticated' })
}
await api.apiFetch('/agent/chat/stream', { method: 'POST', headers: { Accept: 'text/event-stream' } })
assert.equal(calls.at(-1).url, 'https://backend.example/api/agent/chat/stream')
assert.equal(calls.at(-1).options.headers.get('X-API-Key'), firstKey)
assert.equal(calls.at(-1).options.headers.get('Accept'), 'text/event-stream')
assert.equal(calls.at(-1).options.redirect, 'error')
assert.equal(calls.at(-1).options.credentials, 'omit')
assert.equal(calls.at(-1).url.includes(firstKey), false)

// 证据地址来自后端数据，必须拒绝跨域、路径逃逸及非 HTTP URL，而不能把密钥附给任意链接。
for (const url of ['https://evil.example/api/file', '//evil.example/api/file', 'data:text/plain,test', '../private', '/api/../../private', 'https://backend.example/private']) {
  await assert.rejects(api.apiFetch(url))
}
assert.equal(api.resolveApiUrl('/api/paper/123/evidence-assets/figure-1'), 'https://backend.example/api/paper/123/evidence-assets/figure-1')

await api.verifyAccessKey(nextKey)
assert.equal(calls.at(-1).options.headers.get('X-API-Key'), nextKey)
assert.equal(api.getApiKey(), firstKey)
await assert.rejects(api.verifyAccessKey('short'))

api.request.defaults.adapter = async config => {
  calls.push({ url: config.url, options: config })
  return { data: { ok: true }, status: 200, statusText: 'OK', headers: {}, config }
}
assert.deepEqual(await api.request.get('/stats'), { ok: true })
assert.equal(calls.at(-1).url, 'https://backend.example/api/stats')
assert.equal(calls.at(-1).options.headers.get('X-API-Key'), firstKey)

let authEvents = 0
browser.addEventListener(api.AUTH_REQUIRED_EVENT, () => { authEvents += 1 })
globalThis.fetch = async () => Response.json({ status: 'failed', code: 'invalid_api_key' }, { status: 403 })
await api.apiFetch('/auth/check')
assert.equal(api.getApiKey(), '')
assert.equal(saved.size, 0)
assert.equal(authEvents, 1)

// 旧请求返回 403 时，新登录已更换密钥，迟到错误不能清掉新凭据。
api.setApiKey(firstKey)
let finishOldRequest
globalThis.fetch = () => new Promise(resolve => { finishOldRequest = resolve })
const oldRequest = api.apiFetch('/auth/check')
api.setApiKey(nextKey)
finishOldRequest(Response.json({ status: 'failed', code: 'invalid_api_key' }, { status: 403 }))
await oldRequest
assert.equal(api.getApiKey(), nextKey)

const rawErrors = []
const originalConsoleError = console.error
console.error = (...args) => { rawErrors.push(args) }
api.request.defaults.adapter = async config => {
  const error = new Error('denied')
  error.config = config
  error.response = { status: 403, data: { status: 'failed', code: 'invalid_api_key' } }
  throw error
}
try {
  await assert.rejects(api.request.get('/stats'), error => {
    assert.equal(error.payload.code, 'invalid_api_key')
    assert.equal(JSON.stringify(error).includes(nextKey), false)
    return true
  })
} finally {
  console.error = originalConsoleError
}
assert.equal(rawErrors.length, 0)
assert.equal(api.getApiKey(), '')

api.setApiKey(firstKey)
function streamResponse(text) {
  return new Response(text, { headers: { 'Content-Type': 'text/event-stream' } })
}
globalThis.fetch = async (url, options) => {
  calls.push({ url, options })
  if (url.includes('/resume/stream')) return streamResponse('event: final_response\ndata: {"response":{"status":"success"}}\n\nevent: stream_end\ndata: {}\n\n')
  if (url.includes('/agent/chat/stream')) return streamResponse('event: final_response\ndata: {"event_type":"final_response","data":{"response":{"status":"success"}}}\n\nevent: stream_end\ndata: {"event_type":"stream_end","data":{}}\n\n')
  return streamResponse('event: done\ndata: {"status":"success","answer":"done","sources":[]}\n\n')
}
await api.streamAgentChat({ message: 'hello' })
await api.streamAgentWorkContinuationResume('continuation-1', 'session-1')
await api.qaPaperStream('2401.00001', 'hello')
for (const call of calls.slice(-3)) {
  assert.equal(call.options.headers.get('X-API-Key'), firstKey)
  assert.equal(call.options.headers.get('Accept'), 'text/event-stream')
  assert.equal(call.url.startsWith('https://backend.example/api/'), true)
}

globalThis.fetch = async (url, options) => {
  calls.push({ url, options })
  return new Response('file bytes', { headers: { 'Content-Type': 'image/png' } })
}
const blob = await api.fetchApiBlob('/api/paper/2401.00001/evidence-assets/figure-1')
assert.equal(blob.type, 'image/png')
assert.equal(await blob.text(), 'file bytes')
assert.equal(calls.at(-1).options.headers.get('X-API-Key'), firstKey)
api.clearApiKey()
assert.equal(saved.size, 0)

// 密钥失效撤销当前登录；限流、额度和 IP 策略拒绝不能误清除仍有效的密钥。
for (const code of ['api_key_expired', 'api_key_disabled']) {
  api.setApiKey(firstKey)
  globalThis.fetch = async () => Response.json({ status: 'failed', code }, { status: 403 })
  await api.apiFetch('/auth/check')
  assert.equal(api.getApiKey(), '')
}
for (const [code, status] of [['rate_limit_exceeded', 429], ['daily_quota_exceeded', 429], ['ip_blocked', 403], ['ip_not_allowed', 403], ['ip_temporarily_blocked', 403], ['security_storage_unavailable', 503]]) {
  api.setApiKey(firstKey)
  globalThis.fetch = async () => Response.json({ status: 'failed', code, retry_after: 60 }, { status })
  await api.apiFetch('/auth/check')
  assert.equal(api.getApiKey(), firstKey)
}
api.clearApiKey()

// JWT 登录身份来自 /me，所有传输共用 Bearer，401 之外的权限/配额拒绝不应退出登录。
const jwtFirst = 'eyJ.test-user-one.signature-one'
const jwtNext = 'eyJ.test-user-two.signature-two'
const user = { user_id: 'user-one', username: 'reader', email: 'reader@example.com', role: 'researcher', is_active: true, quotas: [] }
function loginFetch(token, profile = user) {
  return async url => Response.json(String(url).endsWith('/login') ? { access_token: token, token_type: 'bearer', expires_in: 3600 } : profile)
}
globalThis.fetch = loginFetch(jwtFirst)
await api.loginWithPassword('reader', 'TestPassword123')
assert.equal(api.getApiKey(), '')
assert.equal(api.currentUser.value.user_id, 'user-one')
assert.equal(api.getCredential().value, jwtFirst)
globalThis.fetch = async (url, options) => {
  calls.push({ url, options })
  return Response.json({ status: 'authenticated' })
}
await api.apiFetch('/auth/check', { headers: { 'X-API-Key': firstKey, Authorization: 'Bearer other-user' } })
assert.equal(calls.at(-1).options.headers.get('Authorization'), `Bearer ${jwtFirst}`)
assert.equal(calls.at(-1).options.headers.get('X-API-Key'), null)
api.request.defaults.adapter = async config => {
  calls.push({ url: config.url, options: config })
  return { data: {}, status: 200, statusText: 'OK', headers: {}, config }
}
await api.request.get('/stats')
assert.equal(calls.at(-1).options.headers.get('Authorization'), `Bearer ${jwtFirst}`)
assert.equal(calls.at(-1).options.headers.get('X-API-Key'), undefined)
assert.equal(calls.at(-1).options.fetchOptions.redirect, 'error')

globalThis.fetch = async (url, options) => {
  calls.push({ url, options })
  if (url.includes('/resume/stream')) return streamResponse('event: final_response\ndata: {"response":{"status":"success"}}\n\nevent: stream_end\ndata: {}\n\n')
  if (url.includes('/agent/chat/stream')) return streamResponse('event: final_response\ndata: {"event_type":"final_response","data":{"response":{"status":"success"}}}\n\nevent: stream_end\ndata: {"event_type":"stream_end","data":{}}\n\n')
  if (url.includes('/qa/stream')) return streamResponse('event: done\ndata: {"status":"success","answer":"done","sources":[]}\n\n')
  return new Response('private file', { headers: { 'Content-Type': 'text/markdown' } })
}
await api.streamAgentChat({ message: 'hello' })
await api.streamAgentWorkContinuationResume('continuation-1', 'session-1')
await api.qaPaperStream('2401.00001', 'hello')
await api.fetchApiBlob('/paper/2401.00001/notes/export')
for (const call of calls.slice(-4)) {
  assert.equal(call.options.headers.get('Authorization'), `Bearer ${jwtFirst}`)
  assert.equal(call.options.headers.has('X-API-Key'), false)
  assert.equal(call.url.includes(jwtFirst), false)
}
for (const [code, status] of [['quota_exceeded', 429], ['insufficient_permissions', 403], ['identity_mismatch', 403]]) {
  globalThis.fetch = async () => Response.json({ status: 'failed', code }, { status })
  await api.apiFetch('/auth/check')
  assert.equal(api.getCredential().value, jwtFirst)
}

let finishOldJwt
globalThis.fetch = () => new Promise(resolve => { finishOldJwt = resolve })
const stale = api.apiFetch('/auth/me')
globalThis.fetch = loginFetch(jwtNext, { ...user, user_id: 'user-two' })
await api.loginWithPassword('other', 'TestPassword123')
finishOldJwt(Response.json({ status: 'failed', code: 'invalid_token' }, { status: 401 }))
await stale
assert.equal(api.getCredential().value, jwtNext)
assert.equal(api.currentUser.value.user_id, 'user-two')

// 旧账号成功响应也必须丢弃，不能在登录切换后写回共享页面缓存。
let finishOldSuccess
globalThis.fetch = () => new Promise(resolve => { finishOldSuccess = resolve })
const staleSuccess = api.apiFetch('/paper/p1/notes')
globalThis.fetch = loginFetch(jwtFirst)
await api.loginWithPassword('reader', 'TestPassword123')
finishOldSuccess(Response.json({ private_data: 'old-user-notes' }))
await assert.rejects(staleSuccess, error => error.name === 'AbortError')
globalThis.fetch = loginFetch(jwtNext, { ...user, user_id: 'user-two' })
await api.loginWithPassword('other', 'TestPassword123')

// 撤销失败必须允许重试，成功后清理凭据与用户信息。
globalThis.fetch = async () => Response.json({ status: 'failed', code: 'security_storage_unavailable' }, { status: 503 })
await assert.rejects(api.logout())
assert.equal(api.getCredential().value, jwtNext)
globalThis.fetch = async () => Response.json({ status: 'logged_out' })
await api.logout()
assert.equal(api.getCredential(), null)
assert.equal(api.currentUser.value, null)

// 测试密钥只在运行时注入，打包后的 API 客户端不包含它们。
const bundled = await readFile(outfile, 'utf8')
assert.equal(bundled.includes(firstKey), false)
assert.equal(bundled.includes(nextKey), false)
console.log('Frontend security tests passed')
