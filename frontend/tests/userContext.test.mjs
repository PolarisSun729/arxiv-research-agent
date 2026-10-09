import assert from 'node:assert/strict'
import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { build } from 'esbuild'

const srcPath = fileURLToPath(new URL('../src/', import.meta.url))

const bundled = await build({
  stdin: {
    contents: "export * from './src/composables/useUserContext.ts'; export { loginWithPassword, clearCredentials, setApiKey } from './src/api/auth.ts'",
    resolveDir: fileURLToPath(new URL('../', import.meta.url)),
    loader: 'ts'
  },
  bundle: true,
  write: false,
  platform: 'browser',
  format: 'esm'
})

const saved = new Map()
globalThis.window = {
  sessionStorage: { getItem: key => saved.get(key) || null, setItem: (key, value) => saved.set(key, value), removeItem: key => saved.delete(key) },
  localStorage: { getItem: () => 'forged-admin-user', setItem: () => assert.fail('Cannot persist credentials in localStorage') }
}
const moduleUrl = `data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`
const { getCurrentUserId, isCurrentDemoUser, resetCurrentUserId, setCurrentUserId, loginWithPassword, clearCredentials, setApiKey } = await import(moduleUrl)

// 未认证状态不能读取旧演示命名空间，也不能手动构造一个 JWT 用户身份。
assert.equal(getCurrentUserId(), '')
assert.equal(isCurrentDemoUser(), false)
assert.throws(() => setCurrentUserId('forged-admin-user'))
globalThis.fetch = async url => Response.json(String(url).endsWith('/login')
  ? { access_token: 'eyJ.test-signature', token_type: 'bearer', expires_in: 3600 }
  : { user_id: 'server-assigned-id', username: 'reader', role: 'viewer', is_active: true, quotas: [] })
await loginWithPassword('reader', 'TestPassword123')
assert.equal(getCurrentUserId(), 'server-assigned-id')
assert.throws(() => setCurrentUserId('someone-else'))
assert.throws(() => resetCurrentUserId())
clearCredentials()
assert.equal(getCurrentUserId(), '')

// 只有显式 API Key 兼容模式继续允许旧的团队数据选择行为。
setApiKey('compatibility-' + 'a'.repeat(40))
setCurrentUserId('user-for-test')
assert.equal(getCurrentUserId(), 'user-for-test')
resetCurrentUserId()
assert.equal(getCurrentUserId(), 'local_user')
assert.equal(isCurrentDemoUser(), true)
clearCredentials()
assert.equal(getCurrentUserId(), '')

async function collectFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const files = await Promise.all(entries.map(async entry => {
    const fullPath = join(dir, entry.name)
    return entry.isDirectory() ? collectFiles(fullPath) : fullPath
  }))
  return files.flat()
}

const sourceFiles = (await collectFiles(srcPath)).filter(file => /\.(ts|vue)$/.test(file))

for (const file of sourceFiles) {
  const source = await readFile(file, 'utf8')
  const normalizedPath = file.replaceAll('\\', '/')
  if (!normalizedPath.endsWith('/src/composables/useUserContext.ts')) {
    assert.equal(source.includes('local_user'), false, `${file} should not hardcode demo user id`)
    assert.equal(source.includes('DEFAULT_USER_ID'), false, `${file} should not define DEFAULT_USER_ID`)
  }
}

const ragSource = await readFile(new URL('../src/composables/usePaperRagChat.ts', import.meta.url), 'utf8')
assert.match(ragSource, /useUserContext/)
assert.match(ragSource, /watch\(\(\) => getUserId\(\)/)
assert.doesNotMatch(ragSource, /user_id:\s*['"]local_user['"]/)

const agentSource = await readFile(new URL('../src/composables/useAgentSearchChat.ts', import.meta.url), 'utf8')
assert.match(agentSource, /effectiveUserId/)
assert.match(agentSource, /activeSessionUserId/)
assert.match(agentSource, /buildConfirmationPlaceholder/)
assert.match(agentSource, /skipUserMessage:\s*true/)
assert.match(agentSource, /if\s*\(!options\?\.skipUserMessage\)\s*\{/)
assert.doesNotMatch(agentSource, /user_id:\s*['"]local_user['"]/)

const agentChatSource = await readFile(new URL('../src/types/agentChat.ts', import.meta.url), 'utf8')
assert.match(agentChatSource, /'system'/)

const papersSource = await readFile(new URL('../src/api/papers.ts', import.meta.url), 'utf8')
assert.match(papersSource, /getCurrentUserId/)
assert.match(papersSource, /resolveUserId/)

console.log('user context tests passed')
