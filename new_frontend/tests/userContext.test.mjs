import assert from 'node:assert/strict'
import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { build } from 'esbuild'

const sourceUrl = new URL('../src/composables/useUserContext.ts', import.meta.url)
const sourcePath = fileURLToPath(sourceUrl)
const srcPath = fileURLToPath(new URL('../src/', import.meta.url))

const bundled = await build({
  entryPoints: [sourcePath],
  bundle: true,
  write: false,
  platform: 'node',
  format: 'esm'
})

const moduleUrl = `data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`
const {
  DEFAULT_DEMO_USER_ID,
  getCurrentUserId,
  isCurrentDemoUser,
  resetCurrentUserId,
  setCurrentUserId
} = await import(moduleUrl)

assert.equal(DEFAULT_DEMO_USER_ID, 'local_user')
assert.equal(getCurrentUserId(), 'local_user')
assert.equal(isCurrentDemoUser(), true)

setCurrentUserId('user-for-test')
assert.equal(getCurrentUserId(), 'user-for-test')
assert.equal(isCurrentDemoUser(), false)

setCurrentUserId('   ')
assert.equal(getCurrentUserId(), 'local_user')

setCurrentUserId('another-user')
resetCurrentUserId()
assert.equal(getCurrentUserId(), 'local_user')

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
assert.doesNotMatch(agentSource, /user_id:\s*['"]local_user['"]/)

const papersSource = await readFile(new URL('../src/api/papers.ts', import.meta.url), 'utf8')
assert.match(papersSource, /getCurrentUserId/)
assert.match(papersSource, /resolveUserId/)

console.log('user context tests passed')
