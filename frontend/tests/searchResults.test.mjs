import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { pathToFileURL } from 'node:url'

const outfile = join(tmpdir(), `search-results-${Date.now()}.mjs`)

await build({
  entryPoints: ['src/utils/searchResults.ts'],
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile,
  write: true
})

const { resolveSearchFetchSize, selectSearchResults, SEARCH_FETCH_LIMIT } = await import(pathToFileURL(outfile))

// 多拉的条数等于已标记数，保证过滤后仍能凑满；超过后端上限时截到上限。
assert.equal(resolveSearchFetchSize(30, 0), 30)
assert.equal(resolveSearchFetchSize(30, 12), 42)
assert.equal(resolveSearchFetchSize(30, 500), SEARCH_FETCH_LIMIT)
// 用户自己要的条数超过上限时不截断，交给后端校验报错。
assert.equal(resolveSearchFetchSize(150, 10), 150)

const papers = ['a', 'b', 'c', 'd', 'e'].map(id => ({ id, arxivId: id }))
const marked = new Set(['b', 'd'])

// 不过滤时只按用户要的条数截断。
assert.deepEqual(selectSearchResults(papers, marked, 3, false), {
  items: papers.slice(0, 3),
  hiddenCount: 0
})

// 过滤时跳过已标记论文并用后面的结果补满。
const filtered = selectSearchResults(papers, marked, 3, true)
assert.deepEqual(filtered.items.map(p => p.id), ['a', 'c', 'e'])
assert.equal(filtered.hiddenCount, 2)

// 截断位置之后的已标记论文不计入隐藏数。
const early = selectSearchResults(papers, marked, 1, true)
assert.deepEqual(early.items.map(p => p.id), ['a'])
assert.equal(early.hiddenCount, 0)

// 没有 arxivId 时用 id 匹配标记。
assert.equal(selectSearchResults([{ id: 'b' }], marked, 3, true).items.length, 0)

console.log('search results filtering ok')
