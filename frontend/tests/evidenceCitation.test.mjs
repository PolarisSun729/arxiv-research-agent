import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { pathToFileURL } from 'node:url'

const outfile = join(tmpdir(), `evidence-citation-${Date.now()}.mjs`)

await build({
  entryPoints: ['src/utils/evidence.ts'],
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile,
  write: true
})

const { buildEvidenceAssetUrl, extractCitedSourceIds, normalizeEvidenceSources, orderCitedSourceIds, toCitationLinks } = await import(pathToFileURL(outfile))
const sources = normalizeEvidenceSources([
  { source_id: 'text-1', content: 'text' },
  { source_id: 'figure-2', chunk_type: 'figure', asset_summary: 'figure' },
  { chunk_id: 'legacy-chunk', content: 'must not become evidence' }
], '2401.00001')

assert.equal(buildEvidenceAssetUrl('2401.00001', 'figure-2'), '/api/paper/2401.00001/evidence-assets/figure-2')
assert.equal(sources[1].asset_url, '/api/paper/2401.00001/evidence-assets/figure-2')
assert.equal(sources.length, 2)
assert.deepEqual(extractCitedSourceIds('结论 [source:figure-2] [Source 1] [source:missing]', sources), ['figure-2'])

// 正文角标与引用列表共用一套编号：先按正文出现顺序，再补后端声明但正文没出现的引用。
const numbered = normalizeEvidenceSources([
  { source_id: '11', content: '认知图 "G"\n包含五种节点', page_number: '3', section_title: 'Method' },
  { source_id: '54', content: '记忆化', page_number: 'N/A' },
  { source_id: '13', content: '推理' }
])
assert.deepEqual(orderCitedSourceIds('a [source:54][source:11] b [source:54]', numbered, ['13', '11', 'gone']), ['54', '11', '13'])

const linked = toCitationLinks('图 [source:11]。推理 [source:54][source:13][source:missing]。', { sources: numbered, citedSourceIds: ['13'] })
assert.equal(
  linked,
  '图 [\\[1\\]](#evidence-source-11 "第 3 页 · Method · 认知图 \\"G\\" 包含五种节点")。' +
  '推理 [\\[2\\]](#evidence-source-54 "记忆化")[\\[3\\]](#evidence-source-13 "推理")。'
)
// 流式阶段还没有证据列表：按出现顺序编号，不丢标记。
assert.equal(
  toCitationLinks('x [source:a] y [source:b] z [source:a]'),
  'x [\\[1\\]](#evidence-source-a "证据 a") y [\\[2\\]](#evidence-source-b "证据 b") z [\\[1\\]](#evidence-source-a "证据 a")'
)

console.log('evidence citation tests passed')
