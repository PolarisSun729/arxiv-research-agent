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

const { buildEvidenceAssetUrl, extractCitedSourceIds, normalizeEvidenceSources } = await import(pathToFileURL(outfile))
const sources = normalizeEvidenceSources([
  { source_id: 'text-1', content: 'text' },
  { source_id: 'figure-2', chunk_type: 'figure', asset_summary: 'figure' },
  { chunk_id: 'legacy-chunk', content: 'must not become evidence' }
], '2401.00001')

assert.equal(buildEvidenceAssetUrl('2401.00001', 'figure-2'), '/api/paper/2401.00001/evidence-assets/figure-2')
assert.equal(sources[1].asset_url, '/api/paper/2401.00001/evidence-assets/figure-2')
assert.equal(sources.length, 2)
assert.deepEqual(extractCitedSourceIds('结论 [source:figure-2] [Source 1] [source:missing]', sources), ['figure-2'])

console.log('evidence citation tests passed')
