import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { pathToFileURL } from 'node:url'

const outfile = join(tmpdir(), `retrieval-debug-${Date.now()}.mjs`)

await build({
  entryPoints: ['src/utils/retrievalDebug.ts'],
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile,
  write: true
})

const { chunkOutcome, voteShares, routeStats, routeOverlap, buildDebugExport, debugExportFilename } = await import(pathToFileURL(outfile))

function row(key, overrides = {}) {
  return {
    key, chunk_id: key, chunk_type: 'text', section: '', page: null, preview: '', content: '',
    route_ranks: {}, route_scores: {}, keyword_terms: [], fused_rank: null, fused_score: null, rrf_votes: {},
    matched_indexes: [], source_queries: [], rerank_rank: null, rerank_score: null, final_rank: null,
    context_role: '', final_reason: '', ...overrides
  }
}

function vote(value) {
  return { rank: 1, weight: 1, confidence: 1, multiplier: 1, vote: value }
}

const chunks = [
  row('a', { route_ranks: { vec: 1, kw: 2 }, fused_rank: 1, rerank_rank: 1, final_rank: 1, rrf_votes: { vec: vote(3), kw: vote(1) } }),
  row('b', { route_ranks: { kw: 1 }, fused_rank: 2, rerank_rank: 3, rrf_votes: { kw: vote(2) } }),
  row('c', { route_ranks: { vec: 2 }, fused_rank: 5, rrf_votes: { vec: vote(1) } }),
  row('d', { final_rank: 2, context_role: 'neighbor' })
]
const run = {
  config: { rrf_candidate_limit: 3, effective_top_k: 2 },
  rerank: { applied: true },
  chunks
}

// 每个 chunk 按最早把它挡掉的阶段归因；只出现在最终上下文的是扩展补入。
assert.deepEqual(chunks.map(item => chunkOutcome(item, run).code), ['final', 'rerank_dropped', 'fusion_cut', 'expansion'])
// rerank 没生效时，融合保留内但没进输出的不归给 rerank。
assert.equal(chunkOutcome(chunks[1], { ...run, rerank: { applied: false } }).code, 'not_selected')
// 融合名次在保留线以外却出现在输出里，是扩展按锚点补进来的，不算融合保留。
assert.equal(chunkOutcome(row('e', { route_ranks: { vec: 9 }, fused_rank: 9, final_rank: 3 }), run).code, 'expansion')

const shares = voteShares(chunks[0])
assert.deepEqual(shares.map(item => [item.route, item.share]), [['vec', 0.75], ['kw', 0.25]])

const stats = Object.fromEntries(routeStats(run, ['vec', 'kw']).map(item => [item.name, item]))
assert.equal(stats.vec.unique, 1)
assert.equal(stats.kw.unique, 1)
assert.equal(stats.vec.finalHits, 1)
assert.equal(stats.vec.finalVoteShare, 0.75)

// vec={a,c}，kw={a,b}：交集 1，并集 3。
const overlap = routeOverlap(run, ['vec', 'kw', 'empty'])
assert.equal(overlap[0][0], 1)
assert.equal(overlap[0][1], 1 / 3)
assert.equal(overlap[0][2], null)

// 导出带上当次请求，每个 chunk 附页面判定的去向，且不改动原始运行数据。
const exportedAt = new Date('2026-10-22T08:09:10Z')
const request = { arxiv_id: 'hep-th/9901001', question: 'q', enable_llm_rerank: true }
const exported = buildDebugExport(run, request, exportedAt)
assert.equal(exported.exported_at, '2026-10-22T08:09:10.000Z')
assert.equal(exported.request, request)
assert.deepEqual(exported.run.chunks.map(item => item.outcome), ['final', 'rerank_dropped', 'fusion_cut', 'expansion'])
assert.equal(exported.run.chunks[0].outcome_label, '进入检索输出')
assert.equal('outcome' in chunks[0], false)
// 旧式编号里的斜杠不能进文件名。
assert.equal(debugExportFilename('hep-th/9901001', exportedAt), 'retrieval-debug-hep-th_9901001-20261022-080910.json')

console.log('retrievalDebug tests passed')
