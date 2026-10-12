import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { pathToFileURL } from 'node:url'

const outfile = join(tmpdir(), `research-process-${Date.now()}.mjs`)

await build({
  entryPoints: ['src/utils/researchProcess.ts'],
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile,
  write: true
})

const { buildResearchRoundViews, buildResearchStepViews, hasResearchProcess } = await import(pathToFileURL(outfile))

assert.equal(hasResearchProcess({ research_rounds: [] }), true)
assert.equal(hasResearchProcess({ routes: {} }), false)
assert.equal(hasResearchProcess(null), false)

const [round] = buildResearchRoundViews([
  {
    round: 2,
    target_need_id: 'need-1',
    decision_reason: 'VERIFY_STAGES',
    query: 'CogMem stages',
    new_candidate_count: 0,
    duplicate_candidate_count: 3,
    rewrite_enabled: true,
    rewritten_queries: ['CogMem architecture'],
    hyde_enabled: false,
    hyde_text: '',
    keyword_queries: ['CogMem workflow'],
    route_counts: { vector_original: 30, keyword: 12 },
    fusion: 'pure_rrf · RRF k 60',
    rerank: 'dashscope · 30 → 30 条',
    // 落库快照裁剪会在长列表末尾追加截断标记对象，展示时必须被丢弃。
    final_context: ['chunk 54 · figure · p4', 'chunk 5 · text · p2', { truncated_items: 4, original_count: 12 }]
  }
])
assert.equal(round.title, '第 2 轮检索 · need-1')
assert.equal(round.subtitle, '新增 0 条 · 重复 3 条 · VERIFY_STAGES')
const stage = Object.fromEntries(round.stages.map(item => [item.key, item.content]))
assert.equal(stage.rewrite, 'CogMem architecture')
assert.equal(stage.hyde, '未启用')
assert.equal(stage.routes, 'vector_original: 30 条 · keyword: 12 条')
assert.equal(stage['final-context'], 'chunk 54 · figure · p4\nchunk 5 · text · p2')

const [disabledRound] = buildResearchRoundViews([{ round: 3, target_need_id: 'need-2', query: 'q', rewrite_enabled: false }])
const disabled = Object.fromEntries(disabledRound.stages.map(item => [item.key, item.content]))
assert.equal(disabled.rewrite, '未启用')
assert.equal(disabled.routes, '无召回结果')
assert.equal(disabled['final-context'], '无最终上下文数据')

const steps = buildResearchStepViews([
  { kind: 'search', title: '第 1 轮检索（need-1）', detail: '新增 20 条候选' },
  { truncated_items: 2, original_count: 10 },
  { kind: 'done', title: '研究结束', detail: 'partial' }
])
assert.deepEqual(steps.map(step => step.title), ['第 1 轮检索（need-1）', '研究结束'])

console.log('researchProcess tests passed')
