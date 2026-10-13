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

const {
  buildRemovedClaimsView,
  buildResearchRoundViews,
  buildResearchStepViews,
  buildResearchSummaryView,
  hasResearchProcess,
  splitStepDetail
} = await import(pathToFileURL(outfile))

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
  { kind: 'filter', title: '筛掉 2 条无关证据', detail: '剩余 18 条' },
  { kind: 'done', title: '研究结束', detail: 'partial' }
])
assert.deepEqual(steps.map(step => step.title), ['第 1 轮检索（need-1）', '筛掉 2 条无关证据', '研究结束'])
assert.equal(steps[1].content, '剩余 18 条')
assert.deepEqual(steps.map(step => step.kind), ['search', 'filter', 'done'])

// 决策理由是模型的长段说明，单独成段；其余短事实做成标签。
assert.deepEqual(splitStepDetail('新增 39 条候选，重复 0 条 · 决策理由 Need more evidence. · 决策时阅读 0 条证据全文'), {
  facts: ['新增 39 条候选，重复 0 条', '决策时阅读 0 条证据全文'],
  reason: 'Need more evidence.'
})
assert.deepEqual(splitStepDetail(''), { facts: [], reason: '' })

assert.deepEqual(
  buildResearchSummaryView({ outcome: 'partial', retrieval_count: 1, draft_attempt_count: 2, supported_claim_count: 10, removed_claim_count: 1 }),
  {
    outcome: 'partial',
    outcomeLabel: '部分回答',
    stats: [
      { label: '检索', value: '1 轮' },
      { label: '起草', value: '2 版' },
      { label: '受支持主张', value: '10 / 11' }
    ]
  }
)
assert.equal(buildResearchSummaryView({ outcome: 'completed', supported_claim_count: 4 }).stats[2].value, '4 条')
assert.equal(buildResearchSummaryView(null), null)

// 部分回答被删的句子：原因、引用、校验结论都要可见；历史快照的截断标记与超长文本预览也要能展示。
const removedView = buildRemovedClaimsView({
  answer_pruning: 'joined_supported_claims',
  removed_claims: [
    { claim_id: 'c1', text: '根据证据，方法包含两个层面。', importance: 'core', citation_ids: [], verdict: 'unsupported', reason: '没有挂引用，无法对照论文核对' },
    { claim_id: 'c2', text: { preview: '很长的句子...', truncated: true }, citation_ids: ['11'], supporting_evidence_ids: ['13'], verdict: 'citation_mismatch', missing_facets: ['具体数值'], reason: '挂的引用不对' },
    { claim_id: 'c3', text: '' },
    { truncated_items: 2, original_count: 10 }
  ]
})
assert.equal(removedView.joinedFallback, true)
assert.equal(removedView.omittedCount, 2)
assert.deepEqual(removedView.items.map(item => item.text), ['根据证据，方法包含两个层面。', '很长的句子...'])
assert.deepEqual(removedView.items[0].meta, ['核心主张', '无引用', '校验结论 unsupported'])
assert.deepEqual(removedView.items[1].meta, ['引用 11', '校验结论 citation_mismatch', '校验器认定的支持证据 13', '缺少：具体数值'])
assert.deepEqual(buildRemovedClaimsView(null), { items: [], omittedCount: 0, joinedFallback: false })

console.log('researchProcess tests passed')
