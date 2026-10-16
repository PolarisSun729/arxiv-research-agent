import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { pathToFileURL } from 'node:url'

const outfile = join(tmpdir(), `agent-run-progress-${Date.now()}.mjs`)

await build({
  entryPoints: ['src/utils/agentRunProgress.ts'],
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile,
  write: true
})

const {
  applyProgressEvent,
  createRunProgress,
  finishRunProgress,
  formatElapsed,
  getActiveEntry
} = await import(pathToFileURL(outfile))

let sequence = 0
function event(event_type, data = {}) {
  sequence += 1
  return { event_type, sequence, run_id: 'r1', timestamp: '', data }
}

function replay(events) {
  let progress = createRunProgress(0)
  events.forEach((item, index) => {
    progress = applyProgressEvent(progress, item, (index + 1) * 1000)
  })
  return progress
}

// 一次完整的检索：簿记节点不出现在时间线上，工具结果回写到已结束的 execute_step。
const searchState = {
  intent: 'arxiv_search',
  search_spec: { query: 'RAG', submitted_days_ago: 7, max_results: 5, categories: [] },
  execution_plan: [
    { step_id: 's1', tool_name: 'search_arxiv_structured' },
    { step_id: 's2', tool_name: 'validate_arxiv_results' }
  ],
  // 真实运行中 state.papers 要到收尾才回填，检索步骤结束时快照里仍是 0。
  paper_count: 0
}
const midway = replay([
  event('run_start'),
  event('step_start', { step: 'parse_search_request' })
])
assert.equal(getActiveEntry(midway)?.label, '理解你的问题')
assert.equal(midway.status, 'running')

const done = replay([
  event('run_start'),
  event('step_start', { step: 'parse_search_request' }),
  event('step_end', { step: 'parse_search_request', status: 'success', state: searchState }),
  event('step_start', { step: 'build_plan' }),
  event('step_end', { step: 'build_plan', status: 'success', state: searchState }),
  event('step_start', { step: 'select_next_step' }),
  event('step_end', { step: 'select_next_step', status: 'success', state: searchState }),
  event('step_start', { step: 'execute_step' }),
  event('tool_call_start', { step: 'execute_step', tool_call: { tool_name: 'search_arxiv_structured', status: 'running' } }),
  event('step_end', { step: 'execute_step', status: 'success', state: searchState }),
  event('tool_call_end', { step: 'execute_step', tool_call: { tool_name: 'validate_arxiv_results', status: 'success', summary: 'continue', result_counts: { papers: 5 } }, state: searchState }),
  event('step_start', { step: 'finalize' }),
  event('step_end', { step: 'finalize', status: 'success', state: searchState }),
  event('final_response', { response: { debug: {} } })
])
assert.equal(done.status, 'done')
assert.deepEqual(done.entries.map(entry => entry.label), ['理解你的问题', '制定执行计划', '检索 arXiv 论文', '整理最终回答'])
assert.ok(done.entries.every(entry => entry.status === 'success'))
assert.deepEqual(done.entries[0].facts, ['意图：论文检索', '关键词：RAG', '最近 7 天', '最多 5 篇'])
assert.deepEqual(done.entries[1].facts, ['计划：检索 arXiv 论文 → 校验检索结果'])
// 数量取自本步输出而不是 state.paper_count；结束事件里被推断成下一步的工具名不能覆盖开始时的工具名；
// 英文内部摘要不展示。
assert.deepEqual(done.entries[2].facts, ['检索到 5 篇论文'])
assert.equal(done.entries[2].toolName, 'search_arxiv_structured')
assert.equal(done.entries[2].endedAt - done.entries[2].startedAt, 2000)

// 批准恢复时工具先被预告，随后的 step_start 不能再建一条重复步骤。
const resumed = replay([
  event('run_start'),
  event('tool_call_start', { step: 'execute_step', tool_call: { tool_name: 'parse_and_index_paper' } }),
  event('step_start', { step: 'execute_step' })
])
assert.equal(resumed.entries.length, 1)
assert.equal(resumed.entries[0].label, '解析并索引论文全文')

// interrupt 会结束当前步骤并提示用户确认。
const interrupted = replay([
  event('step_start', { step: 'execute_step' }),
  event('step_end', { step: '__interrupt__', status: 'success' })
])
assert.deepEqual(interrupted.entries.map(entry => [entry.label, entry.status]), [['执行步骤', 'success'], ['等待你确认', 'success']])

// 异常或网络中断收尾：最后一个运行中步骤标记失败。
const failed = finishRunProgress(replay([event('step_start', { step: 'build_plan' })]), 9000, 'failed')
assert.equal(failed.status, 'failed')
assert.equal(failed.entries[0].status, 'failed')
assert.equal(failed.endedAt, 9000)
assert.equal(getActiveEntry(failed), null)

assert.equal(formatElapsed(1234), '1.2s')
assert.equal(formatElapsed(65_000), '1分05秒')


// 阶梯检索：解析结果按“概念 + 同义写法”展示，检索步骤逐级列出命中数。
const ladder = replay([
  event('step_start', { step: 'parse_search_request' }),
  event('step_end', {
    step: 'parse_search_request',
    status: 'success',
    state: {
      intent: 'arxiv_search',
      search_spec: {
        concepts: [{ term: 'RAG', aliases: ['retrieval-augmented generation'], source: 'user' }],
        query: 'RAG',
        submitted_days_ago: 7
      }
    }
  }),
  event('step_start', { step: 'execute_step' }),
  event('tool_call_start', { step: 'execute_step', tool_call: { tool_name: 'search_arxiv', status: 'running' } }),
  event('step_end', { step: 'execute_step', status: 'success', state: {} }),
  event('tool_call_end', {
    step: 'execute_step',
    tool_call: {
      tool_name: 'search_arxiv',
      status: 'success',
      result_counts: { papers: 5 },
      search_attempts: [
        { level: 'strict', label: '严格匹配', total_results: 0, ok: true },
        { level: 'drop_inferred_categories', label: '放开推断的分类范围', total_results: 12, ok: true }
      ]
    }
  })
])
assert.deepEqual(ladder.entries[0].facts, ['意图：论文检索', '关键词：RAG / retrieval-augmented generation', '最近 7 天'])
assert.deepEqual(ladder.entries[1].facts, ['严格匹配：0 篇', '放开推断的分类范围：12 篇', '检索到 5 篇论文'])

console.log('agent run progress tests passed')

// 追问被结合上文改写时，“理解你的问题”先展示改写后的请求。
const followUp = replay([
  event('step_start', { step: 'parse_search_request' }),
  event('step_end', {
    step: 'parse_search_request',
    status: 'success',
    state: { intent: 'arxiv_search', contextualized_message: '再多找一些 RAG 的论文，不限时间', search_spec: { query: 'RAG' } }
  })
])
assert.deepEqual(followUp.entries[0].facts, ['理解为：再多找一些 RAG 的论文，不限时间', '意图：论文检索', '关键词：RAG'])
