import type { AgentProgressEntry, AgentProgressStatus, AgentRunProgress, AgentStreamEvent } from '@/types/agent'
import { isRecord } from '@/utils/guards'

// 纯簿记节点：耗时极短、对用户没有信息量，只有失败时才露出到时间线。
const BOOKKEEPING_NODES = new Set(['select_next_step', 'observe_step', 'route_after_observation'])

const NODE_LABELS: Record<string, string> = {
  parse_search_request: '理解你的问题',
  build_goal: '明确任务目标',
  build_plan: '制定执行计划',
  select_next_step: '选择下一步',
  execute_step: '执行步骤',
  observe_step: '检查步骤结果',
  route_after_observation: '决定后续动作',
  replan: '调整执行计划',
  finalize: '整理最终回答',
  error_finalize: '整理异常结果',
  __interrupt__: '等待你确认'
}

// 键是执行计划里的工具名（tool_registry 的 contract 名），部分旧名是后端实际工具名。
const TOOL_LABELS: Record<string, string> = {
  normalize_request: '归一化请求',
  build_arxiv_search_spec: '生成检索条件',
  search_arxiv: '检索 arXiv 论文',
  search_arxiv_structured: '检索 arXiv 论文',
  rewrite_arxiv_query: '改写检索关键词',
  validate_arxiv_results: '校验检索结果',
  personalize_paper_results: '按研究画像排序',
  synthesize_arxiv_response: '撰写检索结果摘要',
  recommend_papers: '生成个性化推荐',
  load_user_profile: '读取研究画像',
  load_candidate_papers: '读取候选论文',
  generate_recommendations: '生成个性化推荐',
  validate_recommendations: '校验推荐结果',
  explain_recommendations: '整理推荐说明',
  resolve_paper: '定位目标论文',
  check_paper_index: '检查论文全文索引',
  check_paper_qa_index: '检查论文全文索引',
  parse_and_index_paper: '解析并索引论文全文',
  build_paper_qa_index: '构建论文问答索引',
  answer_paper_question: '基于论文全文回答',
  assess_paper_qa_quality: '评估回答质量',
  generate_fallback_response: '生成兜底回答',
  resolve_preference_target: '确认偏好目标论文',
  update_preference_store: '更新论文偏好',
  verify_preference_update: '校验偏好写入',
  synthesize_preference_response: '整理偏好操作结果'
}

const INTENT_LABELS: Record<string, string> = {
  arxiv_search: '论文检索',
  paper_detail: '论文细节解读',
  paper_summary: '论文总结',
  paper_qa: '论文问答',
  recommendation: '个性化推荐',
  preference_action: '论文偏好操作',
  unclear: '需求需要澄清',
  unsupported: '超出能力范围',
  confirmation: '等待确认'
}

// 这些工具产出的条目数就是用户最关心的中间结果：工具名 -> [输出字段, 文案]。
const RESULT_COUNT_FACTS: Record<string, [string, (count: number) => string]> = {
  search_arxiv: ['papers', count => `检索到 ${count} 篇论文`],
  search_arxiv_structured: ['papers', count => `检索到 ${count} 篇论文`],
  generate_recommendations: ['recommendations', count => `生成 ${count} 条推荐`],
  recommend_papers: ['recommendations', count => `生成 ${count} 条推荐`]
}

export function getToolLabel(toolName: string | null | undefined) {
  const name = String(toolName || '').trim()
  if (!name) return NODE_LABELS.execute_step
  return TOOL_LABELS[name] || `调用工具 ${name}`
}

function getStepLabel(node: string, toolName?: string | null) {
  if (node === 'execute_step' && toolName) return getToolLabel(toolName)
  return NODE_LABELS[node] || node
}

export function createRunProgress(now: number): AgentRunProgress {
  return { status: 'running', startedAt: now, endedAt: null, entries: [] }
}

function findLastEntryIndex(entries: AgentProgressEntry[], node: string, runningOnly: boolean) {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index]
    if (entry.node === node && (!runningOnly || entry.status === 'running')) return index
  }
  return -1
}

function mergeFacts(current: string[], extra: string[]) {
  return [...new Set([...current, ...extra].filter(Boolean))]
}

function truncate(text: string, limit: number) {
  return text.length > limit ? `${text.slice(0, limit)}…` : text
}

function normalizeStepStatus(value: unknown): AgentProgressStatus {
  return value === 'failed' || value === 'skipped' ? value : 'success'
}

function intentFacts(state: Record<string, any>) {
  const facts: string[] = []
  const intent = typeof state.intent === 'string' ? state.intent : ''
  // 追问会先结合上文改写成完整的一句，展示出来让用户确认理解没有跑偏。
  if (typeof state.contextualized_message === 'string' && state.contextualized_message.trim()) {
    facts.push(`理解为：${truncate(state.contextualized_message.trim(), 80)}`)
  }
  if (intent) facts.push(`意图：${INTENT_LABELS[intent] || intent}`)
  const spec = isRecord(state.search_spec) ? state.search_spec : null
  if (intent === 'arxiv_search' && spec) {
    const concepts = Array.isArray(spec.concepts) ? spec.concepts.filter(isRecord) : []
    if (concepts.length) {
      // 概念之间是“且”，同义写法之间是“或”，和后端阶梯检索的组合方式一致。
      const text = concepts
        .map(concept => [concept.term, ...(Array.isArray(concept.aliases) ? concept.aliases : [])].filter(Boolean).join(' / '))
        .join(' + ')
      facts.push(`关键词：${truncate(text, 80)}`)
      if (concepts.some(concept => concept.source === 'profile')) facts.push('主题按研究画像补全')
    } else {
      const query = [spec.query, spec.title_query, spec.abstract_query].find(item => typeof item === 'string' && item.trim())
      if (query) facts.push(`关键词：${truncate(String(query).trim(), 60)}`)
    }
    if (typeof spec.submitted_days_ago === 'number' && spec.submitted_days_ago > 0) facts.push(`最近 ${spec.submitted_days_ago} 天`)
    if (typeof spec.max_results === 'number' && spec.max_results > 0) facts.push(`最多 ${spec.max_results} 篇`)
    if (Array.isArray(spec.categories) && spec.categories.length) facts.push(`分类：${spec.categories.slice(0, 4).join('、')}`)
  }
  return facts
}

function planFacts(state: Record<string, any>) {
  const steps = Array.isArray(state.execution_plan) ? state.execution_plan : []
  const labels: string[] = []
  for (const step of steps) {
    if (!isRecord(step)) continue
    const label = step.tool_name ? getToolLabel(step.tool_name) : String(step.action_type || '')
    // 连续重复的步骤（如多次检索）只展示一次，避免计划链过长。
    if (label && labels[labels.length - 1] !== label) labels.push(label)
  }
  return labels.length ? [`计划：${labels.join(' → ')}`] : []
}

function toolResultFacts(toolCall: Record<string, any> | null, toolName: string | null) {
  const facts: string[] = []
  // 只认后端随 tool_call_end 给出的本步输出条目数；state.paper_count 要到收尾阶段才回填，执行中读到的是 0。
  // 阶梯检索有多级时逐级列出命中数，让用户看到“严格匹配 0 篇 → 放宽后 N 篇”的过程。
  const attempts = Array.isArray(toolCall?.search_attempts) ? toolCall.search_attempts.filter(isRecord) : []
  if (attempts.length > 1) {
    for (const attempt of attempts) {
      const label = String(attempt.label || attempt.level || '检索')
      facts.push(attempt.ok === false ? `${label}：失败` : `${label}：${Number(attempt.total_results) || 0} 篇`)
    }
  }
  const countFact = toolName ? RESULT_COUNT_FACTS[toolName] : undefined
  const counts = isRecord(toolCall?.result_counts) ? toolCall?.result_counts : null
  if (countFact && counts && typeof counts[countFact[0]] === 'number') {
    facts.push(countFact[1](counts[countFact[0]]))
  }
  const error = isRecord(toolCall?.error) ? toolCall?.error : null
  const errorMessage = error && typeof error.message === 'string' ? error.message : ''
  if (toolCall?.status === 'failed' || errorMessage) {
    facts.push(`失败：${truncate(errorMessage || '工具执行失败', 80)}`)
  }
  // 工具摘要可能是内部英文说明；只有中文摘要才直接展示给用户。
  const summary = typeof toolCall?.summary === 'string' ? toolCall.summary.trim() : ''
  if (summary && /[一-鿿]/.test(summary) && !summary.startsWith('正在')) facts.push(truncate(summary, 80))
  return facts
}

function closeEntry(entry: AgentProgressEntry, status: AgentProgressStatus, now: number): AgentProgressEntry {
  return { ...entry, status, endedAt: entry.endedAt ?? now }
}

function closeRunningEntries(entries: AgentProgressEntry[], now: number, lastStatus: AgentProgressStatus = 'success') {
  const lastRunning = entries.map(entry => entry.status).lastIndexOf('running')
  return entries.map((entry, index) => entry.status === 'running'
    ? closeEntry(entry, index === lastRunning ? lastStatus : 'success', now)
    : entry)
}

export function finishRunProgress(progress: AgentRunProgress, now: number, status: 'done' | 'failed' = 'done'): AgentRunProgress {
  if (progress.status !== 'running') return progress
  return {
    ...progress,
    status,
    endedAt: now,
    entries: closeRunningEntries(progress.entries, now, status === 'failed' ? 'failed' : 'success')
  }
}

/**
 * 把一条 Agent SSE 事件归约进运行进度，返回新对象（不修改入参）。
 *
 * 后端事件顺序：step_start → [tool_call_start] → step_end → [tool_call_end]，
 * 所以 tool_call_end 需要回写到已经结束的同名步骤上。
 */
export function applyProgressEvent(progress: AgentRunProgress, event: AgentStreamEvent, now: number): AgentRunProgress {
  const data = isRecord(event.data) ? event.data : {}
  const node = typeof data.step === 'string' ? data.step : ''
  const state = isRecord(data.state) ? data.state : {}
  const entries = [...progress.entries]

  switch (event.event_type) {
    case 'step_start': {
      if (!node || BOOKKEEPING_NODES.has(node) || node === '__interrupt__') return progress
      // 批准恢复时工具会先于 step_start 被预告；同一节点已在运行就不重复建条目。
      if (findLastEntryIndex(entries, node, true) >= 0) return progress
      entries.push({
        key: `${event.sequence}-${node}`,
        node,
        label: getStepLabel(node),
        status: 'running',
        facts: [],
        toolName: null,
        startedAt: now,
        endedAt: null
      })
      return { ...progress, entries }
    }
    case 'tool_call_start': {
      const toolCall = isRecord(data.tool_call) ? data.tool_call : null
      const toolName = typeof toolCall?.tool_name === 'string' ? toolCall.tool_name : null
      const targetNode = node || 'execute_step'
      const index = findLastEntryIndex(entries, targetNode, true)
      if (index >= 0) {
        entries[index] = { ...entries[index], toolName, label: getStepLabel(targetNode, toolName) }
      } else {
        entries.push({
          key: `${event.sequence}-${targetNode}`,
          node: targetNode,
          label: getStepLabel(targetNode, toolName),
          status: 'running',
          facts: [],
          toolName,
          startedAt: now,
          endedAt: null
        })
      }
      return { ...progress, entries }
    }
    case 'tool_call_end': {
      const toolCall = isRecord(data.tool_call) ? data.tool_call : null
      const targetNode = node || 'execute_step'
      const index = findLastEntryIndex(entries, targetNode, false)
      if (index < 0) return progress
      const entry = entries[index]
      // 结束事件的工具名由执行后的状态推断，可能已指向下一步；优先沿用开始时确定的工具名。
      const toolName = entry.toolName || (typeof toolCall?.tool_name === 'string' && toolCall.tool_name) || null
      const failed = toolCall?.status === 'failed'
      entries[index] = {
        ...entry,
        toolName,
        label: getStepLabel(targetNode, toolName),
        status: failed ? 'failed' : entry.status,
        facts: mergeFacts(entry.facts, toolResultFacts(toolCall, toolName))
      }
      return { ...progress, entries }
    }
    case 'step_end': {
      if (!node) return progress
      const status = normalizeStepStatus(data.status)
      if (node === '__interrupt__') {
        const closed = closeRunningEntries(entries, now)
        closed.push({
          key: `${event.sequence}-${node}`,
          node,
          label: NODE_LABELS.__interrupt__,
          status: 'success',
          facts: ['请在下方确认后继续'],
          toolName: null,
          startedAt: now,
          endedAt: now
        })
        return { ...progress, entries: closed }
      }
      if (BOOKKEEPING_NODES.has(node) && status !== 'failed') return progress

      let facts: string[] = []
      if (node === 'parse_search_request') facts = intentFacts(state)
      else if (node === 'build_plan' || node === 'replan') facts = planFacts(state)
      if (status === 'failed' && isRecord(data.step_detail) && typeof data.step_detail.error === 'string') {
        facts.push(`失败：${truncate(data.step_detail.error, 80)}`)
      }

      const index = findLastEntryIndex(entries, node, true)
      if (index >= 0) {
        const entry = entries[index]
        entries[index] = { ...closeEntry(entry, status, now), facts: mergeFacts(entry.facts, facts) }
      } else {
        // 只有 step_end 没有 step_start（旧后端或补发事件）时，直接补一条已结束的步骤。
        entries.push({
          key: `${event.sequence}-${node}`,
          node,
          label: getStepLabel(node),
          status,
          facts,
          toolName: null,
          startedAt: now,
          endedAt: now
        })
      }
      return { ...progress, entries }
    }
    case 'final_response':
      return finishRunProgress(progress, now, isRecord(data.response?.debug?.runtime_error) ? 'failed' : 'done')
    case 'exception':
      return finishRunProgress(progress, now, 'failed')
    case 'stream_end':
      return finishRunProgress(progress, now, data.status === 'error' ? 'failed' : 'done')
    default:
      return progress
  }
}

export function getActiveEntry(progress: AgentRunProgress | null | undefined) {
  if (!progress || progress.status !== 'running') return null
  for (let index = progress.entries.length - 1; index >= 0; index -= 1) {
    if (progress.entries[index].status === 'running') return progress.entries[index]
  }
  return null
}

export function formatElapsed(ms: number) {
  const safe = Math.max(0, ms)
  if (safe < 60_000) return `${(safe / 1000).toFixed(1)}s`
  const minutes = Math.floor(safe / 60_000)
  const seconds = Math.floor((safe % 60_000) / 1000)
  return `${minutes}分${seconds.toString().padStart(2, '0')}秒`
}
