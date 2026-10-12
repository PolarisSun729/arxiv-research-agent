import type { ResearchRoundDebug, ResearchStepDebug, RetrievalDebug } from '@/api/papers'

export interface ProcessStepView {
  key: string
  title: string
  content: string
}

export interface ResearchRoundView {
  key: string
  title: string
  subtitle: string
  stages: ProcessStepView[]
}

// 历史消息的 debug 快照会被裁剪：长列表末尾追加截断标记对象，这里只保留可展示的字符串。
function textList(values: unknown): string[] {
  if (!Array.isArray(values)) return []
  return values.filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
}

function joinOr(values: string[], fallback: string, separator = ' | ') {
  return values.length ? values.join(separator) : fallback
}

function formatRouteCounts(routeCounts: unknown) {
  if (!routeCounts || typeof routeCounts !== 'object') return '无召回结果'
  const entries = Object.entries(routeCounts as Record<string, unknown>).filter(
    (entry): entry is [string, number] => typeof entry[1] === 'number'
  )
  return entries.length ? entries.map(([name, count]) => `${name}: ${count} 条`).join(' · ') : '无召回结果'
}

export function hasResearchProcess(debug: RetrievalDebug | null | undefined): boolean {
  return Array.isArray(debug?.research_rounds) || Array.isArray(debug?.research_steps)
}

export function buildResearchStepViews(steps: ResearchStepDebug[] | undefined): ProcessStepView[] {
  if (!Array.isArray(steps)) return []
  return steps
    .filter(step => step && typeof step.title === 'string')
    .map((step, index) => ({ key: `step-${index}`, title: step.title, content: step.detail || '' }))
}

export function buildResearchRoundViews(rounds: ResearchRoundDebug[] | undefined): ResearchRoundView[] {
  if (!Array.isArray(rounds)) return []
  return rounds
    .filter(round => round && typeof round.round === 'number')
    .map(round => {
      const finalContext = textList(round.final_context)
      const rewritten = textList(round.rewritten_queries)
      return {
        key: `round-${round.round}`,
        title: `第 ${round.round} 轮检索 · ${round.target_need_id || '-'}`,
        subtitle: `新增 ${round.new_candidate_count ?? 0} 条 · 重复 ${round.duplicate_candidate_count ?? 0} 条${
          round.decision_reason ? ` · ${round.decision_reason}` : ''
        }`,
        stages: [
          { key: 'query', title: '检索 Query', content: round.query || '无' },
          {
            key: 'rewrite',
            title: 'Query Rewrite',
            content: round.rewrite_enabled === false ? '未启用' : joinOr(rewritten, '无')
          },
          { key: 'hyde', title: 'HyDE', content: round.hyde_enabled && round.hyde_text ? round.hyde_text : '未启用' },
          { key: 'keyword', title: 'Keyword Search', content: joinOr(textList(round.keyword_queries), '无') },
          { key: 'routes', title: '多路召回', content: formatRouteCounts(round.route_counts) },
          { key: 'fusion', title: 'RRF / Fusion', content: round.fusion || '-' },
          { key: 'rerank', title: 'Rerank', content: round.rerank || '无 rerank 数据' },
          { key: 'final-context', title: 'Final Context', content: joinOr(finalContext, '无最终上下文数据', '\n') }
        ]
      }
    })
}

export interface RemovedClaimView {
  key: string
  text: string
  reason: string
  meta: string[]
}

export interface RemovedClaimsView {
  items: RemovedClaimView[]
  /** 历史快照裁剪掉、无法展示的条数。 */
  omittedCount: number
  /** 原草稿排版无法原位保留、正文改为拼接受支持句子时为 true。 */
  joinedFallback: boolean
}

const IMPORTANCE_LABELS: Record<string, string> = { core: '核心主张', supporting: '辅助主张', background: '背景' }

// 超长字符串在历史快照里会被压成 {preview, truncated}，这里取回可展示的预览。
function displayText(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  if (value && typeof value === 'object' && typeof (value as { preview?: unknown }).preview === 'string') {
    return String((value as { preview: string }).preview).trim()
  }
  return ''
}

export function buildRemovedClaimsView(debug: RetrievalDebug | null | undefined): RemovedClaimsView {
  const removed = debug?.removed_claims
  const raw: unknown[] = Array.isArray(removed) ? removed : []
  const items: RemovedClaimView[] = []
  let omittedCount = 0
  raw.forEach((entry, index) => {
    if (!entry || typeof entry !== 'object') return
    const record = entry as Record<string, unknown>
    if (typeof record.truncated_items === 'number') {
      omittedCount += record.truncated_items
      return
    }
    const text = displayText(record.text)
    if (!text) return
    const citations = textList(record.citation_ids)
    const supporting = textList(record.supporting_evidence_ids)
    const missing = textList(record.missing_facets)
    const meta = [
      IMPORTANCE_LABELS[String(record.importance || '')] || '',
      citations.length ? `引用 ${citations.join('、')}` : '无引用',
      record.verdict ? `校验结论 ${String(record.verdict)}` : '',
      supporting.length && supporting.join() !== citations.join() ? `校验器认定的支持证据 ${supporting.join('、')}` : '',
      missing.length ? `缺少：${missing.join('；')}` : ''
    ].filter(Boolean)
    items.push({
      key: String(record.claim_id || `removed-${index}`),
      text,
      reason: displayText(record.reason) || '校验未通过',
      meta
    })
  })
  return { items, omittedCount, joinedFallback: debug?.answer_pruning === 'joined_supported_claims' }
}
