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
