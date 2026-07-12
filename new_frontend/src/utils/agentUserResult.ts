import type {
  AgentInteraction,
  AgentPaper,
  AgentPreferenceActionResult,
  ArxivSearchResponse,
  PaperTargetCandidate,
  SideEffectApprovalPayload,
  TargetSelectionPayload
} from '@/types/agent'
import type { NormalizedArxivQueryCapability } from '@/types/arxivCapability'
import type { Paper } from '@/types/paper'
import { normalizeArxivQueryCapability } from '@/utils/arxivQueryCapability'

export interface AgentUserInteractionCandidate {
  id: string
  title: string
  arxivId: string
  authors: string
  rank: number | null
  sourceLabel: string
  isDefault: boolean
}

export interface AgentUserInteraction {
  id: string
  kind: AgentInteraction['kind']
  title: string
  description: string
  confirmLabel: string
  cancelLabel: string
  defaultCandidateId: string
  candidates: AgentUserInteractionCandidate[]
}

export interface AgentUserResult {
  papers: Paper[]
  queryCapability: NormalizedArxivQueryCapability | null
  preferenceFeedback: { status: 'success' | 'failed'; message: string } | null
  interaction: AgentUserInteraction | null
  hasVisibleContent: boolean
}

interface ToAgentUserResultOptions {
  interaction?: AgentInteraction | null
}

function normalizePaper(raw: AgentPaper): Paper {
  const arxivId = raw.arxiv_id || raw.arxivId || (raw.id ? String(raw.id).split('/').pop() : '')
  const authors = Array.isArray(raw.authors) ? raw.authors : String(raw.authors || '').split(',').map(item => item.trim()).filter(Boolean)
  const categories = Array.isArray(raw.categories) ? raw.categories : String(raw.categories || '').split(',').map(item => item.trim()).filter(Boolean)
  return {
    id: arxivId || raw.id || '',
    arxivId: arxivId || raw.id || '',
    title: raw.title || '',
    authors,
    summary: raw.abstract || raw.summary || '',
    publishedAt: raw.publishedAt || raw.published || raw.published_date || '',
    updatedAt: raw.updatedAt || raw.updated || '',
    categories,
    pdfUrl: raw.pdfUrl || raw.pdf_url || '',
    absUrl: raw.absUrl || raw.abs_url || raw.url || '',
    label: raw.label || null,
    query_match_score: raw.query_match_score,
    personalization_score: raw.personalization_score,
    final_score: raw.final_score,
    score_breakdown: raw.score_breakdown || null,
    matched_terms: raw.matched_terms || [],
    personalized_reason: raw.personalized_reason || null,
    match_reason: raw.match_reason || null,
    priority: raw.priority
  }
}

export function getPaperTargetCandidateId(candidate: PaperTargetCandidate, index = 0) {
  return String(candidate.candidate_id || candidate.paper_id || candidate.arxiv_id || candidate.arxivId || candidate.id || `candidate-${index}`)
}

function normalizeCandidate(candidate: PaperTargetCandidate, index: number, defaultId: string): AgentUserInteractionCandidate {
  const id = getPaperTargetCandidateId(candidate, index)
  return {
    id,
    title: candidate.title || candidate.arxiv_id || candidate.arxivId || id,
    arxivId: candidate.arxiv_id || candidate.arxivId || candidate.id || '',
    authors: candidate.authors_summary || (Array.isArray(candidate.authors) ? candidate.authors.slice(0, 3).join(', ') : candidate.authors || ''),
    rank: typeof candidate.rank === 'number' ? candidate.rank : null,
    sourceLabel: candidate.source_label || candidate.list_name || candidate.source_type || candidate.source || '上下文候选',
    isDefault: id === defaultId
  }
}

function normalizeInteraction(interaction: AgentInteraction | null | undefined): AgentUserInteraction | null {
  if (!interaction || interaction.status !== 'pending') return null
  if (interaction.kind === 'target_selection') {
    const payload = interaction.payload as TargetSelectionPayload
    return {
      id: interaction.interaction_id,
      kind: interaction.kind,
      title: '请选择要继续处理的论文',
      description: '找到了多个可能的目标论文，请选择其中一篇继续。',
      confirmLabel: '确认并继续',
      cancelLabel: '取消',
      defaultCandidateId: payload.recommended_candidate_id || '',
      candidates: payload.candidates.map((candidate, index) => normalizeCandidate(candidate, index, payload.recommended_candidate_id || ''))
    }
  }
  const payload = interaction.payload as SideEffectApprovalPayload
  return {
    id: interaction.interaction_id,
    kind: interaction.kind,
    title: '需要批准后继续',
    description: payload.reason || `即将执行 ${payload.tool_name}`,
    confirmLabel: '批准执行',
    cancelLabel: '拒绝',
    defaultCandidateId: '',
    candidates: []
  }
}

function normalizePreferenceFeedback(result: AgentPreferenceActionResult | null | undefined) {
  if (!result) return null
  if (result.status === 'failed') return { status: 'failed' as const, message: result.error || result.message || '偏好操作失败，请稍后重试。' }
  if (result.action === 'like') return { status: 'success' as const, message: '已标记为感兴趣' }
  if (result.action === 'dislike') return { status: 'success' as const, message: '已标记为不感兴趣' }
  return { status: 'success' as const, message: result.message || '偏好已更新' }
}

export function toAgentUserResult(response: ArxivSearchResponse | null | undefined, options: ToAgentUserResultOptions = {}): AgentUserResult {
  const papers = (response?.papers || []).map(normalizePaper)
  const queryCapability = normalizeArxivQueryCapability({ capability: response?.query_capability, warnings: response?.warnings || [] })
  const preferenceFeedback = normalizePreferenceFeedback(response?.preference_action_result)
  const interaction = normalizeInteraction(Object.prototype.hasOwnProperty.call(options, 'interaction') ? options.interaction : response?.interaction)
  return {
    papers,
    queryCapability,
    preferenceFeedback,
    interaction,
    hasVisibleContent: Boolean(papers.length || queryCapability || preferenceFeedback || interaction)
  }
}

export function hasVisibleAgentUserResult(result: AgentUserResult | null | undefined) {
  return Boolean(result?.hasVisibleContent)
}
