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
import { normalizeEvidenceSources } from '@/utils/evidence'
import type { RagChatSource } from '@/types/ragChat'

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
  evidenceSources: RagChatSource[]
  citedSourceIds: string[]
  citationWarning: string | null
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

function sideEffectApprovalTitle(payload: SideEffectApprovalPayload) {
  if (payload.tool_name === 'parse_and_index_paper') return '需要确认构建问答索引'
  return '需要批准后继续'
}

function sideEffectApprovalDescription(payload: SideEffectApprovalPayload) {
  const reason = String(payload.reason || '').trim()
  if (payload.tool_name === 'parse_and_index_paper') {
    return '需要下载和解析论文 PDF，并建立全文问答索引；确认后会继续回答原问题。'
  }
  const reasonLabels: Record<string, string> = {
    explicit_user_confirmation_required: '该操作会产生写入或外部调用，需要你确认后再继续。',
    paper_index_missing: '当前论文还没有可用索引，需要确认后先构建索引。',
    paper_index_stale: '当前论文索引可能已过期，需要确认后重新构建。'
  }
  return reasonLabels[reason] || reason || `即将执行 ${payload.tool_name}`
}

function sideEffectApprovalConfirmLabel(payload: SideEffectApprovalPayload) {
  if (payload.tool_name === 'parse_and_index_paper') return '确认构建索引'
  return '批准执行'
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
    title: sideEffectApprovalTitle(payload),
    description: sideEffectApprovalDescription(payload),
    confirmLabel: sideEffectApprovalConfirmLabel(payload),
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
  const paperQaResult = response?.paper_qa_result && typeof response.paper_qa_result === 'object'
    ? response.paper_qa_result
    : null
  const paperQaArxivId = String(paperQaResult?.arxiv_id || response?.resolved_paper?.arxiv_id || '').trim()
  const evidenceSources = normalizeEvidenceSources(paperQaResult?.sources, paperQaArxivId)
  // Paper QA 嵌套字段是主要契约；顶层字段用于统一 Agent 出站结果，不代表回退到旧格式。
  const citedSourceIds = Array.isArray(paperQaResult?.cited_source_ids)
    ? paperQaResult.cited_source_ids.map(value => String(value))
    : Array.isArray(response?.cited_source_ids)
      ? response.cited_source_ids.map(value => String(value))
      : []
  const citationWarning = typeof paperQaResult?.citation_warning === 'string'
    ? paperQaResult.citation_warning
    : typeof response?.citation_warning === 'string' ? response.citation_warning : null
  return {
    papers,
    queryCapability,
    preferenceFeedback,
    interaction,
    evidenceSources,
    citedSourceIds,
    citationWarning,
    hasVisibleContent: Boolean(papers.length || queryCapability || preferenceFeedback || interaction || evidenceSources.length)
  }
}

export function hasVisibleAgentUserResult(result: AgentUserResult | null | undefined) {
  return Boolean(result?.hasVisibleContent)
}
