import type {
  AgentPaper,
  AgentPendingAction,
  AgentPreferenceActionResult,
  ArxivSearchResponse,
  PaperTargetCandidate
} from '@/types/agent'
import type { NormalizedArxivQueryCapability } from '@/types/arxivCapability'
import type { Paper } from '@/types/paper'
import { normalizeArxivQueryCapability } from '@/utils/arxivQueryCapability'

export interface AgentUserPendingActionCandidate {
  id: string
  title: string
  arxivId: string
  authors: string
  rank: number | null
  sourceLabel: string
  isDefault: boolean
}

export interface AgentUserPendingAction {
  kind: 'paper_target_confirmation' | 'general_confirmation'
  title: string
  description: string
  confirmLabel: string
  cancelLabel: string
  isConfirming: boolean
  defaultCandidateId: string
  candidates: AgentUserPendingActionCandidate[]
}

export interface AgentUserPreferenceFeedback {
  status: 'success' | 'failed'
  message: string
}

export interface AgentUserResult {
  papers: Paper[]
  queryCapability: NormalizedArxivQueryCapability | null
  preferenceFeedback: AgentUserPreferenceFeedback | null
  pendingAction: AgentUserPendingAction | null
  hasVisibleContent: boolean
}

interface ToAgentUserResultOptions {
  pendingAction?: AgentPendingAction | null
}

function normalizePaper(raw: AgentPaper): Paper {
  const arxivId = raw.arxiv_id || raw.arxivId || (raw.id ? String(raw.id).split('/').pop() : '')
  const authors = Array.isArray(raw.authors)
    ? raw.authors
    : String(raw.authors || '')
        .split(',')
        .map(author => author.trim())
        .filter(Boolean)
  const categories = Array.isArray(raw.categories)
    ? raw.categories
    : String(raw.categories || '')
        .split(',')
        .map(category => category.trim())
        .filter(Boolean)

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

export function getAgentPendingActionKey(action: AgentPendingAction | Record<string, any> | null | undefined) {
  if (!action || typeof action !== 'object') return ''
  const pendingActionId = String(action.pending_action_id || action.confirmation_request?.pending_action_id || '').trim()
  if (pendingActionId) return `pending:${pendingActionId}`
  const sessionId = String(action.session_id || action.confirmation_request?.session_id || '').trim()
  const stepId = String(action.step_id || action.confirmation_request?.step_id || '').trim()
  const toolName = String(action.tool_name || action.confirmation_request?.tool_name || '').trim()
  return [sessionId, stepId, toolName].some(Boolean) ? `step:${sessionId}:${stepId}:${toolName}` : ''
}

export function isPaperTargetConfirmation(action: AgentPendingAction | null | undefined) {
  return action?.request_type === 'paper_target_confirmation' || action?.type === 'paper_target_confirmation'
}

export function getPaperTargetCandidateId(candidate: PaperTargetCandidate, index = 0) {
  return String(
    candidate.candidate_id ||
    candidate.paper_id ||
    candidate.arxiv_id ||
    candidate.arxivId ||
    candidate.id ||
    candidate.title ||
    `candidate-${index}`
  )
}

function candidateAuthors(candidate: PaperTargetCandidate) {
  if (candidate.authors_summary) return candidate.authors_summary
  if (Array.isArray(candidate.authors)) return candidate.authors.slice(0, 3).join(', ')
  return candidate.authors || ''
}

function candidateSource(candidate: PaperTargetCandidate) {
  return candidate.source_label || candidate.list_name || candidate.source_type || candidate.source || '上下文候选'
}

function normalizeCandidate(
  candidate: PaperTargetCandidate,
  index: number,
  defaultCandidateId: string
): AgentUserPendingActionCandidate {
  const id = getPaperTargetCandidateId(candidate, index)
  return {
    id,
    title: candidate.title || candidate.arxiv_id || candidate.arxivId || id,
    arxivId: candidate.arxiv_id || candidate.arxivId || candidate.id || '',
    authors: candidateAuthors(candidate),
    rank: typeof candidate.rank === 'number' ? candidate.rank : null,
    sourceLabel: candidateSource(candidate),
    isDefault: Boolean(defaultCandidateId && id === defaultCandidateId)
  }
}

function normalizePreferenceFeedback(
  result: AgentPreferenceActionResult | null | undefined
): AgentUserPreferenceFeedback | null {
  if (!result) return null
  if (result.status === 'failed') {
    return {
      status: 'failed',
      message: result.error || result.message || '偏好操作失败，请稍后重试。'
    }
  }
  if (result.action === 'like') return { status: 'success', message: '已标记为感兴趣' }
  if (result.action === 'dislike') return { status: 'success', message: '已标记为不感兴趣' }
  if (result.action === 'remove') return { status: 'success', message: '已取消偏好标记' }
  return { status: 'success', message: result.message || '偏好已更新' }
}

function normalizePendingAction(action: AgentPendingAction | null | undefined): AgentUserPendingAction | null {
  if (!action || (action.status !== 'waiting_confirmation' && action.status !== 'confirming')) return null

  const isPaperTarget = isPaperTargetConfirmation(action)
  const rawCandidates = Array.isArray(action.candidates) ? action.candidates : []
  const recommended = action.recommended_candidate || action.target_paper || rawCandidates[0]
  const defaultCandidateId = action.default_candidate_id || (recommended ? getPaperTargetCandidateId(recommended) : '')
  const candidates = rawCandidates.map((candidate, index) => normalizeCandidate(candidate, index, defaultCandidateId))

  // pending_action 是后端状态流转对象；这里只保留用户做决策所需的信息，避免泄露 tool/session/step 等运行细节。
  return {
    kind: isPaperTarget ? 'paper_target_confirmation' : 'general_confirmation',
    title: isPaperTarget
      ? '请选择要继续处理的论文'
      : action.title || action.title_text || '需要确认后继续',
    description: isPaperTarget
      ? '我找到了多个可能的目标论文，请选择你要继续处理的那一篇。'
      : action.qa_question || action.original_question || action.description || '确认后会继续当前论文处理流程。',
    confirmLabel: isPaperTarget ? '确认并继续' : '确认执行',
    cancelLabel: '取消',
    isConfirming: action.status === 'confirming',
    defaultCandidateId,
    candidates
  }
}

export function toAgentUserResult(
  response: ArxivSearchResponse | null | undefined,
  options: ToAgentUserResultOptions = {}
): AgentUserResult {
  const papers = (response?.papers || []).map(normalizePaper)
  const queryCapability = normalizeArxivQueryCapability({
    capability: response?.query_capability,
    warnings: response?.warnings || []
  })
  const preferenceFeedback = normalizePreferenceFeedback(response?.preference_action_result)
  // 用户界面只接收调用方确认过的 pending action；这样已消费的旧确认卡不会停留在历史消息里。
  const pendingActionSource = Object.prototype.hasOwnProperty.call(options, 'pendingAction')
    ? options.pendingAction
    : response?.pending_action ?? null
  const pendingAction = normalizePendingAction(pendingActionSource)
  const hasVisibleContent = Boolean(
    papers.length ||
    queryCapability ||
    preferenceFeedback ||
    pendingAction
  )

  return {
    papers,
    queryCapability,
    preferenceFeedback,
    pendingAction,
    hasVisibleContent
  }
}

export function hasVisibleAgentUserResult(result: AgentUserResult | null | undefined) {
  return Boolean(result?.hasVisibleContent)
}
