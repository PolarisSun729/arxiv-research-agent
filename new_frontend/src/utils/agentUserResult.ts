import type {
  AgentPaper,
  AgentPendingAction,
  AgentPreferenceActionResult,
  AgentQaIndexJob,
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
  status: string
  title: string
  description: string
  confirmLabel: string
  cancelLabel: string
  isConfirming: boolean
  isBuildingIndex: boolean
  defaultCandidateId: string
  candidates: AgentUserPendingActionCandidate[]
  indexJob: AgentQaIndexJob | null
  indexProgress: number
  indexStageText: string
  indexErrorMessage: string
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

function clampProgress(value: unknown) {
  const numeric = Number(value || 0)
  if (!Number.isFinite(numeric)) return 0
  return Math.max(0, Math.min(100, Math.round(numeric)))
}

function qaIndexStageText(stage: string) {
  const normalized = String(stage || '').trim()
  const labels: Record<string, string> = {
    pending: '等待后台任务',
    starting: '启动索引任务',
    validate_loading_method: '校验解析方式',
    create_build_version: '创建索引版本',
    mark_index_processing: '标记处理中',
    load_paper_metadata: '读取论文元数据',
    download_pdf: '下载 PDF',
    load_document: '解析 PDF',
    chunk_document: '切分全文内容',
    build_retrieval_indexes: '生成检索索引',
    create_embeddings: '生成向量',
    index_embeddings_to_vector_store: '写入向量库',
    activate_index: '激活索引版本',
    mark_index_success: '索引完成',
    failed: '构建失败',
    stale: '任务超时'
  }
  return labels[normalized] || normalized || '准备构建索引'
}

function isIndexBuildAction(action: AgentPendingAction) {
  const toolName = String(action.tool_name || action.confirmation_request?.tool_name || '').trim()
  return toolName === 'parse_and_index_paper'
}

function normalizePendingAction(action: AgentPendingAction | null | undefined): AgentUserPendingAction | null {
  const status = String(action?.status || '').trim()
  if (!action || !['waiting_confirmation', 'confirming', 'index_building', 'index_failed', 'ready_to_resume'].includes(status)) return null

  const isPaperTarget = isPaperTargetConfirmation(action)
  const rawCandidates = Array.isArray(action.candidates) ? action.candidates : []
  const recommended = action.recommended_candidate || action.target_paper || rawCandidates[0]
  const defaultCandidateId = action.default_candidate_id || (recommended ? getPaperTargetCandidateId(recommended) : '')
  const candidates = rawCandidates.map((candidate, index) => normalizeCandidate(candidate, index, defaultCandidateId))
  const indexJob = action.index_job || action.index_continuation?.job || null
  const isBuildingIndex = status === 'index_building'
  const isIndexFailed = status === 'index_failed'
  const isReadyToResume = status === 'ready_to_resume'
  const indexProgress = clampProgress(indexJob?.progress)
  const indexStageText = qaIndexStageText(String(indexJob?.current_stage || indexJob?.status || ''))
  const indexErrorMessage = String(indexJob?.error_message || action.index_continuation?.error_message || '').trim()
  const isIndexAction = isIndexBuildAction(action)
  const baseTitle = action.title || action.title_text || '需要确认后继续'
  const baseDescription = action.description || action.qa_question || action.original_question || '确认后会继续当前论文处理流程。'
  const confirmLabel = isPaperTarget
    ? '确认并继续'
    : isBuildingIndex
      ? '构建中'
      : isIndexFailed
        ? '重试构建索引'
        : isReadyToResume
          ? '继续回答'
          : isIndexAction
            ? '确认构建索引'
            : '确认执行'

  // pending_action 是后端状态流转对象；这里只保留用户做决策所需的信息，避免泄露 tool/session/step 等运行细节。
  return {
    kind: isPaperTarget ? 'paper_target_confirmation' : 'general_confirmation',
    status,
    title: isPaperTarget
      ? '请选择要继续处理的论文'
      : isBuildingIndex
        ? '正在构建问答索引'
        : isIndexFailed
          ? '问答索引构建失败'
          : isReadyToResume
            ? '问答索引已完成'
            : baseTitle,
    description: isPaperTarget
      ? '我找到了多个可能的目标论文，请选择你要继续处理的那一篇。'
      : isBuildingIndex
        ? '正在下载和解析 PDF，并创建全文检索索引；完成后会自动继续回答原问题。'
        : isIndexFailed
          ? (indexErrorMessage || '索引构建失败，可以重试构建或取消本次问答。')
          : isReadyToResume
            ? '索引已经建立，正在继续回答原问题。'
            : baseDescription,
    confirmLabel,
    cancelLabel: isBuildingIndex || isIndexFailed || isReadyToResume ? '取消本次问答' : '取消',
    isConfirming: status === 'confirming',
    isBuildingIndex,
    defaultCandidateId,
    candidates,
    indexJob,
    indexProgress,
    indexStageText,
    indexErrorMessage
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
