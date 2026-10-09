import { ApiError, normalizeApiError } from '@/api/errors'
import type {
  ArxivQueryCapabilityStatus,
  BackendArxivQueryCapability,
  NormalizedArxivQueryCapability
} from '@/types/arxivCapability'

const LOCAL_ARXIV_ERROR_CODES = new Set([
  'local_arxiv_search_error',
  'unsupported_local_arxiv_query',
  'local_search_index_unavailable'
])

const DEFAULT_SUPPORTED_FIELDS = ['id', 'cat', 'submittedDate', 'ti', 'abs', 'au', 'all']
const DEFAULT_UNSUPPORTED_SYNTAX = ['通配符 * / ?', '未声明字段', '完整远程 arXiv API 语法']

interface ArxivCapabilityInput {
  capability?: unknown
  warnings?: unknown
  status?: ArxivQueryCapabilityStatus
  errorCode?: string | null
  errorMessage?: string | null
  errorDetail?: string | null
  errorDetails?: unknown
}

interface AgentToolCallLike {
  trace?: Record<string, unknown> | null
  error?: Record<string, unknown> | null
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}

function readString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function readStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.map(item => String(item || '').trim()).filter(Boolean)
}

function readBoolean(value: unknown, fallback: boolean) {
  return typeof value === 'boolean' ? value : fallback
}

function dedupe(items: string[]): string[] {
  return Array.from(new Set(items.map(item => item.trim()).filter(Boolean)))
}

function hasArxivCapabilityWarning(warnings: string[]): boolean {
  return warnings.some(warning => /arxiv|oai|查询|语法|本地|relevance/i.test(warning))
}

function formatSourceLabel(source: string) {
  if (source === 'local_oai') return '本地 OAI 镜像库'
  if (source === 'api' || source === 'remote_arxiv_api') return '远程 arXiv API'
  return source || 'arXiv 数据源'
}

function formatModeLabel(mode: string) {
  if (mode === 'unsupported') return '查询未执行'
  if (mode.startsWith('local_oai')) return '高精度语法子集'
  return mode || '未知模式'
}

function getSupportedFields(supportedSubset: string[]) {
  const subset = supportedSubset.length ? supportedSubset : DEFAULT_SUPPORTED_FIELDS
  const fields = subset.filter(item => DEFAULT_SUPPORTED_FIELDS.includes(item))
  return fields.length ? fields : DEFAULT_SUPPORTED_FIELDS
}

function getSupportedOperators(supportedSubset: string[]) {
  const subset = supportedSubset.length ? supportedSubset : ['AND', 'OR', 'ANDNOT', 'phrase']
  const operators = subset.filter(item => ['AND', 'OR', 'ANDNOT'].includes(item))
  if (subset.includes('phrase')) {
    operators.push('短语引号')
  }
  // 后端 parser 支持括号分组，但它不是字段 token；这里作为用户可执行契约补充展示。
  operators.push('括号分组')
  return dedupe(operators)
}

function buildDetailLines(params: {
  source: string
  sourceLabel: string
  fullArxivSyntaxSupported: boolean
  unsupportedReason?: string | null
  suggestedAction?: string | null
}) {
  const lines: string[] = []
  if (params.source === 'local_oai') {
    lines.push('当前使用本地 OAI 镜像库，不会自动切换到远程 arXiv API。')
    lines.push('本地模式只执行可稳定解析的高精度查询子集，宁可少召回也避免误召回。')
  } else {
    lines.push(`当前使用 ${params.sourceLabel}。`)
  }
  lines.push(`完整远程 arXiv API 语法：${params.fullArxivSyntaxSupported ? '支持' : '不支持'}。`)
  if (params.unsupportedReason) {
    lines.push(`未执行原因：${params.unsupportedReason}`)
  }
  if (params.suggestedAction) {
    lines.push(`建议处理：${params.suggestedAction}`)
  }
  return lines
}

export function normalizeArxivQueryCapability(input: ArxivCapabilityInput = {}): NormalizedArxivQueryCapability | null {
  const errorDetails = asRecord(input.errorDetails)
  const capabilityFromDetails = asRecord(errorDetails?.query_capability)
  const capability = (asRecord(input.capability) || capabilityFromDetails) as BackendArxivQueryCapability | null
  const warnings = dedupe(readStringArray(input.warnings))
  const errorCode = readString(input.errorCode)
  const errorMessage = readString(input.errorMessage)

  const hasLocalError = Boolean(errorCode && LOCAL_ARXIV_ERROR_CODES.has(errorCode))
  // 没有能力契约、没有 arXiv 相关 warning、也不是本地 arXiv 错误时，不制造提示噪声。
  if (!capability && !hasLocalError && !hasArxivCapabilityWarning(warnings)) {
    return null
  }

  const source = readString(capability?.source) || readString(errorDetails?.source) || 'local_oai'
  const mode = readString(capability?.mode) || (hasLocalError ? 'unsupported' : 'local_oai')
  const supportedSubset = readStringArray(capability?.supported_subset)
  const fullArxivSyntaxSupported = readBoolean(capability?.full_arxiv_syntax_supported, false)
  const unsupportedReason =
    readString(capability?.unsupported_reason) ||
    readString(errorDetails?.reason) ||
    readString(input.errorDetail)
  const suggestedAction =
    readString(capability?.suggested_action) ||
    readString(errorDetails?.suggested_action) ||
    (hasLocalError ? '请改写为本地支持字段，或由后端配置切换为远程 arXiv API 后重试。' : null)

  const status = input.status ||
    (hasLocalError || mode === 'unsupported' || unsupportedReason ? 'error' : warnings.length ? 'warning' : 'info')
  const sourceLabel = formatSourceLabel(source)
  const modeLabel = formatModeLabel(mode)
  const summary = status === 'error'
    ? `${sourceLabel}未执行这次查询：${unsupportedReason || errorMessage || '查询语法超出本地支持范围'}`
    : `${sourceLabel}正在按高精度 arXiv 查询语法子集执行，不等价完整远程 arXiv API。`

  return {
    status,
    source,
    sourceLabel,
    mode,
    modeLabel,
    summary,
    detailLines: buildDetailLines({
      source,
      sourceLabel,
      fullArxivSyntaxSupported,
      unsupportedReason,
      suggestedAction
    }),
    supportedFields: getSupportedFields(supportedSubset),
    supportedOperators: getSupportedOperators(supportedSubset),
    unsupportedSyntax: DEFAULT_UNSUPPORTED_SYNTAX,
    warnings,
    unsupportedReason,
    suggestedAction,
    fullArxivSyntaxSupported,
    raw: capability,
    errorCode,
    errorMessage
  }
}

export function normalizeArxivQueryCapabilityFromError(error: unknown): NormalizedArxivQueryCapability | null {
  const payload = error instanceof ApiError
    ? error.payload
    : normalizeApiError(error, 'arXiv 搜索失败，请稍后重试。')
  const details = asRecord(payload.details)
  const capability = asRecord(details?.query_capability)
  const code = String(payload.code || '')

  if (!capability && !LOCAL_ARXIV_ERROR_CODES.has(code)) {
    return null
  }

  return normalizeArxivQueryCapability({
    capability,
    status: 'error',
    errorCode: code,
    errorMessage: payload.message,
    errorDetail: payload.detail || null,
    errorDetails: details
  })
}

export function normalizeArxivQueryCapabilityFromToolCalls(
  toolCalls: AgentToolCallLike[],
  warnings: unknown = []
): NormalizedArxivQueryCapability | null {
  for (let index = toolCalls.length - 1; index >= 0; index -= 1) {
    const call = toolCalls[index]
    const traceCapability = asRecord(call.trace?.query_capability)
    if (traceCapability) {
      return normalizeArxivQueryCapability({
        capability: traceCapability,
        warnings
      })
    }

    const error = asRecord(call.error)
    const detail = asRecord(error?.detail)
    const details = asRecord(detail?.details)
    const errorCapability = asRecord(details?.query_capability)
    const errorCode = readString(error?.code) || readString(detail?.code)

    if (errorCapability || (errorCode && LOCAL_ARXIV_ERROR_CODES.has(errorCode))) {
      return normalizeArxivQueryCapability({
        capability: errorCapability,
        warnings,
        status: 'error',
        errorCode,
        errorMessage: readString(error?.message) || readString(detail?.message),
        errorDetails: details
      })
    }
  }

  return normalizeArxivQueryCapability({ warnings })
}
