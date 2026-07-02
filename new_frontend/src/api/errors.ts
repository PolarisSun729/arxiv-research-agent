import type { ApiErrorPayload } from '@/types/error'

const ERROR_MESSAGES: Record<string, string> = {
  request_validation_error: '请求参数不合法，请检查后重试。',
  paper_not_found: '未找到对应论文，请确认论文 ID 是否正确。',
  qa_index_not_found: '这篇论文还没有 QA 索引，请先构建索引。',
  qa_index_build_failed: '论文 QA 索引构建失败，可以重新构建。',
  vector_store_error: '检索服务异常，请稍后重试。',
  llm_generation_failed: '答案生成失败，请稍后重试。',
  aborted: '已取消生成',
  stream_incomplete: '回答中断，请重试。',
  stream_interrupted: '回答中断，请重试。',
  resume_checkpoint_not_found: '原执行现场已失效，请重新发起请求。',
  agent_runtime_error: 'Agent 运行失败，请稍后重试。',
  database_write_failed: '答案保存失败。',
  local_arxiv_search_error: '本地 arXiv 检索失败，请检查查询后重试。',
  unsupported_local_arxiv_query: '本地 OAI 镜像库不支持该查询语法，请按支持字段改写。',
  local_search_index_unavailable: '本地 arXiv 搜索索引不可用，请先重建索引。',
  unknown_error: '请求失败，请稍后重试。'
}

export class ApiError extends Error {
  payload: ApiErrorPayload

  constructor(payload: ApiErrorPayload) {
    super(payload.message)
    this.name = 'ApiError'
    this.payload = payload
  }
}

export function isApiErrorPayload(value: unknown): value is ApiErrorPayload {
  return Boolean(
    value &&
      typeof value === 'object' &&
      (value as { status?: unknown }).status === 'failed' &&
      typeof (value as { code?: unknown }).code === 'string'
  )
}

function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function normalizeDetails(value: unknown): Record<string, any> | null {
  return isRecord(value) ? value : null
}

function normalizeFastApiDetail(value: unknown, fallback: string): ApiErrorPayload | null {
  if (!isRecord(value)) return null
  const detail = isRecord(value.detail) ? value.detail : value
  if (typeof detail.code !== 'string') return null

  const code = String(detail.code || 'unknown_error')
  const message = typeof detail.message === 'string' && detail.message
    ? detail.message
    : fallback
  const rawDetail = typeof detail.detail === 'string'
    ? detail.detail
    : typeof detail.message === 'string' ? detail.message : null

  return {
    status: 'failed',
    code,
    message: getApiErrorMessage(code, message),
    detail: rawDetail,
    // FastAPI 业务错误会把可展示契约放在 detail.details，前端需要保留给页面级 banner。
    details: normalizeDetails(detail.details),
    recoverable: typeof detail.recoverable === 'boolean' ? detail.recoverable : true
  }
}

export function normalizeApiError(value: unknown, fallback = '请求失败，请稍后重试。'): ApiErrorPayload {
  if (isApiErrorPayload(value)) {
    const code = String(value.code || 'unknown_error')
    return {
      status: 'failed',
      code,
      message: getApiErrorMessage(code, value.message || fallback),
      detail: typeof value.detail === 'string' ? value.detail : null,
      details: normalizeDetails(value.details),
      recoverable: Boolean(value.recoverable)
    }
  }

  const fastApiDetail = normalizeFastApiDetail(value, fallback)
  if (fastApiDetail) {
    return fastApiDetail
  }

  if (value && typeof value === 'object' && 'response' in value) {
    const data = (value as { response?: { data?: unknown } }).response?.data
    if (isApiErrorPayload(data)) {
      return normalizeApiError(data, fallback)
    }
    const normalizedFastApiError = normalizeFastApiDetail(data, fallback)
    if (normalizedFastApiError) {
      return normalizedFastApiError
    }
    if (data && typeof data === 'object' && isApiErrorPayload((data as { detail?: unknown }).detail)) {
      return normalizeApiError((data as { detail: unknown }).detail, fallback)
    }
  }

  if (value && typeof value === 'object' && isApiErrorPayload((value as { detail?: unknown }).detail)) {
    return normalizeApiError((value as { detail: unknown }).detail, fallback)
  }

  const nestedFastApiDetail = isRecord(value) ? normalizeFastApiDetail(value.detail, fallback) : null
  if (nestedFastApiDetail) {
    return nestedFastApiDetail
  }

  if (value instanceof Error && value.message) {
    return {
      status: 'failed',
      code: 'unknown_error',
      message: fallback || value.message,
      detail: value.message,
      recoverable: true
    }
  }

  return {
    status: 'failed',
    code: 'unknown_error',
    message: fallback,
    detail: null,
    recoverable: true
  }
}

export function getApiErrorMessage(code: string | null | undefined, fallback = '请求失败，请稍后重试。') {
  if (!code) return fallback
  return ERROR_MESSAGES[code] || fallback
}

export function getErrorMessage(error: unknown, fallback = '请求失败，请稍后重试。') {
  if (error instanceof ApiError) {
    return getApiErrorMessage(error.payload.code, error.payload.message || fallback)
  }
  if (isApiErrorPayload(error)) {
    return getApiErrorMessage(error.code, error.message || fallback)
  }
  if (error && typeof error === 'object' && 'payload' in error && isApiErrorPayload((error as { payload?: unknown }).payload)) {
    const payload = (error as { payload: ApiErrorPayload }).payload
    return getApiErrorMessage(payload.code, payload.message || fallback)
  }
  if (error && typeof error === 'object' && 'message' in error && typeof (error as { message?: unknown }).message === 'string') {
    return (error as { message: string }).message || fallback
  }
  return fallback
}

export async function parseFetchErrorResponse(response: Response, fallback = '请求失败，请稍后重试。'): Promise<ApiError> {
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    try {
      const data = await response.json()
      return new ApiError(normalizeApiError(data, fallback))
    } catch {
      // JSON 解析失败时继续走文本兜底，避免抛出另一个不可识别异常。
    }
  }
  const detail = await response.text()
  return new ApiError({
    status: 'failed',
    code: 'unknown_error',
    message: fallback,
    detail: detail || `Request failed with status ${response.status}`,
    recoverable: true
  })
}
