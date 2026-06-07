import type { ApiErrorPayload } from '@/types/error'

const ERROR_MESSAGES: Record<string, string> = {
  request_validation_error: '请求参数不合法，请检查后重试。',
  paper_not_found: '未找到对应论文，请确认论文 ID 是否正确。',
  qa_index_not_found: '这篇论文还没有 QA 索引，请先构建索引。',
  qa_index_build_failed: '论文 QA 索引构建失败，可以重新构建。',
  vector_store_error: '检索服务异常，请稍后重试。',
  llm_generation_failed: '答案生成失败，请稍后重试。',
  resume_checkpoint_not_found: '原执行现场已失效，请重新发起请求。',
  agent_runtime_error: 'Agent 运行失败，请稍后重试。',
  database_write_failed: '保存失败，请稍后重试。',
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

export function normalizeApiError(value: unknown, fallback = '请求失败，请稍后重试。'): ApiErrorPayload {
  if (isApiErrorPayload(value)) {
    const code = String(value.code || 'unknown_error')
    return {
      status: 'failed',
      code,
      message: getApiErrorMessage(code, value.message || fallback),
      detail: typeof value.detail === 'string' ? value.detail : null,
      recoverable: Boolean(value.recoverable)
    }
  }

  if (value && typeof value === 'object' && 'response' in value) {
    const data = (value as { response?: { data?: unknown } }).response?.data
    if (isApiErrorPayload(data)) {
      return normalizeApiError(data, fallback)
    }
    if (data && typeof data === 'object' && isApiErrorPayload((data as { detail?: unknown }).detail)) {
      return normalizeApiError((data as { detail: unknown }).detail, fallback)
    }
  }

  if (value && typeof value === 'object' && isApiErrorPayload((value as { detail?: unknown }).detail)) {
    return normalizeApiError((value as { detail: unknown }).detail, fallback)
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
