// 0~1 的分数转成 0~100 的整数百分比；非数字按 0 处理，避免进度条拿到 NaN。
export function toPercent(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 0
  return Math.max(0, Math.min(100, Math.round(value * 100)))
}

export function formatQueryList(values?: string[]) {
  if (!values || values.length === 0) {
    return '无'
  }
  return values.join(' | ')
}

// SQLite CURRENT_TIMESTAMP 是不带时区的 UTC 文本，直接 new Date 会被当成本地时间，这里补上 Z。
export function normalizeServerTimestamp(value: string) {
  return /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(value) ? `${value.replace(' ', 'T')}Z` : value
}
