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
