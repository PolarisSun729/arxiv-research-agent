import type { PaperLabel } from '@/types/paper'

export function getLabelClass(label?: PaperLabel | null) {
  if (label === 'liked') return 'el-tag--success'
  if (label === 'disliked') return 'el-tag--danger'
  return ''
}

export function getLabelText(label?: PaperLabel | null) {
  if (label === 'liked') return '感兴趣'
  if (label === 'disliked') return '不感兴趣'
  return ''
}
