import type { PaperLabel } from '@/types/paper'

export function getLabelText(label?: PaperLabel | null) {
  if (label === 'liked') return '感兴趣'
  if (label === 'disliked') return '不感兴趣'
  return ''
}
