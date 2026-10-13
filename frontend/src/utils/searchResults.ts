import type { Paper } from '@/types/paper'

// 与后端 ARXIV_SEARCH_MAX_ALLOWED_RESULTS 的默认值一致；多拉不能超过它，否则整次搜索会被 400 拒绝。
export const SEARCH_FETCH_LIMIT = 100

/**
 * 计算实际向后端请求的条数。
 *
 * 返回结果里已标记的论文最多也就 markedCount 篇，所以多拉 markedCount 篇就能保证过滤后仍有 maxResults 篇
 * （库里候选足够时）；标记很多的用户受 SEARCH_FETCH_LIMIT 限制，过滤后可能略少。
 * 用户自己要的条数超过上限时保持原值，让后端照常给出校验错误，而不是悄悄截断。
 */
export function resolveSearchFetchSize(maxResults: number, markedCount: number, limit: number = SEARCH_FETCH_LIMIT) {
  return Math.max(maxResults, Math.min(maxResults + markedCount, limit))
}

/**
 * 从多拉的结果里选出要展示的论文。
 *
 * hiddenCount 只统计「为了凑满 maxResults 而跳过」的已标记论文，不含排在截断位置之后的，
 * 这样提示里的数字和用户实际少看到的论文一一对应。
 */
export function selectSearchResults<T extends Paper>(
  papers: T[],
  markedIds: ReadonlySet<string>,
  maxResults: number,
  hideMarked: boolean
): { items: T[]; hiddenCount: number } {
  if (!hideMarked) {
    return { items: papers.slice(0, maxResults), hiddenCount: 0 }
  }
  const items: T[] = []
  let hiddenCount = 0
  for (const paper of papers) {
    if (items.length >= maxResults) break
    if (markedIds.has(paper.arxivId || paper.id)) {
      hiddenCount += 1
    } else {
      items.push(paper)
    }
  }
  return { items, hiddenCount }
}
