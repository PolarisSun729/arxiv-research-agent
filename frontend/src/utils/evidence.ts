import type { RagChatSource } from '@/types/ragChat'

export function buildEvidenceAssetUrl(arxivId: string, sourceId: string) {
  const paperId = String(arxivId || '').trim()
  const evidenceId = String(sourceId || '').trim()
  if (!paperId || !evidenceId) return ''
  return `/api/paper/${encodeURIComponent(paperId)}/evidence-assets/${encodeURIComponent(evidenceId)}`
}

export function normalizeEvidenceSources(rawSources: unknown, arxivId = ''): RagChatSource[] {
  if (!Array.isArray(rawSources)) return []
  return rawSources
    .map(raw => {
      const source = raw && typeof raw === 'object' ? raw as Record<string, any> : {}
      // 证据必须由后端分配稳定 source_id；缺少 ID 的旧结构不可追踪，也不应伪造成可引用证据。
      const sourceId = String(source.source_id || '').trim()
      const chunkType = String(source.chunk_type || 'text').trim().toLowerCase()
      const assetUrl = String(source.asset_url || '').trim() || (
        chunkType === 'figure' ? buildEvidenceAssetUrl(arxivId, sourceId) : ''
      )
      return {
        source_id: sourceId,
        content: String(source.content || source.asset_summary || source.asset_preview_text || ''),
        page_number: String(source.page_number || ''),
        source: source.source,
        section_title: source.section_title,
        section_path: source.section_path,
        parent_chunk_id: source.parent_chunk_id,
        chunk_type: chunkType,
        asset_summary: source.asset_summary,
        asset_preview_text: source.asset_preview_text,
        asset_url: assetUrl || undefined
      }
    })
    .filter(source => Boolean(source.source_id))
}

export function extractCitedSourceIds(answer: string, sources: RagChatSource[]) {
  const known = new Set(sources.map(source => source.source_id))
  const cited: string[] = []
  for (const match of String(answer || '').matchAll(/\[source:([^\]\s]+)\]/g)) {
    const sourceId = match[1]
    if (known.has(sourceId) && !cited.includes(sourceId)) cited.push(sourceId)
  }
  return cited
}

/**
 * 回答引用的展示顺序：先按正文中首次出现的顺序，再补上后端声明但正文未出现的引用。
 * 正文角标编号与引用列表编号都用这一顺序，保证"[2]"点过去就是列表里的第 2 条。
 */
export function orderCitedSourceIds(answer: string, sources: RagChatSource[], citedSourceIds: string[] = []) {
  const known = new Set(sources.map(source => source.source_id))
  const ordered = extractCitedSourceIds(answer, sources)
  for (const sourceId of citedSourceIds) {
    if (known.has(sourceId) && !ordered.includes(sourceId)) ordered.push(sourceId)
  }
  return ordered
}

const CITATION_MARKER_RE = /\[source:([^\]\s]+)\]/g
const CITATION_TITLE_PREVIEW_CHARS = 120
export const EVIDENCE_LINK_PREFIX = '#evidence-source-'

function describeCitationSource(sourceId: string, source?: RagChatSource) {
  if (!source) return `证据 ${sourceId}`
  const parts: string[] = []
  const pageNumber = String(source.page_number || '').trim()
  if (pageNumber && pageNumber.toUpperCase() !== 'N/A') parts.push(`第 ${pageNumber} 页`)
  const sectionTitle = String(source.section_title || '').trim()
  if (sectionTitle) parts.push(sectionTitle)
  const preview = String(source.content || '').replace(/\s+/g, ' ').trim()
  if (preview) {
    parts.push(preview.length > CITATION_TITLE_PREVIEW_CHARS ? `${preview.slice(0, CITATION_TITLE_PREVIEW_CHARS)}…` : preview)
  }
  return parts.join(' · ') || `证据 ${sourceId}`
}

/**
 * 把正文里的 [source:id] 换成 Markdown 编号链接 "[1]"，悬停显示页码、章节与原文预览；
 * 聊天面板通过 #evidence-source-{id} 事件代理定位证据。已知证据列表时，指向列表外证据的标记会被去掉，
 * 免得渲染出点了没反应的角标；流式输出阶段还没有证据列表，就按出现顺序编号。
 */
export function toCitationLinks(text: string, options: { sources?: RagChatSource[], citedSourceIds?: string[] } = {}) {
  const sources = options.sources || []
  const sourceById = new Map(sources.map(source => [source.source_id, source]))
  const order = sources.length ? orderCitedSourceIds(text, sources, options.citedSourceIds) : []
  return text.replace(CITATION_MARKER_RE, (_match, sourceId: string) => {
    if (sources.length && !sourceById.has(sourceId)) return ''
    let index = order.indexOf(sourceId)
    if (index < 0) {
      order.push(sourceId)
      index = order.length - 1
    }
    const title = describeCitationSource(sourceId, sourceById.get(sourceId))
      .replace(/\\/g, '\\\\')
      .replace(/"/g, '\\"')
    return `[\\[${index + 1}\\]](${EVIDENCE_LINK_PREFIX}${encodeURIComponent(sourceId)} "${title}")`
  })
}

/**
 * 图片证据的图注：优先用资产摘要；否则取 chunk 正文，去掉 "Figure evidence" 类型行和末尾的 "(宽x高) page N"。
 */
export function getFigureCaption(source: RagChatSource) {
  const summary = String(source.asset_summary || '').trim()
  const lines = (summary || String(source.content || '')).split('\n').map(line => line.trim()).filter(Boolean)
  if (lines.length > 1 && /^figure evidence$/i.test(lines[0])) lines.shift()
  return lines.join(' ').replace(/\s*\(\d+\s*x\s*\d+\)\s*(page\s*\S+)?\s*$/i, '').trim()
}
