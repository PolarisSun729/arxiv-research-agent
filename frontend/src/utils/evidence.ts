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
