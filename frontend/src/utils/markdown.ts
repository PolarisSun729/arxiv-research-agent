import MarkdownIt from 'markdown-it'
import dollarmathPlugin from 'markdown-it-dollarmath'
import { renderToString } from 'katex'
import 'katex/dist/katex.min.css'
import type { RagChatSource } from '@/types/ragChat'
import { EVIDENCE_LINK_PREFIX, toCitationLinks } from '@/utils/evidence'

const markdownRenderer = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  typographer: true
}).use(dollarmathPlugin, {
  allow_space: true,
  allow_digits: true,
  double_inline: true,
  allow_labels: true,
  renderer(content: string, { displayMode }: { displayMode: boolean }) {
    return renderToString(content, {
      displayMode,
      throwOnError: false
    })
  }
})

export interface AnswerFigure {
  source: RagChatSource
  /** 与正文角标一致的引用编号。 */
  label: string
}

export interface AnswerSegment {
  html: string
  /** 在这一段之后展示的图片证据；每张图只跟在它第一次被引用的段落后面。 */
  figures: AnswerFigure[]
}

type MarkdownToken = ReturnType<typeof markdownRenderer.parse>[number]

function collectCitedFigures(
  token: MarkdownToken,
  figureById: Map<string, RagChatSource>,
  shown: Set<string>
): AnswerFigure[] {
  const figures: AnswerFigure[] = []
  const children = token.children || []
  children.forEach((child, index) => {
    const href = child.type === 'link_open' ? child.attrGet('href') || '' : ''
    if (!href.startsWith(EVIDENCE_LINK_PREFIX)) return
    const sourceId = decodeURIComponent(href.slice(EVIDENCE_LINK_PREFIX.length))
    const source = figureById.get(sourceId)
    if (!source || shown.has(sourceId)) return
    shown.add(sourceId)
    figures.push({ source, label: (children[index + 1]?.content || '').replace(/[[\]]/g, '') })
  })
  return figures
}

/**
 * 渲染回答并把引用到的图片证据插到正文里：在第一次引用某张图的顶层块（段落、列表、表格等）之后切段，
 * 由面板在段与段之间放图卡。图放在哪里只由“哪句话引用了哪条证据”决定，模型不输出任何图片地址。
 * 没有可展示的图片证据时只返回一段，即整篇回答的普通 Markdown 渲染结果。
 */
export function renderAnswerSegments(
  text: string,
  emptyText = '回答生成中...',
  citation: { sources?: RagChatSource[], citedSourceIds?: string[] } = {}
): AnswerSegment[] {
  const content = (text || '').replace(/\r\n/g, '\n').trim()
  if (!content) return [{ html: `<p class="md-empty">${emptyText}</p>`, figures: [] }]

  const env = {}
  const tokens = markdownRenderer.parse(toCitationLinks(content, citation), env)
  const render = (slice: MarkdownToken[]) => markdownRenderer.renderer.render(slice, markdownRenderer.options, env)
  const figureById = new Map(
    (citation.sources || [])
      .filter(source => source.chunk_type === 'figure' && source.asset_url)
      .map(source => [source.source_id, source])
  )
  if (!figureById.size) return [{ html: render(tokens), figures: [] }]

  const segments: AnswerSegment[] = []
  const shown = new Set<string>()
  let segmentStart = 0
  let pending: AnswerFigure[] = []
  tokens.forEach((token, index) => {
    if (token.type === 'inline') pending.push(...collectCitedFigures(token, figureById, shown))
    // level 0 且不是开标签，说明一个顶层块到此结束；块内引用了新图就在这里切段。
    if (token.level === 0 && token.nesting <= 0 && pending.length) {
      segments.push({ html: render(tokens.slice(segmentStart, index + 1)), figures: pending })
      segmentStart = index + 1
      pending = []
    }
  })
  if (segmentStart < tokens.length) segments.push({ html: render(tokens.slice(segmentStart)), figures: [] })
  return segments
}
