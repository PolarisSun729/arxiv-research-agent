import MarkdownIt from 'markdown-it'
import dollarmathPlugin from 'markdown-it-dollarmath'
import { renderToString } from 'katex'
import 'katex/dist/katex.min.css'
import type { RagChatSource } from '@/types/ragChat'
import { toCitationLinks } from '@/utils/evidence'

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

export function renderMarkdownWithLatex(
  text: string,
  emptyText = '回答生成中...',
  citation: { sources?: RagChatSource[], citedSourceIds?: string[] } = {}
) {
  const content = (text || '').replace(/\r\n/g, '\n').trim()
  if (!content) {
    return `<p class="md-empty">${emptyText}</p>`
  }
  // 稳定引用转成编号链接，聊天面板再通过事件代理定位对应证据。
  return markdownRenderer.render(toCitationLinks(content, citation))
}
