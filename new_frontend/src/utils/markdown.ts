import MarkdownIt from 'markdown-it'
import dollarmathPlugin from 'markdown-it-dollarmath'
import { renderToString } from 'katex'
import 'katex/dist/katex.min.css'

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

export function renderMarkdownWithLatex(text: string, emptyText = '\u56de\u7b54\u751f\u6210\u4e2d...') {
  const content = (text || '').replace(/\r\n/g, '\n').trim()
  if (!content) {
    return `<p class="md-empty">${emptyText}</p>`
  }
  // 稳定引用转成可点击链接，聊天面板再通过事件代理定位对应证据。
  const withEvidenceLinks = content.replace(/\[source:([^\]\s]+)\]/g, (_match, sourceId: string) => {
    return `[source:${sourceId}](#evidence-source-${encodeURIComponent(sourceId)})`
  })
  return markdownRenderer.render(withEvidenceLinks)
}
