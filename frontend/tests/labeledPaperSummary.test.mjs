import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const currentFile = fileURLToPath(import.meta.url)
const viewPath = path.resolve(path.dirname(currentFile), '../src/views/LabeledPapers.vue')
const source = fs.readFileSync(viewPath, 'utf8')

// 已标记页的摘要必须完整交给浏览器排版，不能在展示层静默丢失论文原文。
assert.doesNotMatch(source, /slice\(0,\s*100\)/)
assert.match(source, /formatPaperSummary\(paper\.summary\)/)
// 摘要默认折叠为三行，但全文始终渲染在 DOM 中，并可按卡片展开查看。
assert.match(source, /class="\['paper-summary', \{ 'clamp-3': !expandedIds\.has\(paper\.id\) \}\]"/)
assert.match(source, /@click="toggleExpanded\(paper\.id\)"/)

console.log('labeled paper summaries are not truncated')
