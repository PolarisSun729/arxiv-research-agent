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
assert.match(source, /\.labeled-card\s*\{[\s\S]*?height:\s*340px/)
assert.match(source, /\.paper-summary\s*\{[\s\S]*?overflow-y:\s*auto/)

console.log('labeled paper summaries are not truncated')
