import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { createRequire } from 'node:module'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

const outfile = join(tmpdir(), `answer-figures-${Date.now()}.cjs`)

// markdown-it 依赖 Node 内置模块，打成 CJS 才能在打包产物里正常 require。
await build({
  stdin: {
    contents: "export { renderAnswerSegments } from './markdown'\nexport { getFigureCaption } from './evidence'",
    resolveDir: 'src/utils',
    loader: 'ts'
  },
  bundle: true,
  format: 'cjs',
  platform: 'node',
  outfile,
  write: true,
  // KaTeX 样式只在浏览器里生效，测试里按空模块处理。
  loader: { '.css': 'empty' }
})

const { renderAnswerSegments, getFigureCaption } = createRequire(import.meta.url)(outfile)

const figure = {
  source_id: '52',
  chunk_type: 'figure',
  page_number: '2',
  content: 'Figure evidence\nFigure 1: The Paradigm Shift from Passive Retrieval. (907x364) page 2',
  asset_url: '/api/paper/2610.11314/evidence-assets/52'
}
const text = { source_id: 't1', chunk_type: 'text', page_number: '1', content: 'intro' }
const sources = [figure, text]

// 图卡跟在第一次引用它的顶层块后面，之后再次引用不重复插图；编号与正文角标一致。
const segments = renderAnswerSegments(
  '背景介绍 [source:t1]。\n\n如图 1 所示，范式从被动检索转向主动重构 [source:52]。\n\n- 列表项 [source:52]\n- 另一项 [source:t1]',
  undefined,
  { sources, citedSourceIds: ['t1', '52'] }
)
assert.equal(segments.length, 2)
assert.match(segments[0].html, /背景介绍/)
assert.match(segments[0].html, /主动重构/)
assert.deepEqual(segments[0].figures.map(item => [item.source.source_id, item.label]), [['52', '2']])
assert.match(segments[1].html, /<ul>/)
assert.deepEqual(segments[1].figures, [])

// 列表内引用图：整个列表作为一个顶层块，图卡放在列表之后，不把列表拆断。
const listSegments = renderAnswerSegments('- 第一项 [source:52]\n- 第二项 [source:t1]\n\n结尾。', undefined, { sources })
assert.equal(listSegments.length, 2)
assert.match(listSegments[0].html, /第二项/)
assert.equal(listSegments[0].figures.length, 1)

// 没有带 asset_url 的图片证据时整篇只渲染成一段。
const plain = renderAnswerSegments('a [source:t1]\n\nb', undefined, { sources: [text, { ...figure, asset_url: undefined }] })
assert.equal(plain.length, 1)
assert.deepEqual(plain[0].figures, [])

// 流式阶段还没有证据列表：不插图，也不丢正文。
assert.equal(renderAnswerSegments('x [source:52]').length, 1)
assert.match(renderAnswerSegments('')[0].html, /回答生成中/)

assert.equal(getFigureCaption(figure), 'Figure 1: The Paradigm Shift from Passive Retrieval.')
assert.equal(getFigureCaption({ ...figure, asset_summary: '模型结构图' }), '模型结构图')

console.log('answer figure tests passed')
