import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { pathToFileURL } from 'node:url'

const outfile = join(tmpdir(), `sse-${Date.now()}.mjs`)

await build({
  entryPoints: ['src/utils/sse.ts'],
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile,
  write: true
})

const { parseSseBlock, readSseBlocks } = await import(pathToFileURL(outfile))

function streamOf(chunks) {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(typeof chunk === 'string' ? encoder.encode(chunk) : chunk)
      controller.close()
    }
  })
}

async function collect(body) {
  const blocks = []
  for await (const block of readSseBlocks(body)) blocks.push(block)
  return blocks
}

// 解析：event 与多行 data 拼接，data 只去掉一个前导空格
assert.deepEqual(parseSseBlock('event: delta\ndata: {"a":1}'), { event: 'delta', dataText: '{"a":1}' })
assert.deepEqual(parseSseBlock('event: x\r\ndata: line1\r\ndata:  line2'), { event: 'x', dataText: 'line1\n line2' })
assert.deepEqual(parseSseBlock('data: only-data'), { event: null, dataText: 'only-data' })
assert.deepEqual(parseSseBlock(': comment\nevent: ping'), { event: 'ping', dataText: '' })

// 读流：多块、跨 chunk 的半个事件、\r\n 分隔
assert.deepEqual(
  await collect(streamOf(['event: a\ndata: 1\n\nevent: b\nda', 'ta: 2\n\n', 'event: c\r\ndata: 3\r\n\r\n'])),
  ['event: a\ndata: 1', 'event: b\ndata: 2', 'event: c\r\ndata: 3']
)

// 读流：没有结尾空行的尾块也要产出；纯空白尾块忽略
assert.deepEqual(await collect(streamOf(['event: a\ndata: 1\n\nevent: tail\ndata: 2'])), ['event: a\ndata: 1', 'event: tail\ndata: 2'])
assert.deepEqual(await collect(streamOf(['event: a\ndata: 1\n\n\n'])), ['event: a\ndata: 1'])

// 读流：多字节 UTF-8 字符被拆在两个 chunk 之间时不能乱码
const bytes = new TextEncoder().encode('event: d\ndata: 中文\n\n')
assert.deepEqual(await collect(streamOf([bytes.slice(0, 17), bytes.slice(17)])), ['event: d\ndata: 中文'])

// 消费方提前退出时释放 reader 锁，流可以被重新获取
const body = streamOf(['event: a\ndata: 1\n\nevent: b\ndata: 2\n\n'])
for await (const block of readSseBlocks(body)) {
  assert.equal(block, 'event: a\ndata: 1')
  break
}
assert.equal(body.locked, false)

console.log('sse tests passed')
