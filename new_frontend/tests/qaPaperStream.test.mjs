import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { build } from 'esbuild'

const outfile = join(tmpdir(), `qa-paper-stream-${Date.now()}.mjs`)

await mkdir(tmpdir(), { recursive: true })
await build({
  entryPoints: [fileURLToPath(new URL('../src/api/papers.ts', import.meta.url))],
  bundle: true,
  format: 'esm',
  platform: 'browser',
  outfile,
  treeShaking: true,
  write: true,
  plugins: [
    {
      name: 'qa-stream-test-stubs',
      setup(buildApi) {
        buildApi.onResolve({ filter: /^\.\/request$/ }, () => ({ path: 'request-stub', namespace: 'qa-stub' }))
        buildApi.onResolve({ filter: /^@\/mock\/papers$/ }, () => ({ path: 'mock-papers-stub', namespace: 'qa-stub' }))
        buildApi.onResolve({ filter: /^@\/composables\/useUserContext$/ }, () => ({
          path: 'user-context-stub',
          namespace: 'qa-stub'
        }))
        buildApi.onLoad({ filter: /.*/, namespace: 'qa-stub' }, args => {
          if (args.path === 'request-stub') {
            return { contents: 'export default {}', loader: 'js' }
          }
          if (args.path === 'mock-papers-stub') {
            return {
              contents: 'export const mockPapers = []; export const mockRecommendedPapers = []; export const mockLabeledPapers = []; export const mockStats = {};',
              loader: 'js'
            }
          }
          return { contents: 'export function getCurrentUserId() { return "test-user"; }', loader: 'js' }
        })
      }
    }
  ]
})

const { qaPaperStream } = await import(pathToFileURL(outfile))

function sse(event, data) {
  return `event: ${event}\ndata: ${typeof data === 'string' ? data : JSON.stringify(data)}\n\n`
}

function streamResponse(chunks, options = {}) {
  const encoder = new TextEncoder()
  const signal = options.signal
  let index = 0
  return new Response(
    new ReadableStream({
      pull(controller) {
        if (signal?.aborted) {
          controller.error(new DOMException('aborted', 'AbortError'))
          return
        }
        if (index >= chunks.length) {
          controller.close()
          return
        }
        controller.enqueue(encoder.encode(chunks[index]))
        index += 1
      }
    }),
    { status: 200, headers: { 'content-type': 'text/event-stream' } }
  )
}

globalThis.fetch = async (_url, init = {}) => {
  return streamResponse(globalThis.__qaStreamChunks || [], { signal: init.signal })
}

globalThis.__qaStreamChunks = [
  sse('meta', { status: 'preparing', arxiv_id: '1234.1', question: 'q', sources: [], retrieval_debug: null }),
  sse('delta', { delta: 'hello ' }),
  sse('delta', { delta: 'world' }),
  sse('done', { status: 'success', answer: 'hello world', sources: [] })
]
const completed = await qaPaperStream('1234.1', 'q')
assert.equal(completed.status, 'completed')
assert.equal(completed.answer, 'hello world')
assert.equal(completed.partial, false)
assert.equal(completed.persistence_status, 'saved')

globalThis.__qaStreamChunks = [
  sse('delta', { delta: 'partial answer' })
]
await assert.rejects(
  qaPaperStream('1234.1', 'q'),
  error => error?.payload?.code === 'stream_incomplete'
)

globalThis.__qaStreamChunks = [
  sse('error', {
    status: 'failed',
    code: 'llm_generation_failed',
    message: 'model failed',
    detail: 'timeout',
    recoverable: true
  })
]
await assert.rejects(
  qaPaperStream('1234.1', 'q'),
  error => error?.payload?.code === 'llm_generation_failed'
)

globalThis.fetch = async (_url, init = {}) => {
  const encoder = new TextEncoder()
  return new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(sse('delta', { delta: 'kept' })))
        init.signal?.addEventListener('abort', () => {
          controller.error(new DOMException('aborted', 'AbortError'))
        })
      }
    }),
    { status: 200, headers: { 'content-type': 'text/event-stream' } }
  )
}
const abortController = new AbortController()
const abortedPromise = qaPaperStream('1234.1', 'q', { signal: abortController.signal })
abortController.abort()
await assert.rejects(
  abortedPromise,
  error => error?.payload?.code === 'aborted'
)

globalThis.fetch = async (_url, init = {}) => {
  return streamResponse([
    'event: debug\ndata: {bad json\n\n',
    sse('delta', { delta: 'still works' }),
    sse('done', { status: 'success', answer: 'still works', sources: [] })
  ], { signal: init.signal })
}
const malformedTolerated = await qaPaperStream('1234.1', 'q')
assert.equal(malformedTolerated.status, 'completed')
assert.equal(malformedTolerated.answer, 'still works')

globalThis.fetch = async (_url, init = {}) => {
  return streamResponse([
    sse('delta', { delta: 'generated' }),
    sse('done', {
      status: 'partial_success',
      answer: 'generated',
      sources: [],
      persistence_status: 'failed'
    })
  ], { signal: init.signal })
}
const persistenceFailed = await qaPaperStream('1234.1', 'q')
assert.equal(persistenceFailed.status, 'persistence_failed')
assert.equal(persistenceFailed.answer, 'generated')
assert.equal(persistenceFailed.partial, true)
assert.equal(persistenceFailed.persistence_status, 'failed')

console.log('qa paper stream tests passed')
