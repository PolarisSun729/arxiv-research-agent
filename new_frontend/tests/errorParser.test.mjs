import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { transform } from 'esbuild'

const sourceUrl = new URL('../src/api/errors.ts', import.meta.url)
const source = await readFile(sourceUrl, 'utf8')
const compiled = await transform(source, {
  loader: 'ts',
  format: 'esm',
  sourcemap: 'inline'
})

const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled.code).toString('base64')}`
const { ApiError, getErrorMessage, normalizeApiError } = await import(moduleUrl)

const qaIndexError = normalizeApiError({
  status: 'failed',
  code: 'qa_index_not_found',
  message: 'backend message',
  detail: 'missing index',
  recoverable: true
})

assert.equal(qaIndexError.code, 'qa_index_not_found')
assert.equal(qaIndexError.message, '这篇论文还没有 QA 索引，请先构建索引。')
assert.equal(qaIndexError.recoverable, true)

const wrapped = new ApiError(qaIndexError)
assert.equal(getErrorMessage(wrapped, 'fallback'), '这篇论文还没有 QA 索引，请先构建索引。')

const axiosLike = normalizeApiError({
  response: {
    data: {
      status: 'failed',
      code: 'database_write_failed',
      message: 'save failed',
      detail: 'sqlite locked',
      recoverable: true
    }
  }
})

assert.equal(axiosLike.code, 'database_write_failed')
assert.equal(axiosLike.message, '答案保存失败。')

const detailWrapped = normalizeApiError({
  detail: {
    status: 'failed',
    code: 'llm_generation_failed',
    message: 'generation failed',
    detail: 'provider timeout',
    recoverable: true
  }
})

assert.equal(detailWrapped.code, 'llm_generation_failed')
assert.equal(detailWrapped.message, '答案生成失败，请稍后重试。')

console.log('error parser tests passed')
