import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { pathToFileURL } from 'node:url'

const outfile = join(tmpdir(), `agent-search-chat-${Date.now()}.mjs`)

await build({
  stdin: {
    contents: `
      import assert from 'node:assert/strict'
      import { nextTick, watchEffect } from 'vue'
      import {
        getResumeCalls,
        releaseToolStream,
        resetContinuationStub,
        setContinuations
      } from '@/api/agent'
      import { useAgentSearchChat } from './src/composables/useAgentSearchChat.ts'
      import { toAgentUserResult } from './src/utils/agentUserResult.ts'

      function continuationsForTest(id) {
        return {
          continuation_id: 'continuation-' + id,
          session_id: 's1',
          status: 'waiting_job',
          display_summary: { paper_title: 'Paper ' + id },
          job_id: 'job-' + id,
          job: { status: 'running', current_stage: 'chunk_document', stage_label: '切分文档内容', progress: 45, attempt_no: 1, max_attempts: 3, error_code: null, error_message: null },
          error_code: null,
          error_message: null,
          can_cancel: true,
          can_resume: false,
          ready_at: null,
          expires_at: null,
          resume_run_id: null
        }
      }

      const chat = useAgentSearchChat()
      chat.setInputMessage('q')
      const snapshots = []

      watchEffect(() => {
        snapshots.push(chat.messages.value.map(message => \`\${message.role}:\${message.loading}:\${message.content}\`).join('|'))
      })

      await chat.submitMessage()
      await nextTick()

      const finalMessage = chat.messages.value[1]
      assert.equal(finalMessage.loading, false)
      assert.equal(finalMessage.content, 'done')
      assert.equal(chat.latestResponse.value?.papers?.length, 1)

      // 这个断言锁住前端真实症状：最终消息不仅数据变了，还必须触发响应式渲染更新。
      assert.equal(snapshots.at(-1), 'user:false:q|assistant:false:done')

      const progressChat = useAgentSearchChat()
      progressChat.setInputMessage('stream-progress')
      const pendingProgress = progressChat.submitMessage()
      await nextTick()

      const progressMessage = progressChat.messages.value[1]
      assert.equal(progressMessage.response?.streaming_state?.active_tool_call?.tool_name, 'parse_and_index_paper')

      const visibleResult = toAgentUserResult(progressMessage.response)
      // 普通 tool event 没有持久 job 快照，不能再推导 12% 等占位进度。
      assert.equal(visibleResult.hasVisibleContent, false)
      assert.equal(JSON.stringify(visibleResult).includes('"percentage":12'), false)

      releaseToolStream()
      await pendingProgress

      resetContinuationStub()
      setContinuations([{
        continuation_id: 'continuation-45',
        session_id: 's1',
        status: 'waiting_job',
        display_summary: { arxiv_id: '2401.00001', paper_title: 'Paper 45' },
        job_id: 'job-45',
        job: {
          status: 'running',
          current_stage: 'chunk_document',
          stage_label: '切分文档内容',
          progress: 45,
          attempt_no: 1,
          max_attempts: 3,
          error_code: null,
          error_message: null
        },
        error_code: null,
        error_message: null,
        can_cancel: true,
        can_resume: false,
        ready_at: null,
        expires_at: null,
        resume_run_id: null
      }])
      await chat.refreshWorkContinuations({ allowAutoResume: false })
      // 任务卡只消费后端返回的真实里程碑，不在前端做插值或默认百分比。
      assert.equal(chat.workContinuations.value[0].job.progress, 45)

      resetContinuationStub()
      setContinuations([{
        continuation_id: 'continuation-ready',
        session_id: 's1',
        status: 'ready_to_resume',
        display_summary: { arxiv_id: '2401.00001', paper_title: 'Ready paper' },
        job_id: 'job-ready',
        job: { status: 'success', current_stage: 'activate_index', stage_label: '激活新索引版本', progress: 100, attempt_no: 1, max_attempts: 3, error_code: null, error_message: null },
        error_code: null,
        error_message: null,
        can_cancel: true,
        can_resume: true,
        ready_at: 'now',
        expires_at: 'later',
        resume_run_id: null
      }])
      await chat.refreshWorkContinuations()
      assert.deepEqual(getResumeCalls(), ['continuation-ready'])
      assert.equal(chat.messages.value.at(-1).content, 'continued answer')

      resetContinuationStub()
      setContinuations(['a', 'b'].map(id => ({
        continuation_id: \`continuation-\${id}\`,
        session_id: 's1',
        status: 'ready_to_resume',
        display_summary: { paper_title: \`Paper \${id}\` },
        job_id: \`job-\${id}\`,
        job: { status: 'success', current_stage: 'activate_index', stage_label: '激活新索引版本', progress: 100, attempt_no: 1, max_attempts: 3, error_code: null, error_message: null },
        error_code: null,
        error_message: null,
        can_cancel: true,
        can_resume: true,
        ready_at: 'now',
        expires_at: 'later',
        resume_run_id: null
      })))
      await chat.refreshWorkContinuations()
      // 同一会话有多条 ready 时必须等待用户选择，禁止批量启动多个 Agent run。
      assert.deepEqual(getResumeCalls(), [])

      const originalSetTimeout = globalThis.setTimeout
      const originalClearTimeout = globalThis.clearTimeout
      let scheduledPolls = 0
      globalThis.setTimeout = () => { scheduledPolls += 1; return 1 }
      globalThis.clearTimeout = () => {}
      try {
        const pollingChat = useAgentSearchChat()
        setContinuations([{
          ...continuationsForTest('terminal'),
          status: 'failed',
          can_cancel: false,
          can_resume: false
        }])
        await pollingChat.startWorkContinuationPolling()
        // 只有终态卡片时停止 2.5 秒轮询，避免失败记录保留期间永久请求。
        assert.equal(scheduledPolls, 0)
        pollingChat.stopWorkContinuationPolling()

        setContinuations([continuationsForTest('running')])
        await pollingChat.startWorkContinuationPolling()
        assert.equal(scheduledPolls, 1)
        pollingChat.stopWorkContinuationPolling()
      } finally {
        globalThis.setTimeout = originalSetTimeout
        globalThis.clearTimeout = originalClearTimeout
      }
    `,
    resolveDir: process.cwd(),
    sourcefile: 'agent-search-chat-harness.mjs',
    loader: 'js'
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile,
  write: true,
  plugins: [
    {
      name: 'agent-search-chat-stubs',
      setup(buildApi) {
        buildApi.onResolve({ filter: /^@\/api\/agent$/ }, () => ({ path: 'agent-api-stub', namespace: 'stub' }))
        buildApi.onResolve({ filter: /^@\/api\/errors$/ }, () => ({ path: 'errors-stub', namespace: 'stub' }))
        buildApi.onResolve({ filter: /^@\/stores\/paperStore$/ }, () => ({ path: 'paper-store-stub', namespace: 'stub' }))
        buildApi.onResolve({ filter: /^@\/composables\/useUserContext$/ }, () => ({ path: 'user-context-stub', namespace: 'stub' }))
        buildApi.onResolve({ filter: /^@\/utils\/arxivQueryCapability$/ }, () => ({ path: 'arxiv-capability-stub', namespace: 'stub' }))
        buildApi.onResolve({ filter: /^element-plus$/ }, () => ({ path: 'element-plus-stub', namespace: 'stub' }))
        buildApi.onLoad({ filter: /.*/, namespace: 'stub' }, args => {
          if (args.path === 'agent-api-stub') {
            return {
              contents: `
                let releaseStream = null
                let continuations = []
                let resumeCalls = []
                export function setContinuations(items) { continuations = items }
                export function getResumeCalls() { return [...resumeCalls] }
                export function resetContinuationStub() { continuations = []; resumeCalls = [] }
                export function releaseToolStream() {
                  releaseStream?.()
                  releaseStream = null
                }
                export async function runAgentChat() { throw new Error('sync fallback should not run') }
                export async function listAgentWorkContinuations() { return continuations }
                export async function cancelAgentWorkContinuation() { throw new Error('not used') }
                export async function getAgentResumeRun() { throw new Error('not used') }
                export async function streamAgentWorkContinuationResume(continuationId, sessionId, handlers = {}) {
                  resumeCalls.push(continuationId)
                  handlers.onEvent?.('run_start', { resume_run_id: 'resume-' + continuationId, status: 'running' })
                  return {
                    session_id: sessionId,
                    intent: 'paper_qa',
                    answer: 'continued answer',
                    papers: [],
                    plan: [],
                    tool_calls: [],
                    warnings: [],
                    next_actions: [],
                    steps: [],
                    debug: {}
                  }
                }
                export async function streamAgentChat(request, handlers = {}) {
                  await Promise.resolve()
                  if (request.message === 'stream-progress') {
                    handlers.onEvent?.({
                      event_type: 'tool_call_start',
                      sequence: 1,
                      run_id: 'r-progress',
                      timestamp: 'now',
                      data: {
                        step: 'execute_step',
                        tool_call: {
                          tool_name: 'parse_and_index_paper',
                          status: 'running',
                          summary: '正在执行 parse_and_index_paper',
                          arguments: { arxiv_id: '2401.00001' },
                          trace: { step_id: 'build-index' }
                        }
                      }
                    })
                    await new Promise(resolve => { releaseStream = resolve })
                  }
                  const response = {
                    session_id: 's1',
                    intent: 'arxiv_search',
                    answer: 'done',
                    papers: [{ arxiv_id: '1', title: 'paper' }],
                    plan: [],
                    tool_calls: [],
                    warnings: [],
                    next_actions: [],
                    steps: [],
                    debug: {}
                  }
                  handlers.onEvent?.({
                    event_type: 'final_response',
                    sequence: 1,
                    run_id: 'r1',
                    timestamp: 'now',
                    data: { response }
                  })
                  handlers.onEvent?.({
                    event_type: 'stream_end',
                    sequence: 2,
                    run_id: 'r1',
                    timestamp: 'now',
                    data: { status: 'success' }
                  })
                  return response
                }
              `,
              loader: 'js'
            }
          }
          if (args.path === 'errors-stub') {
            return {
              contents: `
                export function getApiErrorMessage(_code, fallback) { return fallback }
                export function getErrorMessage(error, fallback) { return error?.message || fallback }
              `,
              loader: 'js'
            }
          }
          if (args.path === 'paper-store-stub') {
            return { contents: 'export function usePaperStore() { return { researchProfile: null } }', loader: 'js' }
          }
          if (args.path === 'user-context-stub') {
            return { contents: 'export function useUserContext() { return { getUserId: () => "test-user" } }', loader: 'js' }
          }
          if (args.path === 'arxiv-capability-stub') {
            return { contents: 'export function normalizeArxivQueryCapability() { return null }', loader: 'js' }
          }
          return { contents: 'export const ElMessage = { error() {} }', loader: 'js' }
        })
      }
    }
  ]
})

await import(pathToFileURL(outfile))

console.log('agent search chat tests passed')
