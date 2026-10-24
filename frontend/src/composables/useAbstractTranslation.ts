import { reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { streamAbstractTranslation } from '@/api/papers'
import { ApiError, getErrorMessage } from '@/api/errors'

interface TranslationEntry {
  text: string
  loading: boolean
  done: boolean
  // 当前是否以译文替换原文展示；译完后可来回切换，不重新请求。
  showing: boolean
}

// 模块级共享：同一篇论文在卡片里译过，进详情页或已标记页时直接复用，不再请求。
const entries = reactive(new Map<string, TranslationEntry>())

const FAILED_MESSAGE = '摘要翻译失败，请稍后重试。'

async function startTranslation(arxivId: string) {
  const entry: TranslationEntry = { text: '', loading: true, done: false, showing: true }
  entries.set(arxivId, entry)
  // 取回响应式代理再写，增量文本才能驱动视图更新。
  const state = entries.get(arxivId)!
  try {
    const translation = await streamAbstractTranslation(arxivId, {
      onDelta: (text) => {
        state.text += text
      }
    })
    state.text = translation
    state.done = true
  } catch (error) {
    entries.delete(arxivId)
    // 后端的 llm_generation_failed 在通用文案里叫「答案生成失败」，这里换成翻译语境的说法。
    const isLlmFailure = error instanceof ApiError && error.payload.code === 'llm_generation_failed'
    ElMessage.error(isLlmFailure ? FAILED_MESSAGE : getErrorMessage(error, FAILED_MESSAGE))
  } finally {
    state.loading = false
  }
}

export function useAbstractTranslation() {
  // 流式翻译中途就切到译文展示；译文为空（刚发出请求）时先保留原文，避免摘要区闪成空白。
  function displaySummary(arxivId: string, original: string): string {
    const entry = entries.get(arxivId)
    return entry?.showing && entry.text ? entry.text : original
  }

  function isTranslating(arxivId: string): boolean {
    return Boolean(entries.get(arxivId)?.loading)
  }

  function translateButtonText(arxivId: string): string {
    const entry = entries.get(arxivId)
    if (entry?.loading) return '翻译中…'
    return entry?.showing ? '原文' : '翻译'
  }

  function toggleTranslation(arxivId: string) {
    const entry = entries.get(arxivId)
    if (entry?.loading) return
    if (entry?.done) {
      entry.showing = !entry.showing
      return
    }
    void startTranslation(arxivId)
  }

  return { displaySummary, isTranslating, translateButtonText, toggleTranslation }
}
