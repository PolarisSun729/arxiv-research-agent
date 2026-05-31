import { ref, type Ref } from 'vue'
import { ElMessage } from 'element-plus'
import { runArxivSearchAgent } from '@/api/agent'
import type { ArxivSearchResponse } from '@/types/agent'
import type { AgentChatMessage } from '@/types/agentChat'

function createMessageId(prefix: 'user' | 'assistant') {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

function getErrorMessage(error: unknown, fallback: string) {
  if (error && typeof error === 'object' && 'message' in error && typeof error.message === 'string' && error.message) {
    return error.message
  }
  return fallback
}

export function useAgentSearchChat() {
  const inputMessage = ref('')
  const loading = ref(false)
  const messages: Ref<AgentChatMessage<ArxivSearchResponse>[]> = ref([])
  const latestResponse = ref<ArxivSearchResponse | null>(null)

  function setInputMessage(value: string) {
    inputMessage.value = value
  }

  function clearConversation() {
    messages.value = []
    latestResponse.value = null
    inputMessage.value = ''
  }

  async function submitMessage(rawMessage?: string) {
    const message = (rawMessage ?? inputMessage.value).trim()
    if (!message || loading.value) return

    const userId = createMessageId('user')
    const assistantId = createMessageId('assistant')
    const createdAt = new Date().toISOString()
    const userMessage: AgentChatMessage<ArxivSearchResponse> = {
      id: userId,
      role: 'user',
      content: message,
      loading: false,
      createdAt,
      response: null,
      error: null
    }
    const assistantMessage: AgentChatMessage<ArxivSearchResponse> = {
      id: assistantId,
      role: 'assistant',
      content: '',
      loading: true,
      createdAt,
      response: null,
      error: null
    }

    messages.value.push(userMessage, assistantMessage)
    latestResponse.value = null
    inputMessage.value = ''
    loading.value = true

    try {
      const response = await runArxivSearchAgent({
        message,
        user_id: 'local_user'
      })

      latestResponse.value = response
      const target = messages.value.find(item => item.id === assistantId)
      if (target) {
        target.content = response.answer || ''
        target.response = response
        target.error = null
        target.loading = false
      }
    } catch (error) {
      const errorMessage = getErrorMessage(error, 'Agent 调用失败')
      ElMessage.error(errorMessage)
      const target = messages.value.find(item => item.id === assistantId)
      if (target) {
        target.content = errorMessage
        target.response = null
        target.error = errorMessage
        target.loading = false
      }
      inputMessage.value = message
    } finally {
      loading.value = false
    }
  }

  return {
    inputMessage,
    loading,
    messages,
    latestResponse,
    setInputMessage,
    submitMessage,
    clearConversation
  }
}
