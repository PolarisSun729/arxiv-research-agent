<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useAgentSearchChat } from '@/composables/useAgentSearchChat'
import AgentResponsePanel from '@/components/agent-search/AgentResponsePanel.vue'
import RagChatPanel from '@/components/rag-chat/RagChatPanel.vue'

const router = useRouter()

const {
  inputMessage,
  loading,
  messages,
  setInputMessage,
  submitMessage,
  clearConversation
} = useAgentSearchChat()

const quickPrompts = [
  '帮我找最近 7 天关于 RAG 的 5 篇论文',
  '检索和 agent search 相关的 arXiv 论文',
  '找一些关于多模态检索增强生成的论文'
]

function handlePromptSelect(prompt: string) {
  setInputMessage(prompt)
}

function handleClear() {
  clearConversation()
}

function handleViewDetail(id: string) {
  router.push(`/paper/${id}`)
}

function toAgentResponse(response: unknown): import('@/types/agent').ArxivSearchResponse | null {
  return response as import('@/types/agent').ArxivSearchResponse | null
}
</script>

<template>
  <div class="agent-search-page">
    <section class="hero-card">
      <div class="hero-copy">
        <p class="eyebrow">Agent Search</p>
        <h1 class="title">自然语言 arXiv 搜索入口</h1>
        <p class="subtitle">
          直接说出你的检索需求，系统会先理解意图、生成搜索计划、调用工具，再把完整响应嵌回到对应的助手消息里。
        </p>
      </div>

      <div class="hero-actions">
        <el-button text :disabled="loading || messages.length === 0" @click="handleClear">
          清空对话
        </el-button>
      </div>
    </section>

    <section class="conversation-shell">
      <RagChatPanel
        v-model:sender-text="inputMessage"
        :messages="messages"
        :loading="loading"
        :quick-prompts="quickPrompts"
        title="Agent 对话"
        description="默认只展示最终回答和关键摘要，详细 timeline、tool trace、spec 和错误信息会收进可展开区域。"
        prompt-title="搜索示例"
        assistant-label="Agent"
        sender-placeholder="请输入自然语言搜索需求，Enter 发送，Shift+Enter 换行"
        @submit-question="submitMessage"
        @select-prompt="handlePromptSelect"
      >
        <template #message-footer="{ item }">
          <AgentResponsePanel
            v-if="item.role === 'assistant' && item.response"
            :response="toAgentResponse(item.response)"
            @view-detail="handleViewDetail"
          />
        </template>
      </RagChatPanel>
    </section>
  </div>
</template>

<style scoped>
.agent-search-page {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.hero-card {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 22px 24px;
  border-radius: 22px;
  background:
    radial-gradient(circle at top right, rgba(56, 189, 248, 0.18), transparent 30%),
    linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.94));
  color: #fff;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
}

.hero-copy {
  min-width: 0;
}

.eyebrow {
  margin: 0 0 8px;
  color: #7dd3fc;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.title {
  margin: 0;
  font-size: 30px;
  line-height: 1.15;
}

.subtitle {
  margin: 10px 0 0;
  max-width: 800px;
  color: rgba(226, 232, 240, 0.88);
}

.hero-actions {
  flex: none;
}

.conversation-shell {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

@media (max-width: 1024px) {
  .hero-card {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
