<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { dislikePaper, likePaper, removePaperPreference } from '@/api/papers'
import { useAgentSearchChat } from '@/composables/useAgentSearchChat'
import AgentResponsePanel from '@/components/agent-search/AgentResponsePanel.vue'
import RagChatPanel from '@/components/rag-chat/RagChatPanel.vue'
import type { Paper } from '@/types/paper'
import { usePaperStore } from '@/stores/paperStore'

const router = useRouter()
const store = usePaperStore()
const agentPaperLabels = new Map<string, 'liked' | 'disliked'>()
const profileTopicPreview = computed(() => (store.researchProfile?.positive_topics || []).slice(0, 4))

const {
  inputMessage,
  loading,
  messages,
  latestResponse,
  pendingAction,
  activeSessionId,
  setInputMessage,
  submitMessage,
  submitResume,
  clearConversation
} = useAgentSearchChat()

const quickPrompts = [
  '帮我找最近 7 天关于 RAG 的 5 篇论文',
  '检索和 agent search 相关的 arXiv 论文',
  '找一些关于多模态检索增强生成的论文'
]

onMounted(() => {
  store.fetchResearchProfile()
})

function handlePromptSelect(prompt: string) {
  setInputMessage(prompt)
}

function handleClear() {
  clearConversation()
}

function handleConfirmPendingAction() {
  submitResume('approve', '用户在确认卡片中批准执行')
}

function handleCancelPendingAction() {
  submitResume('reject', '用户在确认卡片中拒绝执行')
}

function handleViewDetail(id: string) {
  router.push(`/paper/${id}`)
}

function handleGoProfile() {
  router.push('/profile')
}

async function handleLabel(paper: Paper, label: 'liked' | 'disliked' | null) {
  try {
    if (label === null) {
      const currentLabel = agentPaperLabels.get(paper.id)
      if (currentLabel) {
        await removePaperPreference(paper, currentLabel)
        agentPaperLabels.delete(paper.id)
      }
      ElMessage.success('已取消标记')
      return
    }

    if (label === 'liked') {
      await likePaper(paper)
      agentPaperLabels.set(paper.id, label)
      ElMessage.success('已标记为感兴趣')
      return
    }

    await dislikePaper(paper)
    agentPaperLabels.set(paper.id, label)
    ElMessage.success('已标记为不感兴趣')
  } catch (error) {
    ElMessage.error('偏好保存失败')
  }
}

function toAgentResponse(response: unknown): import('@/types/agent').ArxivSearchResponse | null {
  return response as import('@/types/agent').ArxivSearchResponse | null
}

function syncAgentPreferenceState() {
  const response = latestResponse.value
  const result = response?.preference_action_result
  if (!result || result.status !== 'success') return

  const targetLabel = result.label === 'liked' ? 'liked' : result.label === 'disliked' ? 'disliked' : null
  const arxivId = result.arxiv_id || result.paper?.arxiv_id || result.paper?.arxivId || result.paper?.id

  if (arxivId) {
    if (targetLabel) {
      agentPaperLabels.set(arxivId, targetLabel)
    } else {
      agentPaperLabels.delete(arxivId)
    }
  }

  for (let i = messages.value.length - 1; i >= 0; i -= 1) {
    const item = messages.value[i]
    if (item.role !== 'assistant' || !item.response || !Array.isArray(item.response.papers) || !item.response.papers.length) {
      continue
    }

    const matchedPaper = item.response.papers.find((paper: any) => {
      const paperId = paper.arxiv_id || paper.arxivId || paper.id
      return arxivId && paperId === arxivId
    })

    if (matchedPaper) {
      matchedPaper.label = targetLabel
      break
    }
  }
}

watch(latestResponse, () => {
  syncAgentPreferenceState()
}, { deep: true })
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

    <section v-if="store.researchProfile" class="profile-banner">
      <div class="profile-banner__copy">
        <div class="profile-banner__title">Agent 会轻量参考你的长期研究画像</div>
        <div class="profile-banner__desc">
          <template v-if="profileTopicPreview.length">
            当前重点：{{ profileTopicPreview.join('、') }}
          </template>
          <template v-else>
            你可以补充研究方向、偏好分类和回答风格，帮助 Agent 更好理解模糊请求。
          </template>
        </div>
      </div>
      <el-button size="small" @click="handleGoProfile">编辑画像</el-button>
    </section>

    <section class="conversation-shell">
      <div v-if="pendingAction" class="pending-action-banner">
        <div class="pending-action-banner__copy">
          <div class="pending-action-banner__label">待确认任务</div>
          <div class="pending-action-banner__title">
            {{ pendingAction.title || '当前论文' }}
          </div>
          <div class="pending-action-banner__desc">
            {{ pendingAction.qa_question || pendingAction.original_question || '需要先确认是否解析 PDF 并建立全文索引。' }}
          </div>
          <div class="pending-action-banner__meta">
            <span>tool: {{ pendingAction.tool_name || '-' }}</span>
            <span>step: {{ pendingAction.step_id || '-' }}</span>
            <span>session: {{ activeSessionId || pendingAction.session_id || '-' }}</span>
          </div>
        </div>
        <div class="pending-action-banner__actions">
          <el-button type="primary" :disabled="loading" @click="handleConfirmPendingAction">
            解析并回答
          </el-button>
          <el-button :disabled="loading" @click="handleCancelPendingAction">
            取消
          </el-button>
        </div>
      </div>

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
            @label="handleLabel"
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

.profile-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 18px;
  border-radius: 18px;
  border: 1px solid rgba(125, 211, 252, 0.26);
  background: linear-gradient(135deg, rgba(240, 249, 255, 0.96), rgba(248, 250, 252, 0.98));
}

.profile-banner__title {
  font-size: 15px;
  font-weight: 700;
  color: #0f172a;
}

.profile-banner__desc {
  margin-top: 4px;
  color: #475569;
  line-height: 1.6;
}

.pending-action-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 18px;
  border-radius: 18px;
  border: 1px solid rgba(14, 165, 233, 0.18);
  background:
    radial-gradient(circle at top right, rgba(14, 165, 233, 0.16), transparent 35%),
    linear-gradient(135deg, rgba(240, 249, 255, 0.96), rgba(255, 255, 255, 0.98));
}

.pending-action-banner__copy {
  min-width: 0;
}

.pending-action-banner__label {
  color: #0284c7;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.pending-action-banner__title {
  margin-top: 6px;
  font-size: 16px;
  font-weight: 700;
  color: #0f172a;
}

.pending-action-banner__desc {
  margin-top: 4px;
  color: #475569;
  line-height: 1.6;
}

.pending-action-banner__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 8px;
  color: #64748b;
  font-size: 12px;
}

.pending-action-banner__actions {
  display: flex;
  gap: 10px;
  flex: none;
}

@media (max-width: 1024px) {
  .hero-card {
    flex-direction: column;
    align-items: stretch;
  }

  .pending-action-banner {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
