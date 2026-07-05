<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { dislikePaper, likePaper, removePaperPreference } from '@/api/papers'
import { useAgentSearchChat } from '@/composables/useAgentSearchChat'
import AgentUserResultAttachments from '@/components/agent-search/AgentUserResultAttachments.vue'
import RagChatPanel from '@/components/rag-chat/RagChatPanel.vue'
import type { AgentPendingAction, ArxivSearchResponse } from '@/types/agent'
import type { Paper } from '@/types/paper'
import { usePaperStore } from '@/stores/paperStore'
import {
  getAgentPendingActionKey,
  getPaperTargetCandidateId,
  isPaperTargetConfirmation,
  toAgentUserResult,
  type AgentUserResult
} from '@/utils/agentUserResult'

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

function selectedPendingCandidate(candidateId?: string) {
  const candidates = Array.isArray(pendingAction.value?.candidates) ? pendingAction.value.candidates : []
  return candidates.find((candidate, index) => getPaperTargetCandidateId(candidate, index) === candidateId) || null
}

function handleConfirmPendingAction(candidateId?: string) {
  const action = pendingAction.value
  if (!action) return

  if (isPaperTargetConfirmation(action)) {
    const candidate = selectedPendingCandidate(candidateId)
    if (!candidate) {
      ElMessage.warning('请先选择一篇论文')
      return
    }
    // 确认目标论文只提交稳定身份字段，后端会在 pending confirmation 候选集合内再次校验。
    submitResume('approve', '用户确认目标论文', {
      pending_action_id: action.pending_action_id,
      confirmed_paper_id: candidateId,
      confirmed_arxiv_id: candidate.arxiv_id || candidate.arxivId || candidate.id || null
    })
    return
  }

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

function toAgentResponse(response: unknown): ArxivSearchResponse | null {
  return response as ArxivSearchResponse | null
}

function visiblePendingActionForResponse(response: ArxivSearchResponse | null): AgentPendingAction | null {
  if (!response?.pending_action || !pendingAction.value) return null
  const responseKey = getAgentPendingActionKey(response.pending_action)
  const activeKey = getAgentPendingActionKey(pendingAction.value)
  // pending action 是可消费状态，必须和当前活跃确认匹配，避免旧消息残留可点击确认卡。
  return responseKey && activeKey && responseKey === activeKey ? pendingAction.value : null
}

function toAgentUserResultForMessage(response: unknown): AgentUserResult {
  const agentResponse = toAgentResponse(response)
  return toAgentUserResult(agentResponse, {
    pendingAction: visiblePendingActionForResponse(agentResponse)
  })
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
          直接说出你的检索需求，Agent 会返回回答、必要提示和可查看的论文结果。
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
      <RagChatPanel
        v-model:sender-text="inputMessage"
        :messages="messages"
        :loading="loading"
        :quick-prompts="quickPrompts"
        title="Agent 对话"
        description="默认只展示最终回答、必要确认、检索范围提示和论文结果。"
        prompt-title="搜索示例"
        assistant-label="Agent"
        sender-placeholder="请输入自然语言搜索需求，Enter 发送，Shift+Enter 换行"
        @submit-question="submitMessage"
        @select-prompt="handlePromptSelect"
      >
        <template #message-footer="{ item }">
          <AgentUserResultAttachments
            v-if="item.role === 'assistant' && item.response"
            :result="toAgentUserResultForMessage(item.response)"
            :loading="loading"
            @view-detail="handleViewDetail"
            @label="handleLabel"
            @confirm-pending-action="handleConfirmPendingAction"
            @cancel-pending-action="handleCancelPendingAction"
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

@media (max-width: 1024px) {
  .hero-card {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
