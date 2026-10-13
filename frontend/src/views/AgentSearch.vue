<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Compass, Delete } from '@element-plus/icons-vue'
import { useRoute, useRouter } from 'vue-router'
import { dislikePaper, likePaper, removePaperPreference } from '@/api/papers'
import { useAgentSearchChat } from '@/composables/useAgentSearchChat'
import AgentUserResultAttachments from '@/components/agent-search/AgentUserResultAttachments.vue'
import AgentRunTimeline from '@/components/agent-search/AgentRunTimeline.vue'
import AgentWorkCard from '@/components/agent-search/AgentWorkCard.vue'
import RagChatPanel from '@/components/rag-chat/RagChatPanel.vue'
import RagEvidencePanel from '@/components/rag-chat/RagEvidencePanel.vue'
import type { AgentInteraction, ArxivSearchResponse, TargetSelectionPayload } from '@/types/agent'
import type { Paper } from '@/types/paper'
import { usePaperStore } from '@/stores/paperStore'
import { getPaperTargetCandidateId, toAgentUserResult, type AgentUserResult } from '@/utils/agentUserResult'
import { orderCitedSourceIds } from '@/utils/evidence'
import type { RagChatSource } from '@/types/ragChat'

const router = useRouter()
const route = useRoute()
const store = usePaperStore()
const agentPaperLabels = new Map<string, 'liked' | 'disliked'>()
const profileTopicPreview = computed(() => (store.researchProfile?.positive_topics || []).slice(0, 4))
const agentEvidenceDrawerOpen = ref(false)
const activeAgentEvidence = ref<{
  question: string
  sources: RagChatSource[]
  citedSourceIds: string[]
  citationWarning: string | null
  highlightedSourceId: string | null
} | null>(null)

const {
  inputMessage,
  loading,
  messages,
  latestResponse,
  currentInteraction,
  workContinuations,
  resumingContinuationIds,
  retryingContinuationIds,
  setInputMessage,
  submitMessage,
  submitResume,
  startWorkContinuationPolling,
  stopWorkContinuationPolling,
  resumeWorkContinuation,
  cancelWorkContinuation,
  retryWorkContinuation,
  clearConversation,
  selectedPaper
} = useAgentSearchChat()

const quickPrompts = [
  '帮我找最近 7 天关于 RAG 的 5 篇论文',
  '检索和 agent search 相关的 arXiv 论文',
  '找一些关于多模态检索增强生成的论文'
]

onMounted(async () => {
  // 从论文搜索页跳转过来时带着查询预填输入框；不自动发送，让用户先确认或改写。
  const prefill = route.query.q
  if (typeof prefill === 'string' && prefill.trim()) {
    setInputMessage(prefill)
    void router.replace({ query: { ...route.query, q: undefined } })
  }
  void store.fetchResearchProfile()
  // 结果卡片上的「已读」读取行为表，进入页面先加载一次。
  store.fetchPaperActions().catch(() => undefined)
  // 新页面从空会话开始；只有当前页面产生 session 后，才查询该会话自己的后台任务。
  await startWorkContinuationPolling()
})

onBeforeUnmount(() => {
  stopWorkContinuationPolling()
})

function handlePromptSelect(prompt: string) {
  setInputMessage(prompt)
}

function handleClear() {
  void clearConversation()
}

function selectedInteractionCandidate(candidateId?: string) {
  if (currentInteraction.value?.kind !== 'target_selection') return null
  const candidates = (currentInteraction.value.payload as TargetSelectionPayload).candidates
  return candidates.find((candidate, index) => getPaperTargetCandidateId(candidate, index) === candidateId) || null
}

function handleConfirmInteraction(candidateId?: string) {
  const interaction = currentInteraction.value
  if (!interaction) return

  if (interaction.kind === 'target_selection') {
    const candidate = selectedInteractionCandidate(candidateId)
    if (!candidate) {
      ElMessage.warning('请先选择一篇论文')
      return
    }
    // 客户端只提交候选身份，后端必须在 interaction 保存的候选集合内重新校验。
    submitResume('select', '用户选择目标论文', { candidate_id: candidateId })
    return
  }
  submitResume('approve', '用户批准执行副作用操作')
}

function handleCancelInteraction() {
  submitResume(currentInteraction.value?.kind === 'target_selection' ? 'cancel' : 'reject', '用户取消当前交互')
}

function handleViewDetail(id: string) {
  // 卡片点击是用户明确选择论文的动作，先更新现有焦点状态，再进入详情页。
  selectedPaper.value = { arxiv_id: id }
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

function visibleInteractionForResponse(response: ArxivSearchResponse | null): AgentInteraction | null {
  if (!response?.interaction || !currentInteraction.value) return null
  return response.interaction.interaction_id === currentInteraction.value.interaction_id ? currentInteraction.value : null
}

function toAgentUserResultForMessage(response: unknown): AgentUserResult {
  const agentResponse = toAgentResponse(response)
  return toAgentUserResult(agentResponse, {
    interaction: visibleInteractionForResponse(agentResponse)
  })
}

function openAgentEvidence(response: unknown, sourceId?: string) {
  const agentResponse = toAgentResponse(response)
  const result = toAgentUserResultForMessage(response)
  if (!result.evidenceSources.length) return
  // 与正文角标同一套编号：先按回答中出现顺序，再补后端声明的引用。
  const citedSourceIds = orderCitedSourceIds(
    agentResponse?.paper_qa_result?.answer || '',
    result.evidenceSources,
    result.citedSourceIds
  )
  activeAgentEvidence.value = {
    question: agentResponse?.paper_qa_result?.question || 'Agent 回答证据',
    sources: result.evidenceSources,
    citedSourceIds,
    citationWarning: result.citationWarning,
    highlightedSourceId: sourceId || null
  }
  agentEvidenceDrawerOpen.value = true
}

function handleAgentSourceSelect(item: any, sourceId: string) {
  openAgentEvidence(item.response, sourceId)
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
  <div class="page agent-page">
    <header class="page-header">
      <div class="page-header__text">
        <h1 class="page-title">Agent 搜索</h1>
        <p class="page-desc">用自然语言描述你想找的论文，Agent 会规划检索、整理结果并给出可查看的论文列表。</p>
      </div>
      <div class="page-actions">
        <el-button :icon="Delete" :disabled="loading || messages.length === 0" @click="handleClear">清空对话</el-button>
      </div>
    </header>

    <section v-if="store.researchProfile" class="notice notice--info profile-hint">
      <el-icon class="profile-hint__icon"><Compass /></el-icon>
      <div class="profile-hint__copy">
        <template v-if="profileTopicPreview.length">
          Agent 会参考你的研究画像，当前重点：<strong>{{ profileTopicPreview.join('、') }}</strong>
        </template>
        <template v-else>
          补充研究方向、偏好分类和回答风格，可以帮助 Agent 更好地理解模糊请求。
        </template>
      </div>
      <el-button link type="primary" @click="handleGoProfile">编辑画像</el-button>
    </section>

    <section v-if="workContinuations.length" class="work-list" aria-label="后台 Agent 工作">
      <div class="section-head">
        <div>
          <h2 class="section-title">后台任务</h2>
          <p class="section-desc">关闭或刷新页面不会丢失任务；只有唯一的当前会话任务会自动恢复。</p>
        </div>
      </div>
      <AgentWorkCard
        v-for="work in workContinuations"
        :key="work.continuation_id"
        :work="work"
        :resuming="resumingContinuationIds.has(work.continuation_id)"
        :retrying="retryingContinuationIds.has(work.continuation_id)"
        @resume="resumeWorkContinuation"
        @cancel="cancelWorkContinuation"
        @retry="retryWorkContinuation"
      />
    </section>

    <section class="surface conversation">
      <RagChatPanel
        v-model:sender-text="inputMessage"
        :messages="messages"
        :loading="loading"
        :quick-prompts="quickPrompts"
        title="想找什么样的论文？"
        description="回答过程中会实时展示 Agent 的执行步骤，完成后可展开回看。"
        prompt-title="试试这些问题"
        assistant-label="Agent"
        sender-placeholder="描述你的检索需求，Enter 发送，Shift+Enter 换行"
        sender-tip="Agent 会先理解需求，再规划检索并整理结果"
        @submit-question="submitMessage"
        @select-prompt="handlePromptSelect"
        @select-source="handleAgentSourceSelect"
      >
        <template #message-progress="{ item }">
          <AgentRunTimeline v-if="item.progress" :progress="item.progress" />
        </template>
        <template #message-footer="{ item }">
          <AgentUserResultAttachments
            v-if="item.role === 'assistant' && item.response"
            :result="toAgentUserResultForMessage(item.response)"
            :loading="loading"
            @view-detail="handleViewDetail"
            @label="handleLabel"
            @confirm-interaction="handleConfirmInteraction"
            @cancel-interaction="handleCancelInteraction"
            @view-evidence="openAgentEvidence(item.response)"
          />
        </template>
      </RagChatPanel>
    </section>

    <el-drawer v-model="agentEvidenceDrawerOpen" title="回答证据" size="min(560px, 92vw)">
      <RagEvidencePanel
        v-if="activeAgentEvidence"
        :question="activeAgentEvidence.question"
        :sources="activeAgentEvidence.sources"
        :cited-source-ids="activeAgentEvidence.citedSourceIds"
        :citation-warning="activeAgentEvidence.citationWarning"
        :highlighted-source-id="activeAgentEvidence.highlightedSourceId"
        :retrieval-debug="null"
        @select-source="activeAgentEvidence.highlightedSourceId = $event"
      />
    </el-drawer>
  </div>
</template>

<style scoped>
.agent-page {
  max-width: 1080px;
}

.profile-hint {
  align-items: center;
  margin-bottom: 16px;
}

.profile-hint__icon {
  flex-shrink: 0;
  font-size: 16px;
}

.profile-hint__copy {
  flex: 1;
  min-width: 0;
}

.work-list {
  display: grid;
  gap: 10px;
  margin-bottom: 16px;
}

.work-list .section-head {
  margin-bottom: 0;
}

.conversation {
  padding: 18px 20px;
}
</style>
