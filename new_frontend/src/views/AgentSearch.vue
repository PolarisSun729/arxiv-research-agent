<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { dislikePaper, likePaper, removePaperPreference } from '@/api/papers'
import { useAgentSearchChat } from '@/composables/useAgentSearchChat'
import AgentResponsePanel from '@/components/agent-search/AgentResponsePanel.vue'
import RagChatPanel from '@/components/rag-chat/RagChatPanel.vue'
import type { PaperTargetCandidate } from '@/types/agent'
import type { Paper } from '@/types/paper'
import { usePaperStore } from '@/stores/paperStore'

const router = useRouter()
const store = usePaperStore()
const agentPaperLabels = new Map<string, 'liked' | 'disliked'>()
const profileTopicPreview = computed(() => (store.researchProfile?.positive_topics || []).slice(0, 4))
const selectedPendingCandidateId = ref('')

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

const isPaperTargetConfirmation = computed(() => {
  const action = pendingAction.value
  return action?.request_type === 'paper_target_confirmation' || action?.type === 'paper_target_confirmation'
})

const isPendingActionConfirming = computed(() => pendingAction.value?.status === 'confirming')

const pendingTargetCandidates = computed<PaperTargetCandidate[]>(() => {
  const candidates = pendingAction.value?.candidates
  return Array.isArray(candidates) ? candidates : []
})

function paperCandidateId(candidate: PaperTargetCandidate, index = 0) {
  return String(
    candidate.candidate_id ||
    candidate.paper_id ||
    candidate.arxiv_id ||
    candidate.arxivId ||
    candidate.id ||
    candidate.title ||
    `candidate-${index}`
  )
}

function paperCandidateAuthors(candidate: PaperTargetCandidate) {
  if (candidate.authors_summary) return candidate.authors_summary
  if (Array.isArray(candidate.authors)) return candidate.authors.slice(0, 3).join(', ')
  return candidate.authors || ''
}

function paperCandidateSource(candidate: PaperTargetCandidate) {
  return candidate.source_label || candidate.list_name || candidate.source_type || candidate.source || '上下文候选'
}

function selectedPendingCandidate() {
  return pendingTargetCandidates.value.find((candidate, index) => paperCandidateId(candidate, index) === selectedPendingCandidateId.value) || null
}

function handleConfirmPendingAction() {
  if (isPaperTargetConfirmation.value) {
    const candidate = selectedPendingCandidate()
    if (!candidate) {
      ElMessage.warning('请先选择一篇论文')
      return
    }
    // 确认目标论文只提交稳定身份字段，后端会在 pending confirmation 候选集合内再次校验。
    submitResume('approve', '用户确认目标论文', {
      pending_action_id: pendingAction.value?.pending_action_id,
      confirmed_paper_id: selectedPendingCandidateId.value,
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

watch(pendingAction, action => {
  if (!action || !(action.request_type === 'paper_target_confirmation' || action.type === 'paper_target_confirmation')) {
    selectedPendingCandidateId.value = ''
    return
  }

  const candidates = Array.isArray(action.candidates) ? action.candidates : []
  const recommended = action.recommended_candidate || action.target_paper || candidates[0]
  selectedPendingCandidateId.value = action.default_candidate_id
    || (recommended ? paperCandidateId(recommended) : '')
}, { immediate: true })
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
          <div v-if="isPendingActionConfirming" class="pending-action-banner__status">
            确认请求已提交，正在等待后端消费并继续执行。
          </div>
          <div class="pending-action-banner__label">待确认任务</div>
          <div class="pending-action-banner__title">
            {{ pendingAction.title || '当前论文' }}
          </div>
          <div class="pending-action-banner__desc">
            {{ pendingAction.qa_question || pendingAction.original_question || '需要先确认是否解析 PDF 并建立全文索引。' }}
          </div>
          <div v-if="isPaperTargetConfirmation" class="paper-target-candidates">
            <el-radio-group v-model="selectedPendingCandidateId" class="paper-target-candidates__group">
              <el-radio
                v-for="(candidate, index) in pendingTargetCandidates"
                :key="paperCandidateId(candidate, index)"
                :label="paperCandidateId(candidate, index)"
                class="paper-target-candidate"
                border
              >
                <div class="paper-target-candidate__main">
                  <div class="paper-target-candidate__title">
                    {{ candidate.title || candidate.arxiv_id || candidate.arxivId || paperCandidateId(candidate, index) }}
                  </div>
                  <div class="paper-target-candidate__meta">
                    <span v-if="candidate.arxiv_id || candidate.arxivId">arXiv: {{ candidate.arxiv_id || candidate.arxivId }}</span>
                    <span v-if="paperCandidateAuthors(candidate)">作者: {{ paperCandidateAuthors(candidate) }}</span>
                    <span v-if="candidate.rank">排名: #{{ candidate.rank }}</span>
                    <span>来源: {{ paperCandidateSource(candidate) }}</span>
                  </div>
                </div>
                <el-tag
                  v-if="pendingAction.default_candidate_id === paperCandidateId(candidate, index)"
                  size="small"
                  type="success"
                  effect="plain"
                >
                  默认
                </el-tag>
              </el-radio>
            </el-radio-group>
            <el-empty
              v-if="pendingTargetCandidates.length === 0"
              description="No candidate papers. Cancel and start again."
              :image-size="72"
            />
          </div>
          <div class="pending-action-banner__meta">
            <span v-if="pendingAction.pending_action_id">pending: {{ pendingAction.pending_action_id }}</span>
            <span>tool: {{ pendingAction.tool_name || '-' }}</span>
            <span>step: {{ pendingAction.step_id || '-' }}</span>
            <span>session: {{ activeSessionId || pendingAction.session_id || '-' }}</span>
            <span v-if="pendingAction.expires_at">expires: {{ pendingAction.expires_at }}</span>
          </div>
        </div>
        <div class="pending-action-banner__actions">
          <el-button
            type="primary"
            :loading="isPendingActionConfirming"
            :disabled="loading || isPendingActionConfirming || (isPaperTargetConfirmation && !selectedPendingCandidateId)"
            @click="handleConfirmPendingAction"
          >
            <span v-if="isPaperTargetConfirmation">&#30830;&#35748;&#24182;&#32487;&#32493;</span>
            <span v-else>&#30830;&#35748;&#25191;&#34892;</span>
            <span v-pre class="legacy-hidden">
            {{ isPaperTargetConfirmation ? '确认并继续' : '确认执行' }}
            </span>
          </el-button>
          <el-button
            v-if="false"
            type="primary"
            :disabled="loading || (isPaperTargetConfirmation && !selectedPendingCandidateId)"
            @click="handleConfirmPendingAction"
          >
            解析并回答
          </el-button>
          <el-button :disabled="loading || isPendingActionConfirming" @click="handleCancelPendingAction">
            &#21462;&#28040;
          </el-button>
          <el-button v-if="false" :disabled="loading" @click="handleCancelPendingAction">
            取消
          </el-button>
          <el-button v-if="false" :disabled="loading" @click="handleCancelPendingAction">
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

.pending-action-banner__status {
  margin-bottom: 8px;
  color: #0f766e;
  font-size: 13px;
  font-weight: 600;
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

.paper-target-candidates {
  margin-top: 12px;
}

.paper-target-candidates__group {
  display: grid;
  gap: 10px;
}

.paper-target-candidate {
  height: auto;
  margin-right: 0;
  padding: 12px;
  white-space: normal;
}

.paper-target-candidate :deep(.el-radio__label) {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
}

.paper-target-candidate__main {
  min-width: 0;
}

.paper-target-candidate__title {
  color: #0f172a;
  font-weight: 700;
  line-height: 1.4;
}

.paper-target-candidate__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 6px;
  color: #64748b;
  font-size: 12px;
  line-height: 1.5;
}

.legacy-hidden {
  display: none;
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
