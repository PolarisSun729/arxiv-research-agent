<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import PaperCard from '@/components/PaperCard.vue'
import type { Paper } from '@/types/paper'
import type { AgentUserResult } from '@/utils/agentUserResult'

const props = withDefaults(defineProps<{
  result: AgentUserResult
  loading?: boolean
}>(), {
  loading: false
})

const emit = defineEmits<{
  (event: 'view-detail', id: string): void
  (event: 'label', paper: Paper, label: 'liked' | 'disliked' | null): void
  (event: 'confirm-interaction', candidateId?: string): void
  (event: 'cancel-interaction'): void
}>()

const selectedCandidateId = ref('')

const interaction = computed(() => props.result.interaction)
const hasVisibleContent = computed(() => props.result.hasVisibleContent)
const shouldRequireCandidate = computed(() =>
  interaction.value?.kind === 'target_selection' && interaction.value.candidates.length > 0
)
const canConfirm = computed(() =>
  Boolean(interaction.value) &&
  !props.loading &&
  (!shouldRequireCandidate.value || Boolean(selectedCandidateId.value))
)

watch(
  () => interaction.value?.defaultCandidateId,
  value => {
    selectedCandidateId.value = value || interaction.value?.candidates[0]?.id || ''
  },
  { immediate: true }
)

watch(
  () => interaction.value?.candidates.map(candidate => candidate.id).join('|') || '',
  () => {
    if (!interaction.value?.candidates.length) {
      selectedCandidateId.value = ''
      return
    }
    if (!interaction.value.candidates.some(candidate => candidate.id === selectedCandidateId.value)) {
      selectedCandidateId.value = interaction.value.defaultCandidateId || interaction.value.candidates[0].id
    }
  }
)

function handleLabel(id: string, label: 'liked' | 'disliked' | null) {
  const paper = props.result.papers.find(item => item.id === id)
  if (!paper) return
  emit('label', paper, label)
}

function handleConfirm() {
  if (!interaction.value || !canConfirm.value) return
  emit('confirm-interaction', selectedCandidateId.value || undefined)
}
</script>

<template>
  <div v-if="hasVisibleContent" class="agent-user-result-attachments">
    <section
      v-if="result.queryCapability"
      class="agent-user-notice"
      :class="`agent-user-notice--${result.queryCapability.status}`"
      :role="result.queryCapability.status === 'error' ? 'alert' : 'status'"
    >
      <div class="agent-user-notice__title">检索范围提示</div>
      <p>{{ result.queryCapability.summary }}</p>
      <p v-if="result.queryCapability.suggestedAction">
        {{ result.queryCapability.suggestedAction }}
      </p>
    </section>

    <section
      v-if="result.preferenceFeedback"
      class="agent-user-feedback"
      :class="`agent-user-feedback--${result.preferenceFeedback.status}`"
      :role="result.preferenceFeedback.status === 'failed' ? 'alert' : 'status'"
    >
      {{ result.preferenceFeedback.message }}
    </section>

    <section v-if="interaction" class="agent-user-confirmation">
      <div class="agent-user-confirmation__copy">
        <h3>{{ interaction.title }}</h3>
        <p>{{ interaction.description }}</p>
      </div>

      <div v-if="interaction.kind === 'target_selection'" class="agent-user-candidates">
        <el-radio-group
          v-if="interaction.candidates.length"
          v-model="selectedCandidateId"
          class="agent-user-candidates__group"
        >
          <el-radio
            v-for="candidate in interaction.candidates"
            :key="candidate.id"
            :label="candidate.id"
            class="agent-user-candidate"
            border
          >
            <div class="agent-user-candidate__main">
              <div class="agent-user-candidate__title">{{ candidate.title }}</div>
              <div class="agent-user-candidate__meta">
                <span v-if="candidate.arxivId">arXiv: {{ candidate.arxivId }}</span>
                <span v-if="candidate.authors">作者: {{ candidate.authors }}</span>
                <span v-if="candidate.rank">排名: #{{ candidate.rank }}</span>
                <span>来源: {{ candidate.sourceLabel }}</span>
              </div>
            </div>
            <el-tag v-if="candidate.isDefault" size="small" type="success" effect="plain">
              推荐
            </el-tag>
          </el-radio>
        </el-radio-group>
        <el-empty
          v-else
          description="没有可选择的候选论文，请取消后重新发起请求。"
          :image-size="72"
        />
      </div>

      <div class="agent-user-confirmation__actions">
        <el-button
          type="primary"
          :loading="loading"
          :disabled="!canConfirm"
          @click="handleConfirm"
        >
          {{ interaction.confirmLabel }}
        </el-button>
        <el-button :disabled="loading" @click="emit('cancel-interaction')">
          {{ interaction.cancelLabel }}
        </el-button>
      </div>
    </section>

    <div v-if="result.papers.length" class="agent-user-paper-list">
      <PaperCard
        v-for="paper in result.papers"
        :key="paper.id"
        :paper="paper"
        @view-detail="emit('view-detail', $event)"
        @label="handleLabel"
      />
    </div>
  </div>
</template>

<style scoped>
.agent-user-result-attachments {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 12px;
}

.agent-user-notice,
.agent-user-feedback,
.agent-user-confirmation {
  border-radius: 8px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  background: #fff;
}

.agent-user-notice {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px 14px;
  border-left: 4px solid #0f766e;
}

.agent-user-notice--warning {
  border-left-color: #d97706;
  background: #fffbeb;
}

.agent-user-notice--error {
  border-left-color: #dc2626;
  background: #fff7f7;
}

.agent-user-notice__title {
  color: #0f172a;
  font-size: 13px;
  font-weight: 700;
}

.agent-user-notice p {
  margin: 0;
  color: #475569;
  font-size: 13px;
  line-height: 1.6;
}

.agent-user-feedback {
  padding: 10px 12px;
  color: #0f766e;
  font-size: 13px;
  font-weight: 700;
  background: #f0fdfa;
}

.agent-user-feedback--failed {
  color: #b91c1c;
  background: #fef2f2;
}

.agent-user-confirmation {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
  background: #f8fafc;
}

.agent-user-confirmation__status {
  color: #0f766e;
  font-size: 13px;
  font-weight: 700;
}

.agent-user-confirmation__copy h3 {
  margin: 0;
  color: #0f172a;
  font-size: 16px;
  line-height: 1.4;
}

.agent-user-confirmation__copy p {
  margin: 6px 0 0;
  color: #475569;
  line-height: 1.6;
}

.agent-user-index-progress {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.agent-user-index-progress__meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: #475569;
  font-size: 13px;
}

.agent-user-index-progress__meta strong {
  color: #0f172a;
  font-variant-numeric: tabular-nums;
}

.agent-user-index-progress__error {
  margin: 0;
  color: #b91c1c;
  font-size: 13px;
  line-height: 1.6;
}

.agent-user-candidates__group {
  display: grid;
  gap: 10px;
}

.agent-user-candidate {
  height: auto;
  margin-right: 0;
  padding: 12px;
  white-space: normal;
}

.agent-user-candidate :deep(.el-radio__label) {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
}

.agent-user-candidate__main {
  min-width: 0;
}

.agent-user-candidate__title {
  color: #0f172a;
  font-weight: 700;
  line-height: 1.4;
}

.agent-user-candidate__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 6px;
  color: #64748b;
  font-size: 12px;
  line-height: 1.5;
}

.agent-user-confirmation__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.agent-user-paper-list {
  display: grid;
  gap: 12px;
}
</style>
