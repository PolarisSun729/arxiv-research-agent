<script setup lang="ts">
import { computed } from 'vue'
import type { AgentWorkContinuation } from '@/types/agent'

const props = withDefaults(defineProps<{
  work: AgentWorkContinuation
  resuming?: boolean
  retrying?: boolean
}>(), {
  resuming: false,
  retrying: false
})

const emit = defineEmits<{
  (event: 'resume', work: AgentWorkContinuation): void
  (event: 'cancel', work: AgentWorkContinuation): void
  (event: 'retry', work: AgentWorkContinuation): void
}>()

const progress = computed(() => props.work.job?.progress ?? null)
const errorMessage = computed(() => props.work.error_message || props.work.job?.error_message || '')
const isFailed = computed(() => ['failed', 'indeterminate'].includes(props.work.status))
const isReady = computed(() => props.work.status === 'ready_to_resume')
const isResuming = computed(() => props.resuming || props.work.status === 'resuming')
const isCompleted = computed(() => props.work.status === 'resumed')

const statusTitle = computed(() => {
  if (props.work.status === 'submitting') return '正在提交后台索引任务'
  if (props.work.status === 'waiting_job') return '正在后台构建问答索引'
  if (props.work.status === 'ready_to_resume') return '索引已就绪，可以继续回答'
  if (props.work.status === 'resuming') return '正在恢复原问题'
  if (props.work.status === 'resumed') return '原问题已恢复完成'
  if (props.work.status === 'indeterminate') return '恢复结果暂时无法确定'
  if (props.work.status === 'failed') return '后台任务失败'
  return '后台任务已停止'
})

const stageText = computed(() => {
  if (isResuming.value) return '正在从原 LangGraph checkpoint 继续执行'
  if (isReady.value) return '后台索引已经验证完成，等待恢复 Agent'
  if (props.work.status === 'submitting') return '已保存授权，正在创建或挂接持久任务'
  return props.work.job?.stage_label || props.work.job?.current_stage || '等待后台任务状态'
})

const attemptText = computed(() => {
  const attemptNo = props.work.job?.attempt_no
  if (attemptNo == null) return ''
  const maxAttempts = props.work.job?.max_attempts
  return maxAttempts == null ? `第 ${attemptNo} 次尝试` : `第 ${attemptNo}/${maxAttempts} 次尝试`
})
</script>

<template>
  <article class="agent-work-card" :class="{ 'agent-work-card--failed': isFailed }">
    <div class="agent-work-card__header">
      <div>
        <p class="agent-work-card__eyebrow">后台 Agent 工作</p>
        <h3>{{ statusTitle }}</h3>
      </div>
      <el-tag v-if="attemptText" size="small" effect="plain">{{ attemptText }}</el-tag>
    </div>

    <div class="agent-work-card__summary">
      <strong>{{ work.display_summary.paper_title || work.display_summary.arxiv_id || '论文问答索引' }}</strong>
      <span v-if="work.display_summary.arxiv_id">arXiv: {{ work.display_summary.arxiv_id }}</span>
      <p v-if="work.display_summary.question_summary">原问题：{{ work.display_summary.question_summary }}</p>
    </div>

    <div v-if="!isFailed && !isCompleted" class="agent-work-card__progress" role="status">
      <div class="agent-work-card__progress-meta">
        <span>{{ stageText }}</span>
        <strong v-if="progress !== null">{{ progress }}%</strong>
      </div>
      <el-progress
        v-if="progress !== null"
        :percentage="progress"
        :status="work.job?.status === 'success' ? 'success' : undefined"
        :stroke-width="8"
      />
      <el-progress
        v-else
        :percentage="100"
        :show-text="false"
        :indeterminate="true"
        :duration="3"
        :stroke-width="8"
      />
      <p class="agent-work-card__hint">
        百分比来自后台 job 的实际阶段里程碑，不表示剩余耗时。
      </p>
    </div>

    <div v-if="isFailed" class="agent-work-card__error" role="alert">
      {{ errorMessage || '后台任务未完成。旧授权不会自动重试，如需继续请重新发起授权。' }}
    </div>

    <div class="agent-work-card__actions">
      <el-button
        v-if="isReady"
        type="primary"
        :loading="resuming"
        :disabled="resuming"
        @click="emit('resume', work)"
      >
        继续回答原问题
      </el-button>
      <el-button
        v-if="isFailed && work.display_summary.arxiv_id"
        :loading="retrying"
        :disabled="retrying"
        @click="emit('retry', work)"
      >
        重新发起授权
      </el-button>
      <el-button
        v-if="work.can_cancel"
        text
        :disabled="resuming"
        @click="emit('cancel', work)"
      >
        停止等待并不再继续回答
      </el-button>
    </div>
  </article>
</template>

<style scoped>
.agent-work-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px 18px;
  border: 1px solid var(--color-border);
  border-left: 3px solid var(--color-primary);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
  box-shadow: var(--shadow-xs);
}

.agent-work-card--failed {
  border-left-color: var(--color-danger);
}

.agent-work-card__header,
.agent-work-card__progress-meta,
.agent-work-card__actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.agent-work-card__eyebrow {
  margin: 0 0 2px;
  color: var(--color-text-3);
  font-size: 12px;
  font-weight: 550;
}

.agent-work-card h3 {
  margin: 0;
  color: var(--color-text);
  font-size: 15px;
  font-weight: 600;
}

.agent-work-card__summary {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
  color: var(--color-text-3);
  font-size: 13px;
}

.agent-work-card__summary strong {
  width: 100%;
  color: var(--color-text);
  font-size: 14px;
  font-weight: 550;
}

.agent-work-card__summary p,
.agent-work-card__hint {
  width: 100%;
  margin: 0;
  line-height: 1.6;
}

.agent-work-card__progress {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.agent-work-card__progress-meta {
  color: var(--color-text-2);
  font-size: 13px;
}

.agent-work-card__progress-meta strong {
  color: var(--color-text);
  font-variant-numeric: tabular-nums;
}

.agent-work-card__hint {
  color: var(--color-text-4);
  font-size: 12px;
}

.agent-work-card__error {
  padding: 10px 12px;
  border-radius: var(--radius);
  border: 1px solid var(--color-danger-border);
  color: #b91c1c;
  background: var(--color-danger-soft);
  font-size: 13px;
  line-height: 1.6;
}

.agent-work-card__actions {
  justify-content: flex-start;
  flex-wrap: wrap;
}
</style>
