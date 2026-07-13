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
  gap: 14px;
  padding: 16px;
  border: 1px solid rgba(14, 116, 144, 0.18);
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(240, 249, 255, 0.96), #fff);
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
}

.agent-work-card--failed {
  border-color: rgba(220, 38, 38, 0.2);
  background: linear-gradient(135deg, rgba(254, 242, 242, 0.96), #fff);
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
  margin: 0 0 4px;
  color: #0891b2;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.1em;
}

.agent-work-card h3 {
  margin: 0;
  color: #0f172a;
  font-size: 16px;
}

.agent-work-card__summary {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  color: #475569;
  font-size: 13px;
}

.agent-work-card__summary strong {
  width: 100%;
  color: #0f172a;
  font-size: 14px;
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
  gap: 8px;
}

.agent-work-card__progress-meta {
  color: #475569;
  font-size: 13px;
}

.agent-work-card__progress-meta strong {
  color: #0f172a;
  font-variant-numeric: tabular-nums;
}

.agent-work-card__hint {
  color: #94a3b8;
  font-size: 12px;
}

.agent-work-card__error {
  padding: 10px 12px;
  border-radius: 10px;
  color: #b91c1c;
  background: rgba(254, 226, 226, 0.72);
  font-size: 13px;
  line-height: 1.6;
}

.agent-work-card__actions {
  justify-content: flex-start;
  flex-wrap: wrap;
}
</style>
