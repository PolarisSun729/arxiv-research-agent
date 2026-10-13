<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import type { AgentRunProgress } from '@/types/agent'
import { formatElapsed, getActiveEntry } from '@/utils/agentRunProgress'

const props = defineProps<{
  progress: AgentRunProgress
}>()

// 单个步骤超过这个时长时补一句解释，避免用户把 PDF 解析、大模型推理等慢步骤误认为卡死。
const SLOW_STEP_MS = 15_000

const now = ref(Date.now())
let timer: ReturnType<typeof setInterval> | null = null

const running = computed(() => props.progress.status === 'running')
const activeEntry = computed(() => getActiveEntry(props.progress))
// 运行中默认展开，结束后默认收起；用户手动切换后以用户选择为准。
const userExpanded = ref<boolean | null>(null)
const expanded = computed(() => userExpanded.value ?? running.value)

watch(
  running,
  value => {
    if (value && !timer) {
      timer = setInterval(() => { now.value = Date.now() }, 500)
    } else if (!value && timer) {
      clearInterval(timer)
      timer = null
    }
  },
  { immediate: true }
)

onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
})

const totalElapsed = computed(() => (props.progress.endedAt ?? now.value) - props.progress.startedAt)

const headline = computed(() => {
  if (running.value) return activeEntry.value ? `正在${activeEntry.value.label}…` : '正在连接 Agent…'
  if (props.progress.status === 'failed') return '执行中断'
  return '执行完成'
})

const stepCount = computed(() => props.progress.entries.length)

const activeSlow = computed(() => {
  const entry = activeEntry.value
  return Boolean(entry && now.value - entry.startedAt > SLOW_STEP_MS)
})

function entryElapsed(entry: { startedAt: number, endedAt?: number | null }) {
  return formatElapsed((entry.endedAt ?? now.value) - entry.startedAt)
}

function toggle() {
  userExpanded.value = !expanded.value
}
</script>

<template>
  <div class="run-timeline" :class="`run-timeline--${progress.status}`">
    <button type="button" class="run-timeline__head" :aria-expanded="expanded" @click="toggle">
      <span class="run-timeline__icon" aria-hidden="true" />
      <span class="run-timeline__headline">{{ headline }}</span>
      <span class="run-timeline__meta">
        <template v-if="stepCount">{{ stepCount }} 步 · </template>{{ formatElapsed(totalElapsed) }}
      </span>
      <span class="run-timeline__chevron" :class="{ 'run-timeline__chevron--open': expanded }" aria-hidden="true" />
    </button>

    <ol v-if="expanded && progress.entries.length" class="run-timeline__steps">
      <li
        v-for="entry in progress.entries"
        :key="entry.key"
        class="run-step"
        :class="`run-step--${entry.status}`"
      >
        <span class="run-step__dot" aria-hidden="true" />
        <div class="run-step__body">
          <div class="run-step__row">
            <span class="run-step__label">{{ entry.label }}</span>
            <span class="run-step__time">{{ entryElapsed(entry) }}</span>
          </div>
          <div v-if="entry.facts.length" class="run-step__facts">
            <span v-for="fact in entry.facts" :key="fact" class="run-step__fact">{{ fact }}</span>
          </div>
        </div>
      </li>
    </ol>

    <p v-if="running && activeSlow" class="run-timeline__hint">
      这一步需要调用大模型或外部服务，可能要几十秒，Agent 仍在正常运行。
    </p>
  </div>
</template>

<style scoped>
.run-timeline {
  --tone: var(--color-primary);
  margin-bottom: 10px;
  padding: 10px 12px;
  border: 1px solid var(--color-border);
  border-radius: 12px;
  background: var(--color-surface-muted);
  font-size: 13px;
}

.run-timeline--done { --tone: var(--color-success); }
.run-timeline--failed { --tone: var(--color-danger); }

.run-timeline__head {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 0;
  border: 0;
  background: none;
  cursor: pointer;
  font-family: inherit;
  text-align: left;
}

.run-timeline__icon {
  flex: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 2px solid var(--tone);
}

.run-timeline--running .run-timeline__icon {
  border-color: color-mix(in srgb, var(--tone) 25%, transparent);
  border-top-color: var(--tone);
  animation: run-spin 0.8s linear infinite;
}

.run-timeline--done .run-timeline__icon,
.run-timeline--failed .run-timeline__icon {
  background: var(--tone);
}

.run-timeline__headline {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  color: var(--color-text);
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.run-timeline__meta {
  flex: none;
  color: var(--color-text-4);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.run-timeline__chevron {
  flex: none;
  width: 7px;
  height: 7px;
  margin: 0 4px 3px;
  border-right: 1.5px solid var(--color-text-4);
  border-bottom: 1.5px solid var(--color-text-4);
  transform: rotate(45deg);
  transition: transform 0.2s;
}

.run-timeline__chevron--open {
  margin-bottom: -3px;
  transform: rotate(-135deg);
}

.run-timeline__steps {
  margin: 10px 0 0;
  padding: 10px 0 0;
  border-top: 1px solid var(--color-surface-sunken);
  list-style: none;
}

.run-step {
  --dot: var(--color-success);
  position: relative;
  display: flex;
  gap: 10px;
  padding-bottom: 10px;
}

.run-step::before {
  content: '';
  position: absolute;
  top: 12px;
  bottom: -2px;
  left: 3px;
  width: 1px;
  background: var(--color-border);
}

.run-step:last-child {
  padding-bottom: 0;
}

.run-step:last-child::before {
  display: none;
}

.run-step--running { --dot: var(--color-primary); }
.run-step--failed { --dot: var(--color-danger); }
.run-step--skipped { --dot: var(--color-text-4); }

.run-step__dot {
  flex: none;
  width: 7px;
  height: 7px;
  margin-top: 6px;
  border-radius: 50%;
  background: var(--dot);
}

.run-step--running .run-step__dot {
  animation: run-pulse 1.2s ease-in-out infinite;
}

.run-step__body {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  flex: 1;
}

.run-step__row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}

.run-step__label {
  color: var(--color-text-2);
  line-height: 1.5;
}

.run-step--running .run-step__label {
  color: var(--color-primary-hover);
  font-weight: 600;
}

.run-step--failed .run-step__label {
  color: #b91c1c;
}

.run-step__time {
  flex: none;
  color: var(--color-text-4);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.run-step__facts {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.run-step__fact {
  max-width: 100%;
  padding: 1px 8px;
  border-radius: 6px;
  background: var(--color-surface-sunken);
  color: var(--color-text-2);
  font-size: 12px;
  line-height: 1.6;
  overflow-wrap: anywhere;
}

.run-timeline__hint {
  margin: 8px 0 0;
  color: var(--color-text-3);
  font-size: 12px;
  line-height: 1.6;
}

@keyframes run-spin {
  to { transform: rotate(360deg); }
}

@keyframes run-pulse {
  0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--dot) 45%, transparent); }
  50% { box-shadow: 0 0 0 4px color-mix(in srgb, var(--dot) 0%, transparent); }
}

@media (prefers-reduced-motion: reduce) {
  .run-timeline--running .run-timeline__icon,
  .run-step--running .run-step__dot {
    animation: none;
  }
}
</style>
