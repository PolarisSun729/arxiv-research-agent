<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh, Star, View, Plus, Check, Close } from '@element-plus/icons-vue'
import { usePaperStore } from '@/stores/paperStore'
import type { UserResearchProfile } from '@/types/paper'
import { useUserContext } from '@/composables/useUserContext'

const store = usePaperStore()
const { canResearch } = useUserContext()
const saving = ref(false)
const rebuilding = ref(false)
const selectedTopic = ref('')
let pollTimer: number | null = null

const manualForm = reactive({
  positive_topic: '',
  negative_topic: '',
  pinned_topic: '',
  hidden_topic: '',
  preferred_categories: '',
  preferred_answer_style: '',
  common_question_types: ''
})

const detail = computed(() => store.researchProfileDetail)
const effectiveProfile = computed(() => detail.value?.effective_profile || store.researchProfile)
const generatedProfile = computed(() => detail.value?.generated_profile || null)
const manualProfile = computed(() => detail.value?.manual_profile || null)
const latestJob = computed(() => store.latestProfileBuildJob || detail.value?.latest_build_job || null)
const latestJobLogs = computed(() => (latestJob.value?.recent_logs || []).slice(-4).reverse())
const latestJobStageMessage = computed(() => latestJob.value?.stage_message || latestJob.value?.metrics?.stage_message || '')
const latestJobPaperProgress = computed(() => {
  const total = Number(latestJob.value?.total_papers || 0)
  if (!total) return ''
  const processed = Number(latestJob.value?.processed_papers || 0)
  return `论文证据卡：${processed} / ${total}`
})
const latestJobEventProgress = computed(() => {
  const counts = latestJob.value?.evidence_counts || latestJob.value?.metrics?.evidence_counts || {}
  const used = Number(counts.used_events || counts.profile_events || 0)
  const total = Number(counts.total_events || used || 0)
  if (!used && !total) return ''
  const skipped = Number(counts.skipped_events || 0)
  return skipped ? `事件：${used} / ${total}，跳过 ${skipped}` : `事件：${used} / ${total}`
})
const latestJobEvidenceRuntime = computed(() => {
  const job = latestJob.value
  if (!job) return ''
  const parts: string[] = []
  const concurrency = Number(job.evidence_concurrency || job.metrics?.evidence_concurrency || 0)
  const averageSeconds = Number(job.average_seconds_per_paper || job.metrics?.average_seconds_per_paper || 0)
  const totalSeconds = Number(job.total_evidence_extraction_seconds || job.metrics?.total_evidence_extraction_seconds || 0)
  const backoffCount = Number(job.rate_limit_backoff_count || job.metrics?.rate_limit_backoff_count || 0)
  if (concurrency) parts.push(`并发：${concurrency}`)
  if (averageSeconds) parts.push(`单篇均耗时：${averageSeconds.toFixed(1)} 秒`)
  if (totalSeconds) parts.push(`生成耗时：${totalSeconds.toFixed(1)} 秒`)
  if (backoffCount) parts.push(`限流退避：${backoffCount} 次`)
  return parts.join('，')
})
const latestJobFailureSummary = computed(() => {
  const failures = latestJob.value?.paper_evidence_failure_details || latestJob.value?.metrics?.paper_evidence_failure_details || []
  return Array.isArray(failures)
    ? failures.slice(-3).map(item => `${item.arxiv_id || 'unknown'}：${item.error_type || 'failed'}`).join('；')
    : ''
})
const selectedEvidence = computed(() => {
  const topic = selectedTopic.value
  return topic ? effectiveProfile.value?.topic_evidence?.[topic] || generatedProfile.value?.topic_evidence?.[topic] || null : null
})

function listOf(profile: UserResearchProfile | null | undefined, key: keyof UserResearchProfile): string[] {
  const value = profile?.[key]
  return Array.isArray(value) ? value.map(item => String(item)).filter(Boolean) : []
}

function parseList(value: string) {
  return value.split(/[\n,，]/).map(item => item.trim()).filter(Boolean)
}

function syncManualForm() {
  manualForm.preferred_categories = listOf(manualProfile.value, 'preferred_categories').join(', ')
  manualForm.preferred_answer_style = manualProfile.value?.preferred_answer_style || ''
  manualForm.common_question_types = listOf(manualProfile.value, 'common_question_types').join(', ')
}

const TERMINAL_JOB_STATUSES = ['completed', 'needs_review', 'failed']

const JOB_STATUS_LABELS: Record<string, string> = {
  pending: '排队中',
  queued: '排队中',
  running: '构建中',
  completed: '已完成',
  needs_review: '待复核',
  failed: '失败'
}
const jobStatusLabel = computed(() => {
  const status = latestJob.value?.status || ''
  return JOB_STATUS_LABELS[status] || status
})
const jobTone = computed(() => {
  const status = latestJob.value?.status
  if (status === 'completed') return 'success'
  if (status === 'failed') return 'danger'
  if (status === 'needs_review') return 'warning'
  return 'primary'
})

async function loadDetail() {
  await store.fetchResearchProfileDetail()
  syncManualForm()
  // 构建中途离开或刷新页面后重新进入时，接上未结束 job 的轮询，避免进度停在加载时的快照。
  const job = store.latestProfileBuildJob
  if (job?.job_id && !TERMINAL_JOB_STATUSES.includes(job.status)) startPolling(job.job_id)
}

async function saveManualPatch(extra: Partial<UserResearchProfile> = {}) {
  saving.value = true
  try {
    await store.saveResearchProfile({
      preferred_categories: parseList(manualForm.preferred_categories),
      preferred_answer_style: manualForm.preferred_answer_style.trim(),
      common_question_types: parseList(manualForm.common_question_types),
      ...extra
    })
    ElMessage.success('手动画像已保存')
    clearTopicInputs()
  } catch {
    ElMessage.error('保存手动画像失败')
  } finally {
    saving.value = false
  }
}

function clearTopicInputs() {
  manualForm.positive_topic = ''
  manualForm.negative_topic = ''
  manualForm.pinned_topic = ''
  manualForm.hidden_topic = ''
}

async function addManualTopic(field: 'positive_topics' | 'negative_topics' | 'pinned_topics' | 'hidden_topics', value: string) {
  const topic = value.trim()
  if (!topic) return
  const current = listOf(manualProfile.value, field)
  await saveManualPatch({ [field]: Array.from(new Set([...current, topic])) } as Partial<UserResearchProfile>)
}

async function removeManualTopic(field: 'positive_topics' | 'negative_topics' | 'pinned_topics' | 'hidden_topics', topic: string) {
  const next = listOf(manualProfile.value, field).filter(item => item !== topic)
  await saveManualPatch({ [field]: next } as Partial<UserResearchProfile>)
}

async function handleRebuild() {
  rebuilding.value = true
  try {
    const result = await store.rebuildResearchProfile()
    ElMessage.success('已创建画像重建任务')
    if (result.job?.job_id) startPolling(result.job.job_id)
  } catch {
    ElMessage.error('创建画像重建任务失败')
  } finally {
    rebuilding.value = false
  }
}

function startPolling(jobId: string) {
  if (pollTimer) window.clearInterval(pollTimer)
  pollTimer = window.setInterval(async () => {
    const job = await store.pollProfileBuildJob(jobId)
    if (TERMINAL_JOB_STATUSES.includes(job.status) && pollTimer) {
      window.clearInterval(pollTimer)
      pollTimer = null
    }
  }, 2000)
}

async function activateSnapshot(snapshotId: string) {
  await store.activateProfileSnapshot(snapshotId)
  ElMessage.success('已切换 active snapshot')
}

onMounted(loadDetail)
onUnmounted(() => {
  if (pollTimer) window.clearInterval(pollTimer)
})
</script>

<template>
  <div class="page profile-page">
    <header class="page-header">
      <div class="page-header__text">
        <h1 class="page-title">研究画像</h1>
        <p class="page-desc">系统根据你的阅读与标记行为自动生成画像，你可以手动修正。推荐、Agent 和论文问答都读取「最终生效画像」。</p>
      </div>
      <div class="page-actions">
        <el-button :icon="Refresh" @click="loadDetail">刷新</el-button>
        <el-button v-if="canResearch" type="primary" :loading="rebuilding" @click="handleRebuild">重新生成画像</el-button>
      </div>
    </header>

    <section v-if="latestJob" class="surface job-card">
      <div class="job-head">
        <span :class="['chip', `chip--${jobTone}`]">{{ jobStatusLabel }}</span>
        <span v-if="latestJob.current_stage && latestJob.current_stage !== latestJob.status" class="job-stage">{{ latestJob.current_stage }}</span>
        <span v-if="latestJobStageMessage" class="muted">{{ latestJobStageMessage }}</span>
        <span class="job-percent">{{ latestJob.progress || 0 }}%</span>
      </div>
      <el-progress
        :percentage="latestJob.progress || 0"
        :stroke-width="6"
        :show-text="false"
        :status="latestJob.status === 'failed' ? 'exception' : latestJob.status === 'completed' ? 'success' : undefined"
      />
      <div class="job-metrics">
        <span v-if="latestJob.build_mode">模式 {{ latestJob.build_mode }}</span>
        <span v-if="latestJobEventProgress">{{ latestJobEventProgress }}</span>
        <span v-if="latestJob.candidate_papers !== undefined">候选论文 {{ latestJob.candidate_papers }}</span>
        <span v-if="latestJobPaperProgress">{{ latestJobPaperProgress }}</span>
        <span v-if="latestJob.cached_papers !== undefined">缓存 {{ latestJob.cached_papers }}</span>
        <span v-if="latestJob.cache_hit_count !== undefined">缓存命中 {{ latestJob.cache_hit_count }}</span>
        <span v-if="latestJob.uncached_papers !== undefined">待生成 {{ latestJob.uncached_papers }}</span>
        <span v-if="latestJob.generated_count !== undefined">已生成 {{ latestJob.generated_count }}</span>
        <span v-if="latestJob.skipped_paper_count !== undefined">跳过 {{ latestJob.skipped_paper_count }}</span>
        <span v-if="latestJob.failed_papers !== undefined">失败 {{ latestJob.failed_papers }}</span>
        <span v-if="latestJobEvidenceRuntime">{{ latestJobEvidenceRuntime }}</span>
        <span v-if="latestJob.current_arxiv_id" class="mono">当前 {{ latestJob.current_arxiv_id }}</span>
      </div>
      <details v-if="latestJobLogs.length" class="job-logs">
        <summary>最近日志</summary>
        <div class="job-log-list">
          <span v-for="log in latestJobLogs" :key="`${log.ts || ''}-${log.message || ''}`">{{ log.message || log }}</span>
        </div>
      </details>
      <div v-if="latestJobFailureSummary" class="notice notice--danger">失败摘要：{{ latestJobFailureSummary }}</div>
      <div v-if="latestJob.error_message" class="notice notice--danger">{{ latestJob.error_message }}</div>
    </section>

    <div class="grid grid--main">
      <section class="surface panel">
        <header class="panel-head">
          <div>
            <h2 class="section-title">最终生效画像</h2>
            <p class="section-desc">点击主题查看它的来源证据</p>
          </div>
          <span class="chip chip--success">生效中</span>
        </header>
        <div class="topic-group">
          <h3>关注主题</h3>
          <div class="chip-row">
            <button
              v-for="topic in listOf(effectiveProfile, 'positive_topics')"
              :key="topic"
              type="button"
              :class="['topic-chip', 'topic-chip--primary', { 'is-selected': selectedTopic === topic }]"
              @click="selectedTopic = topic"
            >{{ topic }}</button>
            <span v-if="!listOf(effectiveProfile, 'positive_topics').length" class="empty-text">暂无</span>
          </div>
        </div>
        <div class="topic-group">
          <h3>近期关注</h3>
          <div class="chip-row">
            <button
              v-for="topic in listOf(effectiveProfile, 'recent_topics')"
              :key="topic"
              type="button"
              :class="['topic-chip', 'topic-chip--warning', { 'is-selected': selectedTopic === topic }]"
              @click="selectedTopic = topic"
            >{{ topic }}</button>
            <span v-if="!listOf(effectiveProfile, 'recent_topics').length" class="empty-text">暂无</span>
          </div>
        </div>
        <div class="topic-group">
          <h3>排除主题</h3>
          <div class="chip-row">
            <button
              v-for="topic in listOf(effectiveProfile, 'negative_topics')"
              :key="topic"
              type="button"
              :class="['topic-chip', 'topic-chip--danger', { 'is-selected': selectedTopic === topic }]"
              @click="selectedTopic = topic"
            >{{ topic }}</button>
            <span v-if="!listOf(effectiveProfile, 'negative_topics').length" class="empty-text">暂无</span>
          </div>
        </div>
      </section>

      <section class="surface panel evidence-panel">
        <header class="panel-head">
          <div>
            <h2 class="section-title">证据解释</h2>
            <p class="section-desc">{{ selectedTopic || '尚未选择主题' }}</p>
          </div>
          <el-icon class="muted"><View /></el-icon>
        </header>
        <template v-if="selectedTopic && selectedEvidence">
          <div class="evidence-scores">
            <div class="surface-muted evidence-score">
              <span>净分</span>
              <strong>{{ selectedEvidence.net_score ?? '-' }}</strong>
            </div>
            <div class="surface-muted evidence-score">
              <span>置信度</span>
              <strong>{{ selectedEvidence.confidence ?? '-' }}</strong>
            </div>
          </div>
          <div class="topic-group">
            <h3>来源论文</h3>
            <div class="chip-row">
              <span v-for="paper in selectedEvidence.source_papers || []" :key="paper" class="chip mono">{{ paper }}</span>
            </div>
          </div>
          <div class="topic-group">
            <h3>来源行为</h3>
            <div class="chip-row">
              <span v-for="action in selectedEvidence.source_actions || []" :key="action" class="chip">{{ action }}</span>
            </div>
          </div>
        </template>
        <el-empty v-else :image-size="72" :description="selectedTopic ? '该主题暂无证据记录' : '点击左侧主题查看证据'" />
      </section>
    </div>

    <div class="grid grid--half">
      <section class="surface panel">
        <header class="panel-head">
          <div>
            <h2 class="section-title">自动生成画像</h2>
            <p class="section-desc">由系统根据行为证据生成</p>
          </div>
          <div class="quality">
            <span class="muted">质量分</span>
            <strong>{{ generatedProfile?.review_status?.quality_score ?? generatedProfile?.quality_report?.quality_score ?? '-' }}</strong>
          </div>
        </header>
        <div class="topic-group">
          <h3>系统主题</h3>
          <div class="chip-row">
            <button
              v-for="topic in listOf(generatedProfile, 'positive_topics')"
              :key="topic"
              type="button"
              :class="['topic-chip', { 'is-selected': selectedTopic === topic }]"
              @click="selectedTopic = topic"
            >{{ topic }}</button>
            <span v-if="!listOf(generatedProfile, 'positive_topics').length" class="empty-text">暂无</span>
          </div>
        </div>
        <div class="topic-group">
          <h3>代表论文</h3>
          <div class="chip-row">
            <span v-for="paper in listOf(generatedProfile, 'representative_papers')" :key="paper" class="chip mono">{{ paper }}</span>
            <span v-if="!listOf(generatedProfile, 'representative_papers').length" class="empty-text">暂无</span>
          </div>
        </div>
      </section>

      <section class="surface panel">
        <header class="panel-head">
          <div>
            <h2 class="section-title">手动修正</h2>
            <p class="section-desc">手动设置优先于自动生成结果</p>
          </div>
        </header>

        <div class="form-grid">
          <div class="field">
            <span class="field-label">添加关注主题</span>
            <div class="input-row">
              <el-input v-model="manualForm.positive_topic" placeholder="例如 retrieval-augmented generation" @keyup.enter="addManualTopic('positive_topics', manualForm.positive_topic)" />
              <el-button :icon="Plus" @click="addManualTopic('positive_topics', manualForm.positive_topic)" />
            </div>
          </div>
          <div class="field">
            <span class="field-label">固定主题 <span class="muted">始终保留</span></span>
            <div class="input-row">
              <el-input v-model="manualForm.pinned_topic" placeholder="输入要固定的主题" @keyup.enter="addManualTopic('pinned_topics', manualForm.pinned_topic)" />
              <el-button :icon="Star" @click="addManualTopic('pinned_topics', manualForm.pinned_topic)" />
            </div>
            <div v-if="listOf(manualProfile, 'pinned_topics').length" class="chip-row">
              <el-tag v-for="topic in listOf(manualProfile, 'pinned_topics')" :key="topic" closable @close="removeManualTopic('pinned_topics', topic)">{{ topic }}</el-tag>
            </div>
          </div>
          <div class="field">
            <span class="field-label">隐藏主题 <span class="muted">不再出现在画像中</span></span>
            <div class="input-row">
              <el-input v-model="manualForm.hidden_topic" placeholder="输入要隐藏的主题" @keyup.enter="addManualTopic('hidden_topics', manualForm.hidden_topic)" />
              <el-button :icon="Close" @click="addManualTopic('hidden_topics', manualForm.hidden_topic)" />
            </div>
            <div v-if="listOf(manualProfile, 'hidden_topics').length" class="chip-row">
              <el-tag v-for="topic in listOf(manualProfile, 'hidden_topics')" :key="topic" type="danger" closable @close="removeManualTopic('hidden_topics', topic)">{{ topic }}</el-tag>
            </div>
          </div>
          <div class="field">
            <span class="field-label">偏好分类</span>
            <el-input v-model="manualForm.preferred_categories" placeholder="例如 cs.CL, cs.AI" />
          </div>
          <div class="field">
            <span class="field-label">偏好回答风格</span>
            <el-input v-model="manualForm.preferred_answer_style" placeholder="例如 先给结论，再给公式推导" />
          </div>
        </div>
        <div class="form-actions">
          <el-button type="primary" :icon="Check" :loading="saving" @click="saveManualPatch()">保存修正</el-button>
        </div>
      </section>
    </div>

    <section class="surface panel">
      <header class="panel-head">
        <div>
          <h2 class="section-title">构建记录</h2>
          <p class="section-desc">每次重新生成都会保存快照，可切换到任一历史版本</p>
        </div>
      </header>
      <div v-if="detail?.snapshots?.length" class="snapshot-list">
        <div v-for="snapshot in detail.snapshots" :key="snapshot.snapshot_id" :class="['snapshot-row', { 'is-active': snapshot.active }]">
          <div class="snapshot-meta">
            <strong class="mono">{{ snapshot.snapshot_id }}</strong>
            <span class="muted">{{ snapshot.created_at }}</span>
          </div>
          <span v-if="snapshot.active" class="chip chip--success">当前使用</span>
          <el-button v-else size="small" @click="activateSnapshot(snapshot.snapshot_id)">切换到此版本</el-button>
        </div>
      </div>
      <p v-else class="empty-text">暂无构建记录</p>
    </section>
  </div>
</template>

<style scoped>
.job-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px 20px;
  margin-bottom: 16px;
}

.job-head {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 13px;
}

.job-stage {
  font-weight: 550;
  color: var(--color-text);
}

.job-percent {
  margin-left: auto;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.job-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  font-size: 12.5px;
  color: var(--color-text-3);
}

.job-logs summary {
  cursor: pointer;
  font-size: 12.5px;
  color: var(--color-primary);
}

.job-log-list {
  display: grid;
  gap: 4px;
  margin-top: 6px;
  font-size: 12px;
  color: var(--color-text-3);
  font-family: var(--font-mono);
}

.grid {
  display: grid;
  gap: 16px;
  margin-bottom: 16px;
}

.grid--main {
  grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
}

.grid--half {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.panel {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 18px 20px;
}

.panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.topic-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.topic-group h3 {
  font-size: 12.5px;
  font-weight: 550;
  color: var(--color-text-3);
}

.topic-chip {
  --chip-color: var(--color-text-2);
  --chip-bg: var(--color-surface-sunken);
  height: 26px;
  padding: 0 10px;
  border: 1px solid transparent;
  border-radius: var(--radius-pill);
  background: var(--chip-bg);
  color: var(--chip-color);
  font: inherit;
  font-size: 12.5px;
  cursor: pointer;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.topic-chip--primary {
  --chip-color: var(--color-primary);
  --chip-bg: var(--color-primary-soft);
}

.topic-chip--warning {
  --chip-color: var(--color-warning);
  --chip-bg: var(--color-warning-soft);
}

.topic-chip--danger {
  --chip-color: var(--color-danger);
  --chip-bg: var(--color-danger-soft);
}

.topic-chip:hover {
  border-color: currentColor;
}

.topic-chip.is-selected {
  border-color: currentColor;
  box-shadow: 0 0 0 2px var(--color-surface), 0 0 0 3px currentColor;
}

.empty-text {
  font-size: 13px;
  color: var(--color-text-4);
}

.evidence-scores {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.evidence-score {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 12px;
  font-size: 12px;
  color: var(--color-text-3);
}

.evidence-score strong {
  font-size: 20px;
  font-weight: 650;
  color: var(--color-text);
  font-variant-numeric: tabular-nums;
}

.quality {
  display: flex;
  align-items: baseline;
  gap: 6px;
  font-size: 12.5px;
}

.quality strong {
  font-size: 18px;
  color: var(--color-text);
}

.form-grid {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field-label {
  font-size: 12.5px;
  font-weight: 550;
  color: var(--color-text-2);
}

.field-label .muted {
  font-weight: 400;
  margin-left: 4px;
}

.input-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
}

.form-actions {
  display: flex;
  justify-content: flex-end;
}

.snapshot-list {
  display: flex;
  flex-direction: column;
}

.snapshot-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 0;
  border-top: 1px solid var(--color-border);
}

.snapshot-row:first-child {
  border-top: none;
}

.snapshot-meta {
  display: flex;
  flex-direction: column;
  min-width: 0;
  font-size: 12.5px;
}

.snapshot-meta strong {
  font-weight: 550;
  color: var(--color-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

@media (max-width: 1100px) {
  .grid--main,
  .grid--half {
    grid-template-columns: 1fr;
  }
}
</style>
