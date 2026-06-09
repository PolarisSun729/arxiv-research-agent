<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh, Star, View, Plus, Check, Close } from '@element-plus/icons-vue'
import { usePaperStore } from '@/stores/paperStore'
import type { UserResearchProfile } from '@/types/paper'

const store = usePaperStore()
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

async function loadDetail() {
  await store.fetchResearchProfileDetail()
  syncManualForm()
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
    if (['completed', 'needs_review', 'failed'].includes(job.status) && pollTimer) {
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
  <div class="profile-page">
    <section class="profile-toolbar">
      <div>
        <h1>研究画像</h1>
        <p>自动画像、手动修正和最终生效画像分层展示；推荐、Agent 和论文问答读取最终生效画像。</p>
      </div>
      <div class="profile-toolbar__actions">
        <el-button :icon="Refresh" @click="loadDetail">刷新</el-button>
        <el-button type="primary" :icon="Refresh" :loading="rebuilding" @click="handleRebuild">重新生成</el-button>
      </div>
    </section>

    <section v-if="latestJob" class="status-strip">
      <span>构建状态：{{ latestJob.status }}</span>
      <el-progress :percentage="latestJob.progress || 0" :stroke-width="8" />
      <span v-if="latestJob.current_stage">阶段：{{ latestJob.current_stage }}</span>
      <span v-if="latestJob.error_message" class="status-strip__error">{{ latestJob.error_message }}</span>
    </section>

    <section class="profile-grid">
      <div class="profile-panel">
        <header><h2>最终生效画像</h2><el-tag type="success">effective</el-tag></header>
        <div class="topic-section">
          <h3>正向主题</h3>
          <el-tag v-for="topic in listOf(effectiveProfile, 'positive_topics')" :key="topic" @click="selectedTopic = topic">{{ topic }}</el-tag>
        </div>
        <div class="topic-section">
          <h3>负向主题</h3>
          <el-tag v-for="topic in listOf(effectiveProfile, 'negative_topics')" :key="topic" type="danger" @click="selectedTopic = topic">{{ topic }}</el-tag>
        </div>
        <div class="topic-section">
          <h3>近期关注</h3>
          <el-tag v-for="topic in listOf(effectiveProfile, 'recent_topics')" :key="topic" type="warning" @click="selectedTopic = topic">{{ topic }}</el-tag>
        </div>
      </div>

      <div class="profile-panel">
        <header><h2>自动生成画像</h2><el-tag>generated</el-tag></header>
        <div class="metric-row">
          <span>质量分</span>
          <strong>{{ generatedProfile?.review_status?.quality_score ?? generatedProfile?.quality_report?.quality_score ?? '-' }}</strong>
        </div>
        <div class="topic-section">
          <h3>系统主题</h3>
          <el-tag v-for="topic in listOf(generatedProfile, 'positive_topics')" :key="topic" @click="selectedTopic = topic">{{ topic }}</el-tag>
        </div>
        <div class="topic-section">
          <h3>代表论文</h3>
          <span v-for="paper in listOf(generatedProfile, 'representative_papers')" :key="paper" class="paper-id">{{ paper }}</span>
        </div>
      </div>

      <div class="profile-panel">
        <header><h2>手动修正</h2><el-tag type="info">manual</el-tag></header>
        <div class="manual-row">
          <el-input v-model="manualForm.positive_topic" placeholder="添加正向 topic" />
          <el-button :icon="Plus" @click="addManualTopic('positive_topics', manualForm.positive_topic)" />
        </div>
        <div class="manual-row">
          <el-input v-model="manualForm.pinned_topic" placeholder="固定 topic" />
          <el-button :icon="Star" @click="addManualTopic('pinned_topics', manualForm.pinned_topic)" />
        </div>
        <div class="manual-row">
          <el-input v-model="manualForm.hidden_topic" placeholder="隐藏 topic" />
          <el-button :icon="Close" @click="addManualTopic('hidden_topics', manualForm.hidden_topic)" />
        </div>
        <el-input v-model="manualForm.preferred_answer_style" placeholder="偏好回答风格" />
        <el-input v-model="manualForm.preferred_categories" placeholder="偏好分类，例如 cs.CL, cs.AI" />
        <el-button type="primary" :icon="Check" :loading="saving" @click="saveManualPatch()">保存手动修正</el-button>
        <div class="topic-section">
          <h3>已固定</h3>
          <el-tag v-for="topic in listOf(manualProfile, 'pinned_topics')" :key="topic" closable @close="removeManualTopic('pinned_topics', topic)">{{ topic }}</el-tag>
        </div>
        <div class="topic-section">
          <h3>已隐藏</h3>
          <el-tag v-for="topic in listOf(manualProfile, 'hidden_topics')" :key="topic" type="danger" closable @close="removeManualTopic('hidden_topics', topic)">{{ topic }}</el-tag>
        </div>
      </div>
    </section>

    <section class="profile-grid profile-grid--bottom">
      <div class="profile-panel">
        <header><h2>证据解释</h2><el-tag :icon="View">{{ selectedTopic || '选择 topic' }}</el-tag></header>
        <template v-if="selectedTopic && selectedEvidence">
          <div class="metric-row"><span>净分</span><strong>{{ selectedEvidence.net_score ?? '-' }}</strong></div>
          <div class="metric-row"><span>置信度</span><strong>{{ selectedEvidence.confidence ?? '-' }}</strong></div>
          <div class="topic-section">
            <h3>来源论文</h3>
            <span v-for="paper in selectedEvidence.source_papers || []" :key="paper" class="paper-id">{{ paper }}</span>
          </div>
          <div class="topic-section">
            <h3>来源行为</h3>
            <el-tag v-for="action in selectedEvidence.source_actions || []" :key="action" type="info">{{ action }}</el-tag>
          </div>
        </template>
        <el-empty v-else description="点击画像 topic 查看证据" />
      </div>

      <div class="profile-panel">
        <header><h2>构建记录</h2><el-tag>snapshots</el-tag></header>
        <div v-for="snapshot in detail?.snapshots || []" :key="snapshot.snapshot_id" class="snapshot-row">
          <div>
            <strong>{{ snapshot.snapshot_id }}</strong>
            <span>{{ snapshot.created_at }}</span>
          </div>
          <el-button v-if="!snapshot.active" size="small" @click="activateSnapshot(snapshot.snapshot_id)">设为 active</el-button>
          <el-tag v-else type="success">active</el-tag>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.profile-page { display: flex; flex-direction: column; gap: 16px; }
.profile-toolbar, .status-strip, .profile-panel { border: 1px solid #dbe3ef; background: #fff; border-radius: 8px; padding: 16px; }
.profile-toolbar { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; }
.profile-toolbar h1 { margin: 0 0 6px; font-size: 24px; color: #111827; }
.profile-toolbar p { margin: 0; color: #64748b; }
.profile-toolbar__actions { display: flex; gap: 8px; }
.status-strip { display: grid; grid-template-columns: auto minmax(160px, 1fr) auto auto; gap: 12px; align-items: center; }
.status-strip__error { color: #dc2626; }
.profile-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.profile-grid--bottom { grid-template-columns: 1fr 1fr; }
.profile-panel { min-width: 0; display: flex; flex-direction: column; gap: 12px; }
.profile-panel header { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.profile-panel h2 { margin: 0; font-size: 16px; color: #111827; }
.topic-section { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.topic-section h3 { flex-basis: 100%; margin: 0; font-size: 13px; color: #64748b; }
.manual-row { display: grid; grid-template-columns: 1fr 36px; gap: 8px; }
.metric-row, .snapshot-row { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
.snapshot-row { padding: 10px 0; border-top: 1px solid #eef2f7; }
.snapshot-row div { display: flex; flex-direction: column; min-width: 0; }
.snapshot-row strong { font-size: 12px; color: #334155; overflow: hidden; text-overflow: ellipsis; }
.snapshot-row span, .paper-id { font-size: 12px; color: #64748b; }
.paper-id { padding: 2px 6px; border: 1px solid #dbe3ef; border-radius: 6px; }
@media (max-width: 1100px) {
  .profile-grid, .profile-grid--bottom { grid-template-columns: 1fr; }
  .status-strip { grid-template-columns: 1fr; }
  .profile-toolbar { flex-direction: column; }
}
</style>
