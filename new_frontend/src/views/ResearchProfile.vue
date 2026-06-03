<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { usePaperStore } from '@/stores/paperStore'

const store = usePaperStore()
const saving = ref(false)

const form = reactive({
  positive_topics: '',
  negative_topics: '',
  recent_topics: '',
  preferred_categories: '',
  preferred_answer_style: '',
  common_question_types: '',
  representative_papers: ''
})

const hasProfile = computed(() => Boolean(store.researchProfile))

function joinList(values?: string[]) {
  return Array.isArray(values) ? values.join(', ') : ''
}

function parseList(value: string) {
  return value
    .split(/[\n,，;]/)
    .map(item => item.trim())
    .filter(Boolean)
}

function syncForm() {
  const profile = store.researchProfile
  form.positive_topics = joinList(profile?.positive_topics)
  form.negative_topics = joinList(profile?.negative_topics)
  form.recent_topics = joinList(profile?.recent_topics)
  form.preferred_categories = joinList(profile?.preferred_categories)
  form.preferred_answer_style = profile?.preferred_answer_style || ''
  form.common_question_types = joinList(profile?.common_question_types)
  form.representative_papers = joinList(profile?.representative_papers)
}

async function loadProfile() {
  await store.fetchResearchProfile()
  syncForm()
}

async function handleSave() {
  saving.value = true
  try {
    await store.saveResearchProfile({
      positive_topics: parseList(form.positive_topics),
      negative_topics: parseList(form.negative_topics),
      recent_topics: parseList(form.recent_topics),
      preferred_categories: parseList(form.preferred_categories),
      preferred_answer_style: form.preferred_answer_style.trim(),
      common_question_types: parseList(form.common_question_types),
      representative_papers: parseList(form.representative_papers)
    })
    ElMessage.success('研究画像已保存')
    syncForm()
  } catch (error) {
    ElMessage.error('保存研究画像失败')
  } finally {
    saving.value = false
  }
}

onMounted(loadProfile)
</script>

<template>
  <div class="profile-page">
    <section class="profile-hero">
      <div>
        <h1>研究画像</h1>
        <p>
          显式维护你的长期研究兴趣、负向主题、偏好分类和回答风格。系统会轻量用于推荐、Agent 搜索和论文问答表述。
        </p>
      </div>
      <el-tag type="info" effect="plain">
        {{ hasProfile ? '已加载画像' : '尚未设置画像' }}
      </el-tag>
    </section>

    <el-card class="profile-card" shadow="never">
      <el-form label-position="top">
        <el-row :gutter="16">
          <el-col :md="12" :sm="24">
            <el-form-item label="正向主题">
              <el-input v-model="form.positive_topics" type="textarea" :rows="3" placeholder="例如：RAG, agent, long context" />
            </el-form-item>
          </el-col>
          <el-col :md="12" :sm="24">
            <el-form-item label="负向主题">
              <el-input v-model="form.negative_topics" type="textarea" :rows="3" placeholder="例如：纯视觉生成, 硬件设计" />
            </el-form-item>
          </el-col>
          <el-col :md="12" :sm="24">
            <el-form-item label="近期关注主题">
              <el-input v-model="form.recent_topics" type="textarea" :rows="3" placeholder="例如：memory routing, tool use" />
            </el-form-item>
          </el-col>
          <el-col :md="12" :sm="24">
            <el-form-item label="偏好分类">
              <el-input v-model="form.preferred_categories" placeholder="例如：cs.AI, cs.CL, cs.IR" />
            </el-form-item>
          </el-col>
          <el-col :md="12" :sm="24">
            <el-form-item label="偏好回答风格">
              <el-input v-model="form.preferred_answer_style" placeholder="例如：简洁结构化、先结论后细节" />
            </el-form-item>
          </el-col>
          <el-col :md="12" :sm="24">
            <el-form-item label="常见提问类型">
              <el-input v-model="form.common_question_types" placeholder="例如：方法对比, 实验解读, 局限性" />
            </el-form-item>
          </el-col>
          <el-col :span="24">
            <el-form-item label="代表性论文">
              <el-input v-model="form.representative_papers" type="textarea" :rows="3" placeholder="填写 arXiv ID、标题或关键代表论文，逗号分隔" />
            </el-form-item>
          </el-col>
        </el-row>

        <div class="profile-actions">
          <el-button @click="loadProfile">重新加载</el-button>
          <el-button type="primary" :loading="saving" @click="handleSave">保存画像</el-button>
        </div>
      </el-form>
    </el-card>
  </div>
</template>

<style scoped>
.profile-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.profile-hero {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.profile-hero h1 {
  margin: 0 0 8px;
  font-size: 28px;
  color: #0f172a;
}

.profile-hero p {
  margin: 0;
  max-width: 860px;
  color: #475569;
  line-height: 1.7;
}

.profile-card {
  border-radius: 20px;
}

.profile-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

@media (max-width: 900px) {
  .profile-hero {
    flex-direction: column;
  }
}
</style>
