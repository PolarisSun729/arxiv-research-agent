<script setup lang="ts">
import { useRouter } from 'vue-router'
import PaperCard from '@/components/PaperCard.vue'
import RagChatPanel from '@/components/rag-chat/RagChatPanel.vue'
import { useAgentSearchChat } from '@/composables/useAgentSearchChat'
import type { AgentPaper, ArxivSearchResponse } from '@/types/agent'
import type { Paper } from '@/types/paper'

const router = useRouter()

const {
  inputMessage,
  loading,
  messages,
  setInputMessage,
  submitMessage,
  clearConversation
} = useAgentSearchChat()

const quickPrompts = [
  '帮我找最近 7 天关于 RAG 的 5 篇论文',
  '检索和 agent search 相关的 arXiv 论文',
  '找一些关于多模态检索增强生成的论文'
]

function normalizePaper(raw: AgentPaper): Paper {
  const arxivId = raw.arxiv_id || raw.arxivId || (raw.id ? String(raw.id).split('/').pop() : '')
  const authors = Array.isArray(raw.authors)
    ? raw.authors
    : String(raw.authors || '')
        .split(',')
        .map((author: string) => author.trim())
        .filter(Boolean)
  const categories = Array.isArray(raw.categories)
    ? raw.categories
    : String(raw.categories || '')
        .split(',')
        .map((category: string) => category.trim())
        .filter(Boolean)

  return {
    id: arxivId || raw.id || '',
    arxivId: arxivId || raw.id || '',
    title: raw.title || '',
    authors,
    summary: raw.abstract || raw.summary || '',
    publishedAt: raw.publishedAt || raw.published || raw.published_date || '',
    updatedAt: raw.updatedAt || raw.updated || '',
    categories,
    pdfUrl: raw.pdfUrl || raw.pdf_url || '',
    absUrl: raw.absUrl || raw.abs_url || raw.url || ''
  }
}

function normalizePapers(rawPapers: AgentPaper[] | null | undefined) {
  return (rawPapers || []).map(normalizePaper)
}

function formatJson(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function formatToolCallStatus(status: string) {
  if (status === 'success') return '成功'
  if (status === 'failed') return '失败'
  return status || 'unknown'
}

function handleViewDetail(id: string) {
  router.push(`/paper/${id}`)
}

function handlePromptSelect(prompt: string) {
  setInputMessage(prompt)
}

function handleClear() {
  clearConversation()
}

function getResponsePapers(rawPapers: AgentPaper[] | null | undefined) {
  return normalizePapers(rawPapers)
}

// 结果跟随对应的助手消息渲染，避免再退回到页面底部的独立结果面板。

function getResponseData(response: unknown): ArxivSearchResponse {
  return (response || {}) as ArxivSearchResponse
}

function getResponseAnswer(response: unknown) {
  return getResponseData(response).answer || ''
}

function getResponseIntent(response: unknown) {
  return getResponseData(response).intent || '-'
}

function getResponseWarnings(response: unknown) {
  return getResponseData(response).warnings || []
}

function getResponseNextActions(response: unknown) {
  return getResponseData(response).next_actions || []
}

function getResponsePlan(response: unknown) {
  return getResponseData(response).plan || []
}

function getResponseSearchSpec(response: unknown) {
  return getResponseData(response).search_spec || null
}

function getResponseToolCalls(response: unknown) {
  return getResponseData(response).tool_calls || []
}

function getResponsePaperCount(response: unknown) {
  return getResponsePapers(getResponseData(response).papers).length
}
</script>

<template>
  <div class="agent-search-page">
    <section class="hero-card">
      <div class="hero-copy">
        <p class="eyebrow">Agent Search</p>
        <h1 class="title">自然语言 arXiv 搜索入口</h1>
        <p class="subtitle">
          直接说你的检索需求，系统会先理解意图、生成搜索计划、调用工具，然后把完整响应嵌回到对应的助手消息里。
        </p>
      </div>

      <div class="hero-actions">
        <el-button text :disabled="loading || messages.length === 0" @click="handleClear">
          清空对话
        </el-button>
      </div>
    </section>

    <section class="conversation-shell">
      <RagChatPanel
        v-model:sender-text="inputMessage"
        :messages="messages"
        :loading="loading"
        :quick-prompts="quickPrompts"
        title="Agent 对话"
        description="结果不再单独堆在页面下方，而是直接嵌到助手回复中，按消息查看更符合聊天语义。"
        prompt-title="搜索示例"
        assistant-label="Agent"
        sender-placeholder="请输入自然语言搜索需求，Enter 发送，Shift+Enter 换行"
        @submit-question="submitMessage"
        @select-prompt="handlePromptSelect"
      >
        <template #message-footer="{ item }">
          <div v-if="item.role === 'assistant' && item.response" class="agent-response-inline">
            <div class="agent-response-inline__head">
              <div>
                <div class="agent-response-inline__title">结果详情</div>
                <div class="agent-response-inline__subtitle">
                  当前这次搜索的完整响应，直接挂在这条助手消息下面，不再做页面外置侧栏。
                </div>
              </div>

              <div class="agent-response-inline__meta">
                <el-tag size="small" effect="plain" type="success">已返回</el-tag>
                <el-tag size="small" effect="plain" type="info">
                  {{ getResponsePaperCount(item.response) }} 篇论文
                </el-tag>
              </div>
            </div>

            <div class="agent-response-inline__answer">
              <div class="section-label">answer</div>
              <div class="answer-text">{{ getResponseAnswer(item.response) }}</div>
            </div>

            <div class="agent-response-inline__chips">
              <div class="summary-chip">
                <span class="summary-chip__label">intent</span>
                <strong>{{ getResponseIntent(item.response) }}</strong>
              </div>
              <div class="summary-chip">
                <span class="summary-chip__label">warnings</span>
                <strong>{{ getResponseWarnings(item.response).length }}</strong>
              </div>
              <div class="summary-chip">
                <span class="summary-chip__label">next_actions</span>
                <strong>{{ getResponseNextActions(item.response).length }}</strong>
              </div>
              <div class="summary-chip">
                <span class="summary-chip__label">tool_calls</span>
                <strong>{{ getResponseToolCalls(item.response).length }}</strong>
              </div>
            </div>

            <div class="agent-response-inline__detail-grid">
              <div class="detail-block">
                <div class="section-label">意图与计划</div>
                <div class="detail-section">
                  <div class="detail-section__label">intent</div>
                  <div class="detail-section__value">{{ getResponseIntent(item.response) }}</div>
                </div>
                <div class="detail-section">
                  <div class="detail-section__label">next_actions</div>
                  <ul v-if="getResponseNextActions(item.response).length" class="bullet-list">
                    <li v-for="action in getResponseNextActions(item.response)" :key="action">
                      {{ action }}
                    </li>
                  </ul>
                  <div v-else class="empty-state">暂无 next_actions</div>
                </div>
                <div class="detail-section">
                  <div class="detail-section__label">plan</div>
                  <ul v-if="getResponsePlan(item.response).length" class="bullet-list">
                    <li v-for="step in getResponsePlan(item.response)" :key="step">
                      {{ step }}
                    </li>
                  </ul>
                  <div v-else class="empty-state">暂无 plan</div>
                </div>
              </div>

              <div class="detail-block">
                <div class="section-label">search_spec</div>
                <pre class="json-block">
{{ getResponseSearchSpec(item.response) ? formatJson(getResponseSearchSpec(item.response)) : 'null' }}
                </pre>
              </div>

              <div class="detail-block">
                <div class="section-label">工具调用</div>
                <div v-if="!getResponseToolCalls(item.response).length" class="empty-state">暂无工具调用记录</div>
                <div v-else class="tool-call-list">
                  <div
                    v-for="(call, index) in getResponseToolCalls(item.response)"
                    :key="index"
                    class="tool-call-item"
                  >
                    <div class="tool-call-top">
                      <strong>{{ call.tool_name }}</strong>
                      <el-tag size="small" :type="call.status === 'success' ? 'success' : 'danger'">
                        {{ formatToolCallStatus(call.status) }}
                      </el-tag>
                    </div>
                    <div v-if="call.summary" class="tool-call-summary">{{ call.summary }}</div>
                    <pre class="json-block">{{ formatJson(call.arguments) }}</pre>
                    <pre v-if="call.error" class="json-block error-block">{{ formatJson(call.error) }}</pre>
                  </div>
                </div>
              </div>

              <div class="detail-block">
                <div class="section-label">warnings</div>
                <ul v-if="getResponseWarnings(item.response).length" class="bullet-list warning-list">
                  <li v-for="warning in getResponseWarnings(item.response)" :key="warning">
                    {{ warning }}
                  </li>
                </ul>
                <div v-else class="empty-state">暂无 warning</div>
              </div>

              <div class="detail-block agent-response-inline__papers">
                <div class="section-label">Papers</div>
                <div v-if="!getResponsePapers(getResponseData(item.response).papers).length" class="empty-state">
                  暂无论文结果
                </div>
                <div v-else class="paper-list">
                  <PaperCard
                    v-for="paper in getResponsePapers(getResponseData(item.response).papers)"
                    :key="paper.id"
                    :paper="paper"
                    @view-detail="handleViewDetail"
                  />
                </div>
              </div>
            </div>
          </div>
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

.agent-response-inline {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 16px;
  border-radius: 20px;
  background:
    radial-gradient(circle at top right, rgba(96, 165, 250, 0.08), transparent 28%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 251, 255, 0.98));
  border: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
}

.agent-response-inline__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.16);
}

.agent-response-inline__title {
  font-size: 18px;
  font-weight: 800;
  color: #0f172a;
}

.agent-response-inline__subtitle {
  margin-top: 6px;
  color: #64748b;
  font-size: 13px;
  line-height: 1.6;
}

.agent-response-inline__meta {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.agent-response-inline__answer {
  padding: 16px 18px;
  border-radius: 18px;
  background: linear-gradient(180deg, #ffffff, #f8fbff);
  border: 1px solid rgba(148, 163, 184, 0.14);
}

.agent-response-inline__chips {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.agent-response-inline__detail-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.summary-chip,
.detail-block {
  border-radius: 18px;
  background: linear-gradient(180deg, #ffffff, #f8fbff);
  border: 1px solid rgba(148, 163, 184, 0.14);
}

.summary-chip {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 6px;
  min-height: 74px;
  padding: 14px 16px;
}

.summary-chip__label,
.section-label,
.detail-section__label {
  font-size: 12px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.summary-chip strong {
  color: #0f172a;
  font-size: 15px;
}

.answer-text {
  margin-top: 10px;
  color: #1f2937;
  line-height: 1.85;
  font-size: 15px;
}

.detail-block {
  padding: 16px;
}

/* 论文结果需要占满整行，才能按当前容器宽度自适应列数。 */
.agent-response-inline__papers {
  grid-column: 1 / -1;
}

.detail-section + .detail-section {
  margin-top: 14px;
}

.detail-section__value {
  margin-top: 8px;
  font-size: 15px;
  color: #0f172a;
  word-break: break-word;
}

.bullet-list {
  margin: 10px 0 0;
  padding-left: 18px;
  color: #1f2937;
  line-height: 1.8;
}

.warning-list {
  color: #b45309;
}

.json-block {
  margin: 0;
  padding: 14px 16px;
  border-radius: 16px;
  background: #0f172a;
  color: #dbeafe;
  overflow: auto;
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.error-block {
  background: #3f1d1d;
  color: #fee2e2;
}

.tool-call-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.tool-call-item {
  padding: 14px;
  border-radius: 16px;
  background: linear-gradient(180deg, #ffffff, #f8fbff);
  border: 1px solid rgba(148, 163, 184, 0.14);
}

.tool-call-top {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  margin-bottom: 8px;
}

.tool-call-summary {
  margin-bottom: 10px;
  color: #374151;
  font-size: 14px;
}

.paper-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 16px;
  align-items: start;
  grid-auto-rows: auto;
}

.empty-state {
  padding: 18px 2px 6px;
  color: #64748b;
}

@media (max-width: 1200px) {
  .agent-response-inline__chips {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .agent-response-inline__detail-grid {
    grid-template-columns: 1fr;
  }

  .paper-list {
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  }
}

@media (max-width: 900px) {
  .hero-card {
    flex-direction: column;
  }

  .title {
    font-size: 26px;
  }

  .agent-response-inline {
    padding: 14px;
  }

  .agent-response-inline__head {
    flex-direction: column;
  }

  .agent-response-inline__chips,
  .paper-list,
  .agent-response-inline__papers {
    grid-column: auto;
  }

  .agent-response-inline__chips,
  .paper-list {
    grid-template-columns: 1fr;
  }
}
</style>
