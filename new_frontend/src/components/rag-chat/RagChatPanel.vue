<script setup lang="ts">
import { computed, ref } from 'vue'
import { BubbleList, Prompts, Welcome, XSender } from 'vue-element-plus-x'
import type { PromptsItemsProps } from 'vue-element-plus-x/types/Prompts'
import type { RagChatMessage } from '@/types/ragChat'
import { renderMarkdownWithLatex } from '@/utils/markdown'

type RagChatPanelMessage = RagChatMessage & {
  placement: 'start' | 'end'
  variant: 'filled' | 'outlined'
  shape: 'round'
}

interface SenderModelValue {
  html: string
  text: string
}

interface SenderExpose {
  clear: () => void
  getModelValue: () => SenderModelValue
  setText: (text: string) => void
}

const props = defineProps<{
  messages: RagChatMessage[]
  loading: boolean
  quickPrompts: string[]
  paperTitle: string
  paperId: string
}>()

const emit = defineEmits<{
  (event: 'submit-question', question: string): void
  (event: 'select-prompt', prompt: string): void
  (event: 'open-evidence', turnId: string): void
}>()

const senderRef = ref<SenderExpose | null>(null)

const promptItems = computed<PromptsItemsProps[]>(() =>
  props.quickPrompts.map((prompt, index) => ({
    key: `${props.paperId}-${index}`,
    label: prompt,
    disabled: props.loading
  }))
)

const bubbleMessages = computed<RagChatPanelMessage[]>(() =>
  props.messages.map(message => ({
    ...message,
    placement: message.role === 'user' ? 'end' : 'start',
    variant: message.role === 'user' ? 'filled' : 'outlined',
    shape: 'round'
  }))
)

function formatCreatedAt(value: string) {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return value
  }
  return parsed.toLocaleString('zh-CN', {
    hour12: false,
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

function handleSubmit() {
  const question = senderRef.value?.getModelValue().text.trim() || ''
  if (!question || props.loading) return
  emit('submit-question', question)
  senderRef.value?.clear()
}

function handlePromptSelect(item: PromptsItemsProps) {
  const prompt = item.label?.trim() || ''
  if (!prompt || props.loading) return
  senderRef.value?.setText(prompt)
  emit('select-prompt', prompt)
}

function handleOpenEvidence(turnId: string) {
  emit('open-evidence', turnId)
}
</script>

<template>
  <div class="rag-chat-panel">
    <div class="rag-chat-panel__messages">
      <Welcome
        v-if="bubbleMessages.length === 0 && !loading"
        class="rag-chat-panel__welcome"
        variant="borderless"
        title="开始向论文提问"
        :description="`围绕《${paperTitle || paperId}》提问，系统会先召回相关 chunk，再生成答案。`"
      />

      <Prompts
        v-if="promptItems.length"
        class="rag-chat-panel__prompts"
        title="快捷问题"
        :items="promptItems"
        :wrap="true"
        @itemClick="handlePromptSelect"
      />

      <BubbleList
        class="rag-chat-panel__bubble-list"
        :list="bubbleMessages"
        :auto-scroll="true"
        :always-show-scrollbar="true"
        :show-back-button="true"
        :item-key="'id'"
        :max-height="'100%'"
      >
        <template #header="{ item }">
          <div class="rag-chat-panel__message-label">
            {{ item.role === 'user' ? '你' : 'Qwen' }}
          </div>
        </template>

        <template #content="{ item }">
          <div v-if="item.role === 'assistant'" class="assistant-markdown" v-html="renderMarkdownWithLatex(item.content)" />
          <div v-else class="rag-chat-panel__user-content">
            {{ item.content }}
          </div>
        </template>

        <template #footer="{ item }">
          <div class="rag-chat-panel__message-footer">
            <div v-if="item.role === 'assistant' && item.loading" class="streaming-indicator">
              <span class="rag-chat-panel__stream-dot" />
              <span>正在流式生成中...</span>
            </div>

            <div
              v-if="item.role === 'assistant' && item.loading && (item.sources.length || item.retrievalDebug)"
              class="streaming-evidence-note"
            >
              召回内容和调试信息将在回答完成后进入右侧抽屉
            </div>

            <div v-if="item.role === 'assistant'" class="assistant-actions">
              <div class="assistant-badges">
                <el-tag size="small" effect="plain" type="info">
                  {{ item.sources.length }} 条来源
                </el-tag>
                <el-tag v-if="item.retrievalDebug" size="small" effect="plain" type="warning">
                  Debug 可查看
                </el-tag>
              </div>
              <el-button
                v-if="item.sources.length || item.retrievalDebug"
                size="small"
                text
                type="primary"
                class="evidence-button"
                @click="handleOpenEvidence(item.turnId)"
              >
                查看来源与调试
              </el-button>
            </div>

            <div class="answer-meta">
              <span>{{ formatCreatedAt(item.createdAt) }}</span>
            </div>
          </div>
        </template>
      </BubbleList>
    </div>

    <div class="rag-chat-panel__sender">
      <XSender
        ref="senderRef"
        placeholder="输入你的问题，按 Enter 发送，Shift+Enter 换行"
        submit-type="enter"
        :loading="loading"
        :disabled="false"
        :clearable="true"
        @submit="handleSubmit"
      >
        <template #footer>
          <div class="rag-chat-panel__sender-tip">
            先做语义检索，再由模型整合回答，适合论文问答场景
          </div>
        </template>
      </XSender>
    </div>
  </div>
</template>

<style scoped>
.rag-chat-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
  height: 100%;
}

.rag-chat-panel__messages {
  display: flex;
  flex: 1;
  min-height: 0;
  flex-direction: column;
  gap: 14px;
}

.rag-chat-panel__welcome {
  padding: 18px 20px;
  border-radius: 24px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(248, 250, 252, 0.94));
  border: 1px solid rgba(148, 163, 184, 0.18);
}

.rag-chat-panel__prompts {
  flex: none;
}

.rag-chat-panel__bubble-list {
  flex: 1;
  min-height: 0;
}

.rag-chat-panel__message-label {
  margin-bottom: 8px;
  font-size: 12px;
  font-weight: 700;
  color: #64748b;
}

.rag-chat-panel__user-content {
  white-space: pre-wrap;
  line-height: 1.75;
}

.rag-chat-panel__message-footer {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 12px;
}

.assistant-markdown {
  max-width: 100%;
  color: #1e293b;
  line-height: 1.85;
  font-size: 15px;
  letter-spacing: 0.01em;
  overflow-wrap: anywhere;
}

.assistant-markdown :deep(p) {
  margin: 0 0 14px;
}

.assistant-markdown :deep(p:last-child) {
  margin-bottom: 0;
}

.assistant-markdown :deep(.md-paragraph) {
  white-space: pre-wrap;
}

.assistant-markdown :deep(.md-empty) {
  margin: 0;
  color: #64748b;
}

.assistant-markdown :deep(.katex-display) {
  margin: 0.8em 0;
  overflow-x: auto;
}

.assistant-markdown :deep(table) {
  display: block;
  width: 100%;
  margin: 14px 0;
  border-collapse: collapse;
  border-spacing: 0;
  overflow-x: auto;
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 16px;
  background: #ffffff;
}

.assistant-markdown :deep(th),
.assistant-markdown :deep(td) {
  padding: 10px 12px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.16);
  border-right: 1px solid rgba(148, 163, 184, 0.12);
  text-align: left;
  vertical-align: top;
}

.assistant-markdown :deep(th) {
  background: #f8fafc;
  color: #0f172a;
  font-weight: 700;
}

.assistant-markdown :deep(tr:last-child td) {
  border-bottom: none;
}

.assistant-markdown :deep(h1),
.assistant-markdown :deep(h2),
.assistant-markdown :deep(h3),
.assistant-markdown :deep(h4),
.assistant-markdown :deep(h5),
.assistant-markdown :deep(h6) {
  margin: 22px 0 10px;
  line-height: 1.35;
  color: #0f172a;
  font-weight: 800;
  letter-spacing: -0.02em;
}

.assistant-markdown :deep(h1) {
  font-size: 24px;
}

.assistant-markdown :deep(h2) {
  font-size: 20px;
}

.assistant-markdown :deep(h3) {
  font-size: 18px;
}

.assistant-markdown :deep(h4),
.assistant-markdown :deep(h5),
.assistant-markdown :deep(h6) {
  font-size: 16px;
}

.assistant-markdown :deep(blockquote) {
  margin: 14px 0;
  padding: 12px 14px;
  border-left: 4px solid rgba(59, 130, 246, 0.4);
  border-radius: 0 14px 14px 0;
  background: rgba(59, 130, 246, 0.06);
  color: #334155;
}

.assistant-markdown :deep(ul),
.assistant-markdown :deep(ol) {
  margin: 10px 0 14px;
  padding-left: 1.35em;
}

.assistant-markdown :deep(li) {
  margin: 6px 0;
}

.assistant-markdown :deep(code) {
  padding: 0.18em 0.48em;
  border-radius: 8px;
  background: rgba(37, 99, 235, 0.08);
  color: #0f172a;
  font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
  font-size: 0.92em;
}

.assistant-markdown :deep(pre) {
  margin: 14px 0;
  padding: 16px 18px;
  border-radius: 18px;
  background: linear-gradient(180deg, #0f172a, #111827);
  color: #e2e8f0;
  overflow-x: auto;
  box-shadow:
    inset 0 0 0 1px rgba(148, 163, 184, 0.08),
    0 12px 28px rgba(15, 23, 42, 0.12);
}

.assistant-markdown :deep(pre code) {
  display: block;
  padding: 0;
  background: transparent;
  color: inherit;
  white-space: pre;
  font-size: 13px;
  line-height: 1.7;
}

.assistant-markdown :deep(a) {
  color: #2563eb;
  text-decoration: none;
  border-bottom: 1px solid rgba(37, 99, 235, 0.22);
}

.assistant-markdown :deep(a:hover) {
  border-bottom-color: rgba(37, 99, 235, 0.65);
}

.assistant-markdown :deep(strong) {
  color: #0f172a;
  font-weight: 800;
}

.assistant-markdown :deep(em) {
  color: #0f172a;
  font-style: italic;
}

.assistant-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(241, 245, 249, 0.96));
  border: 1px solid rgba(148, 163, 184, 0.16);
}

.assistant-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.evidence-button {
  padding-inline: 10px;
}

.streaming-indicator {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #64748b;
}

.rag-chat-panel__stream-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #2563eb;
  box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.35);
  animation: rag-chat-panel-pulse 1.4s infinite;
}

.streaming-evidence-note {
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(59, 130, 246, 0.08);
  color: #2563eb;
  font-size: 12px;
  line-height: 1.6;
}

.answer-meta {
  font-size: 12px;
  color: #94a3b8;
}

.rag-chat-panel__sender {
  flex: none;
}

.rag-chat-panel__sender-tip {
  padding-top: 6px;
  font-size: 12px;
  color: #64748b;
}

@keyframes rag-chat-panel-pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.35);
  }

  70% {
    box-shadow: 0 0 0 10px rgba(37, 99, 235, 0);
  }

  100% {
    box-shadow: 0 0 0 0 rgba(37, 99, 235, 0);
  }
}

@media (max-width: 900px) {
  .assistant-actions {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
