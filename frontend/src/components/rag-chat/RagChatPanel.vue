<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, useSlots, watch } from 'vue'
import { BubbleList, Prompts, Thinking, Welcome, XSender } from 'vue-element-plus-x'
import type { PromptsItemsProps } from 'vue-element-plus-x/types/Prompts'
import type { RetrievalDebug } from '@/api/papers'
import type { AgentChatMessage } from '@/types/agentChat'
import type { QaTurnForRagChat, RagChatSource } from '@/types/ragChat'
import AuthenticatedImage from '@/components/AuthenticatedImage.vue'
import { getFigureCaption, EVIDENCE_LINK_PREFIX } from '@/utils/evidence'
import { renderAnswerSegments } from '@/utils/markdown'

type RagChatPanelMessage = AgentChatMessage & {
  placement: 'start' | 'end'
  variant: 'filled' | 'outlined'
  shape: 'round'
  maxWidth?: string
  turnId?: string
  sources?: RagChatSource[]
  citedSourceIds?: string[]
  retrievalDebug?: RetrievalDebug | null
  response?: QaTurnForRagChat | unknown | null
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
  messages: AgentChatMessage[]
  loading: boolean
  quickPrompts: string[]
  title?: string
  description?: string
  promptTitle?: string
  assistantLabel?: string
  senderPlaceholder?: string
  senderText?: string
  canStop?: boolean
}>()

const emit = defineEmits<{
  (event: 'submit-question', question: string): void
  (event: 'select-prompt', prompt: string): void
  (event: 'update:senderText', value: string): void
  (event: 'stop-generation'): void
  (event: 'select-source', item: RagChatPanelMessage, sourceId: string): void
}>()

defineSlots<{
  'message-footer'?(props: { item: RagChatPanelMessage }): any
  'message-progress'?(props: { item: RagChatPanelMessage }): any
}>()

// 父组件提供了进度插槽时，由它展示实时执行过程，面板不再叠加通用的“正在生成”占位。
const slots = useSlots()
const hasProgressSlot = computed(() => Boolean(slots['message-progress']))

const senderRef = ref<SenderExpose | null>(null)

function asQaTurnResponse(response: unknown): QaTurnForRagChat | null {
  // 同一个聊天面板也承载 Agent 响应，状态标签只读取 QA turn 的结构化字段。
  if (!response || typeof response !== 'object') return null
  return response as QaTurnForRagChat
}

const promptItems = computed<PromptsItemsProps[]>(() =>
  props.quickPrompts.map((prompt, index) => ({
    key: `${index}-${prompt}`,
    label: prompt,
    disabled: props.loading
  }))
)

const bubbleMessages = computed<RagChatPanelMessage[]>(() =>
  props.messages.map(message => ({
    ...message,
    placement: message.role === 'user' ? 'end' : 'start',
    variant: message.role === 'user' ? 'filled' : 'outlined',
    shape: 'round',
    maxWidth: message.role === 'user' ? 'min(82%, 820px)' : '100%'
  }))
)

const hasMessages = computed(() => bubbleMessages.value.length > 0)

// 矮屏下回答区按 100vh 扣掉固定区域后只剩一两行：已有对话时把快捷问题收成一行按钮，
// 回答区再保底一个可读高度（页面整体滚动即可看到输入框）。
const COMPACT_VIEWPORT_HEIGHT = 760
const viewportHeight = ref(typeof window === 'undefined' ? 900 : window.innerHeight)
const promptsExpanded = ref(false)
const compactPrompts = computed(() => hasMessages.value && viewportHeight.value < COMPACT_VIEWPORT_HEIGHT)
const bubbleListMaxHeight = computed(() => {
  if (!hasMessages.value) return '420px'
  return compactPrompts.value && !promptsExpanded.value ? 'max(calc(100vh - 300px), 300px)' : 'max(calc(100vh - 360px), 300px)'
})

function syncViewportHeight() {
  viewportHeight.value = window.innerHeight
}

onMounted(() => window.addEventListener('resize', syncViewportHeight))
onBeforeUnmount(() => window.removeEventListener('resize', syncViewportHeight))

watch(
  () => props.senderText,
  value => {
    const normalizedValue = value || ''
    const currentValue = senderRef.value?.getModelValue().text ?? ''
    if (currentValue !== normalizedValue) {
      senderRef.value?.setText(normalizedValue)
    }
  },
  { immediate: true }
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

function syncSenderText() {
  const text = senderRef.value?.getModelValue().text ?? ''
  emit('update:senderText', text)
}

function handleSubmit() {
  const question = senderRef.value?.getModelValue().text.trim() || ''
  if (!question || props.loading) return
  emit('submit-question', question)
  senderRef.value?.clear()
  emit('update:senderText', '')
}

function handlePromptSelect(item: PromptsItemsProps) {
  const prompt = item.label?.trim() || ''
  if (!prompt || props.loading) return
  senderRef.value?.setText(prompt)
  emit('update:senderText', prompt)
  emit('select-prompt', prompt)
  promptsExpanded.value = false
}

function getAssistantStatusLabel(item: RagChatPanelMessage) {
  const response = asQaTurnResponse(item.response)
  const status = response?.status
  const persistenceStatus = response?.persistenceStatus
  if (item.loading) return '正在生成'
  if (status === 'completed') return persistenceStatus === 'saved' ? '已保存' : '已完成'
  if (status === 'aborted') return '已取消'
  if (status === 'partial' || status === 'interrupted') return '未完成'
  if (status === 'persistence_failed') return '未保存'
  if (status === 'failed') return '失败'
  return ''
}

function getAssistantStatusType(item: RagChatPanelMessage) {
  const status = asQaTurnResponse(item.response)?.status
  if (item.loading) return 'primary'
  if (status === 'completed') return 'success'
  if (status === 'aborted') return 'info'
  if (status === 'partial' || status === 'interrupted' || status === 'persistence_failed') return 'warning'
  if (status === 'failed') return 'danger'
  return 'info'
}

function handleAssistantContentClick(event: MouseEvent, item: RagChatPanelMessage) {
  const target = event.target instanceof Element ? event.target.closest('a[href^="#evidence-source-"]') : null
  if (!(target instanceof HTMLAnchorElement)) return
  event.preventDefault()
  const sourceId = decodeURIComponent(target.getAttribute('href')?.slice(EVIDENCE_LINK_PREFIX.length) || '')
  if (sourceId) emit('select-source', item, sourceId)
}
</script>

<template>
  <div class="rag-chat-panel">
    <div class="rag-chat-panel__messages">
      <Welcome
        v-if="bubbleMessages.length === 0 && !loading"
        class="rag-chat-panel__welcome"
        variant="borderless"
        :title="title || '开始提问'"
        :description="description || '使用自然语言输入你的搜索需求，系统会在对话中返回结果。'"
      />

      <button
        v-if="promptItems.length && compactPrompts"
        type="button"
        class="rag-chat-panel__prompts-toggle"
        :aria-expanded="promptsExpanded"
        @click="promptsExpanded = !promptsExpanded"
      >
        {{ promptTitle || '快捷提示' }}
        <span class="rag-chat-panel__prompts-chevron" :class="{ 'rag-chat-panel__prompts-chevron--open': promptsExpanded }" />
      </button>

      <Prompts
        v-if="promptItems.length && (!compactPrompts || promptsExpanded)"
        class="rag-chat-panel__prompts"
        :title="compactPrompts ? undefined : promptTitle || '快捷提示'"
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
        :max-height="bubbleListMaxHeight"
      >
        <template #header="{ item }">
          <div class="rag-chat-panel__message-label">
            {{ item.role === 'user' ? '用户' : assistantLabel || 'Assistant' }}
          </div>
        </template>

        <!-- Bubble 在 loading 时会用 loading 插槽整体替换 content；有进度插槽时由它展示实时执行过程。 -->
        <template v-if="hasProgressSlot" #loading="{ item }">
          <div class="rag-chat-panel__progress-loading">
            <slot name="message-progress" :item="item" />
          </div>
        </template>

        <template #content="{ item }">
          <slot v-if="item.role === 'assistant' && hasProgressSlot" name="message-progress" :item="item" />
          <template v-if="item.role === 'assistant' && item.loading && !item.content">
            <Thinking
              v-if="!hasProgressSlot"
              class="rag-chat-panel__thinking"
              :model-value="true"
              status="thinking"
              content="正在生成回复"
              :auto-collapse="false"
            />
          </template>
          <div
            v-else-if="item.role === 'assistant'"
            class="assistant-markdown"
            @click="handleAssistantContentClick($event, item)"
          >
            <template
              v-for="(segment, index) in renderAnswerSegments(item.content, undefined, { sources: item.sources, citedSourceIds: item.citedSourceIds })"
              :key="index"
            >
              <div class="assistant-markdown__segment" v-html="segment.html" />
              <!-- 图卡跟在第一次引用这张图的段落后面；图注里的角标与正文共用同一套证据定位事件。 -->
              <figure v-for="figure in segment.figures" :key="figure.source.source_id" class="answer-figure">
                <AuthenticatedImage class="answer-figure__image" :src="figure.source.asset_url || ''" fit="contain">
                  <template #error><div class="answer-figure__missing">图片暂不可用，可点击角标查看文字证据。</div></template>
                </AuthenticatedImage>
                <figcaption class="answer-figure__caption">
                  <a
                    :href="`${EVIDENCE_LINK_PREFIX}${encodeURIComponent(figure.source.source_id)}`"
                    class="answer-figure__ref"
                    title="查看这条证据"
                  >[{{ figure.label }}]</a>
                  <span v-if="figure.source.page_number" class="answer-figure__page">第 {{ figure.source.page_number }} 页</span>
                  <span class="answer-figure__text">{{ getFigureCaption(figure.source) }}</span>
                </figcaption>
              </figure>
            </template>
          </div>
          <div v-else class="rag-chat-panel__user-content">
            {{ item.content }}
          </div>
        </template>

        <template #footer="{ item }">
          <div class="rag-chat-panel__message-footer">
            <div v-if="item.role === 'assistant' && item.loading && !hasProgressSlot" class="streaming-indicator">
              <span class="rag-chat-panel__stream-dot" />
              <span>正在生成中...</span>
            </div>

            <el-tag
              v-if="item.role === 'assistant' && getAssistantStatusLabel(item) && !(item.loading && hasProgressSlot)"
              class="answer-state-tag"
              size="small"
              effect="plain"
              :type="getAssistantStatusType(item)"
            >
              {{ getAssistantStatusLabel(item) }}
            </el-tag>

            <slot name="message-footer" :item="item" />

            <div class="answer-meta">
              <span>{{ formatCreatedAt(item.createdAt) }}</span>
              <span v-if="item.error" class="answer-meta__error">{{ item.error }}</span>
            </div>
          </div>
        </template>
      </BubbleList>
    </div>

    <div class="rag-chat-panel__sender">
      <XSender
        ref="senderRef"
        :placeholder="senderPlaceholder || '输入你的自然语言需求，Enter 发送，Shift+Enter 换行'"
        submit-type="enter"
        :loading="loading"
        :disabled="false"
        :clearable="true"
        @change="syncSenderText"
        @submit="handleSubmit"
      >
        <template #footer>
          <div class="rag-chat-panel__sender-tip">
            <el-button
              v-if="canStop && loading"
              class="rag-chat-panel__stop-button"
              size="small"
              text
              type="danger"
              @click="emit('stop-generation')"
            >
              <el-icon><VideoPause /></el-icon>
              停止生成
            </el-button>
            先做语义解析，再生成结构化结果，适合 Agent 搜索场景
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
}

.rag-chat-panel__messages {
  display: flex;
  flex: 1 1 auto;
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

.rag-chat-panel__prompts-toggle {
  display: inline-flex;
  align-items: center;
  align-self: flex-start;
  gap: 8px;
  padding: 4px 10px;
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.9);
  color: #64748b;
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
}

.rag-chat-panel__prompts-toggle:hover {
  color: #2563eb;
  border-color: rgba(37, 99, 235, 0.3);
}

.rag-chat-panel__prompts-chevron {
  width: 6px;
  height: 6px;
  border-right: 1.5px solid currentColor;
  border-bottom: 1.5px solid currentColor;
  transform: translateY(-2px) rotate(45deg);
  transition: transform 0.2s;
}

.rag-chat-panel__prompts-chevron--open {
  transform: translateY(1px) rotate(-135deg);
}

.rag-chat-panel__bubble-list {
  flex: 1 1 auto;
  min-height: 0;
}

.rag-chat-panel__bubble-list :deep(.message-start),
.rag-chat-panel__bubble-list :deep(.message-assistant),
.rag-chat-panel__bubble-list :deep([data-placement='start']),
.rag-chat-panel__bubble-list :deep([class*='placement-start']) {
  width: 100%;
}

.rag-chat-panel__bubble-list :deep(.message-start .message-bubble),
.rag-chat-panel__bubble-list :deep(.message-assistant .message-bubble),
.rag-chat-panel__bubble-list :deep([data-placement='start'] .message-bubble),
.rag-chat-panel__bubble-list :deep([class*='placement-start'] .message-bubble),
.rag-chat-panel__bubble-list :deep([data-placement='start'] [class*='bubble']),
.rag-chat-panel__bubble-list :deep([class*='placement-start'] [class*='bubble']) {
  max-width: 100% !important;
  width: 100%;
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

.rag-chat-panel__thinking {
  width: 100%;
}

.assistant-markdown {
  width: 100%;
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

/* 有图片证据时正文按段切开；段内末尾元素的外边距由段间距接管，避免切段后行距变化。 */
.assistant-markdown__segment + .assistant-markdown__segment,
.answer-figure + .assistant-markdown__segment {
  margin-top: 14px;
}

.assistant-markdown__segment > :deep(:last-child) {
  margin-bottom: 0;
}

.answer-figure {
  margin: 14px 0 0;
  padding: 10px;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  background: #f8fafc;
}

.answer-figure__image {
  display: block;
  width: 100%;
  border-radius: 10px;
  background: #ffffff;
  overflow: hidden;
}

.answer-figure__image :deep(img) {
  display: block;
  width: 100%;
  max-height: 360px;
  object-fit: contain;
  cursor: zoom-in;
}

.answer-figure__missing {
  padding: 18px 12px;
  color: #94a3b8;
  font-size: 12px;
  text-align: center;
}

.answer-figure__caption {
  display: flex;
  align-items: baseline;
  gap: 6px;
  margin-top: 8px;
  color: #475569;
  font-size: 12px;
  line-height: 1.6;
}

.assistant-markdown .answer-figure__ref {
  flex: none;
  vertical-align: baseline;
}

.answer-figure__page {
  flex: none;
  color: #94a3b8;
}

.answer-figure__text {
  min-width: 0;
  overflow-wrap: anywhere;
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

/* 引用角标：编号小圆角块，连续引用之间留间距，不再挤成一串链接文字。 */
.assistant-markdown :deep(a[href^="#evidence-source-"]) {
  display: inline-block;
  min-width: 1.6em;
  margin: 0 1px 0 2px;
  padding: 0 4px;
  border: none;
  border-radius: 6px;
  background: rgba(37, 99, 235, 0.1);
  color: #2563eb;
  font-size: 11px;
  font-weight: 700;
  line-height: 1.6;
  text-align: center;
  vertical-align: super;
  cursor: pointer;
  transition: background-color 0.15s, color 0.15s;
}

.assistant-markdown :deep(a[href^="#evidence-source-"]:hover) {
  background: #2563eb;
  color: #fff;
}

.assistant-markdown :deep(strong) {
  color: #0f172a;
  font-weight: 800;
}

.assistant-markdown :deep(em) {
  color: #0f172a;
  font-style: italic;
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

.answer-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 12px;
  color: #94a3b8;
}

.answer-meta__error {
  color: #dc2626;
}

.answer-state-tag {
  width: fit-content;
}

.rag-chat-panel__sender {
  flex: none;
}

.rag-chat-panel__sender-tip {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-top: 6px;
  font-size: 12px;
  color: #64748b;
}

.rag-chat-panel__stop-button {
  flex: none;
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
  .rag-chat-panel__bubble-list :deep(.message-start .message-bubble),
  .rag-chat-panel__bubble-list :deep(.message-assistant .message-bubble),
  .rag-chat-panel__bubble-list :deep([data-placement='start'] .message-bubble),
  .rag-chat-panel__bubble-list :deep([class*='placement-start'] .message-bubble),
  .rag-chat-panel__bubble-list :deep([data-placement='start'] [class*='bubble']),
  .rag-chat-panel__bubble-list :deep([class*='placement-start'] [class*='bubble']) {
    max-width: 100% !important;
    width: 100%;
  }

  .rag-chat-panel__message-footer {
    gap: 10px;
  }
}

/* Bubble 的 loading 容器默认 fit-content 且居中排列三个点；放进度时间线时要撑满气泡宽度。 */
:deep(.elx-bubble__content--loading:has(.rag-chat-panel__progress-loading)) {
  width: 100%;
}

:deep(.elx-bubble__loading-wrap:has(.rag-chat-panel__progress-loading)) {
  display: block;
}

.rag-chat-panel__progress-loading {
  width: 100%;
}
</style>
